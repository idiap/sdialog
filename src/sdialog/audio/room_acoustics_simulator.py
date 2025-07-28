import os
from typing import List, Optional, Union
import numpy as np

# import matplotlib.pyplot as plt
import soundfile as sf
import pyroomacoustics as pra
from sdialog.audio.room import Room, RoomRole, AudioSource, Position3D
from sdialog.audio.room import (
    DoctorPosition,
    PatientPosition,
    SoundEventPosition,
    RecordingDevice,
    MicrophonePosition,
)
from sdialog.audio.room_generator import calculate_room_dimensions, ROOM_SIZES


class RoomAcousticsSimulator:
    """
    Simulates sound based on room acoustics based on room definition,
    sound sources provided and microphone(s) setup.

     Example:
         >>> from sdialog.audio.room_generator import RoomGenerator
         >>> from sdialog.audio.room import MicrophonePosition
         >>>
         >>> # Create room with specific microphone position
         >>> room = RoomGenerator().generate(RoomRole.CONSULTATION)
         >>> room_acoustics = RoomAcousticsSimulator(room)
         >>>
         >>> # Change microphone position using enum
         >>> room_acoustics.set_microphone_position(MicrophonePosition.CEILING_CENTERED)
         >>> # Or use explicit coordinates
         >>> room_acoustics.set_microphone_position([2.0, 1.5, 1.8])
         >>>
         >>> # Add audio sources and simulate
         >>> audio = room_acoustics.simulate(audio_sources)
    """

    def __init__(self, room: Optional[Room] = None, sampling_rate=16000):
        self.sampling_rate = sampling_rate
        self.ref_db = -65  # - 45 dB
        self.audiosources: List[AudioSource] = []

        if room is None:
            self.room = Room(
                role=RoomRole.CONSULTATION,
                name="consultation_room_default",
                dimensions=calculate_room_dimensions(ROOM_SIZES[3]),
                rt60=0.5,
                soundsources_position=[
                    DoctorPosition.AT_DESK_SITTING,
                    PatientPosition.NEXT_TO_DESK_SITTING,
                ],
                mic_type=RecordingDevice.WEBCAM,
                mic_position=MicrophonePosition.MONITOR,
                furnitures=False,
            )
        else:
            self.room = room

        self._pyroom = self._create_pyroom(self.room, self.sampling_rate)
        # Set microphone position based on room's mic_position setting
        self.mic_position = self.microphone_position_to_room_position(
            self.room, self.room.mic_position
        )
        self.add_microphone(self.mic_position.to_list())

    def _create_pyroom(self, room: Room, sampling_rate=16000):
        e_absorption, max_order = pra.inverse_sabine(room.rt60, room.dimensions)
        # max_order = 17  # Number of reflections
        return pra.ShoeBox(
            room.dimensions,
            fs=sampling_rate,
            materials=pra.Material(e_absorption),
            max_order=max_order,
        )

    # room_acoustics.add_microphone( .. )
    def add_microphone(self, mic_position):
        """Add microphone to the room"""
        self.set_microphone_position(mic_position)

    def set_microphone_position(
        self, mic_pos: Union[MicrophonePosition, List[float], Position3D, str]
    ):
        """
        Set microphone position using MicrophonePosition enum or explicit coordinates.

        Args:
            mic_pos: Can be MicrophonePosition enum, list [x,y,z], or Position3D object
        """
        if isinstance(mic_pos, MicrophonePosition):
            position_3d = self.microphone_position_to_room_position(self.room, mic_pos)
        elif isinstance(mic_pos, list):
            position_3d = Position3D.from_list(mic_pos)
        elif isinstance(mic_pos, Position3D):
            position_3d = mic_pos
        elif isinstance(mic_pos, str):
            mic_pos = MicrophonePosition(mic_pos)
            position_3d = self.microphone_position_to_room_position(self.room, mic_pos)
        else:
            raise ValueError(
                "mic_pos must be MicrophonePosition enum, list [x,y,z], or Position3D object"
            )

        self.mic_position = position_3d

        # Remove existing microphone and add new one
        if hasattr(self._pyroom, "mic_array") and self._pyroom.mic_array is not None:
            # Clear existing microphone array
            self._pyroom.mic_array = None

        # Add microphone at new position
        mic_array = pra.MicrophoneArray(
            np.array([self.mic_position.to_list()]).T, self._pyroom.fs
        )
        self._pyroom.add_microphone_array(mic_array)
        print(f"  Microphone set to position {self.mic_position.to_list()}")

    def _add_sources(self, audiosources: List[AudioSource]):
        for i, asource in enumerate(audiosources):
            self.audiosources.append(asource)

            position = self.parse_position(asource.position)
            if position is not SoundEventPosition:
                asource._position3d = self.position_to_room_position(
                    self.room, position
                )
            else:
                room_center = [dim / 2 for dim in self._pyroom.dimensions]
                asource._position3d = Position3D(room_center)

            audio = None
            if hasattr(asource, "_test_audio") and asource._test_audio is not None:
                audio = asource._test_audio  # Use in-memory test audio data
                print(
                    f"✓ Using in-memory audio for '{asource.name}' with {len(audio)} samples"
                )
            elif asource.source_file and os.path.exists(asource.source_file):
                audio, original_fs = sf.read(asource.source_file)
                if audio.ndim > 1:  # Convert to mono if stereo
                    audio = np.mean(audio, axis=1)
                print(
                    f"✓ Loaded audio file '{asource.source_file}' for '{asource.name}' with {len(audio)} samples"
                )
            else:
                print(
                    (
                        f"Warning: No audio data found for '{asource.name}' ",
                        "- file '{asource.source_file}' not found and no test data available.",
                    )
                )
                continue

            if audio is not None:
                print(f"Sound source {asource.name} Peak level: {np.max(np.abs(audio)):.3f}")
                audio = self.apply_snr(audio, asource.snr)
                print(f"Sound source {asource.name} Peak level after: {np.max(np.abs(audio)):.3f}")
                self._pyroom.add_source(asource._position3d.to_list(), signal=audio)

    def simulate(self, sources: List[AudioSource] = [], reset=False):  # -> np.array:
        if reset:
            # see https://github.com/LCAV/pyroomacoustics/issues/311
            self.reset()
            self._pyroom = self._create_pyroom(self.room, self.sampling_rate)

        self._add_sources(sources)
        self._pyroom.simulate()  # snr=
        mixed_signal = self._pyroom.mic_array.signals[0, :]

        # peak_level = np.max(np.abs(mixed_signal))
        # if peak_level > 0.95:
        #     # Soft compression to prevent harsh clipping
        #     compression_ratio = 0.95 / peak_level
        #     mixed_signal = mixed_signal * compression_ratio
        #     print(f"Applied soft compression (ratio: {compression_ratio:.3f}) to prevent clipping")
        # print(f"Simulation complete! Peak level: {np.max(np.abs(mixed_signal)):.3f}")

        mixed_signal = self.apply_snr(mixed_signal, -0.03)  # scale audio to max -0.3dB
        return mixed_signal

    def reset(self):
        del self._pyroom
        self._pyroom = None

    # def plot_room_setup(self):
    #     """Visualize the room setup"""
    #     self._pyroom.compute_rir()
    #     self._pyroom.plot_rir()
    #     self._pyroom.plot()
    #     return

    @staticmethod
    def plot_energy_db(ax, rir, fs=24000):
        """The power of the impulse response in dB"""
        power = rir**2
        energy = np.cumsum(power[::-1])[::-1]  # Integration according to Schroeder
        # remove the possibly all zero tail
        i_nz = np.max(np.where(energy > 0)[0])
        energy = energy[:i_nz]
        energy_db = 10 * np.log10(energy)
        energy_db -= energy_db[0]
        ax.plot(energy_db)

    @staticmethod
    def dbfs_to_linear(dbfs):
        return 10 ** (dbfs / 20)

    @staticmethod
    def apply_snr(x, snr):
        """Scale an audio signal to a given maximum SNR."""
        dbfs = 10 ** (snr / 20)
        x *= dbfs / np.abs(x).max(initial=1e-15)
        return x

    @staticmethod
    def parse_position(
        position: str,
    ) -> Union[DoctorPosition, PatientPosition, SoundEventPosition]:
        """
        Convert a position string to the appropriate position enum.
        """
        if position.startswith("doctor-"):
            try:
                return DoctorPosition(position)
            except ValueError:
                raise ValueError(f"Invalid doctor position: {position}")
        elif position.startswith("patient-"):
            try:
                return PatientPosition(position)
            except ValueError:
                raise ValueError(f"Invalid patient position: {position}")
        elif position.startswith("soundevent-"):
            return SoundEventPosition(position)
        elif position.startswith(
            SoundEventPosition.BACKGROUND.value
        ):  # no_type - background
            return SoundEventPosition(position)
        else:
            raise ValueError(
                f"Position must start with 'doctor-' or 'patient-', got: {position}"
            )

    @staticmethod
    def position_to_room_position(
        room: Room, pos: Union[DoctorPosition, PatientPosition]
    ) -> Position3D:
        """
        Convert semantic position enums to actual 3D coordinates within the room.

        This function maps abstract position descriptions (like "doctor:at_desk_sitting")
        to concrete 3D coordinates that can be used for acoustic simulation.

        Args:
            room: Room object containing dimensions and layout information
            pos: Position enum (DoctorPosition or PatientPosition)

        Returns:
            Position3D: 3D coordinates (x, y, z) in meters within the room

        Standard room layout assumptions as define in enums DoctorPosition or PatientPosition :
        - Door at (0, 0) corner
        - Desk along the width wall at 1/4 from door
        - Examination bench in center area
        - Sink and cupboards along length walls
        - Standard sitting height: 0.5m, standing height: 1.7m

        Example:
            >>> from sdialog.audio.room import Room, RoomRole, Dimensions3D, DoctorPosition
            >>> room = Room(role=RoomRole.CONSULTATION,
            ...              dimensions=Dimensions3D(4.0, 3.0, 3.0))
            >>> pos = DoctorPosition.AT_DESK_SITTING
            >>> coord = RoomAcousticsSimulator.position_to_room_position(room, pos)
            >>> print(f"Doctor position: ({coord.x:.1f}, {coord.y:.1f}, {coord.z:.1f})")
            Doctor position: (1.0, 0.4, 0.5)

        Supported positions:
        Doctor positions:
        - AT_DESK_SITTING: Seated at desk
        - AT_DESK_SIDE_STANDING: Standing beside desk
        - NEXT_TO_BENCH_STANDING: Standing next to examination bench
        - NEXT_TO_SINK_FRONT/BACK: Near sink area
        - NEXT_TO_CUPBOARD_FRONT/BACK: Near cupboard area
        - NEXT_TO_DOOR_STANDING: Standing near entrance

        Patient positions:
        - AT_DOOR_STANDING: Standing at entrance
        - NEXT_TO_DESK_SITTING/STANDING: Near desk area
        - SITTING_ON_BENCH: On examination bench
        - CENTER_ROOM_STANDING: Middle of room
        """
        width, length, height = (
            room.dimensions.width,
            room.dimensions.length,
            room.dimensions.height,
        )

        # Define standard furniture positions as fractions of room dimensions
        desk_pos = (width * 0.25, length * 0.15)  # Near corner, away from door
        bench_pos = (width * 0.6, length * 0.5)  # Center-right area
        door_pos = (0.1, 0.1)  # Near corner
        sink_pos = (width * 0.05, length * 0.8)  # Back wall, near corner
        cupboard_pos = (width * 0.95, length * 0.8)  # Back wall, opposite corner
        center_pos = (width * 0.5, length * 0.5)  # Room center

        # Heights for different postures
        sitting_height = 0.5  # Chair/bench sitting height
        standing_height = 1.7  # Average person standing height

        def clamp_position(x, y, z):
            """Ensure position is within room bounds with safety margin"""
            margin = 0.1  # 10cm safety margin from walls
            x = max(margin, min(x, width - margin))
            y = max(margin, min(y, length - margin))
            z = max(0.1, min(z, height - 0.1))
            return Position3D.from_list([x, y, z])

        # Map doctor positions
        if isinstance(pos, DoctorPosition):
            if pos == DoctorPosition.AT_DESK_SITTING:
                return clamp_position(desk_pos[0], desk_pos[1], sitting_height)
            elif pos == DoctorPosition.AT_DESK_SIDE_STANDING:
                return clamp_position(desk_pos[0] + 0.5, desk_pos[1], standing_height)
            elif pos == DoctorPosition.NEXT_TO_BENCH_STANDING:
                return clamp_position(bench_pos[0] - 0.8, bench_pos[1], standing_height)
            elif pos == DoctorPosition.NEXT_TO_SINK_FRONT:
                return clamp_position(
                    sink_pos[0] + 0.3, sink_pos[1] - 0.5, standing_height
                )
            elif pos == DoctorPosition.NEXT_TO_SINK_BACK:
                return clamp_position(
                    sink_pos[0] - 0.3, sink_pos[1] + 0.3, standing_height
                )
            elif pos == DoctorPosition.NEXT_TO_CUPBOARD_FRONT:
                return clamp_position(
                    cupboard_pos[0] - 0.3, cupboard_pos[1] - 0.5, standing_height
                )
            elif pos == DoctorPosition.NEXT_TO_CUPBOARD_BACK:
                return clamp_position(
                    cupboard_pos[0] + 0.3, cupboard_pos[1] + 0.3, standing_height
                )
            elif pos == DoctorPosition.NEXT_TO_DOOR_STANDING:
                return clamp_position(
                    door_pos[0] + 0.5, door_pos[1] + 0.3, standing_height
                )

        # Map patient positions
        elif isinstance(pos, PatientPosition):
            if pos == PatientPosition.AT_DOOR_STANDING:
                return clamp_position(
                    door_pos[0] + 0.3, door_pos[1] + 0.2, standing_height
                )
            elif pos == PatientPosition.NEXT_TO_DESK_SITTING:
                return clamp_position(
                    desk_pos[0] + 0.8, desk_pos[1] + 0.3, sitting_height
                )
            elif pos == PatientPosition.NEXT_TO_DESK_STANDING:
                return clamp_position(
                    desk_pos[0] + 0.8, desk_pos[1] + 0.3, standing_height
                )
            elif pos == PatientPosition.SITTING_ON_BENCH:
                return clamp_position(bench_pos[0], bench_pos[1], sitting_height)
            elif pos == PatientPosition.CENTER_ROOM_STANDING:
                return clamp_position(center_pos[0], center_pos[1], standing_height)

        # Fallback to center of room if position not recognized
        return clamp_position(center_pos[0], center_pos[1], standing_height)

    @staticmethod
    def microphone_position_to_room_position(
        room: Room, mic_pos: MicrophonePosition
    ) -> Position3D:
        """
        Convert semantic microphone position enum to actual 3D coordinates within the room.

        This function maps microphone placement descriptions to concrete 3D coordinates
        that can be used for acoustic simulation.

        Args:
            room: Room object containing dimensions and layout information
            mic_pos: MicrophonePosition enum value

        Returns:
            Position3D: 3D coordinates (x, y, z) in meters within the room

        Standard microphone placement assumptions as in enum MicrophonePosition:
        - TABLE_SMARTPHONE: On desk surface at standard height
        - MONITOR: Near computer monitor (on/beside desk)
        - WALL_MOUNTED: On wall at speaking height
        - CEILING_CENTERED: Center of ceiling
        - CHEST_POCKET: At doctor's chest level (wearable mic)

        Example:
            >>> from sdialog.audio.room import Room, RoomRole, Dimensions3D, MicrophonePosition
            >>> room = Room(role=RoomRole.CONSULTATION,
            ...              dimensions=Dimensions3D(4.0, 3.0, 3.0))
            >>> mic_pos = MicrophonePosition.MONITOR
            >>> coord = RoomAcousticsSimulator.microphone_position_to_room_position(room, mic_pos)
            >>> print(f"Microphone position: ({coord.x:.1f}, {coord.y:.1f}, {coord.z:.1f})")
            Microphone position: (1.2, 0.4, 1.2)
        """
        width, length, height = (
            room.dimensions.width,
            room.dimensions.length,
            room.dimensions.height,
        )

        # Define standard furniture positions (same as speaker positioning)
        desk_pos = (width * 0.25, length * 0.15)  # Near corner, away from door
        center_pos = (width * 0.5, length * 0.5)  # Room center

        # Heights for different microphone placements
        desk_height = 0.8  # Standard desk height
        monitor_height = 1.2  # Monitor/webcam height
        wall_mount_height = 1.5  # Wall-mounted mic height
        ceiling_height = height - 0.1  # Just below ceiling
        chest_height = 1.4  # Chest-worn microphone height

        def clamp_position(x, y, z):
            """Ensure position is within room bounds with safety margin"""
            margin = 0.1  # 10cm safety margin from walls (except ceiling)
            x = max(margin, min(x, width - margin))
            y = max(margin, min(y, length - margin))
            z = max(0.1, min(z, height - 0.05))  # Smaller top margin for ceiling mics
            return Position3D.from_list([x, y, z])

        # Map microphone positions
        if mic_pos == MicrophonePosition.TABLE_SMARTPHONE:
            return clamp_position(desk_pos[0] + 0.3, desk_pos[1] + 0.2, desk_height)
        elif mic_pos == MicrophonePosition.MONITOR:
            # Webcam/microphone on or near monitor (typically on desk)
            return clamp_position(desk_pos[0] + 0.1, desk_pos[1], monitor_height)
        elif mic_pos == MicrophonePosition.WALL_MOUNTED:
            # Wall-mounted microphone, typically on the wall opposite to the desk
            wall_x = width * 0.95  # Near far wall
            wall_y = length * 0.6  # Center-ish of the wall
            return clamp_position(wall_x, wall_y, wall_mount_height)
        elif mic_pos == MicrophonePosition.CEILING_CENTERED:
            return clamp_position(center_pos[0], center_pos[1], ceiling_height)
        elif mic_pos == MicrophonePosition.CHEST_POCKET:
            # Wearable microphone - place at doctor's typical position
            doctor_pos = (desk_pos[0], desk_pos[1])  # Doctor at desk
            return clamp_position(doctor_pos[0], doctor_pos[1], chest_height)

        # Fallback to center position at monitor height
        return clamp_position(center_pos[0], center_pos[1], monitor_height)

    @staticmethod
    def generate_test_audio_sources(
        sampling_rate=16000, duration=2.0, save_files=True
    ) -> List[AudioSource]:
        """
        Generate synthetic audio sources for testing the room acoustics simulator.

        Args:
            sampling_rate: Audio sampling rate in Hz
            duration: Duration of each audio source in seconds
            save_files: Whether to save audio files to disk (creates temp files)

        Returns:
            List[AudioSource]: List of audio sources with different frequencies and positions
        """
        import tempfile
        import os

        # Create temporary directory for audio files if saving
        temp_dir = tempfile.mkdtemp() if save_files else None

        t = np.linspace(0, duration, int(sampling_rate * duration), False)

        # Define test sources with different characteristics
        test_sources = [
            {
                "name": "doctor",
                "position": DoctorPosition.AT_DESK_SITTING.value,
                "frequency": 440.0,  # A4 note
                "amplitude": 0.3,
                "snr": -6.0,
            },
            {
                "name": "patient",
                "position": PatientPosition.NEXT_TO_DESK_SITTING.value,
                "frequency": 330.0,  # E4 note
                "amplitude": 0.25,
                "snr": -12.0,
            },
            {
                "name": "background_noise",
                "position": "no_type",
                "frequency": None,  # White noise
                "amplitude": 0.1,
                "snr": -20.0,
            },
        ]

        audio_sources = []

        for i, source_config in enumerate(test_sources):
            if source_config["frequency"] is not None:
                # Generate sine wave
                audio = source_config["amplitude"] * np.sin(
                    2 * np.pi * source_config["frequency"] * t
                )
                # Add some envelope to make it more natural
                envelope = np.exp(-t * 0.5)  # Exponential decay
                audio = audio * envelope
            else:
                # Generate white noise
                audio = source_config["amplitude"] * np.random.normal(0, 1, len(t))

            source_file = None
            if save_files:
                source_file = os.path.join(
                    temp_dir, f"test_source_{i}_{source_config['name']}.wav"
                )
                sf.write(source_file, audio, sampling_rate)

            audio_source = AudioSource(
                name=source_config["name"],
                position=source_config["position"],
                snr=source_config["snr"],
                source_file=source_file,
                directivity="omnidirectional",
            )

            if not save_files:
                audio_source._test_audio = audio

            audio_sources.append(audio_source)

        if save_files:
            print(f"Generated {len(audio_sources)} test audio sources in {temp_dir}")
        else:
            print(f"Generated {len(audio_sources)} test audio sources (in-memory)")

        return audio_sources


if __name__ == "__main__":
    print("Room Acoustics Simulator")
    from sdialog.audio.room_generator import RoomGenerator

    generator = RoomGenerator()
    room = generator.generate(RoomRole.CONSULTATION)
    print(f" Room dimensions: {room.dimensions}")

    print("\n Doctor positions:")
    for doc_pos in DoctorPosition:
        pos_3d = RoomAcousticsSimulator.position_to_room_position(room, doc_pos)
        print(f"  {doc_pos.value} -> {pos_3d}")

    print("\n Patient positions:")
    for pat_pos in PatientPosition:
        pos_3d = RoomAcousticsSimulator.position_to_room_position(room, pat_pos)
        print(f"  {pat_pos.value} -> {pos_3d}")

    print("\n Microphone positions:")
    for mic_pos in MicrophonePosition:
        pos_3d = RoomAcousticsSimulator.microphone_position_to_room_position(
            room, mic_pos
        )
        print(f"  {mic_pos.value} -> {pos_3d}")

    room_acoustics = RoomAcousticsSimulator(room)
    print(f"\n Default microphone position: {room_acoustics.mic_position}")

    print("\n Testing microphone position: Monitor")
    room_acoustics.set_microphone_position(MicrophonePosition.MONITOR)
    # room_acoustics.set_microphone_position([2.0, 1.5, 1.0])  # Explicit coordinates

    print("\nGenerating test audio sources:")
    audio_sources = RoomAcousticsSimulator.generate_test_audio_sources(
        sampling_rate=room_acoustics.sampling_rate,
        duration=1.0,
        save_files=False,  # Use in-memory audio for testing
    )

    print("\nRunning acoustic simulation:")
    try:
        mixed_audio = room_acoustics.simulate(audio_sources)
        sf.write("test_audio.wav", mixed_audio, room_acoustics.sampling_rate)

        print(f"✓ Simulation complete! Generated {len(mixed_audio)} audio samples")
        print(
            f"  Audio duration: {len(mixed_audio) / room_acoustics.sampling_rate:.2f} seconds"
        )
        print(f"  Peak level: {np.max(np.abs(mixed_audio)):.3f}")
    except Exception as e:
        print(f"✗ Simulation failed: {e}")

    print("\n" + "=" * 80)
    print("✓ Room Acoustics Simulator test completed!")
