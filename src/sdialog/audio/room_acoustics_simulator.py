import os
from typing import List, Optional, Union
import numpy as np
# import matplotlib.pyplot as plt
import soundfile as sf
import pyroomacoustics as pra
from sdialog.audio.room import Room, RoomRole, AudioSource, Position3D
from sdialog.audio.room import DoctorPosition, PatientPosition, RecordingDevice, MicrophonePosition
from sdialog.audio.room_generator import calculate_room_dimensions, ROOM_SIZES


class RoomAcousticsSimulator:
    """
        Simulates sound based on room acoustics based on room definition,
        sound sources provided and microphone(s) setup.

         Example:
             >>> room = sdialog.audio.room.RoomGenerator.generate()
             >>> room_acoustics = RoomAcousticsSimulator(room)
             >>> room_acoustics.add_microphone( .. )
             >>> room_acoustics.add_sources(audio_sources)
             >>> audio = room_acoustics.simulate()
    """
    def __init__(self, room: Optional[Room] = None, sampling_rate=16000):
        self.sampling_rate = sampling_rate
        self.ref_db = -65   # - 45 dB
        self.audiosources: List[AudioSource] = []

        if room is None:
            self.room = Room(
                role=RoomRole.CONSULTATION,
                name="consultation_room_default",
                dimensions=calculate_room_dimensions(ROOM_SIZES[3]),
                rt60=0.5,
                soundsources_position=[DoctorPosition.AT_DESK_SITTING, PatientPosition.NEXT_TO_DESK_SITTING],
                mic_type=RecordingDevice.WEBCAM,
                mic_position=MicrophonePosition.MONITOR,
                furnitures=False,
            )
        else:
            self.room = room

        self._pyroom = self._create_pyroom(self.room, self.sampling_rate)
        self.mic_position = [0.5, 0.5, 0.5]

    def _create_pyroom(self, room: Room, sampling_rate=16000):
        e_absorption, max_order = pra.inverse_sabine(room.rt60, room.dimensions)
        # max_order = 17  # Number of reflections
        return pra.ShoeBox(room.dimensions, fs=sampling_rate, materials=pra.Material(e_absorption), max_order=max_order)

    # room_acoustics.add_microphone( .. )
    def add_microphone(self, mic_position):
        """Add microphone to the room"""
        self.mic_position = mic_position
        mic_array = pra.MicrophoneArray(
            np.array([mic_position]).T,
            self._pyroom.fs
        )
        self._pyroom.add_microphone_array(mic_array)
        print(f"Added microphone at position {mic_position}")

    def _add_sources(self, audiosources: List[AudioSource]):
        for i, asource in enumerate(audiosources):
            self.audiosources.append(asource)

            position = self.parse_position(asource.position)

            asource._position3d = self.position_to_room_position(self.room, position)

            if os.path.exists(asource.source_file):
                audio, original_fs = sf.read(asource.source_file)
                if audio.ndim > 1:  # Convert to mono if stereo
                    audio = np.mean(audio, axis=1)

                self._pyroom.add_source(asource._position3d.to_list(), signal=audio)
            else:
                print(f"Warning: File {asource.source_file} not found.")

            # print(f"✓ Added source '{name}' at position {position} with {len(audio)} samples")

    def simulate(self, sources: List[AudioSource] = [], reset=False):  # -> np.array:

        if reset:
            # see https://github.com/LCAV/pyroomacoustics/issues/311
            self.reset()
            self._pyroom = self._create_pyroom(self.room, self.sampling_rate)

        self._add_sources(sources)
        self._pyroom.simulate()
        mixed_signal = self._pyroom.mic_array.signals[0, :]

        # peak_level = np.max(np.abs(mixed_signal))
        # if peak_level > 0.95:
        #     # Soft compression to prevent harsh clipping
        #     compression_ratio = 0.95 / peak_level
        #     mixed_signal = mixed_signal * compression_ratio
        #     print(f"Applied soft compression (ratio: {compression_ratio:.3f}) to prevent clipping")
        # print(f"Simulation complete! Peak level: {np.max(np.abs(mixed_signal)):.3f}")

        mixed_signal = self.apply_snr(mixed_signal, 1.0)  # scale audio to max 1dB
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
    def apply_snr(x, snr):
        """Scale an audio signal to a given maximum SNR."""
        x *= snr / np.abs(x).max(initial=1e-15)
        return x

    @staticmethod
    def parse_position(position: str) -> Union[DoctorPosition, PatientPosition]:
        """
        Convert a position string to the appropriate position enum.
        """
        if position.startswith("doctor:"):
            try:
                return DoctorPosition(position)
            except ValueError:
                raise ValueError(f"Invalid doctor position: {position}")
        elif position.startswith("patient:"):
            try:
                return PatientPosition(position)
            except ValueError:
                raise ValueError(f"Invalid patient position: {position}")
        else:
            raise ValueError(f"Position must start with 'doctor:' or 'patient:', got: {position}")

    @staticmethod
    def position_to_room_position(room: Room, pos: Union[DoctorPosition, PatientPosition]) -> Position3D:
        """
        Convert semantic position enums to actual 3D coordinates within the room.

        This function maps abstract position descriptions (like "doctor:at_desk_sitting")
        to concrete 3D coordinates that can be used for acoustic simulation.

        Args:
            room: Room object containing dimensions and layout information
            pos: Position enum (DoctorPosition or PatientPosition)

        Returns:
            Position3D: 3D coordinates (x, y, z) in meters within the room

        Standard room layout assumptions:
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
        width, length, height = room.dimensions.width, room.dimensions.length, room.dimensions.height

        # Define standard furniture positions as fractions of room dimensions
        desk_pos = (width * 0.25, length * 0.15)  # Near corner, away from door
        bench_pos = (width * 0.6, length * 0.5)   # Center-right area
        door_pos = (0.1, 0.1)                     # Near corner
        sink_pos = (width * 0.05, length * 0.8)   # Back wall, near corner
        cupboard_pos = (width * 0.95, length * 0.8)  # Back wall, opposite corner
        center_pos = (width * 0.5, length * 0.5)  # Room center

        # Heights for different postures
        sitting_height = 0.5   # Chair/bench sitting height
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
                return clamp_position(sink_pos[0] + 0.3, sink_pos[1] - 0.5, standing_height)
            elif pos == DoctorPosition.NEXT_TO_SINK_BACK:
                return clamp_position(sink_pos[0] - 0.3, sink_pos[1] + 0.3, standing_height)
            elif pos == DoctorPosition.NEXT_TO_CUPBOARD_FRONT:
                return clamp_position(cupboard_pos[0] - 0.3, cupboard_pos[1] - 0.5, standing_height)
            elif pos == DoctorPosition.NEXT_TO_CUPBOARD_BACK:
                return clamp_position(cupboard_pos[0] + 0.3, cupboard_pos[1] + 0.3, standing_height)
            elif pos == DoctorPosition.NEXT_TO_DOOR_STANDING:
                return clamp_position(door_pos[0] + 0.5, door_pos[1] + 0.3, standing_height)

        # Map patient positions
        elif isinstance(pos, PatientPosition):
            if pos == PatientPosition.AT_DOOR_STANDING:
                return clamp_position(door_pos[0] + 0.3, door_pos[1] + 0.2, standing_height)
            elif pos == PatientPosition.NEXT_TO_DESK_SITTING:
                return clamp_position(desk_pos[0] + 0.8, desk_pos[1] + 0.3, sitting_height)
            elif pos == PatientPosition.NEXT_TO_DESK_STANDING:
                return clamp_position(desk_pos[0] + 0.8, desk_pos[1] + 0.3, standing_height)
            elif pos == PatientPosition.SITTING_ON_BENCH:
                return clamp_position(bench_pos[0], bench_pos[1], sitting_height)
            elif pos == PatientPosition.CENTER_ROOM_STANDING:
                return clamp_position(center_pos[0], center_pos[1], standing_height)

        # Fallback to center of room if position not recognized
        return clamp_position(center_pos[0], center_pos[1], standing_height)


if __name__ == "__main__":
    print(" Room Acoustics Simulator")
    from sdialog.audio.room_generator import RoomGenerator
    generator = RoomGenerator()
    room = generator.generate(RoomRole.CONSULTATION)
    print(f"Room dimensions: {room.dimensions}")

    # Test doctor positions
    print("\nDoctor positions:")
    for doc_pos in DoctorPosition:
        pos_3d = RoomAcousticsSimulator.position_to_room_position(room, doc_pos)
        print(f"  {doc_pos.value} -> {pos_3d}")

    # Test patient positions
    print("\nPatient positions:")
    for pat_pos in PatientPosition:
        pos_3d = RoomAcousticsSimulator.position_to_room_position(room, pat_pos)
        print(f"  {pat_pos.value} -> {pos_3d}")

    # Example usage:
    room_acoustics = RoomAcousticsSimulator(room)
    # room_acoustics.add_microphone([1.0, 1.0, 1.5])
    # room_acoustics.add_sources(audio_sources)
    # audio = room_acoustics.simulate(audio_sources)
