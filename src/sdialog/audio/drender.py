# import os
import json
from typing import List, Optional, Union, Dict, Any
import numpy as np
import matplotlib.pyplot as plt
import pyroomacoustics as pra
from pyroomacoustics.datasets.base import Dataset
from sdialog.audio.room import DoctorPosition, MicrophonePosition, PatientPosition, RecordingDevice, Room, RoomRole
from sdialog.audio.room_generator import ROOM_SIZES, calculate_room_dimensions
from sdialog.audio.room import SoundSource, Position3D


# import soundfile as sf

# from scipy.signal import resample


class Drender:
    def __init__(self, room: Optional[Room] = None):
        self.sampling_rate = 16000
        self.ref_db = 45  # - dB
        self.soundsources: List[SoundSource] = []

        if room is None:
            self.room = Room(
                role=RoomRole.CONSULTATION,
                name="consultation_room_default",
                dimensions=calculate_room_dimensions(ROOM_SIZES[3]),
                rt60=0.5,
                speaker_position=[DoctorPosition.AT_DESK_SITTING, PatientPosition.NEXT_TO_DESK_SITTING],
                mic_type=RecordingDevice.WEBCAM,
                mic_position=MicrophonePosition.MONITOR,
                furnitures=False,
            )
        else:
            self.room = room

        self._pyroom = self._create_pyroom(self.room, self.sampling_rate)

    def _create_pyroom(self, room: Room, sampling_rate=16000):
        e_absorption, max_order = pra.inverse_sabine(room.rt60, room.dimensions)
        # max_order = 17  # Number of reflections
        return pra.ShoeBox(room.dimensions, fs=sampling_rate, materials=pra.Material(e_absorption), max_order=max_order)

    def render(self, sound_sources: List[SoundSource], room: Optional[Room] = None) -> np.ndarray:
        if room is None:
            room = self.room
        else:
            self.room = room
            self._pyroom = self._create_pyroom(self.room)


        print(f"dialogs: {sound_sources}")
        for i, s in enumerate(sound_sources):
            print(f"{s}")
            saudio = np.ones(16000)
            self._pyroom.add_source(s.position.to_list(), signal=saudio)

        # Add default microphone if it doesn't exist
        if self._pyroom.mic_array is None:
            mic_position = [
                self.room.dimensions.length / 2,
                self.room.dimensions.width / 2,
                self.room.dimensions.height / 2,
            ]
            self._pyroom.add_microphone(mic_position)
        # Microphone position (webcam on monitor)
        # mic_position = [1.3, 0.7, 1.6]

        self._pyroom.simulate()

        if self._pyroom.mic_array is not None:
            mixed_signal = self._pyroom.mic_array.signals[0, :]
        else:  # Return empty signal if no microphone array
            mixed_signal = np.array([])

        # Apply soft clipping prevention harsh clipping
        if len(mixed_signal) > 0:
            peak_level = np.max(np.abs(mixed_signal))
            if peak_level > 0.95:
                compression_ratio = 0.95 / peak_level
                mixed_signal = mixed_signal * compression_ratio
                print(f"Applied soft compression (ratio: {compression_ratio:.3f}) to prevent clipping")
                print(f"Peak level: {np.max(np.abs(mixed_signal)):.3f}")

        return mixed_signal

    # def save_results(self, mixed_signal, output_dir="output"):
    #     pass

    # room.mic_array.to_wav(
        #     f"examples/samples/guitar_16k_reverb_{args.method}.wav",
        #     norm=True,
        #     bitdepth=np.int16,
        # )

    def plot_room_setup(self):
        """Visualize the room setup"""
        # _, ax = plt.subplots(1, 1, figsize=(10, 8))
        fig, ax = self._pyroom.plot() #.plot(ax=ax)
        ax.set_xlim([-1, 6])
        ax.set_ylim([-1, 4])
        ax.set_zlim([-1, 4])
        ax.set_title("")
        fig.set_size_inches(10, 5)
        fig.savefig("room_setup.png")

        # print("Plotting Impulse Response...")
        # fig_ir, ax_ir = self._pyroom.plot_rir(kind="ir")
        # fig_ir.suptitle("Impulse Response")
        # fig_ir.savefig("rir_impulse.png", dpi=300, bbox_inches='tight')

        # print("Plotting Transfer Function...")
        # fig_tf, ax_tf = self._pyroom.plot_rir(kind="tf")
        # fig_tf.suptitle("Transfer Function")
        # fig_tf.savefig("rir_transfer.png", dpi=300, bbox_inches='tight')

        # print("Plotting Spectrogram...")
        # fig_spec, ax_spec = self._pyroom.plot_rir(kind="spec")
        # fig_spec.suptitle("Spectrogram")
        # fig_spec.savefig("rir_spectrogram.png", dpi=300, bbox_inches='tight')


    @staticmethod
    def load_sound_sources_from_json(json_data: Union[str, List[Dict[str, Any]]]) -> List[SoundSource]:
        """
        Load SoundSource objects from JSON data.

        Args:
            json_data: Either a JSON string or a list of dictionaries

        Returns:
            List of SoundSource objects

        Raises:
            ValueError: If required fields are missing or invalid
            json.JSONDecodeError: If JSON string is invalid
        """
        sound_sources = []
        if isinstance(json_data, str):
            try:
                data = json.loads(json_data)
            except json.JSONDecodeError as e:
                raise json.JSONDecodeError(str(e), e.doc, e.pos)
        else:
            data = json_data

        if not isinstance(data, list):
            raise ValueError("JSON data must be a list of audio source configurations")

        for i, source_data in enumerate(data):
            try:
                sound_source = SoundSource.from_dict(source_data)
                sound_sources.append(sound_source)
            except (ValueError, TypeError) as e:
                source_name = source_data.get("name", f"index {i}") if isinstance(source_data, dict) else f"index {i}"
                raise ValueError(f"Error processing source '{source_name}': {e}")

        return sound_sources

    @staticmethod
    def load_sound_sources_from_file(filepath: str) -> List[SoundSource]:
        """
        Load SoundSource objects from a JSON file.

        Args:
            filepath: Path to the JSON file

        Returns:
            List of SoundSource objects

        Raises:
                FileNotFoundError: If file doesn't exist
                PermissionError: If file cannot be read
                ValueError: If JSON data is invalid
        """
        try:
            with open(filepath, "r", encoding="utf-8") as file:
                json_data = json.load(file)
            return Drender.load_sound_sources_from_json(json_data)
        except FileNotFoundError:
            raise FileNotFoundError(f"JSON file not found: {filepath}")
        except PermissionError:
            raise PermissionError(f"Permission denied reading file: {filepath}")
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in file {filepath}: {e}")

    @staticmethod
    def load_sound_sources_from_jams(jams_infile_path: str) -> List[SoundSource]:
        """
        Args:
            jams_infile_path: Path to the JAMS file

        Returns:
            List of SoundSource objects

        Raises:
            FileNotFoundError: If JAMS file doesn't exist
            ImportError: If jams library is not installed
            ValueError: If JAMS file structure is invalid
        """
        try:
            import jams
        except ImportError:
            raise ImportError("jams library is required. Install with: pip install jams")

        try:
            soundscape_jam = jams.load(jams_infile_path, fmt='jams')
        except FileNotFoundError:
            raise FileNotFoundError(f"JAMS file not found: {jams_infile_path}")
        except Exception as e:
            raise ValueError(f"Error loading JAMS file {jams_infile_path}: {e}")

        # Search for scaper annotations
        scaper_annotations = soundscape_jam.search(namespace='scaper')
        if not scaper_annotations:
            raise ValueError(f"No scaper annotations found in JAMS file: {jams_infile_path}")

        sound_sources = []

        for ann in scaper_annotations:

            # fg_path = ann.sandbox.scaper.get('fg_path', '') if hasattr(ann, 'sandbox') and hasattr(ann.sandbox, 'scaper') else ''
            # bg_path = ann.sandbox.scaper.get('bg_path', '') if hasattr(ann, 'sandbox') and hasattr(ann.sandbox, 'scaper') else ''

            # print(f"Processing JAMS annotation with {len(ann.data)} sound events")
            # if fg_path:
            #     print(f"Foreground path: {fg_path}")
            # if bg_path:
            #     print(f"Background path: {bg_path}")

            for i, obs in enumerate(ann.data):
                try:
                    event_value = obs.value
                    event_label = event_value.get('label', f'unknown_{i}')
                    event_role = event_value.get('role', 'unknown')

                    source_name = f"{event_role}_{event_label}_{i}" if event_role != event_label else f"{event_role}_{i}"


                    # Extract timing information (available for future use)
                    start_time = float(obs.time) if obs.time is not None else 0.0
                    duration = float(obs.duration) if obs.duration is not None else 0.0
                    stop_time = start_time + duration

                    position = ''
                    if 'position' in event_value:
                        pos_data = event_value['position']
                        if pos_data is not None:
                            position = pos_data
                        # if isinstance(pos_data, (list, tuple)) and len(pos_data) >= 3:
                        #     position = [float(pos_data[0]), float(pos_data[1]), float(pos_data[2])]\

                    snr = float(event_value['snr'])

                    sourcefile = event_value.get('source_file')

                    sound_source = SoundSource(
                        name=source_name,
                        position=Position3D.from_list([0.0, 0.0, 0.0]),
                        snr=snr,
                        source_file=sourcefile,
                        directivity="omnidirectional",
                        is_primary=False
                    )
                    # Note: start_time and stop_time are not part of SoundSource dataclass

                    sound_sources.append(sound_source)

                    print(f"  Event {i+1}: {event_role} '{event_label}' {source_name} at ({position}), SNR={snr}dB, time: [ {start_time} : {stop_time} ]")
                    # ({position[0]:.1f}, {position[1]:.1f}, {position[2]:.1f})

                except (KeyError, ValueError, TypeError) as e:
                    print(f"Warning: Skipping malformed sound event {i}: {e}")
                    continue

        if not sound_sources:
            print("Warning: No valid sound sources extracted from JAMS file")

        print(f"Successfully loaded {len(sound_sources)} sound sources from JAMS file")
        return sound_sources


        #        "soundscape.jams"
        # jamsfile = os.path.join(generate_dir, "soundscape.jams")
        # base, ext = os.path.splitext(file_name)
        # if ext.lower() == ".json" or ext.lower() == ".jams":
        #     # If the file is a JSON or TXT file, read it as metadata
        #     with open(file_path, "r") as f:
        #         data_json = f.read()




                # if ext.lower() in [".wav", ".mp3", ".flac", ".ogg"]:


# Usage examples:
if __name__ == "__main__":
    # Your original data
    source_config = [
        {"name": "doctor", "position": [1.0, 0.8, 1.4], "directivity": "directional", "snr_db": -9},
        {"name": "patient", "position": [1.6, 0.8, 1.4], "directivity": "directional", "snr_db": -18},
        {"name": "background_noise", "position": [1.25, 1.6, 2.5], "directivity": "omnidirectional", "snr_db": -15},
    ]


    sound_sources = Drender.load_sound_sources_from_json(source_config)
    for source in sound_sources:
        print(f"Source: {source.name}")
        print(f"  Position: ({source.position.x}, {source.position.y}, {source.position.z})")
        print(f"  Directivity: {source.directivity}")
        print(f"  SNR: {source.snr} dB")

    dr = Drender()

    output_mixed_sound = dr.render(sound_sources)

    if len(output_mixed_sound) > 0:
        # Basic signal statistics
        duration = len(output_mixed_sound) / dr.sampling_rate
        peak_amplitude = np.max(np.abs(output_mixed_sound))
        rms_level = np.sqrt(np.mean(output_mixed_sound**2))
        dynamic_range = 20 * np.log10(peak_amplitude / (rms_level + 1e-10))

        print("\n=== Audio Rendering Diagnostics ===")
        print(
            f"Room dimensions: {dr.room.dimensions.length:.1f}m"\
            f"× {dr.room.dimensions.width:.1f}m × {dr.room.dimensions.height:.1f}m"
        )
        print(f"Room RT60: {dr.room.rt60:.2f} seconds")
        print(f"Sampling rate: {dr.sampling_rate} Hz")
        print(f"Generated signal duration: {duration:.2f} seconds ({len(output_mixed_sound)} samples)")
        print(f"Peak amplitude: {peak_amplitude:.4f}")
        print(f"RMS level: {rms_level:.4f}")
        print(f"Dynamic range: {dynamic_range:.1f} dB")
        print("\nSource configuration summary:")
        for i, source in enumerate(sound_sources):
            print(
                f"  {i + 1}. {source.name}: SNR={source.snr}dB, \
                pos=({source.position.x:.1f},{source.position.y:.1f},{source.position.z:.1f})"
            )

        print("\nGenerating room visualization...")
        dr.plot_room_setup()

    else:
        print("❌ ERROR: No audio signal generated - check source configuration and room setup")
