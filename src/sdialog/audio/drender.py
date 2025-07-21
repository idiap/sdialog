# import os
import json
from typing import List, Optional, Union, Dict, Any
import numpy as np
import matplotlib.pyplot as plt
import pyroomacoustics as pra
from sdialog.audio.room import DoctorPosition, MicrophonePosition, PatientPosition, RecordingDevice, Room, RoomRole
from sdialog.audio.room_generator import ROOM_SIZES, calculate_room_dimensions
from sdialog.audio.room import SoundSource  # , Position3D


# import soundfile as sf

# from scipy.signal import resample


class dRender:
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
        for s in enumerate(sound_sources):
            print(f"{s}")

        # Add default microphone if it doesn't exist
        if self._pyroom.mic_array is None:
            mic_position = [
                self.room.dimensions.length / 2,
                self.room.dimensions.width / 2,
                self.room.dimensions.height / 2,
            ]
            self._pyroom.add_microphone(mic_position)

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

    def plot_room_setup(self):
        """Visualize the room setup"""
        _, ax = plt.subplots(1, 1, figsize=(10, 8))
        self._pyroom.plot(ax=ax)
        ax.set_title("Room Setup with Sources and Microphone")
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig("room_setup.png", dpi=300, bbox_inches="tight")
        plt.show()

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
            return dRender.load_sound_sources_from_json(json_data)
        except FileNotFoundError:
            raise FileNotFoundError(f"JSON file not found: {filepath}")
        except PermissionError:
            raise PermissionError(f"Permission denied reading file: {filepath}")
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in file {filepath}: {e}")


# Usage examples:
if __name__ == "__main__":
    # Your original data
    source_config = [
        {"name": "doctor", "position": [1.0, 0.8, 1.4], "directivity": "directional", "snr_db": -9},
        {"name": "patient", "position": [1.6, 0.8, 1.4], "directivity": "directional", "snr_db": -18},
        {"name": "background_noise", "position": [1.25, 1.6, 2.5], "directivity": "omnidirectional", "snr_db": -15},
    ]

    sound_sources = dRender.load_sound_sources_from_json(source_config)
    for source in sound_sources:
        print(f"Source: {source.name}")
        print(f"  Position: ({source.position.x}, {source.position.y}, {source.position.z})")
        print(f"  Directivity: {source.directivity}")
        print(f"  SNR: {source.snr_db} dB")

    dr = dRender()

    output_mixed_sound = dr.render(sound_sources)

    if len(output_mixed_sound) > 0:
        # Basic signal statistics
        duration = len(output_mixed_sound) / dr.sampling_rate
        peak_amplitude = np.max(np.abs(output_mixed_sound))
        rms_level = np.sqrt(np.mean(output_mixed_sound**2))
        dynamic_range = 20 * np.log10(peak_amplitude / (rms_level + 1e-10))

        print("\n=== Audio Rendering Diagnostics ===")
        print(
            f"Room dimensions: {dr.room.dimensions.length:.1f}m \
                × {dr.room.dimensions.width:.1f}m × {dr.room.dimensions.height:.1f}m"
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
                f"  {i + 1}. {source.name}: SNR={source.snr_db}dB, \
                pos=({source.position.x:.1f},{source.position.y:.1f},{source.position.z:.1f})"
            )

        print("\nGenerating room visualization...")
        dr.plot_room_setup()

    else:
        print("❌ ERROR: No audio signal generated - check source configuration and room setup")
