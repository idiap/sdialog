"""
Telecommunications backend implementation.
"""

import os
import librosa
import numpy as np
import soundfile as sf
from scipy.signal import butter, lfilter
from typing import Any, Callable, Optional, Dict

from sdialog.audio.utils import logger
from sdialog.audio.dialog import AudioDialog
from sdialog.audio import RoomAcousticsConfig
from sdialog.audio.room import Room, RoomPosition
from sdialog.audio.acoustics_simulator import AcousticsSimulator

from .base import BaseRoomAcousticsBackend


class TelecommunicationsBackend(BaseRoomAcousticsBackend):
    """
    Room acoustics backend simulating telecommunications (e.g., telephone calls).

    This backend allows performing room acoustics for each speaker individually
    (simulating that they are in different rooms) and applies a telephone-like
    bandpass filter and downsampling to their audio before mixing them together.
    """

    requires_room = True
    name = "telecommunications"

    def _apply_telephone_effect(self, audio: np.ndarray, sr: int, codec: str = "default") -> np.ndarray:
        """
        Applies a telephone effect and simulates codec compression.

        :param audio: Input audio signal.
        :param sr: Sampling rate of the input audio.
        :param codec: Codec to simulate (e.g., 'g.711', 'g.729', 'amr', 'amr-wb', 'default').
        :return: Processed audio signal.
        """
        codec = codec.lower()

        # Define codec parameters
        if codec == "amr-wb":
            lowcut = 50.0
            highcut = 7000.0
            target_sr = 16000
            quantization_levels = 32  # 5-bit equivalent for simulation
        elif codec == "amr":
            lowcut = 200.0
            highcut = 3400.0
            target_sr = 8000
            quantization_levels = 16  # 4-bit equivalent for simulation
        elif codec == "g.729":
            lowcut = 300.0
            highcut = 3400.0
            target_sr = 8000
            quantization_levels = 8   # 3-bit equivalent for simulation
        elif codec == "g.711":
            lowcut = 300.0
            highcut = 3400.0
            target_sr = 8000
            quantization_levels = 128  # 8-bit equivalent for simulation
        else:  # default
            lowcut = 300.0
            highcut = 3400.0
            target_sr = 8000
            quantization_levels = None

        # Bandpass filter
        nyq = 0.5 * sr
        low = min(lowcut / nyq, 0.99)
        high = min(highcut / nyq, 0.99)

        if low < high:
            b, a = butter(5, [low, high], btype='band')
            audio = lfilter(b, a, audio)

        # Simulate telephone bandwidth by downsampling
        if sr != target_sr:
            audio = librosa.resample(y=audio, orig_sr=sr, target_sr=target_sr)

        # Apply codec compression simulation (mu-law companding + quantization)
        if quantization_levels is not None:
            max_val = np.max(np.abs(audio))
            if max_val > 0:
                audio = audio / max_val

            mu = 255.0
            # Compand
            y = np.sign(audio) * np.log(1 + mu * np.abs(audio)) / np.log(1 + mu)
            # Quantize
            y = np.round(y * quantization_levels) / quantization_levels
            # Expand
            audio = np.sign(y) * (1 / mu) * ((1 + mu) ** np.abs(y) - 1)

            if max_val > 0:
                audio = audio * max_val

        # Upsample back to original sr
        if sr != target_sr:
            audio = librosa.resample(y=audio, orig_sr=target_sr, target_sr=sr)

        return audio

    def simulate(
        self,
        dialog: AudioDialog,
        room: Optional[Any],
        dialog_directory: str,
        room_name: str,
        audio_file_format: str = "wav",
        environment: Optional[dict] = None,
        callback_mix_fn: Optional[Callable] = None,
        callback_mix_kwargs: Optional[dict] = None,
        sampling_rate: int = 44_100,
    ) -> AudioDialog:
        """
        Generate telecommunications audio with individual room acoustics.

        :param dialog: Audio dialog object to update.
        :param room: Default room configuration used if individual rooms are not provided.
        :param dialog_directory: Relative output directory for generated files.
        :param room_name: Name of the room profile to generate.
        :param audio_file_format: Audio format for exported files (default: "wav").
        :param environment: Optional environment overrides. Can contain `speaker_rooms`
                            (dict mapping speaker roles to Room objects) and `speaker_codecs`
                            (dict mapping speaker roles to codec strings like 'g.711', 'amr', etc.
                            or a global `codec` string).
        :param callback_mix_fn: Optional callback used during audio mixing.
        :param callback_mix_kwargs: Optional keyword arguments for the mix callback.
        :param sampling_rate: Sampling rate used for generated audio.
        :return: Updated dialog with room acoustics outputs.
        """
        env = environment or {}
        kwargs_pyroom = env.get("kwargs_pyroom", {})
        source_volumes = env.get("source_volumes", {})

        # Determine rooms for each speaker
        # Users can pass a dictionary of rooms via environment["speaker_rooms"]
        speaker_rooms: Dict[str, Room] = env.get("speaker_rooms", {})

        # If no specific rooms are provided, fallback to the default room for both speakers
        if not speaker_rooms and isinstance(room, Room):
            speaker_rooms = {
                "speaker_1": room,
                "speaker_2": room
            }
        elif not speaker_rooms:
            raise ValueError(
                "TelecommunicationsBackend requires either a default `room` (Room object) "
                "or `speaker_rooms` in the `environment` dictionary."
            )

        all_sources = dialog.get_audio_sources()
        mixed_signals = []

        _callback_mix_kwargs = callback_mix_kwargs.copy() if callback_mix_kwargs is not None else {}
        if "dialog" not in _callback_mix_kwargs:
            _callback_mix_kwargs["dialog"] = dialog

        # Process each speaker individually
        codec = env.get("codec", "default")
        speaker_codecs = env.get("speaker_codecs", {})

        for speaker_role, speaker_room in speaker_rooms.items():

            # Filter sources for this speaker
            # We assume sources positioned at `speaker_role` belong to this speaker.
            # Also include sources like `speaker_1_no_type`, `speaker_1_door`, etc.
            speaker_sources = []
            for s in all_sources:
                if s.position == speaker_role or s.position.startswith(f"{speaker_role}_"):
                    # Create a copy to avoid modifying the original dialog sources
                    s_copy = s.model_copy()

                    # Strip the speaker prefix so AcousticsSimulator can understand the position
                    if s_copy.position.startswith(f"{speaker_role}_"):
                        s_copy.position = s_copy.position[len(f"{speaker_role}_"):]

                    speaker_sources.append(s_copy)

            if not speaker_sources:
                logger.warning(f"No audio sources found for {speaker_role}. Skipping simulation for this speaker.")
                continue

            logger.info(f"Simulating room acoustics for {speaker_role} in telecommunications backend...")
            simulator = AcousticsSimulator(room=speaker_room, sampling_rate=sampling_rate, kwargs_pyroom=kwargs_pyroom)

            speaker_audio = simulator.simulate(
                sources=speaker_sources,
                source_volumes=source_volumes,
                callback_mix_fn=callback_mix_fn,
                callback_mix_kwargs=_callback_mix_kwargs,
            )

            # Apply telephone effect
            speaker_codec = speaker_codecs.get(speaker_role, codec)
            logger.info(f"Applying telephone effect (codec: {speaker_codec}) for {speaker_role}...")
            telephone_audio = self._apply_telephone_effect(speaker_audio, sr=sampling_rate, codec=speaker_codec)
            mixed_signals.append(telephone_audio)

        if not mixed_signals:
            raise ValueError("No audio signals generated. Check speaker roles and sources.")

        mixing_strategy = env.get("mixing_strategy", "mono")
        print("#########################")
        print("mixing_strategy: ", mixing_strategy)
        print("#########################")

        # Mix all speakers together
        # Pad to same length if necessary
        max_len = max(len(sig) for sig in mixed_signals)
        padded_signals = [np.pad(sig, (0, max_len - len(sig))) for sig in mixed_signals]

        if mixing_strategy == "stereo":
            if len(padded_signals) == 1:
                print("usecase 1")
                # Strictly one speaker on the left channel, silence on the right
                final_mix = np.stack([padded_signals[0], np.zeros_like(padded_signals[0])], axis=1)
            else:
                print("usecase 2")

                # Strictly first speaker on the left, second speaker on the right
                left_channel = padded_signals[0]
                right_channel = padded_signals[1]

                # If there are more than 2 speakers, mix the rest into both channels (center)
                if len(padded_signals) > 2:
                    center_mix = np.sum(padded_signals[2:], axis=0)
                    left_channel = left_channel + center_mix
                    right_channel = right_channel + center_mix

                final_mix = np.stack([left_channel, right_channel], axis=1)
        elif mixing_strategy == "mono":
            final_mix = np.sum(padded_signals, axis=0)
        else:
            raise ValueError(f"Unknown mixing strategy: {mixing_strategy}. Supported strategies: 'mono', 'stereo'.")

        # Normalize to prevent clipping (scale to max -1dB roughly)
        max_val = np.max(np.abs(final_mix))
        if max_val > 0:
            final_mix = final_mix / max_val * (10 ** (-1 / 20))

        # Save the audio file
        current_room_audio_path = os.path.join(
            dialog.audio_dir_path,
            dialog_directory,
            "exported_audios",
            "rooms",
            f"audio_pipeline_step3-{room_name}.{audio_file_format}"
        )

        os.makedirs(os.path.dirname(current_room_audio_path), exist_ok=True)
        sf.write(current_room_audio_path, final_mix, sampling_rate)

        # Save the audio path and configuration into the dialog
        if room_name in dialog.audio_step_3_filepaths:
            logger.warning(f"Room '{room_name}' already exists in the dialog")

        audio_paths_post_processing = {}
        if (
            room_name in dialog.audio_step_3_filepaths
            and dialog.audio_step_3_filepaths[room_name].audio_paths_post_processing is not None
        ):
            audio_paths_post_processing = dialog.audio_step_3_filepaths[room_name].audio_paths_post_processing

        # Use the default room or the first speaker's room for the configuration metadata
        rep_room = room if isinstance(room, Room) else list(speaker_rooms.values())[0]

        dialog.audio_step_3_filepaths[room_name] = RoomAcousticsConfig(
            audio_path=current_room_audio_path,
            microphone_position=rep_room.mic_position,
            room_name=room_name,
            room=rep_room,
            source_volumes=source_volumes,
            kwargs_pyroom=kwargs_pyroom,
            background_effect=env.get("background_effect", "white_noise"),
            foreground_effect=env.get("foreground_effect", "ac_noise_minimal"),
            foreground_effect_position=env.get("foreground_effect_position", RoomPosition.TOP_RIGHT),
            audio_paths_post_processing=audio_paths_post_processing,
        )

        return dialog
