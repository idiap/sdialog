"""
This module provides classes for the enrichment of audio events.
Generate audio events from text utterances in a dialog using the markup language format.
"""
# SPDX-FileCopyrightText: Copyright © 2025 Idiap Research Institute <contact@idiap.ch>
# SPDX-FileContributor: Yanis Labrak <yanis.labrak@univ-avignon.fr>
# SPDX-License-Identifier: MIT
import re
import torch
import logging
import whisper
from typing import List
from random import choice
from sdialog import config
from jinja2 import Template
from sdialog.audio.room import Room, MicrophonePosition
from sdialog.generators import DialogGenerator
from sdialog.audio.audio_dialog import AudioDialog
from sdialog.audio.audio_events import Timeline, AudioEvent

logger = logging.getLogger(__name__)

device = "cuda" if torch.cuda.is_available() else "cpu"
whisper_model = whisper.load_model("large-v3", device=device)


def _parse_duration_ms(duration_str: str) -> int:
    if not duration_str:
        return 0
    duration_str = duration_str.lower().strip()
    try:
        if duration_str.endswith('ms'):
            return int(float(duration_str[:-2]))
        if duration_str.endswith('s'):
            return int(float(duration_str[:-1]) * 1000)
        return int(float(duration_str))  # Assume ms if no unit
    except ValueError:
        return 0


class AudioEventsEnricher:
    """
    Audio events enricher pipeline.
    """

    def extract_events(self, dialog: AudioDialog) -> AudioDialog:
        """
        Extract the audio events from the dialog.
        """

        print("Generating audio events from dialog...")
        dialog = self._enrich(dialog)

        print("Computing alignment...")
        dialog = self._compute_alignment(dialog)

        return dialog

    def _enrich(self, dialog: AudioDialog) -> AudioDialog:
        """
        Use an LLM to enrich the audio events in the dialog.
        """

        # Load and populate the prompt with the dialog
        with open(config.config["prompts"]["audio"]["enricher"], "r") as f:
            prompt = Template(f.read()).render(dialog=str(dialog))

        return DialogGenerator(dialogue_details=prompt).generate()

    # TODO: Parse outputs correctly and verify the SNR values
    def generate_snr(self, dialog: AudioDialog) -> AudioDialog:
        """
        Use an LLM to compute the SNR of the speakers in the dialog.
        """

        with open(config.config["prompts"]["audio"]["snr"], "r") as f:
            prompt = Template(f.read()).render(dialog=str(dialog))

        dialog_with_snr = DialogGenerator(dialogue_details=prompt).generate()

        # Extract the SNR from the dialog
        for turn in dialog_with_snr.turns:
            snr = re.search(r'snr="(\d+)"', turn.text)
            turn.snr = int(snr.group(1)) if snr else None

        return dialog_with_snr

    def generate_room_position(self, dialog: AudioDialog) -> AudioDialog:
        """
        Use an LLM to compute the position of the speakers in the room based on predefined positions and the dialog.
        """

        with open(config.config["prompts"]["audio"]["room_position"], "r") as f:
            prompt = Template(f.read()).render(dialog=str(dialog))

        dialog_with_room_position = DialogGenerator(dialogue_details=prompt).generate()

        # Extract the position and microphone position from the dialog
        for turn in dialog_with_room_position.turns:

            turn.position = re.search(
                r'position="([^"]+)"',
                turn.text).group(1) if re.search(r'position="([^"]+)"', turn.text) else None

        return dialog_with_room_position

    def generate_microphone_position(self, dialog: AudioDialog) -> AudioDialog:
        """
        Randomly sample the microphone position for the whole dialogue.
        """

        # Randomly sample the microphone position for the whole dialogue
        microphone_position = choice(
            [p.value for p in MicrophonePosition]
        )

        for turn in dialog.turns:
            turn.microphone_position = microphone_position

        return dialog

    def _structure_markup_language(self, dialog: AudioDialog) -> List[dict]:
        """
        Extract the markup language structure of the dialog and align the events
        at the words level by considering the position to be before the word and
        could finish after the last word of the utterance.

        :return: A list of dictionaries containing the markup language structure of the dialog.
        Each dictionary contains the following keys:
        - "begin_token": The beginning token of the event.
        - "end_token": The ending token of the event (optional and default None).
        - "label": The label of the event.
        - "overlap": The overlap of the event with another event like stopping speaking when typing
        on a keyboard. By default it's defined at True.
        - "duration": The duration of the event (optional).
        """
        events = []
        # Find any tag, span or point
        tag_pattern = re.compile(r'(<(\w+)\s*.*?>(.*?)</\2>)|(<(\w+)[^>]*?/>)|(<(\w+)>)')

        full_text_with_tags = ""
        for turn in dialog.turns:
            full_text_with_tags += f"[{turn.speaker}] {turn.text}\n"

        for match in tag_pattern.finditer(full_text_with_tags):
            text_before = full_text_with_tags[:match.start()]
            clean_text_before = re.sub(r'<[^>]+>', '', text_before)
            begin_word_index = len(clean_text_before.split())

            # The regex has 3 main groups for 3 types of tags
            # Span tag: <label>content</label>
            if match.group(1):
                label, content, *_ = match.groups()
                clean_content = re.sub(r'<[^>]+>', '', content)
                end_word_index = begin_word_index + len(clean_content.split())
                events.append({
                    "begin_token": begin_word_index,
                    "end_token": end_word_index,
                    "label": label,
                    "overlap": True,
                    "duration": None
                })
            # Point tag: <label ... />
            elif match.group(4):
                label = match.group(5)
                # crude attribute parsing
                overlap = "overlapping=\"False\"" not in match.group(4)
                duration_match = re.search(r'duration="([^"]+)"', match.group(4))
                duration = duration_match.group(1) if duration_match else None
                events.append({
                    "begin_token": begin_word_index,
                    "end_token": None,
                    "label": label,
                    "overlap": overlap,
                    "duration": duration
                })
            # Simple tag: <label>
            elif match.group(6):
                label = match.group(7)
                events.append({
                    "begin_token": begin_word_index,
                    "end_token": None,
                    "label": label,
                    "overlap": True,
                    "duration": None
                })

        return events

    # def remove_markup_language(self):
    #     """
    #     Remove the markup language tags from the dialog.
    #     """
    #     for turn in dialog.turns:
    #         # This regex finds all XML-like tags (e.g., <tag>, </tag>, <tag/>)
    #         # and removes them, keeping the inner text of span tags.
    #         turn.text = re.sub(r'<[^>]+>', '', turn.text)
    #     return dialog

    def _compute_alignment(self, dialog: AudioDialog) -> AudioDialog:
        """
        Compute the alignment of the audio events in the dialog based on the position
        of the anchors tokens (begin_token and end_token) and the utterances audios.
        """
        structured_events = self._structure_markup_language(dialog)
        timeline = Timeline()
        dialog_word_timings = []
        turn_word_offsets = [0]
        cumulative_words = 0

        # We assume audio is 16kHz for whisper
        sample_rate = 16000

        current_time_offset_s = 0.0

        for i, turn in enumerate(dialog.turns):
            # Add utterance to timeline
            utterance_start_time_s = current_time_offset_s
            utterance_duration_s = len(turn.get_audio()) / sample_rate if sample_rate > 0 else 0
            clean_text_for_label = re.sub(r'<[^>]+>', '', turn.text).strip()

            if clean_text_for_label or utterance_duration_s > 0:
                utterance_event = AudioEvent(
                    label=clean_text_for_label if clean_text_for_label else "speech",
                    source_file="<utterance_audio>",  # This is not a real file
                    start_time=int(utterance_start_time_s * 1000),
                    duration=int(utterance_duration_s * 1000),
                    role=turn.speaker
                )
                timeline.add_event(utterance_event)

            # Add placeholder for speaker tag, as counted in structure_markup_language
            dialog_word_timings.append({
                'word': f'[{turn.speaker}]',
                'start': current_time_offset_s,
                'end': current_time_offset_s
            })
            cumulative_words += 1

            clean_text = re.sub(r'<[^>]+>', '', turn.text)

            if not clean_text.strip():
                turn_word_offsets.append(cumulative_words)
                current_time_offset_s += utterance_duration_s
                continue

            # Resample audio to 16kHz if necessary, assuming we know original sr
            # For now, we assume it is already 16kHz from the TTS engine
            result = whisper_model.transcribe(turn.get_audio(), word_timestamps=True, fp16=False, language='en')

            turn_words_with_ts = []
            for segment in result.get('segments', []):
                turn_words_with_ts.extend(segment.get('words', []))

            if len(clean_text.split()) != len(turn_words_with_ts):
                logger.warning(
                    f"Word count mismatch in turn {i}. "
                    f"Expected {len(clean_text.split())} words, but whisper found {len(turn_words_with_ts)}. "
                    "Alignment may be inaccurate."
                )

            for word_info in turn_words_with_ts:
                word_info['start'] += current_time_offset_s
                word_info['end'] += current_time_offset_s
                dialog_word_timings.append(word_info)

            cumulative_words += len(turn_words_with_ts)
            turn_word_offsets.append(cumulative_words)

            if turn_words_with_ts:
                current_time_offset_s = turn_words_with_ts[-1]['end']
            else:
                current_time_offset_s += utterance_duration_s

        for event in structured_events:
            begin_token_idx = event['begin_token']

            if begin_token_idx >= len(dialog_word_timings):
                logger.warning(f"Event '{event['label']}' begin token index {begin_token_idx} is out of bounds.")
                continue

            start_time_ms = dialog_word_timings[begin_token_idx]['start'] * 1000

            turn_index = next((
                i - 1 for i, offset in enumerate(turn_word_offsets) if begin_token_idx < offset),
                len(dialog.turns) - 1
            )
            speaker_role = dialog.turns[turn_index].speaker

            duration_ms = 0
            if event['end_token'] is not None:
                end_token_idx = event['end_token']
                if end_token_idx > 0 and end_token_idx <= len(dialog_word_timings):
                    end_time_ms = dialog_word_timings[end_token_idx - 1]['end'] * 1000
                    duration_ms = end_time_ms - start_time_ms
            else:
                duration_ms = _parse_duration_ms(event.get('duration'))

            audio_event = AudioEvent(
                label=event['label'],
                source_file=event['label'],
                start_time=int(start_time_ms),
                duration=int(duration_ms),
                role=speaker_role
            )
            timeline.add_event(audio_event)

        dialog.timeline = timeline

        return dialog
