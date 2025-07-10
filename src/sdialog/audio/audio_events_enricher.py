"""
This module provides classes for the enrichment of audio events.
Generate audio events from text utterances in a dialog using the markup language format.
"""
# SPDX-FileCopyrightText: Copyright © 2025 Idiap Research Institute <contact@idiap.ch>
# SPDX-FileContributor: Yanis Labrak <yanis.labrak@univ-avignon.fr>
# SPDX-License-Identifier: MIT
import re
from jinja2 import Template
from typing import List
from sdialog import Dialog, config
from sdialog.generators import DialogGenerator
from sdialog.audio.audio_events import Timeline, AudioEvent


class AudioEventsEnricher:
    """
    Audio events enricher pipeline.
    """

    def enrich(dialog: Dialog) -> Dialog:
        """
        Use an LLM to enrich the audio events in the dialog.
        """

        # Load and populate the prompt with the dialog
        with open(config.config["audio"]["enricher"], "r") as f:
            prompt = Template(f.read()).render(dialog=str(dialog))

        edited_dialog = DialogGenerator(dialogue_details=prompt).generate()

        return edited_dialog

    def structure_markup_language(dialog: Dialog) -> List[dict]:
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

    def remove_markup_language(dialog: Dialog) -> Dialog:
        """
        Remove the markup language tags from the dialog.
        """
        for turn in dialog.turns:
            # This regex finds all XML-like tags (e.g., <tag>, </tag>, <tag/>)
            # and removes them, keeping the inner text of span tags.
            turn.text = re.sub(r'<[^>]+>', '', turn.text)
        return dialog

    # TODO: Implement this method
    def compute_alignment(dialog: Dialog) -> Timeline:
        """
        Compute the alignment of the audio events in the dialog based on the position
        of the anchors tokens (begin_token and end_token).
        """
        timeline = Timeline()
        timeline.add_event(AudioEvent(
            label="keyboard_typing",
            source_file="keyboard_typing.wav",
            start_time=10,
            duration=20,
            role="background"
        ))
        raise NotImplementedError("Can't use this method yet.")
        return timeline
