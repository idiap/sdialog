"""
This module provides classes for the enrichment of audio events.
Generate audio events from text utterances in a dialog using the markup language format.
"""
# SPDX-FileCopyrightText: Copyright © 2025 Idiap Research Institute <contact@idiap.ch>
# SPDX-FileContributor: Yanis Labrak <yanis.labrak@univ-avignon.fr>
# SPDX-License-Identifier: MIT
import re
from typing import List
from sdialog import Dialog
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

        edited_dialog = DialogGenerator(
            dialogue_details=f"""
Your task is to rewrite the following dialogue by inserting appropriate audio clue tags. The tags should follow an XML-like syntax. There are two types of tags:

1. **Span tags**: For events that last for a duration of text. The format is `<label>text</label>`.
   - `label` is the name of the audio event (e.g., `keyboard_typing`, `ac_noise`).
   - Example: `I need to <keyboard_typing>finish this email</keyboard_typing> before I leave.`

2. **Point tags**: For instantaneous events. The format is `<label duration="Xms" overlapping="True|False" />`.
   - `label` is the name of the audio event (e.g., `cough`, `door_slam`).
   - `duration` is an optional attribute for the duration in milliseconds (e.g., `duration="20ms"`).
   - `overlapping` should be `True` if the audio can happen during speech, and `False` otherwise.
   - Example: `Please Mr. Dupont enter the room <door_slam overlapping="False" />, how are you today?`

You can use labels like `steps`, `door_opening`, `door_closing`, `footsteps`, `car_engine`, `ac_noise`, `keyboard_typing`, `phone_ringing`, `doorbell_ringing`, `cough`, `sneeze`, `sigh`, `laughter`. Use snake_case for labels with multiple words.

The dialogue content, speakers, and order must remain exactly the same. You should only add the audio tags.

Here is the original dialogue:
{dialog}

Return only the modified dialogue with audio clues included.
"""
        ).generate()
    
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
        - "overlap": The overlap of the event with another event like stopping speaking when typing on a keyboard. By default it's defined at True.
        - "duration": The duration of the event (optional).
        """
        text = str(dialog)
        events = []

        # Regex for span tags: <label>text</label>
        span_pattern = re.compile(r'<(\w+)>(.*?)</\1>')
        
        # Regex for point tags: <label duration="Xms" overlapping="True|False" />
        point_pattern = re.compile(r'<(\w+)\s*(?:duration="(\d+m?s)")?\s*(?:overlapping="(True|False)")?\s*/>')

        # Find all point tags
        for match in point_pattern.finditer(text):
            label, duration, overlap = match.groups()
            events.append({
                "begin_token": text[:match.start()].split(),
                "end_token": None,
                "label": label,
                "overlap": overlap == 'True' if overlap is not None else True,
                "duration": duration
            })

        # Find all span tags
        for match in span_pattern.finditer(text):
            label, content = match.groups()
            events.append({
                "begin_token": text[:match.start()].split(),
                "end_token": (text[:match.start()] + content).split(),
                "label": label,
                "overlap": True, # Overlapping is default True for span tags based on new prompt
                "duration": None
            })

        return events
    
    def remove_markup_language(dialog: Dialog) -> Dialog:
        """
        Remove the markup language tags from the dialog.
        """
        for turn in dialog.turns:
            text = turn.text
            span_pattern = re.compile(r'<(\w+)(?:[^>]*)>(.*?)</\1>')
            text = span_pattern.sub(r'\2', text)

            point_pattern = re.compile(r'<(\w+)\s*.*?/\s*>')
            text = point_pattern.sub('', text)
            turn.text = text
        
        return dialog
    
    def compute_alignment(dialog: Dialog) -> Timeline:
        """
        Compute the alignment of the audio events in the dialog based on the position of the anchors tokens (begin_token and end_token).
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

