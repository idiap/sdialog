"""
This module provides classes for the enrichment of audio events.
Generate audio events from text utterances in a dialog using the SSML format.
"""
# SPDX-FileCopyrightText: Copyright © 2025 Idiap Research Institute <contact@idiap.ch>
# SPDX-FileContributor: Yanis Labrak <yanis.labrak@univ-avignon.fr>
# SPDX-License-Identifier: MIT
from typing import List
from sdialog import Dialog
from sdialog.audio.audio_events import Timeline, AudioEvent


class AudioEventsEnricher:
    """
    Audio events enricher pipeline.
    """

    def enrich(dialog: Dialog) -> Dialog:
        """
        Use an LLM to enrich the audio events in the dialog.
        """
        # TODO: Write the audio_events_enricher.j2 prompt template to enrich the audio events in the dialog.
        raise NotImplementedError("Can't use this method yet.")
        return dialog
    
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
        raise NotImplementedError("Can't use this method yet.")
        return []
    
    def remove_markup_language(dialog: Dialog) -> Dialog:
        """
        Remove the markup language tags from the dialog.
        """
        raise NotImplementedError("Can't use this method yet.")
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

