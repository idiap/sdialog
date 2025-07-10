"""
This module provides classes to manage audio events and timelines.
"""
# SPDX-FileCopyrightText: Copyright © 2025 Idiap Research Institute <contact@idiap.ch>
# SPDX-FileContributor: Yanis Labrak <yanis.labrak@univ-avignon.fr>
# SPDX-License-Identifier: MIT
from typing import List


class AudioEvent:
    """
    Base class for audio events.
    """

    def __init__(self, label: str, source_file: str, start_time: int, duration: int, role: str):
        """
        Initialize an audio event.

        :param label: label of the event.
        :param source_file: Source file of the event.
        :param start_time: Start time of the event in milliseconds.
        :param duration: Duration of the event in milliseconds.
        :param role: Role of the event (speaker_1, speaker_2, foreground, background, etc.).
        """
        self.label = label
        self.source_file = source_file
        self.start_time = start_time
        self.duration = duration
        self.role = role

    def __str__(self):
        return f"{self.label} {self.role} {self.start_time} {self.duration} {self.source_file}"


class Timeline:
    """
    Timeline of audio events.
    """
    def __init__(self):
        self.events: List[AudioEvent] = []

    def add_event(self, event: AudioEvent) -> None:
        """
        Add an event to the timeline.
        """
        self.events.append(event)
