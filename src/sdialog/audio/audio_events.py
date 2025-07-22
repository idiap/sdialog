"""
This module provides classes to manage audio events and timelines.
"""
# SPDX-FileCopyrightText: Copyright © 2025 Idiap Research Institute <contact@idiap.ch>
# SPDX-FileContributor: Yanis Labrak <yanis.labrak@univ-avignon.fr>
# SPDX-License-Identifier: MIT
from typing import List
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pydantic import BaseModel, Field


class AudioEvent(BaseModel):
    """
    Base class for audio events.
    """
    label: str = Field(default=None)
    source_file: str = Field(default=None)
    start_time: int = Field(default=None)
    duration: int = Field(default=None)
    role: str = Field(default=None)

    def __str__(self):
        return f"{self.label} {self.role} {self.start_time} {self.duration} {self.source_file}"


class Timeline(BaseModel):
    """
    Timeline of audio events.
    """
    events: List[AudioEvent] = []

    def add_event(self, event: AudioEvent) -> None:
        """
        Add an event to the timeline.
        """
        self.events.append(event)

    def print(self) -> None:
        """
        Print the timeline.
        """
        print(f"Timeline with {len(self.events)} events:")
        for event in sorted(self.events, key=lambda e: e.start_time):
            print(event)

    def draw(self, output_file: str, time_scale: float = 1000.0) -> None:
        """
        Draw the timeline and save it as a PNG file.

        :param output_file: The path to save the PNG file.
        :param time_scale: The scale to convert time units to seconds (default 1000 for ms).
        """
        if not self.events:
            print("Timeline is empty, nothing to draw.")
            return

        fig, ax = plt.subplots(figsize=(20, 10))

        roles = sorted(list(set(event.role for event in self.events)))
        role_y_map = {role: i for i, role in enumerate(roles)}

        # Color mapping for roles
        colors = plt.get_cmap('tab20', len(roles))
        role_colors = {role: colors(i) for i, role in enumerate(roles)}

        for event in self.events:
            y_pos = role_y_map[event.role]
            start_sec = event.start_time / time_scale
            duration_sec = event.duration / time_scale

            # Draw a rectangle for the event
            rect = patches.Rectangle(
                (start_sec, y_pos - 0.4),  # x, y
                duration_sec,              # width
                0.8,                       # height
                facecolor=role_colors[event.role],
                edgecolor='black',
                alpha=0.7
            )
            ax.add_patch(rect)

            # Add the event label
            text_x = start_sec + duration_sec / 2
            text_y = y_pos
            ax.text(text_x, text_y, event.label, ha='center', va='center', fontsize=9, clip_on=True)

        # Set plot limits and labels
        max_time = max((e.start_time + e.duration) / time_scale for e in self.events) if self.events else 1
        ax.set_xlim(0, max_time * 1.05)
        ax.set_ylim(-0.5, len(roles) - 0.5)

        ax.set_yticks(range(len(roles)))
        ax.set_yticklabels(roles)
        ax.set_xlabel("Time (seconds)")
        ax.set_ylabel("Role")
        ax.set_title("Audio Event Timeline")

        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        plt.tight_layout()

        plt.savefig(output_file, dpi=300)
        plt.close(fig)
        print(f"Timeline saved to {output_file}")
