"""
Room acoustics backend package.

This package exposes room acoustics backend contracts, built-in backends,
and the backend resolver utility.
"""

from .base import BaseRoomAcousticsBackend
from .pyroomacoustics import PyroomAcousticsBackend
from .resolver import resolve_room_acoustics_backend
from .telecommunications import TelecommunicationsBackend

__all__ = [
    "BaseRoomAcousticsBackend",
    "PyroomAcousticsBackend",
    "TelecommunicationsBackend",
    "resolve_room_acoustics_backend",
]
