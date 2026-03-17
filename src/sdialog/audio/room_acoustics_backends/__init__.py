"""
Room acoustics backend package.

This package exposes room acoustics backend contracts, built-in backends,
and the backend resolver utility.
"""

from .base import BaseRoomAcousticsBackend
from .pyroomacoustics import PyroomAcousticsBackend
from .resolver import resolve_room_acoustics_backend

__all__ = [
    "BaseRoomAcousticsBackend",
    "PyroomAcousticsBackend",
    "resolve_room_acoustics_backend",
]
