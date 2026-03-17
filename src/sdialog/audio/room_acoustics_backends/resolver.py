"""
Backend resolver for room acoustics simulation.
"""

from typing import Any, Optional

from .base import BaseRoomAcousticsBackend
from .pyroomacoustics import PyroomAcousticsBackend


def resolve_room_acoustics_backend(
    room_acoustics_backend: Optional[Any] = None,
    backend_kwargs: Optional[dict] = None,
) -> BaseRoomAcousticsBackend:
    """
    Resolve room acoustics backend specification into a backend instance.

    :param room_acoustics_backend: Backend specification. Supports ``None``, backend classes,
        backend instances, legacy aliases, or objects exposing ``simulate(...)``.
    :type room_acoustics_backend: Optional[Any]
    :param backend_kwargs: Keyword arguments used when instantiating backend classes.
    :type backend_kwargs: Optional[dict]
    :return: Resolved room acoustics backend instance.
    :rtype: BaseRoomAcousticsBackend
    :raises ValueError: If the backend specification is not supported.
    """
    kwargs = backend_kwargs or {}

    if room_acoustics_backend is None:
        return PyroomAcousticsBackend()

    if isinstance(room_acoustics_backend, BaseRoomAcousticsBackend):
        return room_acoustics_backend

    if (
        isinstance(room_acoustics_backend, type)
        and issubclass(room_acoustics_backend, BaseRoomAcousticsBackend)
    ):
        return room_acoustics_backend(**kwargs)

    if hasattr(room_acoustics_backend, "simulate") and callable(room_acoustics_backend.simulate):
        return room_acoustics_backend

    raise ValueError(
        "Unsupported `room_acoustics_backend`. Use None, a backend class/instance, "
        "or an object exposing `simulate(...)`."
    )
