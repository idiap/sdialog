"""
Sliding window rate limiting, shared by the SDialog servers (REST API and WhatsApp).
"""
# SPDX-FileCopyrightText: Copyright © 2025 Idiap Research Institute <contact@idiap.ch>
# SPDX-FileContributor: Sergio Burdisso <sergio.burdisso@idiap.ch>
# SPDX-License-Identifier: MIT
import time
import threading

from collections import deque
from typing import Deque, Dict, Optional, Tuple

PRUNE_EVERY = 1000  # Hits between purges of keys with no recent requests


class RateLimiter:
    """
    Per-key sliding window rate limiter (e.g. one key per user or client IP) enforcing several limits at once,
    such as a per-minute limit for bursts and a per-hour limit for sustained use.

    Rejected requests do not count towards the limits, so a client that keeps retrying while blocked is not
    blocked for longer.

    Example:

        .. code-block:: python

            from sdialog.rate_limit import RateLimiter

            limiter = RateLimiter({"minute": (30, 60), "hour": (300, 3600)})

            exceeded = limiter.hit("10.0.0.1")
            if exceeded:
                limit_name, retry_after = exceeded  # e.g. ("minute", 12.5)

    :param limits: Limit name -> (maximum requests, window in seconds). Limits whose maximum is None
                   (or 0) are disabled. When several limits are exceeded, the first one is reported.
    :type limits: Dict[str, Tuple[Optional[int], float]]
    """

    def __init__(self, limits: Dict[str, Tuple[Optional[int], float]]):
        self.limits = [(name, limit, seconds) for name, (limit, seconds) in limits.items() if limit]
        self._windows: Dict[str, Deque[float]] = {}  # key -> times of accepted requests within the longest window
        self._lock = threading.Lock()
        self._hits = 0

    def hit(self, key: str) -> Optional[Tuple[str, float]]:
        """
        Record a request from ``key`` if it is within all limits.

        :param key: Who the request comes from (e.g. user ID or client IP).
        :type key: str
        :return: None if the request is allowed, otherwise ``(limit_name, retry_after)``: the name of the
                 exceeded limit and the seconds until a request from ``key`` would be allowed again.
        :rtype: Optional[Tuple[str, float]]
        """
        if not self.limits:
            return None

        with self._lock:
            now = time.monotonic()
            longest = max(seconds for _, _, seconds in self.limits)

            self._hits += 1
            if self._hits % PRUNE_EVERY == 0:
                self._windows = {k: w for k, w in self._windows.items() if w and now - w[-1] < longest}

            window = self._windows.setdefault(key, deque())
            while window and now - window[0] >= longest:
                window.popleft()

            for name, limit, seconds in self.limits:
                recent = [t for t in window if now - t < seconds]
                if len(recent) >= limit:
                    # Allowed again once enough of the recent requests leave the window
                    return name, seconds - (now - recent[len(recent) - limit])

            window.append(now)
            return None

    def clear(self) -> None:
        """Forget all recorded requests."""
        with self._lock:
            self._windows.clear()
            self._hits = 0
