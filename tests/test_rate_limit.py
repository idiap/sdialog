"""
Unit tests for the sliding window rate limiter shared by the SDialog servers.
"""
from unittest.mock import patch

import pytest

from sdialog import rate_limit
from sdialog.rate_limit import RateLimiter


@pytest.fixture
def clock():
    now = {"t": 1000.0}
    with patch("time.monotonic", side_effect=lambda: now["t"]):
        yield now


def test_allows_up_to_limit_then_rejects(clock):
    limiter = RateLimiter({"minute": (3, 60)})
    assert [limiter.hit("a") for _ in range(3)] == [None] * 3
    assert limiter.hit("a") == ("minute", 60)
    assert limiter.hit("b") is None  # Other keys are independent


def test_window_slides(clock):
    limiter = RateLimiter({"minute": (2, 60)})
    limiter.hit("a")
    clock["t"] += 30
    limiter.hit("a")
    clock["t"] += 10
    assert limiter.hit("a") == ("minute", 20)  # The first request leaves the window in 20s
    clock["t"] += 20
    assert limiter.hit("a") is None


def test_rejected_requests_do_not_count(clock):
    limiter = RateLimiter({"minute": (1, 60)})
    limiter.hit("a")
    for _ in range(10):
        clock["t"] += 5
        assert limiter.hit("a") is not None
    clock["t"] += 10  # 60s after the only accepted request
    assert limiter.hit("a") is None


def test_several_limits(clock):
    limiter = RateLimiter({"hour": (3, 3600), "minute": (2, 60)})
    assert [limiter.hit("a") for _ in range(2)] == [None, None]
    assert limiter.hit("a")[0] == "minute"
    clock["t"] += 60
    assert limiter.hit("a") is None
    clock["t"] += 60
    name, retry_after = limiter.hit("a")
    assert name == "hour" and retry_after == 3600 - 120


def test_disabled_limits():
    limiter = RateLimiter({"hour": (None, 3600), "minute": (0, 60)})
    assert limiter.limits == []
    assert all(limiter.hit("a") is None for _ in range(100))


def test_idle_keys_are_pruned(clock, monkeypatch):
    monkeypatch.setattr(rate_limit, "PRUNE_EVERY", 10)
    limiter = RateLimiter({"minute": (5, 60)})
    for i in range(5):
        limiter.hit(f"old-{i}")
    clock["t"] += 61
    for _ in range(5):
        limiter.hit("new")
    assert set(limiter._windows) == {"new"}


def test_clear(clock):
    limiter = RateLimiter({"minute": (1, 60)})
    limiter.hit("a")
    limiter.clear()
    assert limiter.hit("a") is None
