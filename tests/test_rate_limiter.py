
from rate_limiter import InMemoryRateLimiter


def test_requests_are_allowed_up_to_limit():
    limiter = InMemoryRateLimiter(
        max_requests=2,
        window_seconds=60,
    )

    assert limiter.check("client-a") is None
    assert limiter.check("client-a") is None

    retry_after = limiter.check("client-a")

    assert isinstance(retry_after, int)
    assert retry_after >= 1


def test_clients_have_independent_limits():
    limiter = InMemoryRateLimiter(
        max_requests=1,
        window_seconds=60,
    )

    assert limiter.check("client-a") is None
    assert limiter.check("client-b") is None

    assert limiter.check("client-a") is not None
    assert limiter.check("client-b") is not None


def test_expired_requests_leave_the_window(monkeypatch):
    import rate_limiter

    clock = [100.0]
    monkeypatch.setattr(
        rate_limiter.time,
        "monotonic",
        lambda: clock[0],
    )

    limiter = InMemoryRateLimiter(
        max_requests=1,
        window_seconds=10,
    )

    assert limiter.check("client-a") is None
    assert limiter.check("client-a") is not None

    clock[0] = 111.0

    assert limiter.check("client-a") is None


def test_reset_clears_recorded_requests():
    limiter = InMemoryRateLimiter(
        max_requests=1,
        window_seconds=60,
    )

    assert limiter.check("client-a") is None
    assert limiter.check("client-a") is not None

    limiter.reset()

    assert limiter.check("client-a") is None


def test_invalid_configuration_is_rejected():
    import pytest

    with pytest.raises(ValueError):
        InMemoryRateLimiter(max_requests=0)

    with pytest.raises(ValueError):
        InMemoryRateLimiter(window_seconds=0)
