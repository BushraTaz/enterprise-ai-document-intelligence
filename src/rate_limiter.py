
import threading
import time
from collections import defaultdict, deque


class InMemoryRateLimiter:
    """Thread-safe, per-client sliding-window rate limiter."""

    def __init__(self, max_requests=10, window_seconds=60):
        if max_requests < 1:
            raise ValueError("max_requests must be at least 1")

        if window_seconds <= 0:
            raise ValueError("window_seconds must be positive")

        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self._requests = defaultdict(deque)
        self._lock = threading.Lock()

    def check(self, client_key):
        """
        Return None if allowed, otherwise return retry time in seconds.
        Uses monotonic time so clock adjustments don't affect the window.
        """
        now = time.monotonic()
        cutoff = now - self.window_seconds

        with self._lock:
            timestamps = self._requests[client_key]

            while timestamps and timestamps[0] <= cutoff:
                timestamps.popleft()

            if len(timestamps) >= self.max_requests:
                retry_after = (
                    timestamps[0] + self.window_seconds - now
                )
                return max(1, int(retry_after + 0.999))

            timestamps.append(now)
            return None

    def reset(self):
        """Clear all recorded requests; useful for automated tests."""
        with self._lock:
            self._requests.clear()
