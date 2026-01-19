"""Middleware helpers for observability integration.

This module provides utilities for integrating observability
into FastAPI applications including:
- Correlation ID extraction from headers
- Request timing utilities
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterator, Mapping

from noa_swarm.observability.correlation import generate_correlation_id

# Standard header name for correlation ID
CORRELATION_ID_HEADER = "X-Correlation-ID"

# Alternative header names to check
CORRELATION_ID_HEADERS = [
    "X-Correlation-ID",
    "X-Request-ID",
    "X-Trace-ID",
]


def extract_correlation_id(headers: Mapping[str, str]) -> str:
    """Extract correlation ID from request headers.

    Checks multiple header names and generates a new ID if none found.

    Args:
        headers: Request headers mapping.

    Returns:
        The correlation ID (extracted or generated).
    """
    for header_name in CORRELATION_ID_HEADERS:
        # Check both exact case and lowercase
        if header_name in headers:
            return headers[header_name]
        if header_name.lower() in headers:
            return headers[header_name.lower()]

    # Check case-insensitive
    lower_headers = {k.lower(): v for k, v in headers.items()}
    for header_name in CORRELATION_ID_HEADERS:
        if header_name.lower() in lower_headers:
            return lower_headers[header_name.lower()]

    # Generate new ID if not found
    return generate_correlation_id()


@dataclass
class Timer:
    """Timer for measuring elapsed time."""

    start_time: float
    end_time: float | None = None

    @property
    def elapsed(self) -> float:
        """Get elapsed time in seconds.

        Returns:
            Elapsed time since start (or until stop if stopped).
        """
        if self.end_time is not None:
            return self.end_time - self.start_time
        return time.perf_counter() - self.start_time

    def stop(self) -> float:
        """Stop the timer and return elapsed time.

        Returns:
            Elapsed time in seconds.
        """
        self.end_time = time.perf_counter()
        return self.elapsed


@contextmanager
def timer_context() -> Iterator[Timer]:
    """Context manager for timing code blocks.

    Yields:
        A Timer object that tracks elapsed time.

    Example:
        with timer_context() as timer:
            do_something()
        print(f"Elapsed: {timer.elapsed:.3f}s")
    """
    timer = Timer(start_time=time.perf_counter())
    try:
        yield timer
    finally:
        timer.stop()
