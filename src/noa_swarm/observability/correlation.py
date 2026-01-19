"""Correlation ID management for distributed tracing.

This module provides correlation ID propagation for tracking
requests across distributed services.
"""

from __future__ import annotations

import uuid
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Iterator

# Context variable for storing correlation ID
_correlation_id: ContextVar[str | None] = ContextVar("correlation_id", default=None)


def get_correlation_id() -> str:
    """Get the current correlation ID.

    If no correlation ID is set, generates a new one.

    Returns:
        The current correlation ID.
    """
    corr_id = _correlation_id.get()
    if corr_id is None:
        corr_id = generate_correlation_id()
        _correlation_id.set(corr_id)
    return corr_id


def set_correlation_id(correlation_id: str) -> None:
    """Set the correlation ID for the current context.

    Args:
        correlation_id: The correlation ID to set.
    """
    _correlation_id.set(correlation_id)


def clear_correlation_id() -> None:
    """Clear the correlation ID from the current context."""
    _correlation_id.set(None)


def generate_correlation_id() -> str:
    """Generate a new correlation ID.

    Returns:
        A new unique correlation ID.
    """
    return str(uuid.uuid4())


@contextmanager
def correlation_id_context(correlation_id: str) -> Iterator[str]:
    """Context manager for setting a temporary correlation ID.

    Restores the previous correlation ID when exiting the context.

    Args:
        correlation_id: The correlation ID to use within the context.

    Yields:
        The correlation ID.
    """
    previous_id = _correlation_id.get()
    _correlation_id.set(correlation_id)
    try:
        yield correlation_id
    finally:
        _correlation_id.set(previous_id)
