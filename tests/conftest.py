"""Pytest configuration and shared fixtures."""

import pytest


@pytest.fixture
def anyio_backend() -> str:
    """Use asyncio as the async backend for tests."""
    return "asyncio"
