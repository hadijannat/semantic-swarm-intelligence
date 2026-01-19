"""File system connectors for tag import/export.

This module provides functions for importing and exporting tag data
from various file formats (CSV, JSON). Currently contains stubs for
future implementation.
"""

from __future__ import annotations

from pathlib import Path

from noa_swarm.common.schemas import TagRecord


async def import_from_csv(path: Path) -> list[TagRecord]:
    """Import tags from a CSV file.

    Args:
        path: Path to the CSV file.

    Returns:
        List of TagRecord instances.

    Raises:
        NotImplementedError: This function is not yet implemented.
    """
    raise NotImplementedError("CSV import not yet implemented")


async def export_to_csv(tags: list[TagRecord], path: Path) -> None:
    """Export tags to a CSV file.

    Args:
        tags: List of TagRecord instances to export.
        path: Path to write the CSV file.

    Raises:
        NotImplementedError: This function is not yet implemented.
    """
    raise NotImplementedError("CSV export not yet implemented")


async def import_from_json(path: Path) -> list[TagRecord]:
    """Import tags from a JSON file.

    Args:
        path: Path to the JSON file.

    Returns:
        List of TagRecord instances.

    Raises:
        NotImplementedError: This function is not yet implemented.
    """
    raise NotImplementedError("JSON import not yet implemented")


async def export_to_json(tags: list[TagRecord], path: Path) -> None:
    """Export tags to a JSON file.

    Args:
        tags: List of TagRecord instances to export.
        path: Path to write the JSON file.

    Raises:
        NotImplementedError: This function is not yet implemented.
    """
    raise NotImplementedError("JSON export not yet implemented")
