"""File system connectors for tag import/export.

This module provides functions for importing and exporting tag data
from various file formats (CSV, JSON).
"""

from __future__ import annotations

import asyncio
import csv
import json
from typing import TYPE_CHECKING

from noa_swarm.common.schemas import TagRecord

if TYPE_CHECKING:
    from pathlib import Path


async def import_from_csv(path: Path) -> list[TagRecord]:
    """Import tags from a CSV file.

    Args:
        path: Path to the CSV file.

    Returns:
        List of TagRecord instances.

    """
    return await asyncio.to_thread(_import_from_csv_sync, path)


async def export_to_csv(tags: list[TagRecord], path: Path) -> None:
    """Export tags to a CSV file.

    Args:
        tags: List of TagRecord instances to export.
        path: Path to write the CSV file.

    """
    await asyncio.to_thread(_export_to_csv_sync, tags, path)


async def import_from_json(path: Path) -> list[TagRecord]:
    """Import tags from a JSON file.

    Args:
        path: Path to the JSON file.

    Returns:
        List of TagRecord instances.

    """
    return await asyncio.to_thread(_import_from_json_sync, path)


async def export_to_json(tags: list[TagRecord], path: Path) -> None:
    """Export tags to a JSON file.

    Args:
        tags: List of TagRecord instances to export.
        path: Path to write the JSON file.

    """
    await asyncio.to_thread(_export_to_json_sync, tags, path)


def _import_from_csv_sync(path: Path) -> list[TagRecord]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        records: list[TagRecord] = []
        for row in reader:
            parent_path = (
                row.get("parent_path", "").split("/") if row.get("parent_path") else []
            )
            records.append(
                TagRecord(
                    node_id=row.get("node_id", ""),
                    browse_name=row.get("browse_name", ""),
                    display_name=row.get("display_name") or None,
                    data_type=row.get("data_type") or None,
                    description=row.get("description") or None,
                    parent_path=parent_path,
                    source_server=row.get("source_server", ""),
                    engineering_unit=row.get("engineering_unit") or None,
                    access_level=int(row["access_level"]) if row.get("access_level") else None,
                )
            )
        return records


def _export_to_csv_sync(tags: list[TagRecord], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = [
            "node_id",
            "browse_name",
            "display_name",
            "data_type",
            "description",
            "parent_path",
            "source_server",
            "engineering_unit",
            "access_level",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for tag in tags:
            writer.writerow(
                {
                    "node_id": tag.node_id,
                    "browse_name": tag.browse_name,
                    "display_name": tag.display_name or "",
                    "data_type": tag.data_type or "",
                    "description": tag.description or "",
                    "parent_path": "/".join(tag.parent_path),
                    "source_server": tag.source_server,
                    "engineering_unit": tag.engineering_unit or "",
                    "access_level": tag.access_level or "",
                }
            )


def _import_from_json_sync(path: Path) -> list[TagRecord]:
    with path.open(encoding="utf-8") as handle:
        data = json.load(handle)
    records: list[TagRecord] = []
    for item in data:
        records.append(TagRecord(**item))
    return records


def _export_to_json_sync(tags: list[TagRecord], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump([tag.model_dump(mode="json") for tag in tags], handle, indent=2)
