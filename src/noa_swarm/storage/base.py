"""Repository interfaces for persistent storage.

These interfaces keep storage concerns separate from domain logic.
Concrete implementations can target in-memory, SQLite, or Postgres backends.
"""

from __future__ import annotations

from typing import Protocol

from noa_swarm.common.schemas import ConsensusRecord, TagMappingRecord, TagRecord


class TagRepository(Protocol):
    """Repository interface for discovered tags."""

    async def upsert(self, tag: TagRecord) -> TagRecord:
        """Insert or update a tag record."""

    async def list(
        self,
        *,
        offset: int = 0,
        limit: int = 1000,
        filter_pattern: str | None = None,
    ) -> list[TagRecord]:
        """List tags with optional regex filtering."""

    async def get(self, tag_id: str) -> TagRecord | None:
        """Get a tag by tag_id."""

    async def clear(self) -> None:
        """Remove all tags."""


class MappingRepository(Protocol):
    """Repository interface for tag mapping records."""

    async def upsert(self, mapping: TagMappingRecord) -> TagMappingRecord:
        """Insert or update a mapping record."""

    async def list(
        self,
        *,
        status: str | None = None,
        offset: int = 0,
        limit: int = 1000,
    ) -> list[TagMappingRecord]:
        """List mappings with optional status filter."""

    async def get(self, tag_name: str) -> TagMappingRecord | None:
        """Get a mapping by tag name."""

    async def delete(self, tag_name: str) -> bool:
        """Delete a mapping by tag name."""

    async def clear(self) -> None:
        """Remove all mappings."""


class ConsensusRepository(Protocol):
    """Repository interface for consensus records."""

    async def record(self, consensus: ConsensusRecord) -> ConsensusRecord:
        """Insert a consensus record."""

    async def list(
        self,
        *,
        tag_id: str | None = None,
        offset: int = 0,
        limit: int = 1000,
    ) -> list[ConsensusRecord]:
        """List consensus records with optional tag filter."""

    async def get(self, tag_id: str) -> ConsensusRecord | None:
        """Get consensus record by tag_id."""

    async def clear(self) -> None:
        """Remove all consensus records."""
