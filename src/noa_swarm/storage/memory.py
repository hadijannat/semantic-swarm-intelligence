"""In-memory repository implementations.

These repositories provide functional storage for development and testing.
They are thread-safe for async usage via an asyncio lock.
"""

from __future__ import annotations

import asyncio
import re

from noa_swarm.common.schemas import ConsensusRecord, TagMappingRecord, TagRecord, utc_now
from noa_swarm.storage.base import ConsensusRepository, MappingRepository, TagRepository


class InMemoryTagRepository(TagRepository):
    """In-memory tag repository."""

    def __init__(self) -> None:
        self._tags: dict[str, TagRecord] = {}
        self._lock = asyncio.Lock()

    async def upsert(self, tag: TagRecord) -> TagRecord:
        async with self._lock:
            self._tags[tag.tag_id] = tag
        return tag

    async def list(
        self,
        *,
        offset: int = 0,
        limit: int = 1000,
        filter_pattern: str | None = None,
    ) -> list[TagRecord]:
        async with self._lock:
            tags = list(self._tags.values())

        if filter_pattern:
            try:
                pattern = re.compile(filter_pattern, re.IGNORECASE)
                tags = [
                    tag
                    for tag in tags
                    if pattern.search(tag.browse_name)
                    or pattern.search(tag.display_name or "")
                    or pattern.search(tag.full_path)
                ]
            except re.error:
                # Invalid regex: return unfiltered list
                pass

        return tags[offset : offset + limit]

    async def get(self, tag_id: str) -> TagRecord | None:
        async with self._lock:
            return self._tags.get(tag_id)

    async def clear(self) -> None:
        async with self._lock:
            self._tags.clear()


class InMemoryMappingRepository(MappingRepository):
    """In-memory mapping repository."""

    def __init__(self) -> None:
        self._mappings: dict[str, TagMappingRecord] = {}
        self._lock = asyncio.Lock()

    async def upsert(self, mapping: TagMappingRecord) -> TagMappingRecord:
        async with self._lock:
            updated = mapping.model_copy(update={"updated_at": utc_now()})
            self._mappings[mapping.tag_name] = updated
        return updated

    async def list(
        self,
        *,
        status: str | None = None,
        offset: int = 0,
        limit: int = 1000,
    ) -> list[TagMappingRecord]:
        async with self._lock:
            mappings = list(self._mappings.values())

        if status:
            mappings = [m for m in mappings if m.status == status]

        # Sort by updated_at descending for convenience
        mappings.sort(key=lambda m: m.updated_at, reverse=True)
        return mappings[offset : offset + limit]

    async def get(self, tag_name: str) -> TagMappingRecord | None:
        async with self._lock:
            return self._mappings.get(tag_name)

    async def delete(self, tag_name: str) -> bool:
        async with self._lock:
            return self._mappings.pop(tag_name, None) is not None

    async def clear(self) -> None:
        async with self._lock:
            self._mappings.clear()


class InMemoryConsensusRepository(ConsensusRepository):
    """In-memory consensus repository."""

    def __init__(self) -> None:
        self._records: dict[str, ConsensusRecord] = {}
        self._lock = asyncio.Lock()

    async def record(self, consensus: ConsensusRecord) -> ConsensusRecord:
        async with self._lock:
            self._records[consensus.tag_id] = consensus
        return consensus

    async def list(
        self,
        *,
        tag_id: str | None = None,
        offset: int = 0,
        limit: int = 1000,
    ) -> list[ConsensusRecord]:
        async with self._lock:
            records = list(self._records.values())

        if tag_id:
            records = [record for record in records if record.tag_id == tag_id]

        records.sort(key=lambda r: r.created_at, reverse=True)
        return records[offset : offset + limit]

    async def get(self, tag_id: str) -> ConsensusRecord | None:
        async with self._lock:
            return self._records.get(tag_id)

    async def clear(self) -> None:
        async with self._lock:
            self._records.clear()
