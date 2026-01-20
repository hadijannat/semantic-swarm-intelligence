"""Mapping service for tag-to-IRDI records."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from noa_swarm.common.logging import get_logger
from noa_swarm.common.schemas import MappingStatus, TagMappingRecord, utc_now

if TYPE_CHECKING:
    from noa_swarm.dictionaries import ProviderRegistry
    from noa_swarm.storage.base import MappingRepository, TagRepository

logger = get_logger(__name__)


@dataclass
class MappingStats:
    """Aggregated mapping statistics."""

    pending: int = 0
    mapped: int = 0
    verified: int = 0
    rejected: int = 0
    conflict: int = 0

    @property
    def total(self) -> int:
        return self.pending + self.mapped + self.verified + self.rejected + self.conflict


class MappingService:
    """Service for managing tag mappings."""

    def __init__(
        self,
        mapping_repo: MappingRepository,
        tag_repo: TagRepository,
        registry: ProviderRegistry,
    ) -> None:
        self._mapping_repo = mapping_repo
        self._tag_repo = tag_repo
        self._registry = registry

    async def list_mappings(
        self,
        *,
        status: str | None = None,
        offset: int = 0,
        limit: int = 1000,
    ) -> tuple[list[TagMappingRecord], MappingStats]:
        mappings = await self._mapping_repo.list(
            status=status,
            offset=offset,
            limit=limit,
        )
        stats = await self._compute_stats()
        return mappings, stats

    async def create_mapping(
        self,
        *,
        tag_name: str,
        browse_path: str,
        irdi: str | None = None,
    ) -> TagMappingRecord:
        tag_id = await self._resolve_tag_id(tag_name, browse_path)

        preferred_name = None
        if irdi:
            concept = await self._registry.lookup_any(irdi)
            preferred_name = concept.preferred_name if concept else None

        status: MappingStatus = "mapped" if irdi else "pending"
        record = TagMappingRecord(
            tag_id=tag_id,
            tag_name=tag_name,
            browse_path=browse_path,
            irdi=irdi,
            preferred_name=preferred_name,
            status=status,
        )
        return await self._mapping_repo.upsert(record)

    async def get_mapping(self, tag_name: str) -> TagMappingRecord | None:
        return await self._mapping_repo.get(tag_name)

    async def update_mapping(
        self,
        tag_name: str,
        *,
        irdi: str | None = None,
        status: MappingStatus | None = None,
    ) -> TagMappingRecord | None:
        existing = await self._mapping_repo.get(tag_name)
        if existing is None:
            return None

        preferred_name = existing.preferred_name
        if irdi:
            concept = await self._registry.lookup_any(irdi)
            preferred_name = concept.preferred_name if concept else preferred_name

        updated = existing.model_copy(
            update={
                "irdi": irdi if irdi is not None else existing.irdi,
                "preferred_name": preferred_name,
                "status": status or existing.status,
                "updated_at": utc_now(),
            }
        )
        return await self._mapping_repo.upsert(updated)

    async def delete_mapping(self, tag_name: str) -> bool:
        return await self._mapping_repo.delete(tag_name)

    async def approve_mapping(self, tag_name: str) -> TagMappingRecord | None:
        return await self.update_mapping(tag_name, status="verified")

    async def reject_mapping(self, tag_name: str) -> TagMappingRecord | None:
        return await self.update_mapping(tag_name, status="rejected")

    async def _compute_stats(self) -> MappingStats:
        mappings = await self._mapping_repo.list(limit=10_000)
        stats = MappingStats()
        for mapping in mappings:
            match mapping.status:
                case "pending":
                    stats.pending += 1
                case "mapped":
                    stats.mapped += 1
                case "verified":
                    stats.verified += 1
                case "rejected":
                    stats.rejected += 1
                case "conflict":
                    stats.conflict += 1
        return stats

    async def _resolve_tag_id(self, tag_name: str, browse_path: str) -> str:
        tags = await self._tag_repo.list(limit=10_000)
        for tag in tags:
            if tag.browse_name == tag_name or tag.full_path == browse_path:
                return tag.tag_id
        return f"manual|{tag_name}"
