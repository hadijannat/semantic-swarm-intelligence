"""Discovery service for OPC UA tag browsing."""

from __future__ import annotations

import asyncio
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from noa_swarm.common.logging import get_logger
from noa_swarm.common.schemas import TagMappingRecord, TagRecord
from noa_swarm.connectors.opcua_asyncua import OPCUABrowser

if TYPE_CHECKING:
    from noa_swarm.storage.base import MappingRepository, TagRepository

logger = get_logger(__name__)


@dataclass
class DiscoveryStatus:
    """Current discovery status."""

    is_running: bool = False
    progress: float = 0.0
    discovered_count: int = 0
    server_url: str | None = None
    operation_id: str | None = None
    started_at: datetime | None = None
    finished_at: datetime | None = None
    error: str | None = None


class DiscoveryService:
    """Service for managing OPC UA discovery tasks."""

    def __init__(
        self,
        tag_repo: TagRepository,
        mapping_repo: MappingRepository | None = None,
    ) -> None:
        self._tag_repo = tag_repo
        self._mapping_repo = mapping_repo
        self._status = DiscoveryStatus()
        self._task: asyncio.Task[None] | None = None
        self._lock = asyncio.Lock()

    async def start(
        self,
        server_url: str,
        root_node: str | None = None,
        max_depth: int = 10,
    ) -> DiscoveryStatus:
        """Start a discovery task."""
        async with self._lock:
            if self._task and not self._task.done():
                return self._status

            operation_id = f"discovery-{uuid.uuid4().hex[:8]}"
            self._status = DiscoveryStatus(
                is_running=True,
                progress=0.0,
                discovered_count=0,
                server_url=server_url,
                operation_id=operation_id,
                started_at=datetime.now(UTC),
            )

            self._task = asyncio.create_task(
                self._run_discovery(server_url, root_node, max_depth),
                name=f"discovery-{operation_id}",
            )

            return self._snapshot()

    async def stop(self) -> DiscoveryStatus:
        """Stop the current discovery task if running."""
        async with self._lock:
            if self._task and not self._task.done():
                self._task.cancel()
                self._status.is_running = False
                self._status.finished_at = datetime.now(UTC)
            return self._snapshot()

    def get_status(self) -> DiscoveryStatus:
        """Return current discovery status."""
        return self._snapshot()

    def _snapshot(self) -> DiscoveryStatus:
        return DiscoveryStatus(
            is_running=self._status.is_running,
            progress=self._status.progress,
            discovered_count=self._status.discovered_count,
            server_url=self._status.server_url,
            operation_id=self._status.operation_id,
            started_at=self._status.started_at,
            finished_at=self._status.finished_at,
            error=self._status.error,
        )

    async def list_nodes(
        self,
        *,
        offset: int = 0,
        limit: int = 1000,
        filter_pattern: str | None = None,
    ) -> list[TagRecord]:
        """Return discovered nodes."""
        return await self._tag_repo.list(
            offset=offset,
            limit=limit,
            filter_pattern=filter_pattern,
        )

    async def get_node(self, node_id: str) -> TagRecord | None:
        """Find a node by NodeId or tag_id."""
        # Try direct tag_id lookup
        record = await self._tag_repo.get(node_id)
        if record:
            return record

        # Fall back to search by node_id
        tags = await self._tag_repo.list(limit=10_000)
        for tag in tags:
            if tag.node_id == node_id:
                return tag
        return None

    async def _run_discovery(
        self,
        server_url: str,
        root_node: str | None,
        max_depth: int,
    ) -> None:
        """Run a discovery task and update status."""
        try:
            async with OPCUABrowser(server_url) as browser:
                tags = await browser.browse_all_tags(start_node=root_node, max_depth=max_depth)

            for tag in tags:
                await self._tag_repo.upsert(tag)
                if self._mapping_repo:
                    existing = await self._mapping_repo.get(tag.browse_name)
                    if existing is None:
                        mapping = TagMappingRecord(
                            tag_id=tag.tag_id,
                            tag_name=tag.browse_name,
                            browse_path=tag.full_path,
                            status="pending",
                        )
                        await self._mapping_repo.upsert(mapping)

            self._status.discovered_count = len(tags)
            self._status.progress = 1.0
            self._status.is_running = False
            self._status.finished_at = datetime.now(UTC)
            logger.info(
                "Discovery completed",
                server_url=server_url,
                discovered=len(tags),
            )
        except asyncio.CancelledError:
            self._status.is_running = False
            self._status.finished_at = datetime.now(UTC)
            logger.warning("Discovery cancelled", server_url=server_url)
        except Exception as exc:  # pragma: no cover - unexpected failures
            self._status.is_running = False
            self._status.error = str(exc)
            self._status.finished_at = datetime.now(UTC)
            logger.error("Discovery failed", error=str(exc), server_url=server_url)
