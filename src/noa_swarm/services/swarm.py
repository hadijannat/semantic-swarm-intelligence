"""Swarm service for consensus and agent registry."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

from noa_swarm.common.schemas import ConsensusRecord
from noa_swarm.storage.base import ConsensusRepository


@dataclass
class AgentStatus:
    """Minimal agent status snapshot."""

    agent_id: str
    status: str = "unknown"
    reliability_score: float = 0.0
    model_version: str | None = None
    last_seen: datetime | None = None
    tags_processed: int = 0
    accuracy: float | None = None


class SwarmService:
    """Service for swarm status and consensus history."""

    def __init__(self, consensus_repo: ConsensusRepository) -> None:
        self._consensus_repo = consensus_repo
        self._agents: dict[str, AgentStatus] = {}

    def list_agents(self) -> list[AgentStatus]:
        return list(self._agents.values())

    def upsert_agent(self, agent: AgentStatus) -> None:
        self._agents[agent.agent_id] = agent

    async def list_consensus(
        self,
        *,
        tag_id: str | None = None,
        offset: int = 0,
        limit: int = 1000,
    ) -> list[ConsensusRecord]:
        return await self._consensus_repo.list(
            tag_id=tag_id,
            offset=offset,
            limit=limit,
        )

    async def get_consensus(self, tag_id: str) -> ConsensusRecord | None:
        return await self._consensus_repo.get(tag_id)

    async def get_status(self) -> dict[str, int]:
        agents = self.list_agents()
        records = await self._consensus_repo.list(limit=10_000)

        completed_today = 0
        today = datetime.now(timezone.utc).date()
        for record in records:
            if record.created_at.date() == today:
                completed_today += 1

        return {
            "total_agents": len(agents),
            "active_agents": len([a for a in agents if a.status == "active"]),
            "consensus_in_progress": 0,
            "completed_today": completed_today,
        }
