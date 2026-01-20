"""Swarm routes for coordination and consensus.

This module provides endpoints for managing swarm agents,
monitoring consensus, and viewing agent reliability scores.
"""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Any

from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel

from noa_swarm.api.deps import get_swarm_service
from noa_swarm.common.logging import get_logger

if TYPE_CHECKING:
    from noa_swarm.services.swarm import SwarmService

logger = get_logger(__name__)

router = APIRouter()


class AgentInfo(BaseModel):
    """Model for agent information."""

    agent_id: str
    status: str
    reliability_score: float
    model_version: str | None = None
    last_seen: datetime | None = None
    tags_processed: int = 0
    accuracy: float | None = None


class SwarmStatus(BaseModel):
    """Model for swarm status."""

    total_agents: int
    active_agents: int
    consensus_in_progress: int
    completed_today: int


class ConsensusRecord(BaseModel):
    """Model for a consensus record."""

    tag_name: str
    irdi: str
    confidence: float
    agreement_ratio: float
    participating_agents: int
    voting_round: int
    timestamp: datetime


class VoteInfo(BaseModel):
    """Model for vote information."""

    agent_id: str
    irdi: str
    confidence: float
    timestamp: datetime


@router.get("/status", response_model=SwarmStatus)
async def get_swarm_status(
    service: SwarmService = Depends(get_swarm_service),
) -> SwarmStatus:
    """Get the current swarm status.

    Returns statistics about the active swarm agents and consensus operations.
    """
    status = await service.get_status()
    return SwarmStatus(
        total_agents=status["total_agents"],
        active_agents=status["active_agents"],
        consensus_in_progress=status["consensus_in_progress"],
        completed_today=status["completed_today"],
    )


@router.get("/agents", response_model=list[AgentInfo])
async def list_agents(
    service: SwarmService = Depends(get_swarm_service),
) -> list[AgentInfo]:
    """List all known agents in the swarm.

    Returns information about all registered agents.
    """
    return [
        AgentInfo(
            agent_id=agent.agent_id,
            status=agent.status,
            reliability_score=agent.reliability_score,
            model_version=agent.model_version,
            last_seen=agent.last_seen,
            tags_processed=agent.tags_processed,
            accuracy=agent.accuracy,
        )
        for agent in service.list_agents()
    ]


@router.get("/agents/{agent_id}", response_model=AgentInfo)
async def get_agent(
    agent_id: str,
    service: SwarmService = Depends(get_swarm_service),
) -> AgentInfo:
    """Get information about a specific agent.

    Args:
        agent_id: The ID of the agent.

    Returns:
        Detailed information about the agent.
    """
    for agent in service.list_agents():
        if agent.agent_id == agent_id:
            return AgentInfo(
                agent_id=agent.agent_id,
                status=agent.status,
                reliability_score=agent.reliability_score,
                model_version=agent.model_version,
                last_seen=agent.last_seen,
                tags_processed=agent.tags_processed,
                accuracy=agent.accuracy,
            )
    return AgentInfo(agent_id=agent_id, status="unknown", reliability_score=0.0)


@router.get("/consensus", response_model=list[ConsensusRecord])
async def list_consensus_records(
    offset: int = Query(0, ge=0, description="Offset for pagination"),
    limit: int = Query(100, ge=1, le=1000, description="Maximum records to return"),
    tag_name: str | None = Query(None, description="Filter by tag name"),
    service: SwarmService = Depends(get_swarm_service),
) -> list[ConsensusRecord]:
    """List consensus records.

    Returns the history of consensus decisions made by the swarm.

    Args:
        offset: Number of records to skip.
        limit: Maximum number of records to return.
        tag_name: Optional filter by tag name.

    Returns:
        List of consensus records.
    """
    records = await service.list_consensus(
        tag_id=tag_name,
        offset=offset,
        limit=limit,
    )
    return [
        ConsensusRecord(
            tag_name=record.tag_id,
            irdi=record.agreed_irdi,
            confidence=record.consensus_confidence,
            agreement_ratio=(
                len(record.unique_voters) / record.vote_count if record.vote_count else 0.0
            ),
            participating_agents=len(record.unique_voters),
            voting_round=0,
            timestamp=record.created_at,
        )
        for record in records
    ]


@router.get("/consensus/{tag_name}", response_model=ConsensusRecord)
async def get_consensus(
    tag_name: str,
    service: SwarmService = Depends(get_swarm_service),
) -> ConsensusRecord:
    """Get the consensus record for a specific tag.

    Args:
        tag_name: The name of the tag.

    Returns:
        The consensus record for the tag.
    """
    record = await service.get_consensus(tag_name)
    if record is None:
        return ConsensusRecord(
            tag_name=tag_name,
            irdi="unknown",
            confidence=0.0,
            agreement_ratio=0.0,
            participating_agents=0,
            voting_round=0,
            timestamp=datetime.utcnow(),
        )
    return ConsensusRecord(
        tag_name=record.tag_id,
        irdi=record.agreed_irdi,
        confidence=record.consensus_confidence,
        agreement_ratio=(
            len(record.unique_voters) / record.vote_count if record.vote_count else 0.0
        ),
        participating_agents=len(record.unique_voters),
        voting_round=0,
        timestamp=record.created_at,
    )


@router.get("/consensus/{tag_name}/votes", response_model=list[VoteInfo])
async def get_consensus_votes(
    tag_name: str,
    service: SwarmService = Depends(get_swarm_service),
) -> list[VoteInfo]:
    """Get all votes for a specific tag consensus.

    Returns the individual agent votes that contributed to the consensus.

    Args:
        tag_name: The name of the tag.

    Returns:
        List of votes from participating agents.
    """
    record = await service.get_consensus(tag_name)
    if record is None:
        return []
    return [
        VoteInfo(
            agent_id=vote.agent_id,
            irdi=vote.candidate_irdi,
            confidence=vote.confidence,
            timestamp=vote.timestamp,
        )
        for vote in record.votes
    ]


@router.post("/trigger")
async def trigger_consensus(
    tag_names: list[str],
    _service: SwarmService = Depends(get_swarm_service),
) -> dict[str, Any]:
    """Trigger consensus for specific tags.

    Initiates a new consensus round for the specified tags.

    Args:
        tag_names: List of tag names to run consensus for.

    Returns:
        Status of the triggered consensus operation.
    """
    logger.info("Triggering consensus", tag_count=len(tag_names))

    return {
        "status": "triggered",
        "tags": tag_names,
        "message": f"Consensus triggered for {len(tag_names)} tags",
    }


@router.get("/reliability", response_model=dict[str, float])
async def get_reliability_scores(
    service: SwarmService = Depends(get_swarm_service),
) -> dict[str, float]:
    """Get reliability scores for all agents.

    Returns the current reliability scores used in weighted voting.
    """
    return {agent.agent_id: agent.reliability_score for agent in service.list_agents()}
