"""Mapping routes for tag-to-IRDI operations.

This module provides endpoints for managing tag mappings,
including creation, retrieval, and approval workflows.
"""

from __future__ import annotations

from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel

from noa_swarm.api.deps import get_mapping_service
from noa_swarm.common.logging import get_logger
from noa_swarm.services.mapping import MappingService

logger = get_logger(__name__)

router = APIRouter()


class MappingCandidate(BaseModel):
    """Model for a mapping candidate."""

    irdi: str
    preferred_name: str
    confidence: float
    source: str


class TagMapping(BaseModel):
    """Model for a tag mapping."""

    tag_name: str
    browse_path: str
    irdi: str | None = None
    preferred_name: str | None = None
    status: str = "pending"
    confidence: float | None = None
    candidates: list[MappingCandidate] = []
    created_at: datetime | None = None
    updated_at: datetime | None = None


class MappingListResponse(BaseModel):
    """Response model for mapping list."""

    mappings: list[TagMapping]
    total: int
    stats: dict[str, int]


class MappingCreateRequest(BaseModel):
    """Request model for creating a mapping."""

    tag_name: str
    browse_path: str
    irdi: str | None = None


class MappingUpdateRequest(BaseModel):
    """Request model for updating a mapping."""

    irdi: str | None = None
    status: str | None = None


@router.get("/", response_model=MappingListResponse)
async def list_mappings(
    status: str | None = Query(None, description="Filter by status"),
    offset: int = Query(0, ge=0, description="Offset for pagination"),
    limit: int = Query(100, ge=1, le=1000, description="Maximum mappings to return"),
    service: MappingService = Depends(get_mapping_service),
) -> MappingListResponse:
    """List all tag mappings.

    Returns the list of tag mappings with optional filtering.

    Args:
        status: Filter by mapping status.
        offset: Number of mappings to skip.
        limit: Maximum number of mappings to return.

    Returns:
        List of mappings with total count and statistics.
    """
    mappings, stats = await service.list_mappings(
        status=status,
        offset=offset,
        limit=limit,
    )
    return MappingListResponse(
        mappings=[
            TagMapping(
                tag_name=mapping.tag_name,
                browse_path=mapping.browse_path,
                irdi=mapping.irdi,
                preferred_name=mapping.preferred_name,
                status=mapping.status,
                confidence=mapping.confidence,
                candidates=[
                    MappingCandidate(
                        irdi=candidate.irdi,
                        preferred_name="",
                        confidence=candidate.confidence,
                        source=candidate.source_model,
                    )
                    for candidate in mapping.candidates
                ],
                created_at=mapping.created_at,
                updated_at=mapping.updated_at,
            )
            for mapping in mappings
        ],
        total=stats.total,
        stats={
            "pending": stats.pending,
            "mapped": stats.mapped,
            "verified": stats.verified,
            "rejected": stats.rejected,
            "conflict": stats.conflict,
        },
    )


@router.post("/", response_model=TagMapping)
async def create_mapping(
    request: MappingCreateRequest,
    service: MappingService = Depends(get_mapping_service),
) -> TagMapping:
    """Create a new tag mapping.

    Creates a mapping entry for a tag, optionally with an initial IRDI.

    Args:
        request: The mapping creation request.

    Returns:
        The created mapping.
    """
    logger.info("Creating mapping", tag_name=request.tag_name)

    mapping = await service.create_mapping(
        tag_name=request.tag_name,
        browse_path=request.browse_path,
        irdi=request.irdi,
    )
    return TagMapping(
        tag_name=mapping.tag_name,
        browse_path=mapping.browse_path,
        irdi=mapping.irdi,
        preferred_name=mapping.preferred_name,
        status=mapping.status,
        confidence=mapping.confidence,
        created_at=mapping.created_at,
        updated_at=mapping.updated_at,
    )


@router.get("/{tag_name}", response_model=TagMapping)
async def get_mapping(
    tag_name: str,
    service: MappingService = Depends(get_mapping_service),
) -> TagMapping:
    """Get a specific mapping by tag name.

    Args:
        tag_name: The name of the tag.

    Returns:
        The mapping for the specified tag.

    Raises:
        HTTPException: If mapping is not found.
    """
    mapping = await service.get_mapping(tag_name)
    if mapping is None:
        raise HTTPException(status_code=404, detail=f"Mapping for {tag_name} not found")

    return TagMapping(
        tag_name=mapping.tag_name,
        browse_path=mapping.browse_path,
        irdi=mapping.irdi,
        preferred_name=mapping.preferred_name,
        status=mapping.status,
        confidence=mapping.confidence,
        created_at=mapping.created_at,
        updated_at=mapping.updated_at,
    )


@router.put("/{tag_name}", response_model=TagMapping)
async def update_mapping(
    tag_name: str,
    request: MappingUpdateRequest,
    service: MappingService = Depends(get_mapping_service),
) -> TagMapping:
    """Update a tag mapping.

    Updates the IRDI or status of an existing mapping.

    Args:
        tag_name: The name of the tag.
        request: The update request.

    Returns:
        The updated mapping.

    Raises:
        HTTPException: If mapping is not found.
    """
    mapping = await service.update_mapping(
        tag_name,
        irdi=request.irdi,
        status=request.status,
    )
    if mapping is None:
        raise HTTPException(status_code=404, detail=f"Mapping for {tag_name} not found")

    return TagMapping(
        tag_name=mapping.tag_name,
        browse_path=mapping.browse_path,
        irdi=mapping.irdi,
        preferred_name=mapping.preferred_name,
        status=mapping.status,
        confidence=mapping.confidence,
        created_at=mapping.created_at,
        updated_at=mapping.updated_at,
    )


@router.delete("/{tag_name}")
async def delete_mapping(
    tag_name: str,
    service: MappingService = Depends(get_mapping_service),
) -> dict[str, str]:
    """Delete a tag mapping.

    Args:
        tag_name: The name of the tag.

    Returns:
        Confirmation of deletion.

    Raises:
        HTTPException: If mapping is not found.
    """
    deleted = await service.delete_mapping(tag_name)
    if not deleted:
        raise HTTPException(status_code=404, detail=f"Mapping for {tag_name} not found")
    return {"status": "deleted", "tag_name": tag_name}


@router.post("/{tag_name}/approve")
async def approve_mapping(
    tag_name: str,
    service: MappingService = Depends(get_mapping_service),
) -> TagMapping:
    """Approve a mapping (change status to verified).

    Args:
        tag_name: The name of the tag.

    Returns:
        The updated mapping.

    Raises:
        HTTPException: If mapping is not found.
    """
    mapping = await service.approve_mapping(tag_name)
    if mapping is None:
        raise HTTPException(status_code=404, detail=f"Mapping for {tag_name} not found")
    return TagMapping(
        tag_name=mapping.tag_name,
        browse_path=mapping.browse_path,
        irdi=mapping.irdi,
        preferred_name=mapping.preferred_name,
        status=mapping.status,
        confidence=mapping.confidence,
        created_at=mapping.created_at,
        updated_at=mapping.updated_at,
    )


@router.post("/{tag_name}/reject")
async def reject_mapping(
    tag_name: str,
    service: MappingService = Depends(get_mapping_service),
) -> TagMapping:
    """Reject a mapping (change status to rejected).

    Args:
        tag_name: The name of the tag.

    Returns:
        The updated mapping.

    Raises:
        HTTPException: If mapping is not found.
    """
    mapping = await service.reject_mapping(tag_name)
    if mapping is None:
        raise HTTPException(status_code=404, detail=f"Mapping for {tag_name} not found")
    return TagMapping(
        tag_name=mapping.tag_name,
        browse_path=mapping.browse_path,
        irdi=mapping.irdi,
        preferred_name=mapping.preferred_name,
        status=mapping.status,
        confidence=mapping.confidence,
        created_at=mapping.created_at,
        updated_at=mapping.updated_at,
    )


@router.get("/{tag_name}/candidates", response_model=list[MappingCandidate])
async def get_mapping_candidates(tag_name: str) -> list[MappingCandidate]:
    """Get candidate IRDIs for a tag.

    Returns the list of candidate mappings suggested by the ML model.

    Args:
        tag_name: The name of the tag.

    Returns:
        List of candidate mappings with confidence scores.

    Raises:
        HTTPException: If tag is not found.
    """
    # TODO: Integrate with actual ML inference
    raise HTTPException(status_code=404, detail=f"Tag {tag_name} not found")
