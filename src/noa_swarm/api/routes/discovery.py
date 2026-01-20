"""Discovery routes for OPC UA tag browsing.

This module provides endpoints for discovering and browsing
OPC UA tags from connected servers.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel

from noa_swarm.api.deps import get_discovery_service
from noa_swarm.common.logging import get_logger
from noa_swarm.services.discovery import DiscoveryService

logger = get_logger(__name__)

router = APIRouter()


class DiscoveredNode(BaseModel):
    """Model for a discovered OPC UA node."""

    node_id: str
    browse_name: str
    display_name: str
    node_class: str
    data_type: str | None = None
    browse_path: str


class DiscoveryResponse(BaseModel):
    """Response model for discovery operations."""

    nodes: list[DiscoveredNode]
    total: int
    server_url: str | None = None


class DiscoveryStatus(BaseModel):
    """Model for discovery status."""

    is_running: bool
    progress: float
    discovered_count: int
    server_url: str | None = None


@router.get("/status", response_model=DiscoveryStatus)
async def get_discovery_status(
    service: DiscoveryService = Depends(get_discovery_service),
) -> DiscoveryStatus:
    """Get the current discovery status.

    Returns the progress and state of any ongoing discovery operation.
    """
    status = service.get_status()
    return DiscoveryStatus(
        is_running=status.is_running,
        progress=status.progress,
        discovered_count=status.discovered_count,
        server_url=status.server_url,
    )


@router.post("/start")
async def start_discovery(
    server_url: str = Query(..., description="OPC UA server URL"),
    root_node: str = Query("i=85", description="Root node to start browsing from"),
    max_depth: int = Query(10, ge=1, le=100, description="Maximum browse depth"),
    service: DiscoveryService = Depends(get_discovery_service),
) -> dict[str, Any]:
    """Start OPC UA tag discovery.

    Initiates an asynchronous discovery operation on the specified
    OPC UA server.

    Args:
        server_url: URL of the OPC UA server.
        root_node: NodeId to start browsing from.
        max_depth: Maximum depth to browse.

    Returns:
        Discovery operation ID and status.
    """
    logger.info(
        "Starting discovery",
        server_url=server_url,
        root_node=root_node,
        max_depth=max_depth,
    )

    status = await service.start(server_url, root_node, max_depth)
    return {
        "operation_id": status.operation_id,
        "status": "started" if status.is_running else "idle",
        "server_url": status.server_url,
        "message": "Discovery operation started" if status.is_running else "Discovery idle",
    }


@router.post("/stop")
async def stop_discovery(
    service: DiscoveryService = Depends(get_discovery_service),
) -> dict[str, str]:
    """Stop the current discovery operation.

    Returns:
        Confirmation of discovery stop.
    """
    status = await service.stop()
    return {
        "status": "stopped" if not status.is_running else "running",
        "message": "Discovery operation stopped",
    }


@router.get("/nodes", response_model=DiscoveryResponse)
async def get_discovered_nodes(
    offset: int = Query(0, ge=0, description="Offset for pagination"),
    limit: int = Query(100, ge=1, le=1000, description="Maximum nodes to return"),
    filter_pattern: str | None = Query(None, description="Filter by name pattern"),
    service: DiscoveryService = Depends(get_discovery_service),
) -> DiscoveryResponse:
    """Get discovered nodes.

    Returns the list of discovered OPC UA nodes with pagination.

    Args:
        offset: Number of nodes to skip.
        limit: Maximum number of nodes to return.
        filter_pattern: Optional regex pattern to filter nodes.

    Returns:
        List of discovered nodes with total count.
    """
    tags = await service.list_nodes(offset=offset, limit=limit, filter_pattern=filter_pattern)
    nodes = [
        DiscoveredNode(
            node_id=tag.node_id,
            browse_name=tag.browse_name,
            display_name=tag.display_name or tag.browse_name,
            node_class="Variable",
            data_type=tag.data_type,
            browse_path=tag.full_path,
        )
        for tag in tags
    ]
    return DiscoveryResponse(
        nodes=nodes,
        total=len(nodes),
        server_url=service.get_status().server_url,
    )


@router.get("/nodes/{node_id}")
async def get_node_details(
    node_id: str,
    service: DiscoveryService = Depends(get_discovery_service),
) -> dict[str, Any]:
    """Get details for a specific node.

    Args:
        node_id: The NodeId of the node to retrieve.

    Returns:
        Detailed information about the node.

    Raises:
        HTTPException: If node is not found.
    """
    record = await service.get_node(node_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"Node {node_id} not found")

    return {
        "node_id": record.node_id,
        "browse_name": record.browse_name,
        "display_name": record.display_name or record.browse_name,
        "data_type": record.data_type,
        "description": record.description,
        "browse_path": record.full_path,
        "source_server": record.source_server,
        "engineering_unit": record.engineering_unit,
        "access_level": record.access_level,
    }
