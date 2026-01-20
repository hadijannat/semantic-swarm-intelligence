"""AAS routes for Asset Administration Shell export.

This module provides endpoints for creating and exporting
Asset Administration Shell packages.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, Query
from fastapi.responses import FileResponse
from pydantic import BaseModel

from noa_swarm.api.deps import get_aas_service
from noa_swarm.common.logging import get_logger
from noa_swarm.services.aas import AASService
from noa_swarm.aas import AASExporter, ExportConfig, ExportFormat, create_tag_mapping_aas

logger = get_logger(__name__)

router = APIRouter()


class SubmodelInfo(BaseModel):
    """Model for submodel information."""

    submodel_id: str
    semantic_id: str
    tag_count: int
    mapping_rate: float
    statistics: dict[str, int]


class ExportRequest(BaseModel):
    """Request model for AAS export."""

    aas_id: str = "urn:noa:aas:tagmapping:default"
    asset_id: str = "urn:noa:asset:plant:default"
    submodel_id: str = "urn:noa:submodel:tagmapping:default"
    include_timestamps: bool = True
    include_statistics: bool = True


class ExportResponse(BaseModel):
    """Response model for export operations."""

    success: bool
    format: str
    file_size: int | None = None
    message: str


@router.get("/submodel", response_model=SubmodelInfo)
async def get_submodel_info(
    service: AASService = Depends(get_aas_service),
) -> SubmodelInfo:
    """Get information about the current tag mapping submodel.

    Returns statistics about the current tag mapping state.
    """
    submodel = await service.build_submodel("urn:noa:submodel:tagmapping:default")
    stats = submodel.get_statistics()
    return SubmodelInfo(
        submodel_id=submodel.submodel_id,
        semantic_id=submodel.semantic_id,
        tag_count=len(submodel.tags),
        mapping_rate=stats.mapping_rate,
        statistics={
            "total": stats.total_tags,
            "pending": stats.pending_tags,
            "mapped": stats.mapped_tags,
            "verified": stats.verified_tags,
            "rejected": stats.rejected_tags,
            "conflict": stats.conflict_tags,
        },
    )


@router.get("/submodel/json")
async def get_submodel_json(
    service: AASService = Depends(get_aas_service),
) -> dict[str, Any]:
    """Get the tag mapping submodel as JSON.

    Returns the complete submodel structure in AAS JSON format.
    """
    submodel = await service.build_submodel("urn:noa:submodel:tagmapping:default")
    aas, sm = create_tag_mapping_aas(
        submodel=submodel,
        aas_id="urn:noa:aas:tagmapping:default",
        asset_id="urn:noa:asset:plant:default",
    )

    exporter = AASExporter(ExportConfig(pretty_print=True))
    json_str = exporter.export_json(aas, sm)

    import json
    return json.loads(json_str)


@router.post("/export/json")
async def export_json(
    request: ExportRequest,
    service: AASService = Depends(get_aas_service),
) -> FileResponse:
    """Export the AAS package as JSON.

    Creates a JSON file containing the AAS and submodel.

    Args:
        request: Export configuration options.

    Returns:
        JSON file download.
    """
    submodel = await service.build_submodel(request.submodel_id)
    aas, sm = create_tag_mapping_aas(
        submodel=submodel,
        aas_id=request.aas_id,
        asset_id=request.asset_id,
    )

    config = ExportConfig(
        include_timestamps=request.include_timestamps,
        include_statistics=request.include_statistics,
    )
    exporter = AASExporter(config=config)

    # Create temporary file
    temp_file = tempfile.NamedTemporaryFile(
        suffix=".json",
        delete=False,
        prefix="aas_export_",
    )
    output_path = Path(temp_file.name)
    temp_file.close()

    exporter.export_to_file(aas, sm, output_path, ExportFormat.JSON)

    logger.info("Exported AAS to JSON", path=str(output_path))

    return FileResponse(
        path=output_path,
        filename="tag_mapping_aas.json",
        media_type="application/json",
    )


@router.post("/export/xml")
async def export_xml(
    request: ExportRequest,
    service: AASService = Depends(get_aas_service),
) -> FileResponse:
    """Export the AAS package as XML.

    Creates an XML file containing the AAS and submodel.

    Args:
        request: Export configuration options.

    Returns:
        XML file download.
    """
    submodel = await service.build_submodel(request.submodel_id)
    aas, sm = create_tag_mapping_aas(
        submodel=submodel,
        aas_id=request.aas_id,
        asset_id=request.asset_id,
    )

    config = ExportConfig(
        include_timestamps=request.include_timestamps,
        include_statistics=request.include_statistics,
    )
    exporter = AASExporter(config=config)

    # Create temporary file
    temp_file = tempfile.NamedTemporaryFile(
        suffix=".xml",
        delete=False,
        prefix="aas_export_",
    )
    output_path = Path(temp_file.name)
    temp_file.close()

    exporter.export_to_file(aas, sm, output_path, ExportFormat.XML)

    logger.info("Exported AAS to XML", path=str(output_path))

    return FileResponse(
        path=output_path,
        filename="tag_mapping_aas.xml",
        media_type="application/xml",
    )


@router.post("/export/aasx")
async def export_aasx(
    request: ExportRequest,
    service: AASService = Depends(get_aas_service),
) -> FileResponse:
    """Export the AAS package as AASX.

    Creates an AASX package file containing the AAS and submodel.

    Args:
        request: Export configuration options.

    Returns:
        AASX file download.
    """
    submodel = await service.build_submodel(request.submodel_id)
    aas, sm = create_tag_mapping_aas(
        submodel=submodel,
        aas_id=request.aas_id,
        asset_id=request.asset_id,
    )

    config = ExportConfig(
        include_timestamps=request.include_timestamps,
        include_statistics=request.include_statistics,
    )
    exporter = AASExporter(config=config)

    # Create temporary file
    temp_file = tempfile.NamedTemporaryFile(
        suffix=".aasx",
        delete=False,
        prefix="aas_export_",
    )
    output_path = Path(temp_file.name)
    temp_file.close()

    exporter.export_to_file(aas, sm, output_path, ExportFormat.AASX)

    logger.info("Exported AAS to AASX", path=str(output_path))

    return FileResponse(
        path=output_path,
        filename="tag_mapping_aas.aasx",
        media_type="application/octet-stream",
    )


@router.get("/formats")
async def list_export_formats() -> list[dict[str, str]]:
    """List available export formats.

    Returns information about supported AAS export formats.
    """
    return [
        {
            "format": "json",
            "name": "AAS JSON",
            "description": "JSON serialization compliant with AAS Part 1",
            "extension": ".json",
        },
        {
            "format": "xml",
            "name": "AAS XML",
            "description": "XML serialization compliant with AAS Part 1",
            "extension": ".xml",
        },
        {
            "format": "aasx",
            "name": "AASX Package",
            "description": "OPC UA-compliant package format for AAS exchange",
            "extension": ".aasx",
        },
    ]
