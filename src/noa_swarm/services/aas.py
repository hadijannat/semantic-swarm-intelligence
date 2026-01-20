"""Service for building and exporting AAS tag mapping submodels."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from noa_swarm.aas import (
    AASExporter,
    ConsensusInfo,
    DiscoveredTag,
    ExportConfig,
    ExportFormat,
    MappingStatus,
    TagMappingSubmodel,
    create_tag_mapping_aas,
)

if TYPE_CHECKING:
    from noa_swarm.common.schemas import TagMappingRecord
    from noa_swarm.storage.base import MappingRepository


class AASService:
    """Service for AAS export and submodel creation."""

    def __init__(self, mapping_repo: MappingRepository) -> None:
        self._mapping_repo = mapping_repo

    async def build_submodel(self, submodel_id: str) -> TagMappingSubmodel:
        submodel = TagMappingSubmodel(submodel_id=submodel_id)
        mappings = await self._mapping_repo.list(limit=10_000)

        for mapping in mappings:
            submodel.add_tag(self._to_discovered_tag(mapping))

        return submodel

    async def export(
        self,
        *,
        submodel_id: str,
        aas_id: str,
        asset_id: str,
        output_path: str | Path,
        export_format: ExportFormat,
        include_timestamps: bool = True,
        include_statistics: bool = True,
    ) -> None:
        submodel = await self.build_submodel(submodel_id)
        aas, sm = create_tag_mapping_aas(submodel, aas_id, asset_id)

        config = ExportConfig(
            include_timestamps=include_timestamps,
            include_statistics=include_statistics,
        )
        exporter = AASExporter(config=config)
        path = Path(output_path)
        exporter.export_to_file(aas, sm, path, export_format)

    @staticmethod
    def _to_discovered_tag(mapping: TagMappingRecord) -> DiscoveredTag:
        status_map = {
            "pending": MappingStatus.PENDING,
            "mapped": MappingStatus.MAPPED,
            "verified": MappingStatus.VERIFIED,
            "rejected": MappingStatus.REJECTED,
            "conflict": MappingStatus.CONFLICT,
        }
        status = status_map.get(mapping.status, MappingStatus.PENDING)

        consensus = None
        if mapping.confidence is not None:
            consensus = ConsensusInfo(confidence=mapping.confidence)

        return DiscoveredTag(
            tag_name=mapping.tag_name,
            browse_path=mapping.browse_path,
            irdi=mapping.irdi,
            preferred_name=mapping.preferred_name,
            status=status,
            consensus=consensus,
            updated_at=mapping.updated_at,
        )
