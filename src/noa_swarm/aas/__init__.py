"""AAS (Asset Administration Shell) export module.

This module provides functionality for creating and exporting AAS structures
following the Industry 4.0 Asset Administration Shell specification.

Components:
- **submodels**: TagMappingSubmodel for discovered tag data
- **basyx_export**: AAS export to JSON, XML, and AASX formats

Example usage:
    >>> from noa_swarm.aas import (
    ...     TagMappingSubmodel,
    ...     DiscoveredTag,
    ...     MappingStatus,
    ...     AASExporter,
    ...     ExportFormat,
    ...     create_tag_mapping_aas,
    ... )
    >>> submodel = TagMappingSubmodel(submodel_id="urn:example:sm:1")
    >>> submodel.add_tag(DiscoveredTag(
    ...     tag_name="TIC-101.PV",
    ...     browse_path="/Objects/TIC-101/PV",
    ... ))
    >>> aas, sm = create_tag_mapping_aas(submodel, "urn:aas:1", "urn:asset:1")
    >>> exporter = AASExporter()
    >>> json_str = exporter.export_json(aas, sm)
"""

from noa_swarm.aas.submodels import (
    ConsensusInfo,
    DiscoveredTag,
    MappingStatistics,
    MappingStatus,
    TagMappingSubmodel,
)
from noa_swarm.aas.basyx_export import (
    AASExporter,
    ExportConfig,
    ExportFormat,
    create_tag_mapping_aas,
)

__all__ = [
    # Submodel types
    "ConsensusInfo",
    "DiscoveredTag",
    "MappingStatistics",
    "MappingStatus",
    "TagMappingSubmodel",
    # Export functionality
    "AASExporter",
    "ExportConfig",
    "ExportFormat",
    "create_tag_mapping_aas",
]
