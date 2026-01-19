"""BaSyx AAS export functionality.

This module provides export capabilities for Asset Administration Shells
using the BaSyx Python SDK. Supports multiple formats:

- **JSON**: AAS JSON serialization (Part 1 compliant)
- **XML**: AAS XML serialization (Part 1 compliant)
- **AASX**: OPC UA package format for AAS exchange

Example usage:
    >>> from noa_swarm.aas.basyx_export import AASExporter, create_tag_mapping_aas
    >>> from noa_swarm.aas.submodels import TagMappingSubmodel, DiscoveredTag
    >>>
    >>> submodel = TagMappingSubmodel(submodel_id="urn:test:sm:1")
    >>> submodel.add_tag(DiscoveredTag(tag_name="TIC-101", browse_path="/TIC-101"))
    >>> aas, sm = create_tag_mapping_aas(submodel, "urn:aas:1", "urn:asset:1")
    >>> exporter = AASExporter()
    >>> exporter.export_to_file(aas, sm, Path("output.aasx"), ExportFormat.AASX)

References:
    - AAS Part 1: https://www.plattform-i40.de/IP/Redaktion/EN/Downloads/Publikation/Details_of_the_Asset_Administration_Shell_Part1_V3.html
    - BaSyx SDK: https://github.com/eclipse-basyx/basyx-python-sdk
"""

from __future__ import annotations

import io
import json as json_module
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING

from basyx.aas import model
from basyx.aas.adapter import json as aas_json
from basyx.aas.adapter import xml as aas_xml
from basyx.aas.adapter import aasx as aas_aasx

from noa_swarm.common.logging import get_logger

if TYPE_CHECKING:
    from noa_swarm.aas.submodels import TagMappingSubmodel

logger = get_logger(__name__)


class ExportFormat(str, Enum):
    """Supported export formats for AAS.

    Attributes:
        JSON: AAS JSON serialization format.
        XML: AAS XML serialization format.
        AASX: OPC UA package format (ZIP-based).
    """

    JSON = "json"
    XML = "xml"
    AASX = "aasx"


@dataclass
class ExportConfig:
    """Configuration for AAS export.

    Attributes:
        include_timestamps: Include timestamp properties in export.
        include_statistics: Include statistics collection in export.
        pretty_print: Use indented formatting for JSON/XML.
    """

    include_timestamps: bool = True
    include_statistics: bool = True
    pretty_print: bool = True


def create_tag_mapping_aas(
    submodel: TagMappingSubmodel,
    aas_id: str,
    asset_id: str,
    aas_id_short: str = "TagMappingAAS",
    asset_id_short: str = "MappedAsset",
) -> tuple[model.AssetAdministrationShell, model.Submodel]:
    """Create an AAS containing the tag mapping submodel.

    Creates a complete Asset Administration Shell structure with:
    - Asset information
    - AAS shell referencing the submodel
    - Converted BaSyx Submodel

    Args:
        submodel: The TagMappingSubmodel to include.
        aas_id: Unique identifier for the AAS.
        asset_id: Unique identifier for the asset.
        aas_id_short: Short identifier for the AAS.
        asset_id_short: Short identifier for the asset.

    Returns:
        Tuple of (AssetAdministrationShell, Submodel).
    """
    # Convert our submodel to BaSyx submodel
    basyx_submodel = submodel.to_basyx_submodel()

    # Create asset information
    asset_info = model.AssetInformation(
        asset_kind=model.AssetKind.INSTANCE,
        global_asset_id=asset_id,
    )

    # Create submodel reference
    submodel_ref = model.ModelReference.from_referable(basyx_submodel)

    # Create the AAS
    aas = model.AssetAdministrationShell(
        id_=aas_id,
        id_short=aas_id_short,
        asset_information=asset_info,
        submodel={submodel_ref},
    )

    logger.info(
        "Created AAS for tag mapping",
        aas_id=aas_id,
        asset_id=asset_id,
        tag_count=len(submodel.tags),
    )

    return aas, basyx_submodel


class AASExporter:
    """Export AAS and submodels to various formats.

    Provides methods for exporting Asset Administration Shells and
    their submodels to JSON, XML, and AASX package formats using
    the BaSyx Python SDK.

    Attributes:
        _config: Export configuration settings.
    """

    def __init__(self, config: ExportConfig | None = None) -> None:
        """Initialize the AAS exporter.

        Args:
            config: Optional export configuration. Uses defaults if not provided.
        """
        self._config = config or ExportConfig()

        logger.debug(
            "Initialized AASExporter",
            pretty_print=self._config.pretty_print,
        )

    def export_json(
        self,
        aas: model.AssetAdministrationShell,
        submodel: model.Submodel,
    ) -> str:
        """Export AAS and submodel to JSON string.

        Args:
            aas: The AAS to export.
            submodel: The submodel to export.

        Returns:
            JSON string representation.
        """
        # Create object store with both elements
        object_store: model.DictObjectStore[model.Identifiable] = model.DictObjectStore(
            [aas, submodel]
        )

        # Serialize to JSON
        output = io.BytesIO()
        aas_json.write_aas_json_file(output, object_store)

        json_str = output.getvalue().decode("utf-8")

        if self._config.pretty_print:
            # Re-format with indentation
            data = json_module.loads(json_str)
            json_str = json_module.dumps(data, indent=2)

        logger.debug("Exported AAS to JSON", aas_id=str(aas.id))
        return json_str

    def export_xml(
        self,
        aas: model.AssetAdministrationShell,
        submodel: model.Submodel,
    ) -> str:
        """Export AAS and submodel to XML string.

        Args:
            aas: The AAS to export.
            submodel: The submodel to export.

        Returns:
            XML string representation.
        """
        # Create object store with both elements
        object_store: model.DictObjectStore[model.Identifiable] = model.DictObjectStore(
            [aas, submodel]
        )

        # Serialize to XML
        output = io.BytesIO()
        aas_xml.write_aas_xml_file(output, object_store)

        xml_str = output.getvalue().decode("utf-8")

        logger.debug("Exported AAS to XML", aas_id=str(aas.id))
        return xml_str

    def export_aasx(
        self,
        aas: model.AssetAdministrationShell,
        submodel: model.Submodel,
        output_path: Path,
    ) -> None:
        """Export AAS and submodel to AASX package.

        Creates an OPC UA-compliant AASX package containing
        the AAS and submodel in JSON format.

        Args:
            aas: The AAS to export.
            submodel: The submodel to export.
            output_path: Path for the output .aasx file.
        """
        # Create object store with both elements
        object_store: model.DictObjectStore[model.Identifiable] = model.DictObjectStore(
            [aas, submodel]
        )

        # Create file store for any supplementary files
        file_store: aas_aasx.DictSupplementaryFileContainer = (
            aas_aasx.DictSupplementaryFileContainer()
        )

        # Write AASX package
        with aas_aasx.AASXWriter(str(output_path)) as writer:
            writer.write_aas(
                aas_ids=aas.id,
                object_store=object_store,
                file_store=file_store,
            )

        logger.info(
            "Exported AAS to AASX package",
            aas_id=str(aas.id),
            output_path=str(output_path),
        )

    def export_to_file(
        self,
        aas: model.AssetAdministrationShell,
        submodel: model.Submodel,
        output_path: Path,
        format: ExportFormat,
    ) -> None:
        """Export AAS to file in specified format.

        Args:
            aas: The AAS to export.
            submodel: The submodel to export.
            output_path: Path for the output file.
            format: The export format to use.

        Raises:
            ValueError: If format is not supported.
        """
        if format == ExportFormat.JSON:
            content = self.export_json(aas, submodel)
            output_path.write_text(content, encoding="utf-8")

        elif format == ExportFormat.XML:
            content = self.export_xml(aas, submodel)
            output_path.write_text(content, encoding="utf-8")

        elif format == ExportFormat.AASX:
            self.export_aasx(aas, submodel, output_path)

        else:
            raise ValueError(f"Unsupported export format: {format}")

        logger.info(
            "Exported AAS to file",
            format=format.value,
            output_path=str(output_path),
        )
