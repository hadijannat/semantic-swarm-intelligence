"""Unit tests for BaSyx AAS export functionality."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING
from zipfile import ZipFile

import pytest

from noa_swarm.aas.submodels import (
    TagMappingSubmodel,
    DiscoveredTag,
    MappingStatus,
    ConsensusInfo,
)
from noa_swarm.aas.basyx_export import (
    AASExporter,
    ExportConfig,
    ExportFormat,
    create_tag_mapping_aas,
)

if TYPE_CHECKING:
    pass


class TestExportConfig:
    """Tests for ExportConfig dataclass."""

    def test_default_values(self) -> None:
        """Test ExportConfig has correct default values."""
        config = ExportConfig()

        assert config.include_timestamps is True
        assert config.include_statistics is True
        assert config.pretty_print is True

    def test_custom_values(self) -> None:
        """Test ExportConfig accepts custom values."""
        config = ExportConfig(
            include_timestamps=False,
            include_statistics=False,
            pretty_print=False,
        )

        assert config.include_timestamps is False
        assert config.include_statistics is False
        assert config.pretty_print is False


class TestExportFormat:
    """Tests for ExportFormat enum."""

    def test_all_formats_defined(self) -> None:
        """Test all expected export formats are defined."""
        assert hasattr(ExportFormat, "JSON")
        assert hasattr(ExportFormat, "XML")
        assert hasattr(ExportFormat, "AASX")

    def test_format_values(self) -> None:
        """Test format enum values."""
        assert ExportFormat.JSON.value == "json"
        assert ExportFormat.XML.value == "xml"
        assert ExportFormat.AASX.value == "aasx"


class TestCreateTagMappingAAS:
    """Tests for create_tag_mapping_aas function."""

    def test_creates_aas_with_submodel(self) -> None:
        """Test creating AAS with tag mapping submodel."""
        submodel = TagMappingSubmodel(
            submodel_id="urn:test:submodel:1",
        )
        submodel.add_tag(DiscoveredTag(
            tag_name="TIC-101.PV",
            browse_path="/Objects/TIC-101/PV",
        ))

        aas, sm = create_tag_mapping_aas(
            submodel=submodel,
            aas_id="urn:test:aas:plant001",
            asset_id="urn:test:asset:plant001",
        )

        # Should return AAS and Submodel objects
        from basyx.aas.model import AssetAdministrationShell, Submodel
        assert isinstance(aas, AssetAdministrationShell)
        assert isinstance(sm, Submodel)
        assert str(aas.id) == "urn:test:aas:plant001"

    def test_aas_references_submodel(self) -> None:
        """Test that AAS references the submodel."""
        submodel = TagMappingSubmodel(submodel_id="urn:test:sm:1")

        aas, sm = create_tag_mapping_aas(
            submodel=submodel,
            aas_id="urn:test:aas:1",
            asset_id="urn:test:asset:1",
        )

        # AAS should have reference to submodel
        assert len(aas.submodel) > 0


class TestAASExporter:
    """Tests for AASExporter class."""

    @pytest.fixture
    def sample_submodel(self) -> TagMappingSubmodel:
        """Create a sample submodel for testing."""
        submodel = TagMappingSubmodel(
            submodel_id="urn:test:submodel:export",
        )
        submodel.add_tag(DiscoveredTag(
            tag_name="TIC-101.PV",
            browse_path="/Objects/TIC-101/PV",
            irdi="0173-1#02-AAB663#001",
            status=MappingStatus.MAPPED,
            consensus=ConsensusInfo(confidence=0.95, participating_agents=3),
        ))
        submodel.add_tag(DiscoveredTag(
            tag_name="FIC-200.SP",
            browse_path="/Objects/FIC-200/SP",
            status=MappingStatus.PENDING,
        ))
        return submodel

    @pytest.fixture
    def exporter(self) -> AASExporter:
        """Create an exporter for testing."""
        return AASExporter()

    def test_create_exporter(self) -> None:
        """Test creating an AAS exporter."""
        exporter = AASExporter()
        assert exporter is not None

    def test_create_exporter_with_config(self) -> None:
        """Test creating exporter with custom config."""
        config = ExportConfig(pretty_print=False)
        exporter = AASExporter(config=config)
        assert exporter._config.pretty_print is False

    def test_export_to_json(
        self, exporter: AASExporter, sample_submodel: TagMappingSubmodel
    ) -> None:
        """Test exporting to JSON format."""
        aas, sm = create_tag_mapping_aas(
            submodel=sample_submodel,
            aas_id="urn:test:aas:json",
            asset_id="urn:test:asset:json",
        )

        json_str = exporter.export_json(aas, sm)

        # Should be valid JSON
        data = json.loads(json_str)
        assert "assetAdministrationShells" in data or "submodels" in data

    def test_export_to_json_file(
        self, exporter: AASExporter, sample_submodel: TagMappingSubmodel
    ) -> None:
        """Test exporting JSON to file."""
        aas, sm = create_tag_mapping_aas(
            submodel=sample_submodel,
            aas_id="urn:test:aas:jsonfile",
            asset_id="urn:test:asset:jsonfile",
        )

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            output_path = Path(f.name)

        try:
            exporter.export_to_file(aas, sm, output_path, ExportFormat.JSON)

            assert output_path.exists()
            content = output_path.read_text()
            data = json.loads(content)
            assert data is not None
        finally:
            output_path.unlink(missing_ok=True)

    def test_export_to_xml(
        self, exporter: AASExporter, sample_submodel: TagMappingSubmodel
    ) -> None:
        """Test exporting to XML format."""
        aas, sm = create_tag_mapping_aas(
            submodel=sample_submodel,
            aas_id="urn:test:aas:xml",
            asset_id="urn:test:asset:xml",
        )

        xml_str = exporter.export_xml(aas, sm)

        # Should be valid XML string
        assert xml_str.startswith("<?xml") or xml_str.startswith("<")
        assert "aas:" in xml_str or "AssetAdministrationShell" in xml_str

    def test_export_to_xml_file(
        self, exporter: AASExporter, sample_submodel: TagMappingSubmodel
    ) -> None:
        """Test exporting XML to file."""
        aas, sm = create_tag_mapping_aas(
            submodel=sample_submodel,
            aas_id="urn:test:aas:xmlfile",
            asset_id="urn:test:asset:xmlfile",
        )

        with tempfile.NamedTemporaryFile(suffix=".xml", delete=False) as f:
            output_path = Path(f.name)

        try:
            exporter.export_to_file(aas, sm, output_path, ExportFormat.XML)

            assert output_path.exists()
            content = output_path.read_text()
            assert "<?xml" in content or "<" in content
        finally:
            output_path.unlink(missing_ok=True)

    def test_export_to_aasx(
        self, exporter: AASExporter, sample_submodel: TagMappingSubmodel
    ) -> None:
        """Test exporting to AASX package."""
        aas, sm = create_tag_mapping_aas(
            submodel=sample_submodel,
            aas_id="urn:test:aas:aasx",
            asset_id="urn:test:asset:aasx",
        )

        with tempfile.NamedTemporaryFile(suffix=".aasx", delete=False) as f:
            output_path = Path(f.name)

        try:
            exporter.export_to_file(aas, sm, output_path, ExportFormat.AASX)

            assert output_path.exists()
            # AASX is a ZIP file
            assert ZipFile(output_path).namelist()
        finally:
            output_path.unlink(missing_ok=True)

    def test_aasx_contains_required_files(
        self, exporter: AASExporter, sample_submodel: TagMappingSubmodel
    ) -> None:
        """Test AASX package contains required structure."""
        aas, sm = create_tag_mapping_aas(
            submodel=sample_submodel,
            aas_id="urn:test:aas:structure",
            asset_id="urn:test:asset:structure",
        )

        with tempfile.NamedTemporaryFile(suffix=".aasx", delete=False) as f:
            output_path = Path(f.name)

        try:
            exporter.export_to_file(aas, sm, output_path, ExportFormat.AASX)

            with ZipFile(output_path) as zf:
                names = zf.namelist()
                # AASX should contain relationships and content
                assert any("rels" in n.lower() or "aas" in n.lower() for n in names)
        finally:
            output_path.unlink(missing_ok=True)


class TestAASExporterWithMultipleTags:
    """Tests for exporting submodels with multiple tags."""

    @pytest.fixture
    def large_submodel(self) -> TagMappingSubmodel:
        """Create a submodel with many tags."""
        submodel = TagMappingSubmodel(submodel_id="urn:test:large")

        for i in range(50):
            status = [
                MappingStatus.PENDING,
                MappingStatus.MAPPED,
                MappingStatus.VERIFIED,
            ][i % 3]
            submodel.add_tag(DiscoveredTag(
                tag_name=f"TAG-{i:03d}.PV",
                browse_path=f"/Objects/TAG-{i:03d}/PV",
                status=status,
            ))
        return submodel

    def test_export_large_submodel_json(
        self, large_submodel: TagMappingSubmodel
    ) -> None:
        """Test exporting large submodel to JSON."""
        aas, sm = create_tag_mapping_aas(
            submodel=large_submodel,
            aas_id="urn:test:aas:large",
            asset_id="urn:test:asset:large",
        )

        exporter = AASExporter()
        json_str = exporter.export_json(aas, sm)

        data = json.loads(json_str)
        assert data is not None

    def test_export_large_submodel_aasx(
        self, large_submodel: TagMappingSubmodel
    ) -> None:
        """Test exporting large submodel to AASX."""
        aas, sm = create_tag_mapping_aas(
            submodel=large_submodel,
            aas_id="urn:test:aas:largeaasx",
            asset_id="urn:test:asset:largeaasx",
        )

        with tempfile.NamedTemporaryFile(suffix=".aasx", delete=False) as f:
            output_path = Path(f.name)

        try:
            exporter = AASExporter()
            exporter.export_to_file(aas, sm, output_path, ExportFormat.AASX)

            assert output_path.exists()
            assert output_path.stat().st_size > 0
        finally:
            output_path.unlink(missing_ok=True)


class TestExportRoundTrip:
    """Tests for export/import round-trip validation."""

    def test_json_roundtrip_preserves_structure(self) -> None:
        """Test that JSON export preserves AAS structure."""
        submodel = TagMappingSubmodel(submodel_id="urn:roundtrip:sm:1")
        submodel.add_tag(DiscoveredTag(
            tag_name="TEST-001",
            browse_path="/Test/001",
            irdi="0173-1#02-AAB663#001",
            status=MappingStatus.MAPPED,
        ))

        aas, sm = create_tag_mapping_aas(
            submodel=submodel,
            aas_id="urn:roundtrip:aas:1",
            asset_id="urn:roundtrip:asset:1",
        )

        exporter = AASExporter()
        json_str = exporter.export_json(aas, sm)
        data = json.loads(json_str)

        # Check structure is preserved
        assert "assetAdministrationShells" in data or "submodels" in data
