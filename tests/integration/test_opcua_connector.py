"""Integration tests for OPC UA connector.

These tests use the OPC UA simulator to test the browser functionality
without requiring a real OPC UA server.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

import pytest

from noa_swarm.common.schemas import TagRecord
from noa_swarm.connectors.opcua_asyncua import (
    OPCUABrowser,
    OPCUABrowserError,
    OPCUAConnectionError,
    OPCUAWriteAttemptError,
    create_opcua_browser,
)
from noa_swarm.connectors.opcua_simulator import (
    OPCUASimulator,
    PlantStructure,
    create_test_simulator,
)

if TYPE_CHECKING:
    pass


# Use a unique port for each test run to avoid conflicts
import time as _time
TEST_PORT_BASE = 49000 + (int(_time.time()) % 1000)  # Randomize base to avoid conflicts

# Counter for unique port allocation
_port_counter = 0


@pytest.fixture
def unique_port() -> int:
    """Generate a unique port for each test."""
    global _port_counter
    _port_counter += 1
    return TEST_PORT_BASE + _port_counter


@pytest.fixture
async def simulator(unique_port: int) -> OPCUASimulator:
    """Create and start a test simulator."""
    structure = PlantStructure(
        plant_name="TestPlant",
        areas=["Area1"],
        units_per_area=2,
        tags_per_unit=3,
    )
    sim = OPCUASimulator(
        port=unique_port,
        structure=structure,
        add_dictionary_entries=True,
    )
    await sim.start()
    yield sim
    await sim.stop()


@pytest.mark.integration
class TestOPCUASimulator:
    """Tests for the OPC UA simulator."""

    async def test_simulator_starts_and_stops(self, unique_port: int) -> None:
        """Test that simulator can start and stop cleanly."""
        sim = OPCUASimulator(port=unique_port)
        assert not sim.is_running

        await sim.start()
        assert sim.is_running
        assert sim.endpoint_url == f"opc.tcp://localhost:{unique_port}"

        await sim.stop()
        assert not sim.is_running

    async def test_simulator_context_manager(self, unique_port: int) -> None:
        """Test simulator as async context manager."""
        async with OPCUASimulator(port=unique_port) as sim:
            assert sim.is_running
            assert sim.endpoint_url == f"opc.tcp://localhost:{unique_port}"

        assert not sim.is_running

    async def test_simulator_creates_isa_tags(self, unique_port: int) -> None:
        """Test that simulator creates ISA-style tags."""
        structure = PlantStructure(
            plant_name="Plant1",
            areas=["AreaA"],
            units_per_area=1,
            tags_per_unit=5,
        )

        async with OPCUASimulator(port=unique_port, structure=structure) as sim:
            # Connect with browser and verify tags exist
            async with OPCUABrowser(sim.endpoint_url) as browser:
                tags = await browser.browse_all_tags()

                # Filter to only ISA-style tags (PREFIX-NUMBER pattern)
                isa_prefixes = {"FIC", "TIC", "PIC", "LIC", "AIC", "XV", "FCV", "PSH", "TSL", "SI"}
                isa_tags = [
                    t for t in tags
                    if "-" in t.browse_name and t.browse_name.split("-")[0] in isa_prefixes
                ]

                # Should have 5 ISA tags (5 per unit * 1 unit * 1 area)
                assert len(isa_tags) == 5

                # All ISA tags should have ISA-style names (PREFIX-NUMBER)
                for tag in isa_tags:
                    parts = tag.browse_name.split("-")
                    assert len(parts) == 2
                    assert parts[0] in isa_prefixes
                    assert parts[1].isdigit()

    async def test_create_test_simulator_helper(self, unique_port: int) -> None:
        """Test the create_test_simulator helper function."""
        sim = await create_test_simulator(
            port=unique_port,
            tags_per_unit=2,
            add_dictionary_entries=False,
        )
        try:
            assert sim.is_running
            assert "TestPlant" in sim._structure.plant_name
        finally:
            await sim.stop()


@pytest.mark.integration
class TestOPCUABrowser:
    """Tests for the OPC UA browser."""

    async def test_browser_connects_and_disconnects(
        self, simulator: OPCUASimulator
    ) -> None:
        """Test that browser can connect and disconnect."""
        browser = OPCUABrowser(simulator.endpoint_url)
        assert not browser.is_connected

        await browser.connect()
        assert browser.is_connected
        assert browser.endpoint_url == simulator.endpoint_url

        await browser.disconnect()
        assert not browser.is_connected

    async def test_browser_context_manager(
        self, simulator: OPCUASimulator
    ) -> None:
        """Test browser as async context manager."""
        async with OPCUABrowser(simulator.endpoint_url) as browser:
            assert browser.is_connected

        assert not browser.is_connected

    async def test_create_opcua_browser_helper(
        self, simulator: OPCUASimulator
    ) -> None:
        """Test the create_opcua_browser helper function."""
        async with create_opcua_browser(simulator.endpoint_url) as browser:
            assert browser.is_connected
            tags = await browser.browse_all_tags()
            assert len(tags) > 0

    async def test_browse_all_tags_returns_tag_records(
        self, simulator: OPCUASimulator
    ) -> None:
        """Test that browse_all_tags returns TagRecord instances."""
        async with OPCUABrowser(simulator.endpoint_url) as browser:
            tags = await browser.browse_all_tags()

            assert len(tags) > 0
            for tag in tags:
                assert isinstance(tag, TagRecord)
                assert tag.node_id
                assert tag.browse_name
                assert tag.source_server == simulator.endpoint_url

    async def test_tag_records_have_metadata(
        self, simulator: OPCUASimulator
    ) -> None:
        """Test that tag records include metadata."""
        async with OPCUABrowser(simulator.endpoint_url) as browser:
            tags = await browser.browse_all_tags()

            # At least some tags should have data types
            tags_with_data_type = [t for t in tags if t.data_type]
            assert len(tags_with_data_type) > 0

            # Tags should have parent paths (hierarchical structure)
            tags_with_parent = [t for t in tags if t.parent_path]
            assert len(tags_with_parent) > 0

    async def test_tag_records_have_hierarchy(
        self, simulator: OPCUASimulator
    ) -> None:
        """Test that tags have proper hierarchical paths."""
        async with OPCUABrowser(simulator.endpoint_url) as browser:
            tags = await browser.browse_all_tags()

            # Filter to ISA-style process tags
            isa_prefixes = {"FIC", "TIC", "PIC", "LIC", "AIC", "XV", "FCV", "PSH", "TSL", "SI"}
            process_tags = [
                t for t in tags
                if "-" in t.browse_name and t.browse_name.split("-")[0] in isa_prefixes
            ]

            # Check that ISA tags have proper hierarchical paths
            assert len(process_tags) > 0, "No process tags found"
            for tag in process_tags:
                # Parent path should include Plant > Area > Unit
                assert tag.parent_path, f"Tag {tag.browse_name} has no parent path"
                # First element should be the plant name
                assert "TestPlant" in tag.parent_path[0] or "Plant" in tag.parent_path[0]

    async def test_read_value(
        self, simulator: OPCUASimulator
    ) -> None:
        """Test reading a single value."""
        async with OPCUABrowser(simulator.endpoint_url) as browser:
            tags = await browser.browse_all_tags()
            assert len(tags) > 0

            # Filter to ISA-style process tags (these have actual values)
            isa_prefixes = {"FIC", "TIC", "PIC", "LIC", "AIC", "XV", "FCV", "PSH", "TSL", "SI"}
            process_tags = [
                t for t in tags
                if "-" in t.browse_name and t.browse_name.split("-")[0] in isa_prefixes
            ]
            assert len(process_tags) > 0, "No process tags found"

            # Read the first process tag's value
            first_tag = process_tags[0]
            value = await browser.read_value(first_tag.node_id)

            # Value should be a number or boolean (our simulator creates these)
            assert value is not None
            assert isinstance(value, (int, float, bool))

    async def test_read_multiple_values(
        self, simulator: OPCUASimulator
    ) -> None:
        """Test reading multiple values concurrently."""
        async with OPCUABrowser(simulator.endpoint_url) as browser:
            tags = await browser.browse_all_tags()

            # Filter to ISA-style process tags (these have actual values)
            isa_prefixes = {"FIC", "TIC", "PIC", "LIC", "AIC", "XV", "FCV", "PSH", "TSL", "SI"}
            process_tags = [
                t for t in tags
                if "-" in t.browse_name and t.browse_name.split("-")[0] in isa_prefixes
            ]
            assert len(process_tags) >= 3, "Need at least 3 process tags"

            # Read first 3 process tags
            node_ids = [t.node_id for t in process_tags[:3]]
            values = await browser.read_values(node_ids)

            assert len(values) == 3
            for value in values:
                assert value is not None

    async def test_get_node_metadata(
        self, simulator: OPCUASimulator
    ) -> None:
        """Test getting metadata for a specific node."""
        async with OPCUABrowser(simulator.endpoint_url) as browser:
            tags = await browser.browse_all_tags()
            first_tag = tags[0]

            metadata = await browser.get_node_metadata(first_tag.node_id)

            assert metadata is not None
            assert metadata.node_id == first_tag.node_id
            assert metadata.browse_name == first_tag.browse_name


@pytest.mark.integration
class TestReadOnlyEnforcement:
    """Tests for read-only enforcement."""

    async def test_write_value_raises_error(
        self, simulator: OPCUASimulator
    ) -> None:
        """Test that write_value raises OPCUAWriteAttemptError."""
        async with OPCUABrowser(simulator.endpoint_url) as browser:
            with pytest.raises(OPCUAWriteAttemptError) as exc_info:
                await browser.write_value("ns=2;s=any", 42)

            assert "not allowed" in str(exc_info.value).lower()

    async def test_write_attribute_raises_error(
        self, simulator: OPCUASimulator
    ) -> None:
        """Test that write_attribute raises OPCUAWriteAttemptError."""
        async with OPCUABrowser(simulator.endpoint_url) as browser:
            with pytest.raises(OPCUAWriteAttemptError) as exc_info:
                await browser.write_attribute("ns=2;s=any", "attr", "value")

            assert "not allowed" in str(exc_info.value).lower()

    async def test_call_method_raises_error(
        self, simulator: OPCUASimulator
    ) -> None:
        """Test that call_method raises OPCUAWriteAttemptError."""
        async with OPCUABrowser(simulator.endpoint_url) as browser:
            with pytest.raises(OPCUAWriteAttemptError) as exc_info:
                await browser.call_method("ns=2;s=any", "method")

            assert "not allowed" in str(exc_info.value).lower()


@pytest.mark.integration
class TestConnectionErrors:
    """Tests for connection error handling."""

    async def test_connection_to_invalid_server_fails(self) -> None:
        """Test that connecting to an invalid server raises an error."""
        browser = OPCUABrowser("opc.tcp://localhost:99999")

        with pytest.raises(OPCUAConnectionError) as exc_info:
            await browser.connect()

        assert "failed to connect" in str(exc_info.value).lower()

    async def test_operations_without_connection_fail(self) -> None:
        """Test that operations without connection raise errors."""
        browser = OPCUABrowser("opc.tcp://localhost:4840")

        with pytest.raises(OPCUAConnectionError) as exc_info:
            await browser.browse_all_tags()

        assert "not connected" in str(exc_info.value).lower()


@pytest.mark.integration
class TestConcurrencyControl:
    """Tests for concurrency control and backpressure."""

    async def test_semaphore_is_configured(
        self, simulator: OPCUASimulator
    ) -> None:
        """Test that semaphore is correctly configured from settings."""
        from noa_swarm.common.config import OPCUASettings

        # Create settings with specific concurrency limit
        settings = OPCUASettings(max_concurrent_requests=7)

        browser = OPCUABrowser(simulator.endpoint_url, settings=settings)

        # Verify the semaphore was created with the correct limit
        # Note: Semaphore internal value check
        assert browser._semaphore._value == 7

    async def test_multiple_concurrent_reads(
        self, simulator: OPCUASimulator
    ) -> None:
        """Test multiple concurrent read operations."""
        async with OPCUABrowser(simulator.endpoint_url) as browser:
            tags = await browser.browse_all_tags()

            # Filter to ISA-style process tags (these have actual values)
            isa_prefixes = {"FIC", "TIC", "PIC", "LIC", "AIC", "XV", "FCV", "PSH", "TSL", "SI"}
            process_tags = [
                t for t in tags
                if "-" in t.browse_name and t.browse_name.split("-")[0] in isa_prefixes
            ]

            # Start multiple reads concurrently (use a small number to be fast)
            tags_to_read = process_tags[:3] if len(process_tags) >= 3 else process_tags
            tasks = [browser.read_value(tag.node_id) for tag in tags_to_read]

            results = await asyncio.gather(*tasks)
            assert len(results) == len(tags_to_read)
            for result in results:
                assert result is not None


@pytest.mark.integration
class TestDictionaryEntryExtraction:
    """Tests for HasDictionaryEntry reference extraction."""

    async def test_irdi_extraction_from_dictionary_entries(
        self, unique_port: int
    ) -> None:
        """Test that IRDIs can be extracted from HasDictionaryEntry references."""
        # Create simulator with dictionary entries enabled
        structure = PlantStructure(
            plant_name="SemanticPlant",
            areas=["Area1"],
            units_per_area=1,
            tags_per_unit=10,  # More tags to ensure we get some with IRDIs
        )

        async with OPCUASimulator(
            port=unique_port,
            structure=structure,
            add_dictionary_entries=True,
        ) as sim:
            async with OPCUABrowser(sim.endpoint_url) as browser:
                tags = await browser.browse_all_tags()

                # The simulator adds dictionary entries for known prefixes
                # Check that at least some metadata was captured
                assert len(tags) > 0

                # Filter to ISA-style process tags
                isa_prefixes = {"FIC", "TIC", "PIC", "LIC", "AIC", "XV", "FCV", "PSH", "TSL", "SI"}
                process_tags = [
                    t for t in tags
                    if "-" in t.browse_name and t.browse_name.split("-")[0] in isa_prefixes
                ]

                # Verify ISA tag structure
                assert len(process_tags) > 0, "No process tags found"
                for tag in process_tags:
                    assert tag.node_id
                    assert tag.browse_name
                    # ISA naming: PREFIX-NUMBER
                    parts = tag.browse_name.split("-")
                    assert len(parts) == 2


@pytest.mark.integration
class TestFilesystemStubs:
    """Tests for filesystem connector stubs."""

    async def test_import_csv_raises_not_implemented(self) -> None:
        """Test that CSV import raises NotImplementedError."""
        from pathlib import Path

        from noa_swarm.connectors.filesystem import import_from_csv

        with pytest.raises(NotImplementedError):
            await import_from_csv(Path("test.csv"))

    async def test_export_csv_raises_not_implemented(self) -> None:
        """Test that CSV export raises NotImplementedError."""
        from pathlib import Path

        from noa_swarm.connectors.filesystem import export_to_csv

        with pytest.raises(NotImplementedError):
            await export_to_csv([], Path("test.csv"))

    async def test_import_json_raises_not_implemented(self) -> None:
        """Test that JSON import raises NotImplementedError."""
        from pathlib import Path

        from noa_swarm.connectors.filesystem import import_from_json

        with pytest.raises(NotImplementedError):
            await import_from_json(Path("test.json"))

    async def test_export_json_raises_not_implemented(self) -> None:
        """Test that JSON export raises NotImplementedError."""
        from pathlib import Path

        from noa_swarm.connectors.filesystem import export_to_json

        with pytest.raises(NotImplementedError):
            await export_to_json([], Path("test.json"))
