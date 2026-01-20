"""OPC UA Simulator for testing.

This module provides a simulated OPC UA server that generates ISA-95 style
tags for testing purposes. It creates a realistic hierarchical structure
with Plant > Area > Unit > Tags.
"""

from __future__ import annotations

import asyncio
import random
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Self

from asyncua import Server, ua
from loguru import logger

if TYPE_CHECKING:
    from asyncua.common.node import Node


# ISA-95 Tag Type Definitions
@dataclass
class TagDefinition:
    """Definition of a tag type following ISA naming conventions."""

    prefix: str  # e.g., "FIC", "TIC"
    name: str  # Full name, e.g., "Flow Indicator Controller"
    unit: str  # Engineering unit, e.g., "m3/h", "degC"
    data_type: ua.NodeId  # OPC UA data type
    min_value: float = 0.0
    max_value: float = 100.0
    description_template: str = "{name} for {unit_name}"


# Standard ISA tag types
ISA_TAG_TYPES = [
    TagDefinition(
        prefix="FIC",
        name="Flow Indicator Controller",
        unit="m3/h",
        data_type=ua.NodeId(ua.ObjectIds.Double),
        min_value=0.0,
        max_value=1000.0,
        description_template="Flow rate measurement and control",
    ),
    TagDefinition(
        prefix="TIC",
        name="Temperature Indicator Controller",
        unit="degC",
        data_type=ua.NodeId(ua.ObjectIds.Double),
        min_value=-50.0,
        max_value=500.0,
        description_template="Temperature measurement and control",
    ),
    TagDefinition(
        prefix="PIC",
        name="Pressure Indicator Controller",
        unit="bar",
        data_type=ua.NodeId(ua.ObjectIds.Double),
        min_value=0.0,
        max_value=100.0,
        description_template="Pressure measurement and control",
    ),
    TagDefinition(
        prefix="LIC",
        name="Level Indicator Controller",
        unit="%",
        data_type=ua.NodeId(ua.ObjectIds.Double),
        min_value=0.0,
        max_value=100.0,
        description_template="Level measurement and control",
    ),
    TagDefinition(
        prefix="AIC",
        name="Analyzer Indicator Controller",
        unit="ppm",
        data_type=ua.NodeId(ua.ObjectIds.Double),
        min_value=0.0,
        max_value=10000.0,
        description_template="Analytical measurement and control",
    ),
    TagDefinition(
        prefix="XV",
        name="On/Off Valve",
        unit="",
        data_type=ua.NodeId(ua.ObjectIds.Boolean),
        description_template="Discrete on/off valve position",
    ),
    TagDefinition(
        prefix="FCV",
        name="Flow Control Valve",
        unit="%",
        data_type=ua.NodeId(ua.ObjectIds.Double),
        min_value=0.0,
        max_value=100.0,
        description_template="Flow control valve position",
    ),
    TagDefinition(
        prefix="PSH",
        name="Pressure Switch High",
        unit="",
        data_type=ua.NodeId(ua.ObjectIds.Boolean),
        description_template="High pressure alarm switch",
    ),
    TagDefinition(
        prefix="TSL",
        name="Temperature Switch Low",
        unit="",
        data_type=ua.NodeId(ua.ObjectIds.Boolean),
        description_template="Low temperature alarm switch",
    ),
    TagDefinition(
        prefix="SI",
        name="Speed Indicator",
        unit="rpm",
        data_type=ua.NodeId(ua.ObjectIds.Double),
        min_value=0.0,
        max_value=5000.0,
        description_template="Rotational speed measurement",
    ),
]

# Sample IRDI mappings for semantic enrichment
SAMPLE_IRDIS = {
    "FIC": "0173-1#02-AAB123#001",  # Flow measurement
    "TIC": "0173-1#02-AAB234#001",  # Temperature measurement
    "PIC": "0173-1#02-AAB345#001",  # Pressure measurement
    "LIC": "0173-1#02-AAB456#001",  # Level measurement
    "AIC": "0173-1#02-AAB567#001",  # Analyzer measurement
    "XV": "0173-1#02-AAC123#001",  # Valve - on/off
    "FCV": "0173-1#02-AAC234#001",  # Valve - control
    "PSH": "0173-1#02-AAD123#001",  # Pressure switch
    "TSL": "0173-1#02-AAD234#001",  # Temperature switch
    "SI": "0173-1#02-AAE123#001",  # Speed measurement
}


@dataclass
class PlantStructure:
    """Configuration for the simulated plant structure."""

    plant_name: str = "Plant1"
    areas: list[str] = field(default_factory=lambda: ["AreaA", "AreaB"])
    units_per_area: int = 3
    tags_per_unit: int = 5


class OPCUASimulator:
    """Simulated OPC UA server for testing.

    Creates an ISA-95 style hierarchical structure with realistic tag names
    and metadata. Supports optional HasDictionaryEntry references for
    semantic linkage testing.

    Usage:
        async with OPCUASimulator(port=4840) as simulator:
            endpoint = simulator.endpoint_url
            # Use endpoint with OPCUABrowser for testing
    """

    def __init__(
        self,
        port: int = 4840,
        structure: PlantStructure | None = None,
        add_dictionary_entries: bool = True,
    ) -> None:
        """Initialize the OPC UA simulator.

        Args:
            port: Port to run the server on.
            structure: Plant structure configuration. Uses defaults if not provided.
            add_dictionary_entries: Whether to add HasDictionaryEntry references.
        """
        self._port = port
        self._structure = structure or PlantStructure()
        self._add_dictionary_entries = add_dictionary_entries
        self._server: Server | None = None
        self._running = False
        self._created_nodes: list[Node] = []

    @property
    def endpoint_url(self) -> str:
        """Return the OPC UA server endpoint URL."""
        return f"opc.tcp://localhost:{self._port}"

    @property
    def is_running(self) -> bool:
        """Return True if the server is running."""
        return self._running

    async def __aenter__(self) -> Self:
        """Enter async context manager and start server."""
        await self.start()
        return self

    async def __aexit__(
        self, exc_type: type | None, exc_val: Exception | None, exc_tb: object
    ) -> None:
        """Exit async context manager and stop server."""
        await self.stop()

    async def start(self) -> None:
        """Start the OPC UA server and populate the address space."""
        if self._running:
            logger.warning("Simulator already running")
            return

        self._server = Server()
        await self._server.init()

        # Configure server
        self._server.set_endpoint(self.endpoint_url)
        self._server.set_server_name("NOA Test OPC UA Server")

        # Set up namespace
        namespace_uri = "urn:noa:opcua:simulator"
        idx = await self._server.register_namespace(namespace_uri)

        # Create the address space structure
        await self._create_address_space(idx)

        # Start the server
        await self._server.start()
        self._running = True

        logger.info(f"OPC UA Simulator started on {self.endpoint_url}")

    async def stop(self) -> None:
        """Stop the OPC UA server."""
        if self._server and self._running:
            try:
                await self._server.stop()
                logger.info("OPC UA Simulator stopped")
            except Exception as e:
                logger.warning(f"Error stopping simulator: {e}")
            finally:
                self._running = False
                self._server = None
                self._created_nodes.clear()

    async def _create_address_space(self, idx: int) -> None:
        """Create the hierarchical address space.

        Args:
            idx: Namespace index.
        """
        if self._server is None:
            return

        objects = self._server.nodes.objects

        # Create Plant node
        plant_node = await objects.add_object(idx, self._structure.plant_name)
        self._created_nodes.append(plant_node)

        # Create Areas under Plant
        for area_name in self._structure.areas:
            area_node = await plant_node.add_object(idx, area_name)
            self._created_nodes.append(area_node)

            # Create Units under each Area
            for unit_num in range(1, self._structure.units_per_area + 1):
                unit_name = f"Unit{unit_num:02d}"
                unit_node = await area_node.add_object(idx, unit_name)
                self._created_nodes.append(unit_node)

                # Create Tags under each Unit
                await self._create_tags_for_unit(idx, unit_node, unit_num)

    async def _create_tags_for_unit(
        self,
        idx: int,
        unit_node: Node,
        unit_num: int,
    ) -> None:
        """Create tags for a unit.

        Args:
            idx: Namespace index.
            unit_node: Parent unit node.
            unit_num: Unit number for tag numbering.
        """
        if self._server is None:
            return

        # Select random tag types for this unit
        selected_types = random.sample(
            ISA_TAG_TYPES,
            min(self._structure.tags_per_unit, len(ISA_TAG_TYPES)),
        )

        for i, tag_type in enumerate(selected_types):
            # Create ISA-style tag name: PREFIX-XYZ (e.g., FIC-101)
            tag_number = unit_num * 100 + (i + 1)
            tag_name = f"{tag_type.prefix}-{tag_number}"

            # Create the variable node
            initial_value = self._get_initial_value(tag_type)

            tag_node = await unit_node.add_variable(
                idx,
                tag_name,
                initial_value,
                varianttype=self._get_variant_type(tag_type.data_type),
            )
            self._created_nodes.append(tag_node)

            # Set writable (for simulation updates, but browser won't write)
            await tag_node.set_writable()

            # Add description
            await self._set_description(tag_node, tag_type)

            # Add engineering unit if applicable
            if tag_type.unit:
                await self._add_engineering_unit(idx, tag_node, tag_type)

            # Add HasDictionaryEntry reference if configured
            if self._add_dictionary_entries and tag_type.prefix in SAMPLE_IRDIS:
                await self._add_dictionary_entry(idx, tag_node, tag_type.prefix)

    def _get_initial_value(self, tag_type: TagDefinition) -> float | bool:
        """Get an initial value for a tag.

        Args:
            tag_type: Tag type definition.

        Returns:
            Initial value for the tag.
        """
        if tag_type.data_type == ua.NodeId(ua.ObjectIds.Boolean):
            return random.choice([True, False])
        else:
            return random.uniform(tag_type.min_value, tag_type.max_value)

    def _get_variant_type(self, data_type: ua.NodeId) -> ua.VariantType:
        """Convert OPC UA data type to variant type.

        Args:
            data_type: OPC UA data type node ID.

        Returns:
            Corresponding variant type.
        """
        type_map = {
            ua.ObjectIds.Boolean: ua.VariantType.Boolean,
            ua.ObjectIds.Double: ua.VariantType.Double,
            ua.ObjectIds.Float: ua.VariantType.Float,
            ua.ObjectIds.Int32: ua.VariantType.Int32,
            ua.ObjectIds.UInt32: ua.VariantType.UInt32,
            ua.ObjectIds.String: ua.VariantType.String,
        }
        return type_map.get(data_type.Identifier, ua.VariantType.Double)

    async def _set_description(self, node: Node, tag_type: TagDefinition) -> None:
        """Set the description attribute on a node.

        Args:
            node: Node to set description on.
            tag_type: Tag type definition.
        """
        try:
            description = ua.LocalizedText(tag_type.description_template)
            await node.write_attribute(
                ua.AttributeIds.Description,
                ua.DataValue(ua.Variant(description, ua.VariantType.LocalizedText)),
            )
        except Exception as e:
            logger.debug(f"Could not set description: {e}")

    async def _add_engineering_unit(
        self,
        idx: int,
        tag_node: Node,
        tag_type: TagDefinition,
    ) -> None:
        """Add engineering unit property to a tag.

        Args:
            idx: Namespace index.
            tag_node: Tag node.
            tag_type: Tag type definition.
        """
        if self._server is None:
            return

        try:
            # Create EUInformation structure
            eu_info = ua.EUInformation()
            eu_info.DisplayName = ua.LocalizedText(tag_type.unit)
            eu_info.Description = ua.LocalizedText(f"Engineering unit: {tag_type.unit}")

            # Add as property
            await tag_node.add_property(
                idx,
                "EngineeringUnits",
                eu_info,
            )
        except Exception as e:
            logger.debug(f"Could not add engineering unit: {e}")

    async def _add_dictionary_entry(
        self,
        idx: int,
        tag_node: Node,
        tag_prefix: str,
    ) -> None:
        """Add HasDictionaryEntry reference to a tag.

        This simulates semantic linkage to a dictionary entry (like ECLASS).

        Args:
            idx: Namespace index.
            tag_node: Tag node.
            tag_prefix: Tag prefix to look up IRDI.
        """
        if self._server is None:
            return

        irdi = SAMPLE_IRDIS.get(tag_prefix)
        if not irdi:
            return

        try:
            # Create a dictionary entry node containing the IRDI
            entry_name = f"DictionaryEntry_{tag_prefix}"

            # Get or create DictionaryEntries folder
            objects = self._server.nodes.objects
            dict_folder = None

            children = await objects.get_children()
            for child in children:
                browse_name = await child.read_browse_name()
                if browse_name.Name == "DictionaryEntries":
                    dict_folder = child
                    break

            if dict_folder is None:
                dict_folder = await objects.add_folder(idx, "DictionaryEntries")

            # Create entry node with IRDI as value
            entry_node = await dict_folder.add_variable(
                idx,
                entry_name,
                irdi,
            )

            # Add HasDictionaryEntry reference from tag to entry
            await tag_node.add_reference(
                entry_node.nodeid,
                ua.NodeId(ua.ObjectIds.HasDictionaryEntry),
                forward=True,
            )

        except Exception as e:
            logger.debug(f"Could not add dictionary entry: {e}")

    async def update_tag_values(self) -> None:
        """Update all tag values with simulated changes.

        This can be called periodically to simulate value changes.
        """
        if not self._running or self._server is None:
            return

        for node in self._created_nodes:
            try:
                node_class = await node.read_node_class()
                if node_class != ua.NodeClass.Variable:
                    continue

                # Get current value and add some variation
                current_value = await node.read_value()
                if isinstance(current_value, bool):
                    # Randomly flip boolean with 10% probability
                    if random.random() < 0.1:
                        await node.write_value(not current_value)
                elif isinstance(current_value, int | float):
                    # Add small random variation
                    variation = current_value * random.uniform(-0.05, 0.05)
                    new_value = current_value + variation
                    await node.write_value(new_value)

            except Exception as e:
                logger.debug(f"Error updating node value: {e}")

    async def run_with_updates(self, interval: float = 1.0) -> None:
        """Run the server continuously with periodic value updates.

        Args:
            interval: Update interval in seconds.

        Note:
            This method runs forever until cancelled.
        """
        if not self._running:
            await self.start()

        try:
            while self._running:
                await self.update_tag_values()
                await asyncio.sleep(interval)
        except asyncio.CancelledError:
            logger.info("Simulator update loop cancelled")


async def create_test_simulator(
    port: int = 4840,
    tags_per_unit: int = 5,
    add_dictionary_entries: bool = True,
) -> OPCUASimulator:
    """Create and start a test simulator with default configuration.

    Args:
        port: Port to run on.
        tags_per_unit: Number of tags per unit.
        add_dictionary_entries: Whether to add IRDI references.

    Returns:
        Running OPCUASimulator instance.

    Example:
        simulator = await create_test_simulator(port=4841)
        try:
            # Use simulator
            pass
        finally:
            await simulator.stop()
    """
    structure = PlantStructure(
        plant_name="TestPlant",
        areas=["Production", "Utilities"],
        units_per_area=2,
        tags_per_unit=tags_per_unit,
    )

    simulator = OPCUASimulator(
        port=port,
        structure=structure,
        add_dictionary_entries=add_dictionary_entries,
    )

    await simulator.start()
    return simulator
