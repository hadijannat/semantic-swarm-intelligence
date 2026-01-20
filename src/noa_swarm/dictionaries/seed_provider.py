"""Seed dictionary provider with curated process automation concepts.

This module provides a local, offline-capable dictionary provider containing
a curated subset of common process automation concepts from IEC 61987.
It serves as a fallback when external APIs are unavailable.

The seed set focuses on the most common PA-DIM (Process Automation Device
Information Model) concepts used in process industries:
- Process variables (temperature, pressure, flow, level)
- Signal types (analog input/output, discrete input/output)
- Device characteristics (manufacturer, model, serial number)
- Operational parameters (setpoint, alarm limits, etc.)

Example usage:
    >>> from noa_swarm.dictionaries.seed_provider import SeedDictionaryProvider
    >>> provider = SeedDictionaryProvider()
    >>> concept = await provider.lookup("0173-1#02-AAB663#001")
    >>> print(concept.preferred_name)  # "Temperature"
"""

from __future__ import annotations

from noa_swarm.common.logging import get_logger
from noa_swarm.dictionaries.base import (
    DictionaryConcept,
    DictionaryProvider,
    HierarchyNode,
    SearchResult,
)

logger = get_logger(__name__)


def _create_seed_concepts() -> dict[str, DictionaryConcept]:
    """Create the seed set of curated process automation concepts.

    Returns:
        Dictionary mapping IRDIs to DictionaryConcept instances.
    """
    concepts: list[DictionaryConcept] = [
        # Process Variables
        DictionaryConcept(
            irdi="0173-1#02-AAB663#001",
            preferred_name="Temperature",
            definition="Physical quantity measuring the degree of hotness or coldness",
            unit="°C",
            data_type="float",
            alternate_names=["Temp", "Process Temperature"],
            source="seed",
            version="1.0",
        ),
        DictionaryConcept(
            irdi="0173-1#02-AAB664#001",
            preferred_name="Pressure",
            definition="Force per unit area exerted by a fluid",
            unit="bar",
            data_type="float",
            alternate_names=["Press", "Process Pressure"],
            source="seed",
            version="1.0",
        ),
        DictionaryConcept(
            irdi="0173-1#02-AAB665#001",
            preferred_name="Flow Rate",
            definition="Volume of fluid passing through a point per unit time",
            unit="m³/h",
            data_type="float",
            alternate_names=["Flow", "Volumetric Flow", "Volume Flow Rate"],
            source="seed",
            version="1.0",
        ),
        DictionaryConcept(
            irdi="0173-1#02-AAB666#001",
            preferred_name="Level",
            definition="Height of a fluid surface relative to a reference point",
            unit="m",
            data_type="float",
            alternate_names=["Tank Level", "Liquid Level"],
            source="seed",
            version="1.0",
        ),
        DictionaryConcept(
            irdi="0173-1#02-AAB667#001",
            preferred_name="Mass Flow Rate",
            definition="Mass of fluid passing through a point per unit time",
            unit="kg/h",
            data_type="float",
            alternate_names=["Mass Flow"],
            source="seed",
            version="1.0",
        ),
        DictionaryConcept(
            irdi="0173-1#02-AAB668#001",
            preferred_name="Density",
            definition="Mass per unit volume of a substance",
            unit="kg/m³",
            data_type="float",
            alternate_names=["Specific Gravity"],
            source="seed",
            version="1.0",
        ),
        DictionaryConcept(
            irdi="0173-1#02-AAB669#001",
            preferred_name="Viscosity",
            definition="Measure of a fluid's resistance to flow",
            unit="Pa·s",
            data_type="float",
            alternate_names=["Dynamic Viscosity"],
            source="seed",
            version="1.0",
        ),
        DictionaryConcept(
            irdi="0173-1#02-AAB670#001",
            preferred_name="pH Value",
            definition="Measure of acidity or basicity of a solution",
            unit=None,
            data_type="float",
            alternate_names=["pH", "Acidity"],
            source="seed",
            version="1.0",
        ),
        DictionaryConcept(
            irdi="0173-1#02-AAB671#001",
            preferred_name="Conductivity",
            definition="Ability of a material to conduct electric current",
            unit="S/m",
            data_type="float",
            alternate_names=["Electrical Conductivity"],
            source="seed",
            version="1.0",
        ),
        DictionaryConcept(
            irdi="0173-1#02-AAB672#001",
            preferred_name="Humidity",
            definition="Amount of water vapor present in air",
            unit="%",
            data_type="float",
            alternate_names=["Relative Humidity", "RH"],
            source="seed",
            version="1.0",
        ),
        # Differential values
        DictionaryConcept(
            irdi="0173-1#02-AAB673#001",
            preferred_name="Differential Pressure",
            definition="Difference in pressure between two points",
            unit="bar",
            data_type="float",
            alternate_names=["Delta P", "Pressure Drop"],
            source="seed",
            version="1.0",
        ),
        DictionaryConcept(
            irdi="0173-1#02-AAB674#001",
            preferred_name="Differential Temperature",
            definition="Difference in temperature between two points",
            unit="K",
            data_type="float",
            alternate_names=["Delta T", "Temperature Difference"],
            source="seed",
            version="1.0",
        ),
        # Signal Types
        DictionaryConcept(
            irdi="0173-1#02-AAC001#001",
            preferred_name="Analog Input",
            definition="Continuous electrical signal representing a process variable",
            unit=None,
            data_type="float",
            alternate_names=["AI", "4-20mA Input"],
            source="seed",
            version="1.0",
        ),
        DictionaryConcept(
            irdi="0173-1#02-AAC002#001",
            preferred_name="Analog Output",
            definition="Continuous electrical signal for process control",
            unit=None,
            data_type="float",
            alternate_names=["AO", "4-20mA Output"],
            source="seed",
            version="1.0",
        ),
        DictionaryConcept(
            irdi="0173-1#02-AAC003#001",
            preferred_name="Discrete Input",
            definition="Binary on/off signal from a field device",
            unit=None,
            data_type="boolean",
            alternate_names=["DI", "Digital Input"],
            source="seed",
            version="1.0",
        ),
        DictionaryConcept(
            irdi="0173-1#02-AAC004#001",
            preferred_name="Discrete Output",
            definition="Binary on/off signal to a field device",
            unit=None,
            data_type="boolean",
            alternate_names=["DO", "Digital Output"],
            source="seed",
            version="1.0",
        ),
        # Control Parameters
        DictionaryConcept(
            irdi="0173-1#02-AAD001#001",
            preferred_name="Setpoint",
            definition="Target value for a controlled process variable",
            unit=None,
            data_type="float",
            alternate_names=["SP", "Set Point", "Target Value"],
            source="seed",
            version="1.0",
        ),
        DictionaryConcept(
            irdi="0173-1#02-AAD002#001",
            preferred_name="Process Value",
            definition="Current measured value of a process variable",
            unit=None,
            data_type="float",
            alternate_names=["PV", "Measured Value"],
            source="seed",
            version="1.0",
        ),
        DictionaryConcept(
            irdi="0173-1#02-AAD003#001",
            preferred_name="Output Value",
            definition="Control output signal percentage",
            unit="%",
            data_type="float",
            alternate_names=["MV", "Manipulated Variable", "Control Output"],
            source="seed",
            version="1.0",
        ),
        DictionaryConcept(
            irdi="0173-1#02-AAD004#001",
            preferred_name="Proportional Gain",
            definition="Proportional term coefficient in PID control",
            unit=None,
            data_type="float",
            alternate_names=["Kp", "P Gain"],
            source="seed",
            version="1.0",
        ),
        DictionaryConcept(
            irdi="0173-1#02-AAD005#001",
            preferred_name="Integral Time",
            definition="Integral term time constant in PID control",
            unit="s",
            data_type="float",
            alternate_names=["Ti", "I Time", "Reset Time"],
            source="seed",
            version="1.0",
        ),
        DictionaryConcept(
            irdi="0173-1#02-AAD006#001",
            preferred_name="Derivative Time",
            definition="Derivative term time constant in PID control",
            unit="s",
            data_type="float",
            alternate_names=["Td", "D Time", "Rate Time"],
            source="seed",
            version="1.0",
        ),
        # Alarm Limits
        DictionaryConcept(
            irdi="0173-1#02-AAE001#001",
            preferred_name="High Alarm Limit",
            definition="Upper threshold for process alarm condition",
            unit=None,
            data_type="float",
            alternate_names=["HI Alarm", "High Limit"],
            source="seed",
            version="1.0",
        ),
        DictionaryConcept(
            irdi="0173-1#02-AAE002#001",
            preferred_name="High High Alarm Limit",
            definition="Critical upper threshold for process alarm",
            unit=None,
            data_type="float",
            alternate_names=["HIHI Alarm", "High High Limit"],
            source="seed",
            version="1.0",
        ),
        DictionaryConcept(
            irdi="0173-1#02-AAE003#001",
            preferred_name="Low Alarm Limit",
            definition="Lower threshold for process alarm condition",
            unit=None,
            data_type="float",
            alternate_names=["LO Alarm", "Low Limit"],
            source="seed",
            version="1.0",
        ),
        DictionaryConcept(
            irdi="0173-1#02-AAE004#001",
            preferred_name="Low Low Alarm Limit",
            definition="Critical lower threshold for process alarm",
            unit=None,
            data_type="float",
            alternate_names=["LOLO Alarm", "Low Low Limit"],
            source="seed",
            version="1.0",
        ),
        DictionaryConcept(
            irdi="0173-1#02-AAE005#001",
            preferred_name="Alarm Deadband",
            definition="Hysteresis band around alarm limit to prevent chattering",
            unit=None,
            data_type="float",
            alternate_names=["Deadband", "Hysteresis"],
            source="seed",
            version="1.0",
        ),
        # Device Information
        DictionaryConcept(
            irdi="0173-1#02-AAF001#001",
            preferred_name="Manufacturer Name",
            definition="Name of the device manufacturer",
            unit=None,
            data_type="string",
            alternate_names=["Vendor", "Manufacturer"],
            source="seed",
            version="1.0",
        ),
        DictionaryConcept(
            irdi="0173-1#02-AAF002#001",
            preferred_name="Model Number",
            definition="Manufacturer's model designation",
            unit=None,
            data_type="string",
            alternate_names=["Model", "Part Number"],
            source="seed",
            version="1.0",
        ),
        DictionaryConcept(
            irdi="0173-1#02-AAF003#001",
            preferred_name="Serial Number",
            definition="Unique identifier assigned by manufacturer",
            unit=None,
            data_type="string",
            alternate_names=["Serial", "S/N"],
            source="seed",
            version="1.0",
        ),
        DictionaryConcept(
            irdi="0173-1#02-AAF004#001",
            preferred_name="Firmware Version",
            definition="Version of embedded software in the device",
            unit=None,
            data_type="string",
            alternate_names=["FW Version", "Software Version"],
            source="seed",
            version="1.0",
        ),
        DictionaryConcept(
            irdi="0173-1#02-AAF005#001",
            preferred_name="Hardware Version",
            definition="Version of device hardware",
            unit=None,
            data_type="string",
            alternate_names=["HW Version"],
            source="seed",
            version="1.0",
        ),
        DictionaryConcept(
            irdi="0173-1#02-AAF006#001",
            preferred_name="Device Tag",
            definition="User-assigned identifier for the device",
            unit=None,
            data_type="string",
            alternate_names=["Tag", "Tag Name", "Instrument Tag"],
            source="seed",
            version="1.0",
        ),
        DictionaryConcept(
            irdi="0173-1#02-AAF007#001",
            preferred_name="Device Description",
            definition="Textual description of device purpose or function",
            unit=None,
            data_type="string",
            alternate_names=["Description", "Descriptor"],
            source="seed",
            version="1.0",
        ),
        # Measurement Range
        DictionaryConcept(
            irdi="0173-1#02-AAG001#001",
            preferred_name="Lower Range Value",
            definition="Lower limit of measurement range",
            unit=None,
            data_type="float",
            alternate_names=["LRV", "Range Low"],
            source="seed",
            version="1.0",
        ),
        DictionaryConcept(
            irdi="0173-1#02-AAG002#001",
            preferred_name="Upper Range Value",
            definition="Upper limit of measurement range",
            unit=None,
            data_type="float",
            alternate_names=["URV", "Range High"],
            source="seed",
            version="1.0",
        ),
        DictionaryConcept(
            irdi="0173-1#02-AAG003#001",
            preferred_name="Span",
            definition="Difference between upper and lower range values",
            unit=None,
            data_type="float",
            alternate_names=["Range Span"],
            source="seed",
            version="1.0",
        ),
        DictionaryConcept(
            irdi="0173-1#02-AAG004#001",
            preferred_name="Zero",
            definition="Zero point of measurement scale",
            unit=None,
            data_type="float",
            alternate_names=["Zero Point", "Offset"],
            source="seed",
            version="1.0",
        ),
        # Status and Diagnostics
        DictionaryConcept(
            irdi="0173-1#02-AAH001#001",
            preferred_name="Device Status",
            definition="Current operational status of the device",
            unit=None,
            data_type="integer",
            alternate_names=["Status", "Health Status"],
            source="seed",
            version="1.0",
        ),
        DictionaryConcept(
            irdi="0173-1#02-AAH002#001",
            preferred_name="Fault Status",
            definition="Indicates presence of device fault condition",
            unit=None,
            data_type="boolean",
            alternate_names=["Fault", "Error Status"],
            source="seed",
            version="1.0",
        ),
        DictionaryConcept(
            irdi="0173-1#02-AAH003#001",
            preferred_name="Operating Hours",
            definition="Total time device has been operating",
            unit="h",
            data_type="float",
            alternate_names=["Runtime", "Hours Run"],
            source="seed",
            version="1.0",
        ),
        DictionaryConcept(
            irdi="0173-1#02-AAH004#001",
            preferred_name="Calibration Date",
            definition="Date of last calibration",
            unit=None,
            data_type="string",
            alternate_names=["Last Cal Date"],
            source="seed",
            version="1.0",
        ),
        DictionaryConcept(
            irdi="0173-1#02-AAH005#001",
            preferred_name="Maintenance Required",
            definition="Flag indicating maintenance is needed",
            unit=None,
            data_type="boolean",
            alternate_names=["Needs Maintenance"],
            source="seed",
            version="1.0",
        ),
        # Communication
        DictionaryConcept(
            irdi="0173-1#02-AAI001#001",
            preferred_name="Communication Protocol",
            definition="Protocol used for device communication",
            unit=None,
            data_type="string",
            alternate_names=["Protocol", "Comm Protocol"],
            source="seed",
            version="1.0",
        ),
        DictionaryConcept(
            irdi="0173-1#02-AAI002#001",
            preferred_name="Device Address",
            definition="Network or bus address of the device",
            unit=None,
            data_type="string",
            alternate_names=["Address", "Node Address"],
            source="seed",
            version="1.0",
        ),
        DictionaryConcept(
            irdi="0173-1#02-AAI003#001",
            preferred_name="Baud Rate",
            definition="Communication speed in bits per second",
            unit="bps",
            data_type="integer",
            alternate_names=["Comm Speed"],
            source="seed",
            version="1.0",
        ),
        # Valve-specific
        DictionaryConcept(
            irdi="0173-1#02-AAJ001#001",
            preferred_name="Valve Position",
            definition="Current position of a control valve",
            unit="%",
            data_type="float",
            alternate_names=["Position", "Valve Opening"],
            source="seed",
            version="1.0",
        ),
        DictionaryConcept(
            irdi="0173-1#02-AAJ002#001",
            preferred_name="Valve Characteristic",
            definition="Flow characteristic of the valve",
            unit=None,
            data_type="string",
            alternate_names=["Characteristic", "Inherent Characteristic"],
            source="seed",
            version="1.0",
        ),
        DictionaryConcept(
            irdi="0173-1#02-AAJ003#001",
            preferred_name="Valve Size",
            definition="Nominal size of the valve",
            unit="mm",
            data_type="float",
            alternate_names=["Size", "Nominal Diameter"],
            source="seed",
            version="1.0",
        ),
        DictionaryConcept(
            irdi="0173-1#02-AAJ004#001",
            preferred_name="Cv Value",
            definition="Flow coefficient of the valve",
            unit=None,
            data_type="float",
            alternate_names=["Cv", "Flow Coefficient", "Kv"],
            source="seed",
            version="1.0",
        ),
        # Motor/Pump-specific
        DictionaryConcept(
            irdi="0173-1#02-AAK001#001",
            preferred_name="Motor Speed",
            definition="Rotational speed of motor or pump",
            unit="rpm",
            data_type="float",
            alternate_names=["Speed", "RPM"],
            source="seed",
            version="1.0",
        ),
        DictionaryConcept(
            irdi="0173-1#02-AAK002#001",
            preferred_name="Motor Current",
            definition="Electrical current drawn by motor",
            unit="A",
            data_type="float",
            alternate_names=["Current", "Amps"],
            source="seed",
            version="1.0",
        ),
        DictionaryConcept(
            irdi="0173-1#02-AAK003#001",
            preferred_name="Motor Power",
            definition="Electrical power consumed by motor",
            unit="kW",
            data_type="float",
            alternate_names=["Power"],
            source="seed",
            version="1.0",
        ),
        DictionaryConcept(
            irdi="0173-1#02-AAK004#001",
            preferred_name="Motor Torque",
            definition="Rotational force produced by motor",
            unit="N·m",
            data_type="float",
            alternate_names=["Torque"],
            source="seed",
            version="1.0",
        ),
        DictionaryConcept(
            irdi="0173-1#02-AAK005#001",
            preferred_name="Motor Vibration",
            definition="Vibration level of motor or rotating equipment",
            unit="mm/s",
            data_type="float",
            alternate_names=["Vibration"],
            source="seed",
            version="1.0",
        ),
        # Time-based
        DictionaryConcept(
            irdi="0173-1#02-AAL001#001",
            preferred_name="Timestamp",
            definition="Date and time of measurement or event",
            unit=None,
            data_type="string",
            alternate_names=["Time", "Date Time"],
            source="seed",
            version="1.0",
        ),
        DictionaryConcept(
            irdi="0173-1#02-AAL002#001",
            preferred_name="Sampling Rate",
            definition="Frequency of data acquisition",
            unit="Hz",
            data_type="float",
            alternate_names=["Sample Rate", "Scan Rate"],
            source="seed",
            version="1.0",
        ),
        DictionaryConcept(
            irdi="0173-1#02-AAL003#001",
            preferred_name="Update Rate",
            definition="Frequency of data transmission or display refresh",
            unit="Hz",
            data_type="float",
            alternate_names=["Refresh Rate"],
            source="seed",
            version="1.0",
        ),
        # Quality
        DictionaryConcept(
            irdi="0173-1#02-AAM001#001",
            preferred_name="Signal Quality",
            definition="Quality indicator for measured signal",
            unit=None,
            data_type="integer",
            alternate_names=["Quality", "Signal Status"],
            source="seed",
            version="1.0",
        ),
        DictionaryConcept(
            irdi="0173-1#02-AAM002#001",
            preferred_name="Uncertainty",
            definition="Measurement uncertainty or error bound",
            unit="%",
            data_type="float",
            alternate_names=["Error", "Accuracy"],
            source="seed",
            version="1.0",
        ),
        # Energy
        DictionaryConcept(
            irdi="0173-1#02-AAN001#001",
            preferred_name="Energy Consumption",
            definition="Total energy consumed",
            unit="kWh",
            data_type="float",
            alternate_names=["Energy", "Power Consumption"],
            source="seed",
            version="1.0",
        ),
        DictionaryConcept(
            irdi="0173-1#02-AAN002#001",
            preferred_name="Voltage",
            definition="Electrical potential difference",
            unit="V",
            data_type="float",
            alternate_names=["Electric Voltage"],
            source="seed",
            version="1.0",
        ),
        DictionaryConcept(
            irdi="0173-1#02-AAN003#001",
            preferred_name="Frequency",
            definition="Electrical frequency",
            unit="Hz",
            data_type="float",
            alternate_names=["Electric Frequency"],
            source="seed",
            version="1.0",
        ),
        DictionaryConcept(
            irdi="0173-1#02-AAN004#001",
            preferred_name="Power Factor",
            definition="Ratio of real to apparent power",
            unit=None,
            data_type="float",
            alternate_names=["PF", "Cos Phi"],
            source="seed",
            version="1.0",
        ),
    ]

    return {c.irdi: c for c in concepts}


# Hierarchy definitions - parent -> children mapping
_HIERARCHY_MAP: dict[str, dict[str, list[str] | str | None]] = {
    # Process Variables category
    "0173-1#02-AAB663#001": {  # Temperature
        "parent": None,
        "children": ["0173-1#02-AAB674#001"],  # Differential Temperature
    },
    "0173-1#02-AAB664#001": {  # Pressure
        "parent": None,
        "children": ["0173-1#02-AAB673#001"],  # Differential Pressure
    },
    "0173-1#02-AAB665#001": {  # Flow Rate
        "parent": None,
        "children": ["0173-1#02-AAB667#001"],  # Mass Flow Rate
    },
    "0173-1#02-AAB666#001": {"parent": None, "children": []},  # Level
    "0173-1#02-AAB667#001": {  # Mass Flow Rate
        "parent": "0173-1#02-AAB665#001",  # Flow Rate
        "children": [],
    },
    "0173-1#02-AAB668#001": {"parent": None, "children": []},  # Density
    "0173-1#02-AAB669#001": {"parent": None, "children": []},  # Viscosity
    "0173-1#02-AAB670#001": {"parent": None, "children": []},  # pH
    "0173-1#02-AAB671#001": {"parent": None, "children": []},  # Conductivity
    "0173-1#02-AAB672#001": {"parent": None, "children": []},  # Humidity
    "0173-1#02-AAB673#001": {  # Differential Pressure
        "parent": "0173-1#02-AAB664#001",  # Pressure
        "children": [],
    },
    "0173-1#02-AAB674#001": {  # Differential Temperature
        "parent": "0173-1#02-AAB663#001",  # Temperature
        "children": [],
    },
}


class SeedDictionaryProvider(DictionaryProvider):
    """Dictionary provider with curated process automation concepts.

    This provider contains a curated subset of common IEC 61987 concepts
    used in process industries. It works offline and serves as a fallback
    when external dictionary APIs are unavailable.

    The seed set includes concepts for:
    - Process variables (temperature, pressure, flow, level)
    - Signal types (analog/discrete input/output)
    - Control parameters (setpoint, PID tuning)
    - Alarm limits
    - Device information
    - Diagnostics and status

    Attributes:
        _concepts: Dictionary mapping IRDIs to DictionaryConcept instances.
    """

    def __init__(self) -> None:
        """Initialize the seed dictionary provider."""
        self._concepts = _create_seed_concepts()
        logger.info(
            "Initialized SeedDictionaryProvider",
            concept_count=len(self._concepts),
        )

    @property
    def name(self) -> str:
        """Return the provider name."""
        return "seed"

    async def lookup(self, irdi: str) -> DictionaryConcept | None:
        """Look up a concept by its IRDI.

        Args:
            irdi: The International Registration Data Identifier.

        Returns:
            The concept if found in the seed set, None otherwise.
        """
        return self._concepts.get(irdi)

    async def search(
        self,
        query: str,
        max_results: int = 10,
    ) -> list[SearchResult]:
        """Search for concepts matching a query.

        Performs case-insensitive substring matching on preferred names,
        definitions, and alternate names.

        Args:
            query: Search query string.
            max_results: Maximum number of results to return.

        Returns:
            List of search results ordered by relevance score.
        """
        query_lower = query.lower().strip()
        results: list[SearchResult] = []

        for concept in self._concepts.values():
            score = self._compute_match_score(concept, query_lower)
            if score > 0:
                match_type = "exact" if score >= 1.0 else "partial"
                results.append(
                    SearchResult(
                        concept=concept,
                        score=min(score, 1.0),
                        match_type=match_type,
                    )
                )

        # Sort by score descending
        results.sort(key=lambda r: r.score, reverse=True)
        return results[:max_results]

    async def list_concepts(self, limit: int | None = None) -> list[DictionaryConcept]:
        """Return the locally cached seed concepts.

        Args:
            limit: Optional maximum number of concepts to return.

        Returns:
            List of DictionaryConcept entries.
        """
        concepts = list(self._concepts.values())
        if limit is not None:
            return concepts[:limit]
        return concepts

    def _compute_match_score(self, concept: DictionaryConcept, query: str) -> float:
        """Compute match score for a concept against a query.

        Args:
            concept: The concept to score.
            query: Lowercase search query.

        Returns:
            Score between 0.0 and 1.0+ (higher is better match).
        """
        if not query:
            # Empty query matches everything with base score
            return 0.5

        score = 0.0
        name_lower = concept.preferred_name.lower()

        # Exact match on preferred name
        if name_lower == query:
            score = 1.0
        # Preferred name starts with query
        elif name_lower.startswith(query):
            score = 0.9
        # Query in preferred name
        elif query in name_lower:
            score = 0.8
        # Query in definition
        elif concept.definition and query in concept.definition.lower():
            score = 0.6
        # Query in alternate names
        else:
            for alt_name in concept.alternate_names:
                if query in alt_name.lower():
                    score = 0.7
                    break

        return score

    async def get_hierarchy(
        self,
        irdi: str,
        depth: int = 1,
    ) -> HierarchyNode | None:
        """Get the hierarchy information for a concept.

        Args:
            irdi: The IRDI of the concept.
            depth: How many levels of children to include (0 = none).

        Returns:
            Hierarchy node with parent and children, None if not found.
        """
        concept = self._concepts.get(irdi)
        if concept is None:
            return None

        # Get hierarchy info from map, or create default
        hierarchy_info = _HIERARCHY_MAP.get(irdi, {"parent": None, "children": []})
        parent_irdi = hierarchy_info.get("parent")
        child_irdis = hierarchy_info.get("children", [])

        # Compute depth (0 if no parent)
        node_depth = 0
        if parent_irdi:
            node_depth = 1
            # Could traverse further up but keeping it simple

        return HierarchyNode(
            irdi=irdi,
            preferred_name=concept.preferred_name,
            parent_irdi=parent_irdi,  # type: ignore[arg-type]
            child_irdis=list(child_irdis) if depth > 0 else [],  # type: ignore[arg-type]
            depth=node_depth,
        )

    async def is_available(self) -> bool:
        """Check if the provider is available.

        The seed provider is always available as it works offline.

        Returns:
            Always True.
        """
        return True
