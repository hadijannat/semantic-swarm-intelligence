"""Synthetic ISA-95 tag generator for training ML models.

This module generates realistic ISA-95 style tags with associated IEC 61987 IRDIs
for training the CharCNN semantic mapping model.

ISA Tag Naming Convention:
- First 1-3 letters: Measured/initiating variable (F=Flow, T=Temperature, etc.)
- Next letter: Modifier (I=Indicator, C=Controller, etc.)
- Next letter: Function (C=Control, S=Switch, etc.)
- Followed by: Loop/tag number (e.g., -101, -2301)

Examples:
- FIC-101: Flow Indicator Controller
- TIC-2301: Temperature Indicator Controller
- PIC-1501: Pressure Indicator Controller
- LIC-201: Level Indicator Controller
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, ClassVar

if TYPE_CHECKING:
    from collections.abc import Iterator


class MeasuredVariable(Enum):
    """ISA S5.1 Measured/Initiating Variable Codes."""

    FLOW = "F"
    TEMPERATURE = "T"
    PRESSURE = "P"
    LEVEL = "L"
    ANALYZER = "A"
    SPEED = "S"
    WEIGHT = "W"
    DENSITY = "D"
    VISCOSITY = "V"
    POSITION = "Z"
    CURRENT = "I"
    POWER = "J"
    TIME = "K"
    MOISTURE = "M"
    QUANTITY = "Q"
    RADIATION = "R"
    CONDUCTIVITY = "X"


class ReadoutFunction(Enum):
    """ISA S5.1 Readout/Passive Function Codes."""

    INDICATOR = "I"
    RECORDER = "R"
    TRANSMITTER = "T"
    ELEMENT = "E"
    GLASS = "G"  # Sight glass, gauge glass


class OutputFunction(Enum):
    """ISA S5.1 Output/Active Function Codes."""

    CONTROLLER = "C"
    SWITCH = "S"
    VALVE = "V"
    ALARM = "A"
    RELAY = "Y"
    SOLENOID = "Y"


class AlarmType(Enum):
    """ISA S5.1 Alarm Types."""

    HIGH = "H"
    LOW = "L"
    HIGH_HIGH = "HH"
    LOW_LOW = "LL"


# Seed IRDIs following ECLASS/IEC 61987 patterns
# Format: 0173-1#01-XXXXXX#001 (IEC CDD organization code)
SEED_IRDIS: dict[str, str] = {
    # Flow-related properties
    "flow_rate": "0173-1#01-AAA001#001",
    "flow_rate_volumetric": "0173-1#01-AAA011#001",
    "flow_rate_mass": "0173-1#01-AAA012#001",
    "flow_totalizer": "0173-1#01-AAA013#001",
    # Temperature-related properties
    "temperature": "0173-1#01-AAA002#001",
    "temperature_process": "0173-1#01-AAA021#001",
    "temperature_ambient": "0173-1#01-AAA022#001",
    "temperature_differential": "0173-1#01-AAA023#001",
    # Pressure-related properties
    "pressure": "0173-1#01-AAA003#001",
    "pressure_absolute": "0173-1#01-AAA031#001",
    "pressure_gauge": "0173-1#01-AAA032#001",
    "pressure_differential": "0173-1#01-AAA033#001",
    # Level-related properties
    "level": "0173-1#01-AAA004#001",
    "level_percentage": "0173-1#01-AAA041#001",
    "level_volume": "0173-1#01-AAA042#001",
    # Valve-related properties
    "valve_position": "0173-1#01-AAA005#001",
    "valve_position_percentage": "0173-1#01-AAA051#001",
    "valve_command": "0173-1#01-AAA052#001",
    "valve_feedback": "0173-1#01-AAA053#001",
    # Composition/Analyzer properties
    "composition": "0173-1#01-AAA006#001",
    "composition_percentage": "0173-1#01-AAA061#001",
    "composition_ppm": "0173-1#01-AAA062#001",
    "ph_value": "0173-1#01-AAA063#001",
    "conductivity": "0173-1#01-AAA064#001",
    # Speed-related properties
    "speed": "0173-1#01-AAA007#001",
    "speed_rpm": "0173-1#01-AAA071#001",
    "speed_percentage": "0173-1#01-AAA072#001",
    # Weight-related properties
    "weight": "0173-1#01-AAA008#001",
    "weight_mass": "0173-1#01-AAA081#001",
    "weight_force": "0173-1#01-AAA082#001",
    # Control-related properties
    "setpoint": "0173-1#01-AAA101#001",
    "process_value": "0173-1#01-AAA102#001",
    "output_value": "0173-1#01-AAA103#001",
    "controller_mode": "0173-1#01-AAA104#001",
    # Switch/Alarm properties
    "switch_status": "0173-1#01-AAA201#001",
    "alarm_high": "0173-1#01-AAA202#001",
    "alarm_low": "0173-1#01-AAA203#001",
    "alarm_high_high": "0173-1#01-AAA204#001",
    "alarm_low_low": "0173-1#01-AAA205#001",
    # Motor/Pump properties
    "motor_running": "0173-1#01-AAA301#001",
    "motor_current": "0173-1#01-AAA302#001",
    "motor_speed": "0173-1#01-AAA303#001",
    "pump_status": "0173-1#01-AAA304#001",
    # Miscellaneous
    "density": "0173-1#01-AAA401#001",
    "viscosity": "0173-1#01-AAA402#001",
    "position": "0173-1#01-AAA403#001",
    "power": "0173-1#01-AAA404#001",
    "energy": "0173-1#01-AAA405#001",
    "vibration": "0173-1#01-AAA406#001",
}

# Mapping from tag pattern to IRDI
TAG_PATTERN_TO_IRDI: dict[str, str] = {
    # Flow patterns
    "FI": SEED_IRDIS["flow_rate"],
    "FIC": SEED_IRDIS["flow_rate"],
    "FIT": SEED_IRDIS["flow_rate"],
    "FT": SEED_IRDIS["flow_rate"],
    "FE": SEED_IRDIS["flow_rate"],
    "FCV": SEED_IRDIS["valve_position"],
    "FV": SEED_IRDIS["valve_position"],
    "FQ": SEED_IRDIS["flow_totalizer"],
    "FQI": SEED_IRDIS["flow_totalizer"],
    # Temperature patterns
    "TI": SEED_IRDIS["temperature"],
    "TIC": SEED_IRDIS["temperature"],
    "TIT": SEED_IRDIS["temperature"],
    "TT": SEED_IRDIS["temperature"],
    "TE": SEED_IRDIS["temperature"],
    "TCV": SEED_IRDIS["valve_position"],
    "TSH": SEED_IRDIS["alarm_high"],
    "TSL": SEED_IRDIS["alarm_low"],
    "TSHH": SEED_IRDIS["alarm_high_high"],
    "TSLL": SEED_IRDIS["alarm_low_low"],
    # Pressure patterns
    "PI": SEED_IRDIS["pressure"],
    "PIC": SEED_IRDIS["pressure"],
    "PIT": SEED_IRDIS["pressure"],
    "PT": SEED_IRDIS["pressure"],
    "PE": SEED_IRDIS["pressure"],
    "PCV": SEED_IRDIS["valve_position"],
    "PDI": SEED_IRDIS["pressure_differential"],
    "PDIC": SEED_IRDIS["pressure_differential"],
    "PDT": SEED_IRDIS["pressure_differential"],
    "PSH": SEED_IRDIS["alarm_high"],
    "PSL": SEED_IRDIS["alarm_low"],
    "PSHH": SEED_IRDIS["alarm_high_high"],
    "PSLL": SEED_IRDIS["alarm_low_low"],
    # Level patterns
    "LI": SEED_IRDIS["level"],
    "LIC": SEED_IRDIS["level"],
    "LIT": SEED_IRDIS["level"],
    "LT": SEED_IRDIS["level"],
    "LE": SEED_IRDIS["level"],
    "LCV": SEED_IRDIS["valve_position"],
    "LV": SEED_IRDIS["valve_position"],
    "LSH": SEED_IRDIS["alarm_high"],
    "LSL": SEED_IRDIS["alarm_low"],
    "LSHH": SEED_IRDIS["alarm_high_high"],
    "LSLL": SEED_IRDIS["alarm_low_low"],
    # Analyzer patterns
    "AI": SEED_IRDIS["composition"],
    "AIC": SEED_IRDIS["composition"],
    "AIT": SEED_IRDIS["composition"],
    "AT": SEED_IRDIS["composition"],
    "AE": SEED_IRDIS["composition"],
    # Speed patterns
    "SI": SEED_IRDIS["speed"],
    "SIC": SEED_IRDIS["speed"],
    "ST": SEED_IRDIS["speed"],
    # Weight patterns
    "WI": SEED_IRDIS["weight"],
    "WIC": SEED_IRDIS["weight"],
    "WT": SEED_IRDIS["weight"],
    # Generic valve patterns
    "XV": SEED_IRDIS["valve_position"],
    "HV": SEED_IRDIS["valve_position"],  # Hand valve
    "CV": SEED_IRDIS["valve_position"],  # Control valve
    # Setpoint/Output patterns
    "SP": SEED_IRDIS["setpoint"],
    "PV": SEED_IRDIS["process_value"],
    "OP": SEED_IRDIS["output_value"],
    "OUT": SEED_IRDIS["output_value"],
}

# Engineering unit mappings
ENGINEERING_UNITS: dict[str, list[str]] = {
    "flow_rate": ["m3/h", "gpm", "l/min", "kg/h", "lb/h", "SCFM", "Nm3/h"],
    "temperature": ["degC", "degF", "K"],
    "pressure": ["bar", "psi", "kPa", "MPa", "mbar", "inH2O", "mmHg"],
    "level": ["%", "m", "mm", "ft", "in"],
    "valve_position": ["%"],
    "composition": ["%", "ppm", "ppb", "mol%", "wt%"],
    "speed": ["rpm", "Hz", "rad/s", "%"],
    "weight": ["kg", "lb", "ton", "g"],
}

# Description templates for different tag types
DESCRIPTION_TEMPLATES: dict[str, list[str]] = {
    "FI": [
        "{area} Feed Flow Indicator",
        "{area} Flow Measurement",
        "{area} Product Flow Rate",
        "{area} Inlet Flow",
        "{area} Outlet Flow",
    ],
    "FIC": [
        "{area} Feed Flow Controller",
        "{area} Flow Control Loop",
        "{area} Product Flow Control",
        "{area} Recycle Flow Controller",
    ],
    "TI": [
        "{area} Temperature Indicator",
        "{area} Process Temperature",
        "{area} Reactor Temperature",
        "{area} Column Temperature",
    ],
    "TIC": [
        "{area} Temperature Controller",
        "{area} Reactor Temperature Control",
        "{area} Heater Temperature Control",
        "{area} Cooler Temperature Control",
    ],
    "PI": [
        "{area} Pressure Indicator",
        "{area} Process Pressure",
        "{area} Vessel Pressure",
        "{area} Line Pressure",
    ],
    "PIC": [
        "{area} Pressure Controller",
        "{area} Back Pressure Control",
        "{area} Reactor Pressure Control",
    ],
    "LI": [
        "{area} Level Indicator",
        "{area} Tank Level",
        "{area} Vessel Level",
        "{area} Drum Level",
    ],
    "LIC": [
        "{area} Level Controller",
        "{area} Tank Level Control",
        "{area} Vessel Level Control",
    ],
    "AI": [
        "{area} Analyzer Indicator",
        "{area} Composition Analyzer",
        "{area} Quality Measurement",
    ],
    "AIC": [
        "{area} Analyzer Controller",
        "{area} Composition Controller",
        "{area} Quality Control Loop",
    ],
    "default": [
        "{area} Process Measurement",
        "{area} Control Loop",
        "{area} Indicator",
    ],
}

# Process area names for realistic tag generation
PROCESS_AREAS: list[str] = [
    "Reactor",
    "Distillation",
    "Heat Exchanger",
    "Separator",
    "Compressor",
    "Pump",
    "Tank",
    "Vessel",
    "Furnace",
    "Condenser",
    "Reboiler",
    "Stripper",
    "Absorber",
    "Filter",
    "Dryer",
    "Mixer",
    "Crystallizer",
    "Evaporator",
    "Boiler",
    "Cooling Tower",
]


@dataclass(frozen=True, slots=True)
class TagSample:
    """A single tagged sample for ML training.

    Attributes:
        tag_name: ISA-style tag name (e.g., FIC-101)
        irdi: Associated IEC 61987 IRDI
        description: Human-readable description
        engineering_unit: Engineering unit if applicable
        features: Optional feature dictionary for additional context
    """

    tag_name: str
    irdi: str
    description: str = ""
    engineering_unit: str = ""
    features: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, str | dict[str, str]]:
        """Convert to dictionary representation."""
        return {
            "tag_name": self.tag_name,
            "irdi": self.irdi,
            "description": self.description,
            "engineering_unit": self.engineering_unit,
            "features": self.features,
        }


@dataclass
class DatasetSplit:
    """Train/validation/test split of tag samples.

    Attributes:
        train: Training samples
        val: Validation samples
        test: Test samples
    """

    train: list[TagSample]
    val: list[TagSample]
    test: list[TagSample]

    @property
    def total_size(self) -> int:
        """Return total number of samples across all splits."""
        return len(self.train) + len(self.val) + len(self.test)

    def get_irdi_distribution(self) -> dict[str, int]:
        """Get distribution of IRDIs across all splits."""
        distribution: dict[str, int] = {}
        for sample in self.train + self.val + self.test:
            distribution[sample.irdi] = distribution.get(sample.irdi, 0) + 1
        return distribution


class SyntheticTagGenerator:
    """Generate synthetic ISA-95 tags with IEC 61987 IRDI mappings.

    This generator creates realistic industrial process control tags following
    ISA S5.1 naming conventions, mapped to IEC 61987 IRDIs for training
    semantic mapping models.

    Attributes:
        seed: Random seed for reproducibility
        rng: Random number generator instance

    Example:
        >>> generator = SyntheticTagGenerator(seed=42)
        >>> samples = generator.generate(num_samples=100)
        >>> len(samples)
        100
        >>> split = generator.generate_split(total=1000, train_ratio=0.8, val_ratio=0.1)
        >>> len(split.train), len(split.val), len(split.test)
        (800, 100, 100)
    """

    # Tag type definitions with their configurations
    TAG_TYPES: ClassVar[list[dict[str, str | list[str]]]] = [
        # Flow tags
        {"prefix": "FI", "category": "flow_rate", "patterns": ["FI"]},
        {"prefix": "FIC", "category": "flow_rate", "patterns": ["FIC"]},
        {"prefix": "FT", "category": "flow_rate", "patterns": ["FT", "FIT"]},
        {"prefix": "FCV", "category": "valve_position", "patterns": ["FCV", "FV"]},
        {"prefix": "FQ", "category": "flow_totalizer", "patterns": ["FQ", "FQI"]},
        # Temperature tags
        {"prefix": "TI", "category": "temperature", "patterns": ["TI"]},
        {"prefix": "TIC", "category": "temperature", "patterns": ["TIC"]},
        {"prefix": "TT", "category": "temperature", "patterns": ["TT", "TIT", "TE"]},
        {"prefix": "TSH", "category": "alarm_high", "patterns": ["TSH"]},
        {"prefix": "TSL", "category": "alarm_low", "patterns": ["TSL"]},
        {"prefix": "TSHH", "category": "alarm_high_high", "patterns": ["TSHH"]},
        {"prefix": "TSLL", "category": "alarm_low_low", "patterns": ["TSLL"]},
        # Pressure tags
        {"prefix": "PI", "category": "pressure", "patterns": ["PI"]},
        {"prefix": "PIC", "category": "pressure", "patterns": ["PIC"]},
        {"prefix": "PT", "category": "pressure", "patterns": ["PT", "PIT", "PE"]},
        {"prefix": "PDI", "category": "pressure_differential", "patterns": ["PDI"]},
        {"prefix": "PDIC", "category": "pressure_differential", "patterns": ["PDIC"]},
        {"prefix": "PSH", "category": "alarm_high", "patterns": ["PSH"]},
        {"prefix": "PSL", "category": "alarm_low", "patterns": ["PSL"]},
        # Level tags
        {"prefix": "LI", "category": "level", "patterns": ["LI"]},
        {"prefix": "LIC", "category": "level", "patterns": ["LIC"]},
        {"prefix": "LT", "category": "level", "patterns": ["LT", "LIT", "LE"]},
        {"prefix": "LCV", "category": "valve_position", "patterns": ["LCV", "LV"]},
        {"prefix": "LSH", "category": "alarm_high", "patterns": ["LSH"]},
        {"prefix": "LSL", "category": "alarm_low", "patterns": ["LSL"]},
        # Analyzer tags
        {"prefix": "AI", "category": "composition", "patterns": ["AI"]},
        {"prefix": "AIC", "category": "composition", "patterns": ["AIC"]},
        {"prefix": "AT", "category": "composition", "patterns": ["AT", "AIT", "AE"]},
        # Speed tags
        {"prefix": "SI", "category": "speed", "patterns": ["SI"]},
        {"prefix": "SIC", "category": "speed", "patterns": ["SIC"]},
        {"prefix": "ST", "category": "speed", "patterns": ["ST"]},
        # Weight tags
        {"prefix": "WI", "category": "weight", "patterns": ["WI"]},
        {"prefix": "WIC", "category": "weight", "patterns": ["WIC"]},
        {"prefix": "WT", "category": "weight", "patterns": ["WT"]},
        # Generic valves
        {"prefix": "XV", "category": "valve_position", "patterns": ["XV"]},
        {"prefix": "HV", "category": "valve_position", "patterns": ["HV"]},
        {"prefix": "CV", "category": "valve_position", "patterns": ["CV"]},
    ]

    def __init__(self, seed: int = 42) -> None:
        """Initialize the generator with a random seed.

        Args:
            seed: Random seed for reproducibility
        """
        self.seed = seed
        self.rng = random.Random(seed)

    def _get_tag_number(self, style: str = "standard") -> str:
        """Generate a realistic tag number.

        Args:
            style: Number style - 'standard' (101-999), 'extended' (1001-9999),
                   or 'area_based' (e.g., 2301 = area 23, loop 01)

        Returns:
            Tag number string
        """
        if style == "standard":
            return str(self.rng.randint(101, 999))
        elif style == "extended":
            return str(self.rng.randint(1001, 9999))
        else:  # area_based
            area = self.rng.randint(10, 99)
            loop = self.rng.randint(1, 99)
            return f"{area}{loop:02d}"

    def _get_delimiter(self) -> str:
        """Get a random tag delimiter.

        Returns:
            Delimiter character
        """
        return self.rng.choice(["-", "_", ""])

    def _get_suffix(self) -> str:
        """Get an optional tag suffix.

        Returns:
            Suffix string (may be empty)
        """
        if self.rng.random() < 0.3:
            return self.rng.choice([".PV", ".SP", ".OP", ".CV", "_PV", "_SP", ""])
        return ""

    def _get_engineering_unit(self, category: str) -> str:
        """Get a random engineering unit for a category.

        Args:
            category: Tag category (e.g., 'flow_rate', 'temperature')

        Returns:
            Engineering unit string
        """
        # Map category to unit type
        unit_mapping: dict[str, str] = {
            "flow_rate": "flow_rate",
            "flow_totalizer": "flow_rate",
            "temperature": "temperature",
            "pressure": "pressure",
            "pressure_differential": "pressure",
            "level": "level",
            "valve_position": "valve_position",
            "composition": "composition",
            "speed": "speed",
            "weight": "weight",
            "alarm_high": "level",  # Alarms usually have same units as measured variable
            "alarm_low": "level",
            "alarm_high_high": "level",
            "alarm_low_low": "level",
        }
        unit_type = unit_mapping.get(category, "level")
        units = ENGINEERING_UNITS.get(unit_type, ["%"])
        return self.rng.choice(units)

    def _get_description(self, prefix: str) -> str:
        """Generate a realistic description for a tag.

        Args:
            prefix: Tag prefix (e.g., 'FIC', 'TI')

        Returns:
            Description string
        """
        area = self.rng.choice(PROCESS_AREAS)

        # Get templates for this prefix, fall back to default
        templates = DESCRIPTION_TEMPLATES.get(prefix, DESCRIPTION_TEMPLATES["default"])
        template = self.rng.choice(templates)

        return template.format(area=area)

    def _generate_single_tag(self) -> TagSample:
        """Generate a single synthetic tag sample.

        Returns:
            TagSample instance
        """
        # Select a random tag type
        tag_type = self.rng.choice(self.TAG_TYPES)
        prefix = str(tag_type["prefix"])
        category = str(tag_type["category"])

        # Generate tag components
        style = self.rng.choice(["standard", "extended", "area_based"])
        number = self._get_tag_number(style)
        delimiter = self._get_delimiter()
        suffix = self._get_suffix()

        # Construct tag name
        tag_name = f"{prefix}{delimiter}{number}{suffix}"

        # Get IRDI
        irdi = SEED_IRDIS.get(category, SEED_IRDIS["process_value"])

        # Generate description and unit
        description = self._get_description(prefix)
        engineering_unit = self._get_engineering_unit(category)

        # Build features dict
        features: dict[str, str] = {
            "prefix": prefix,
            "number": number,
            "category": category,
        }

        return TagSample(
            tag_name=tag_name,
            irdi=irdi,
            description=description,
            engineering_unit=engineering_unit,
            features=features,
        )

    def generate(self, num_samples: int) -> list[TagSample]:
        """Generate a list of synthetic tag samples.

        Args:
            num_samples: Number of samples to generate

        Returns:
            List of TagSample instances
        """
        return [self._generate_single_tag() for _ in range(num_samples)]

    def generate_iter(self, num_samples: int) -> Iterator[TagSample]:
        """Generate tag samples as an iterator.

        Args:
            num_samples: Number of samples to generate

        Yields:
            TagSample instances
        """
        for _ in range(num_samples):
            yield self._generate_single_tag()

    def generate_split(
        self,
        total: int,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        test_ratio: float | None = None,
    ) -> DatasetSplit:
        """Generate train/val/test split of synthetic tags.

        Args:
            total: Total number of samples to generate
            train_ratio: Fraction for training (default 0.8)
            val_ratio: Fraction for validation (default 0.1)
            test_ratio: Fraction for test (default: 1 - train_ratio - val_ratio)

        Returns:
            DatasetSplit with train, val, and test samples

        Raises:
            ValueError: If ratios don't sum to 1.0
        """
        if test_ratio is None:
            test_ratio = 1.0 - train_ratio - val_ratio

        if not (0.99 <= train_ratio + val_ratio + test_ratio <= 1.01):
            raise ValueError(
                f"Split ratios must sum to 1.0, got {train_ratio + val_ratio + test_ratio}"
            )

        # Generate all samples
        all_samples = self.generate(total)

        # Calculate split sizes
        train_size = int(total * train_ratio)
        val_size = int(total * val_ratio)

        # Shuffle with fixed seed for reproducibility
        self.rng.shuffle(all_samples)

        # Split
        train = all_samples[:train_size]
        val = all_samples[train_size : train_size + val_size]
        test = all_samples[train_size + val_size :]

        return DatasetSplit(train=train, val=val, test=test)

    def generate_balanced(self, samples_per_irdi: int) -> list[TagSample]:
        """Generate balanced samples with equal representation of each IRDI.

        Args:
            samples_per_irdi: Number of samples per unique IRDI

        Returns:
            List of TagSample instances with balanced IRDI distribution
        """
        samples: list[TagSample] = []

        # Group tag types by IRDI
        irdi_to_types: dict[str, list[dict[str, str | list[str]]]] = {}
        for tag_type in self.TAG_TYPES:
            category = str(tag_type["category"])
            irdi = SEED_IRDIS.get(category, SEED_IRDIS["process_value"])
            if irdi not in irdi_to_types:
                irdi_to_types[irdi] = []
            irdi_to_types[irdi].append(tag_type)

        # Generate samples for each IRDI
        for irdi, types in irdi_to_types.items():
            for _ in range(samples_per_irdi):
                # Select a random type that maps to this IRDI
                tag_type = self.rng.choice(types)
                prefix = str(tag_type["prefix"])
                category = str(tag_type["category"])

                # Generate tag components
                style = self.rng.choice(["standard", "extended", "area_based"])
                number = self._get_tag_number(style)
                delimiter = self._get_delimiter()
                suffix = self._get_suffix()

                tag_name = f"{prefix}{delimiter}{number}{suffix}"
                description = self._get_description(prefix)
                engineering_unit = self._get_engineering_unit(category)

                samples.append(
                    TagSample(
                        tag_name=tag_name,
                        irdi=irdi,
                        description=description,
                        engineering_unit=engineering_unit,
                        features={"prefix": prefix, "number": number, "category": category},
                    )
                )

        # Shuffle for randomness
        self.rng.shuffle(samples)
        return samples

    @staticmethod
    def get_irdi_for_tag(tag_name: str) -> str | None:
        """Get the IRDI for a given tag name based on its prefix.

        Args:
            tag_name: ISA-style tag name

        Returns:
            IRDI string if pattern is recognized, None otherwise
        """
        # Extract prefix from tag name (letters before delimiter or number)
        match = re.match(r"^([A-Za-z]+)", tag_name)
        if not match:
            return None

        prefix = match.group(1).upper()

        # Look up in pattern mapping
        return TAG_PATTERN_TO_IRDI.get(prefix)

    @staticmethod
    def get_all_irdis() -> dict[str, str]:
        """Get all seed IRDIs.

        Returns:
            Dictionary mapping concept names to IRDIs
        """
        return SEED_IRDIS.copy()

    @staticmethod
    def get_irdi_categories() -> list[str]:
        """Get list of all IRDI categories.

        Returns:
            List of category names
        """
        return list(SEED_IRDIS.keys())
