"""NASA C-MAPSS turbofan engine degradation dataset loader.

The Commercial Modular Aero-Propulsion System Simulation (C-MAPSS) dataset
contains run-to-failure simulations of turbofan engines under various
conditions. The dataset is widely used for prognostics and health management
(PHM) research.

Dataset structure:
- Operational settings (3): Altitude, Mach number, Throttle Resolver Angle
- Sensor measurements (21): Temperatures, pressures, speeds, flow rates

Subsets:
- FD001: Single operating condition, single fault mode
- FD002: Six operating conditions, single fault mode
- FD003: Single operating condition, two fault modes
- FD004: Six operating conditions, two fault modes

This module provides a CMAPSSDataset class that:
- Loads C-MAPSS text files (train_FD001.txt, test_FD001.txt, etc.)
- Maps sensors to ISA-style names
- Associates each sensor with seed IRDIs for training
- Returns (tag_name, irdi, features) tuples for ML training

Reference:
    Saxena, A., Goebel, K., Simon, D., Eklund, N. (2008). Damage propagation
    modeling for aircraft engine run-to-failure simulation. IEEE PHM Conference.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
from numpy.typing import NDArray

from noa_swarm.ml.datasets.synth_tags import SEED_IRDIS, DatasetSplit, TagSample

if TYPE_CHECKING:
    from collections.abc import Iterator


# C-MAPSS Sensor Definitions
# Based on the PHM08 data challenge documentation

CMAPSS_OPERATIONAL_SETTINGS: dict[int, dict[str, str]] = {
    1: {
        "name": "Altitude",
        "unit": "ft",
        "isa_tag": "ZI-101",
        "category": "position",
    },
    2: {
        "name": "Mach Number",
        "unit": "Ma",
        "isa_tag": "SI-101",
        "category": "speed",
    },
    3: {
        "name": "Throttle Resolver Angle (TRA)",
        "unit": "deg",
        "isa_tag": "ZI-102",
        "category": "position",
    },
}

CMAPSS_SENSORS: dict[int, dict[str, str]] = {
    1: {
        "name": "Total Temperature at Fan Inlet (T2)",
        "unit": "degR",
        "isa_tag": "TI-201",
        "category": "temperature",
    },
    2: {
        "name": "Total Temperature at LPC Outlet (T24)",
        "unit": "degR",
        "isa_tag": "TI-202",
        "category": "temperature",
    },
    3: {
        "name": "Total Temperature at HPC Outlet (T30)",
        "unit": "degR",
        "isa_tag": "TI-203",
        "category": "temperature",
    },
    4: {
        "name": "Total Temperature at LPT Outlet (T50)",
        "unit": "degR",
        "isa_tag": "TI-204",
        "category": "temperature",
    },
    5: {
        "name": "Pressure at Fan Inlet (P2)",
        "unit": "psia",
        "isa_tag": "PI-201",
        "category": "pressure",
    },
    6: {
        "name": "Total Pressure in Bypass-Duct (Ps30)",
        "unit": "psia",
        "isa_tag": "PI-202",
        "category": "pressure",
    },
    7: {
        "name": "Physical Fan Speed (Nf)",
        "unit": "rpm",
        "isa_tag": "SI-201",
        "category": "speed",
    },
    8: {
        "name": "Physical Core Speed (Nc)",
        "unit": "rpm",
        "isa_tag": "SI-202",
        "category": "speed",
    },
    9: {
        "name": "Engine Pressure Ratio (epr)",
        "unit": "",
        "isa_tag": "PDI-201",
        "category": "pressure_differential",
    },
    10: {
        "name": "Static Pressure at HPC Outlet (Ps30)",
        "unit": "psia",
        "isa_tag": "PI-203",
        "category": "pressure",
    },
    11: {
        "name": "Ratio of Fuel Flow to Ps30 (phi)",
        "unit": "pps/psia",
        "isa_tag": "FI-201",
        "category": "flow_rate",
    },
    12: {
        "name": "Corrected Fan Speed (NRf)",
        "unit": "rpm",
        "isa_tag": "SI-203",
        "category": "speed",
    },
    13: {
        "name": "Corrected Core Speed (NRc)",
        "unit": "rpm",
        "isa_tag": "SI-204",
        "category": "speed",
    },
    14: {
        "name": "Bypass Ratio (BPR)",
        "unit": "",
        "isa_tag": "AI-201",
        "category": "composition",
    },
    15: {
        "name": "Burner Fuel-Air Ratio (farB)",
        "unit": "",
        "isa_tag": "AI-202",
        "category": "composition",
    },
    16: {
        "name": "Bleed Enthalpy (htBleed)",
        "unit": "",
        "isa_tag": "TI-205",
        "category": "temperature",
    },
    17: {
        "name": "Demanded Fan Speed (Nf_dmd)",
        "unit": "rpm",
        "isa_tag": "SIC-201",
        "category": "setpoint",
    },
    18: {
        "name": "Demanded Corrected Fan Speed (PCNfR_dmd)",
        "unit": "rpm",
        "isa_tag": "SIC-202",
        "category": "setpoint",
    },
    19: {
        "name": "High-Pressure Turbine Cool Air Flow (W31)",
        "unit": "lbm/s",
        "isa_tag": "FI-202",
        "category": "flow_rate",
    },
    20: {
        "name": "Low-Pressure Turbine Cool Air Flow (W32)",
        "unit": "lbm/s",
        "isa_tag": "FI-203",
        "category": "flow_rate",
    },
    21: {
        "name": "HPT Coolant Bleed (hpBleed)",
        "unit": "",
        "isa_tag": "FI-204",
        "category": "flow_rate",
    },
}

# Dataset subset definitions
CMAPSSSubset = Literal["FD001", "FD002", "FD003", "FD004"]

CMAPSS_SUBSETS: dict[str, dict[str, str | int]] = {
    "FD001": {
        "description": "Single operating condition, HPC degradation",
        "operating_conditions": 1,
        "fault_modes": 1,
    },
    "FD002": {
        "description": "Six operating conditions, HPC degradation",
        "operating_conditions": 6,
        "fault_modes": 1,
    },
    "FD003": {
        "description": "Single operating condition, HPC and Fan degradation",
        "operating_conditions": 1,
        "fault_modes": 2,
    },
    "FD004": {
        "description": "Six operating conditions, HPC and Fan degradation",
        "operating_conditions": 6,
        "fault_modes": 2,
    },
}


@dataclass
class CMAPSSSensor:
    """Single C-MAPSS sensor or setting with its metadata.

    Attributes:
        index: Sensor index (1-based)
        sensor_type: Sensor type ('operational_setting' or 'sensor')
        name: Sensor description
        unit: Engineering unit
        isa_tag: ISA-style tag name
        category: Semantic category for IRDI mapping
        irdi: Associated IRDI
    """

    index: int
    sensor_type: str
    name: str
    unit: str
    isa_tag: str
    category: str
    irdi: str

    @property
    def original_name(self) -> str:
        """Return original C-MAPSS variable name."""
        if self.sensor_type == "operational_setting":
            return f"setting_{self.index}"
        return f"sensor_{self.index}"

    def to_tag_sample(self, features: dict[str, float] | None = None) -> TagSample:
        """Convert to TagSample for ML training.

        Args:
            features: Optional feature values for this sensor

        Returns:
            TagSample instance
        """
        feature_dict: dict[str, str] = {
            "sensor_type": self.sensor_type,
            "index": str(self.index),
            "category": self.category,
        }
        if features:
            for k, v in features.items():
                feature_dict[k] = str(v)

        return TagSample(
            tag_name=self.isa_tag,
            irdi=self.irdi,
            description=self.name,
            engineering_unit=self.unit,
            features=feature_dict,
        )


class CMAPSSDataset:
    """NASA C-MAPSS turbofan engine degradation dataset loader.

    Loads C-MAPSS data files and provides mappings from sensors to ISA-style
    tags and IEC 61987 IRDIs.

    Attributes:
        data_dir: Path to data directory
        subset: Dataset subset (FD001, FD002, FD003, FD004)
        sensors: List of all sensors with metadata
        train_data: Loaded training data
        test_data: Loaded test data
        rul_data: Remaining useful life labels

    Example:
        >>> dataset = CMAPSSDataset(data_dir=Path("data/cmapss"), subset="FD001")
        >>> sensors = dataset.get_sensors()
        >>> for sensor in sensors[:5]:
        ...     print(f"{sensor.isa_tag}: {sensor.name} -> {sensor.irdi}")
        ZI-101: Altitude -> 0173-1#01-AAA403#001
        ...
    """

    # Column names for C-MAPSS data files
    COLUMN_NAMES: list[str] = (
        ["unit_id", "cycle"]
        + [f"setting_{i}" for i in range(1, 4)]
        + [f"sensor_{i}" for i in range(1, 22)]
    )

    def __init__(
        self,
        data_dir: Path | str | None = None,
        subset: CMAPSSSubset = "FD001",
    ) -> None:
        """Initialize C-MAPSS dataset.

        Args:
            data_dir: Path to directory containing C-MAPSS data files.
                      If None, only metadata is available.
            subset: Dataset subset to use (FD001, FD002, FD003, FD004)
        """
        self.data_dir = Path(data_dir) if data_dir else None
        self.subset = subset
        self.sensors: list[CMAPSSSensor] = self._build_sensors()
        self.train_data: NDArray[np.floating[Any]] | None = None
        self.test_data: NDArray[np.floating[Any]] | None = None
        self.rul_data: NDArray[np.floating[Any]] | None = None

    def _build_sensors(self) -> list[CMAPSSSensor]:
        """Build list of all C-MAPSS sensors with metadata.

        Returns:
            List of CMAPSSSensor instances
        """
        sensors: list[CMAPSSSensor] = []

        # Add operational settings
        for idx, info in CMAPSS_OPERATIONAL_SETTINGS.items():
            irdi = SEED_IRDIS.get(info["category"], SEED_IRDIS["process_value"])
            sensors.append(
                CMAPSSSensor(
                    index=idx,
                    sensor_type="operational_setting",
                    name=info["name"],
                    unit=info["unit"],
                    isa_tag=info["isa_tag"],
                    category=info["category"],
                    irdi=irdi,
                )
            )

        # Add sensors
        for idx, info in CMAPSS_SENSORS.items():
            irdi = SEED_IRDIS.get(info["category"], SEED_IRDIS["process_value"])
            sensors.append(
                CMAPSSSensor(
                    index=idx,
                    sensor_type="sensor",
                    name=info["name"],
                    unit=info["unit"],
                    isa_tag=info["isa_tag"],
                    category=info["category"],
                    irdi=irdi,
                )
            )

        return sensors

    def get_sensors(self) -> list[CMAPSSSensor]:
        """Get all C-MAPSS sensors.

        Returns:
            List of CMAPSSSensor instances
        """
        return self.sensors.copy()

    def get_sensor_by_tag(self, tag: str) -> CMAPSSSensor | None:
        """Find a sensor by its ISA tag name.

        Args:
            tag: ISA-style tag name (e.g., 'TI-201')

        Returns:
            CMAPSSSensor if found, None otherwise
        """
        for sensor in self.sensors:
            if sensor.isa_tag == tag:
                return sensor
        return None

    def get_sensor_by_index(
        self, sensor_type: str, index: int
    ) -> CMAPSSSensor | None:
        """Find a sensor by its type and index.

        Args:
            sensor_type: Sensor type ('operational_setting' or 'sensor')
            index: Sensor index (1-based)

        Returns:
            CMAPSSSensor if found, None otherwise
        """
        for sensor in self.sensors:
            if sensor.sensor_type == sensor_type and sensor.index == index:
                return sensor
        return None

    def get_tag_samples(self) -> list[TagSample]:
        """Get all sensors as TagSample instances.

        Returns:
            List of TagSample instances for ML training
        """
        return [sensor.to_tag_sample() for sensor in self.sensors]

    def get_tag_samples_iter(self) -> Iterator[TagSample]:
        """Iterate over sensors as TagSample instances.

        Yields:
            TagSample instances
        """
        for sensor in self.sensors:
            yield sensor.to_tag_sample()

    def get_split(
        self,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        seed: int = 42,
    ) -> DatasetSplit:
        """Get train/val/test split of C-MAPSS sensors.

        Args:
            train_ratio: Fraction for training
            val_ratio: Fraction for validation
            seed: Random seed for reproducibility

        Returns:
            DatasetSplit with train, val, and test samples
        """
        import random

        rng = random.Random(seed)
        samples = self.get_tag_samples()
        rng.shuffle(samples)

        total = len(samples)
        train_size = int(total * train_ratio)
        val_size = int(total * val_ratio)

        train = samples[:train_size]
        val = samples[train_size : train_size + val_size]
        test = samples[train_size + val_size :]

        return DatasetSplit(train=train, val=val, test=test)

    def load_train_data(self) -> NDArray[np.floating[Any]]:
        """Load training data for the specified subset.

        Returns:
            NumPy array of shape (samples, 26) with columns:
            [unit_id, cycle, setting1-3, sensor1-21]

        Raises:
            FileNotFoundError: If data file is not found
            ValueError: If data_dir is not set
        """
        if self.data_dir is None:
            raise ValueError("data_dir must be set to load data")

        filename = f"train_{self.subset}.txt"
        filepath = self.data_dir / filename
        if not filepath.exists():
            raise FileNotFoundError(f"C-MAPSS training file not found: {filepath}")

        self.train_data = np.loadtxt(filepath)
        return self.train_data

    def load_test_data(self) -> NDArray[np.floating[Any]]:
        """Load test data for the specified subset.

        Returns:
            NumPy array of shape (samples, 26)

        Raises:
            FileNotFoundError: If data file is not found
            ValueError: If data_dir is not set
        """
        if self.data_dir is None:
            raise ValueError("data_dir must be set to load data")

        filename = f"test_{self.subset}.txt"
        filepath = self.data_dir / filename
        if not filepath.exists():
            raise FileNotFoundError(f"C-MAPSS test file not found: {filepath}")

        self.test_data = np.loadtxt(filepath)
        return self.test_data

    def load_rul_data(self) -> NDArray[np.floating[Any]]:
        """Load RUL (Remaining Useful Life) labels for the test set.

        Returns:
            NumPy array of shape (num_engines,)

        Raises:
            FileNotFoundError: If data file is not found
            ValueError: If data_dir is not set
        """
        if self.data_dir is None:
            raise ValueError("data_dir must be set to load data")

        filename = f"RUL_{self.subset}.txt"
        filepath = self.data_dir / filename
        if not filepath.exists():
            raise FileNotFoundError(f"C-MAPSS RUL file not found: {filepath}")

        self.rul_data = np.loadtxt(filepath)
        return self.rul_data

    def load_all(
        self,
    ) -> tuple[
        NDArray[np.floating[Any]], NDArray[np.floating[Any]], NDArray[np.floating[Any]]
    ]:
        """Load all data files for the subset.

        Returns:
            Tuple of (train_data, test_data, rul_data)

        Raises:
            FileNotFoundError: If any data file is not found
            ValueError: If data_dir is not set
        """
        train = self.load_train_data()
        test = self.load_test_data()
        rul = self.load_rul_data()
        return train, test, rul

    def get_operational_settings(self) -> list[CMAPSSSensor]:
        """Get only operational setting sensors.

        Returns:
            List of CMAPSSSensor instances for settings
        """
        return [s for s in self.sensors if s.sensor_type == "operational_setting"]

    def get_measurement_sensors(self) -> list[CMAPSSSensor]:
        """Get only measurement sensors (not settings).

        Returns:
            List of CMAPSSSensor instances for measurements
        """
        return [s for s in self.sensors if s.sensor_type == "sensor"]

    def get_sensors_by_category(self, category: str) -> list[CMAPSSSensor]:
        """Get sensors by semantic category.

        Args:
            category: Semantic category (e.g., 'temperature', 'pressure')

        Returns:
            List of matching CMAPSSSensor instances
        """
        return [s for s in self.sensors if s.category == category]

    def get_irdi_mapping(self) -> dict[str, str]:
        """Get mapping from ISA tags to IRDIs.

        Returns:
            Dictionary mapping ISA tag names to IRDIs
        """
        return {s.isa_tag: s.irdi for s in self.sensors}

    def get_subset_info(self) -> dict[str, str | int]:
        """Get information about the current subset.

        Returns:
            Dictionary with subset details
        """
        return CMAPSS_SUBSETS[self.subset]

    @staticmethod
    def get_available_subsets() -> list[str]:
        """Get list of available C-MAPSS subsets.

        Returns:
            List of subset names
        """
        return list(CMAPSS_SUBSETS.keys())

    @property
    def num_sensors(self) -> int:
        """Return total number of sensors."""
        return len(self.sensors)

    @property
    def num_operational_settings(self) -> int:
        """Return number of operational settings."""
        return len(CMAPSS_OPERATIONAL_SETTINGS)

    @property
    def num_measurement_sensors(self) -> int:
        """Return number of measurement sensors."""
        return len(CMAPSS_SENSORS)

    def __repr__(self) -> str:
        """Return string representation."""
        return (
            f"CMAPSSDataset(subset={self.subset!r}, sensors={self.num_sensors}, "
            f"train_loaded={self.train_data is not None}, "
            f"test_loaded={self.test_data is not None})"
        )
