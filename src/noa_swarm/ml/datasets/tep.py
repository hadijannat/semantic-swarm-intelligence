"""Tennessee Eastman Process (TEP) dataset loader.

The Tennessee Eastman Process is a benchmark simulation of an industrial chemical
process widely used in process control research. The dataset contains 52 variables:
- XMEAS(1-41): Process measurements (flows, temps, pressures, levels, compositions)
- XMV(1-11): Manipulated variables (valves, setpoints)

This module provides a TEPDataset class that:
- Loads TEP data from CSV or MAT format files
- Maps variables to ISA-style names (e.g., XMEAS_1 -> FI-101)
- Associates each variable with seed IRDIs for training
- Returns (tag_name, irdi, features) tuples for ML training

Reference:
    Downs, J.J., Vogel, E.F. (1993). A plant-wide industrial process control problem.
    Computers & Chemical Engineering, 17(3), 245-255.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from noa_swarm.ml.datasets.synth_tags import SEED_IRDIS, DatasetSplit, TagSample

if TYPE_CHECKING:
    from collections.abc import Iterator

    from numpy.typing import NDArray


# TEP Variable Definitions
# XMEAS: Process measurements (1-41)
# XMV: Manipulated variables (1-11)

TEP_XMEAS_VARIABLES: dict[int, dict[str, str]] = {
    1: {
        "name": "A Feed (stream 1)",
        "unit": "kscmh",
        "isa_tag": "FI-101",
        "category": "flow_rate",
    },
    2: {
        "name": "D Feed (stream 2)",
        "unit": "kg/h",
        "isa_tag": "FI-102",
        "category": "flow_rate",
    },
    3: {
        "name": "E Feed (stream 3)",
        "unit": "kg/h",
        "isa_tag": "FI-103",
        "category": "flow_rate",
    },
    4: {
        "name": "A and C Feed (stream 4)",
        "unit": "kscmh",
        "isa_tag": "FI-104",
        "category": "flow_rate",
    },
    5: {
        "name": "Recycle Flow (stream 8)",
        "unit": "kscmh",
        "isa_tag": "FI-108",
        "category": "flow_rate",
    },
    6: {
        "name": "Reactor Feed Rate (stream 6)",
        "unit": "kscmh",
        "isa_tag": "FI-106",
        "category": "flow_rate",
    },
    7: {
        "name": "Reactor Pressure",
        "unit": "kPa gauge",
        "isa_tag": "PI-101",
        "category": "pressure",
    },
    8: {
        "name": "Reactor Level",
        "unit": "%",
        "isa_tag": "LI-101",
        "category": "level",
    },
    9: {
        "name": "Reactor Temperature",
        "unit": "degC",
        "isa_tag": "TI-101",
        "category": "temperature",
    },
    10: {
        "name": "Purge Rate (stream 9)",
        "unit": "kscmh",
        "isa_tag": "FI-109",
        "category": "flow_rate",
    },
    11: {
        "name": "Product Separator Temperature",
        "unit": "degC",
        "isa_tag": "TI-102",
        "category": "temperature",
    },
    12: {
        "name": "Product Separator Level",
        "unit": "%",
        "isa_tag": "LI-102",
        "category": "level",
    },
    13: {
        "name": "Product Separator Pressure",
        "unit": "kPa gauge",
        "isa_tag": "PI-102",
        "category": "pressure",
    },
    14: {
        "name": "Product Separator Underflow (stream 10)",
        "unit": "m3/h",
        "isa_tag": "FI-110",
        "category": "flow_rate",
    },
    15: {
        "name": "Stripper Level",
        "unit": "%",
        "isa_tag": "LI-103",
        "category": "level",
    },
    16: {
        "name": "Stripper Pressure",
        "unit": "kPa gauge",
        "isa_tag": "PI-103",
        "category": "pressure",
    },
    17: {
        "name": "Stripper Underflow (stream 11)",
        "unit": "m3/h",
        "isa_tag": "FI-111",
        "category": "flow_rate",
    },
    18: {
        "name": "Stripper Temperature",
        "unit": "degC",
        "isa_tag": "TI-103",
        "category": "temperature",
    },
    19: {
        "name": "Stripper Steam Flow",
        "unit": "kg/h",
        "isa_tag": "FI-112",
        "category": "flow_rate",
    },
    20: {
        "name": "Compressor Work",
        "unit": "kW",
        "isa_tag": "JI-101",
        "category": "power",
    },
    21: {
        "name": "Reactor Cooling Water Outlet Temperature",
        "unit": "degC",
        "isa_tag": "TI-104",
        "category": "temperature",
    },
    22: {
        "name": "Separator Cooling Water Outlet Temperature",
        "unit": "degC",
        "isa_tag": "TI-105",
        "category": "temperature",
    },
    # Composition measurements (stream 6 components)
    23: {
        "name": "Component A (stream 6)",
        "unit": "mol%",
        "isa_tag": "AI-101",
        "category": "composition",
    },
    24: {
        "name": "Component B (stream 6)",
        "unit": "mol%",
        "isa_tag": "AI-102",
        "category": "composition",
    },
    25: {
        "name": "Component C (stream 6)",
        "unit": "mol%",
        "isa_tag": "AI-103",
        "category": "composition",
    },
    26: {
        "name": "Component D (stream 6)",
        "unit": "mol%",
        "isa_tag": "AI-104",
        "category": "composition",
    },
    27: {
        "name": "Component E (stream 6)",
        "unit": "mol%",
        "isa_tag": "AI-105",
        "category": "composition",
    },
    28: {
        "name": "Component F (stream 6)",
        "unit": "mol%",
        "isa_tag": "AI-106",
        "category": "composition",
    },
    # Composition measurements (stream 9 - purge)
    29: {
        "name": "Component A (stream 9)",
        "unit": "mol%",
        "isa_tag": "AI-107",
        "category": "composition",
    },
    30: {
        "name": "Component B (stream 9)",
        "unit": "mol%",
        "isa_tag": "AI-108",
        "category": "composition",
    },
    31: {
        "name": "Component C (stream 9)",
        "unit": "mol%",
        "isa_tag": "AI-109",
        "category": "composition",
    },
    32: {
        "name": "Component D (stream 9)",
        "unit": "mol%",
        "isa_tag": "AI-110",
        "category": "composition",
    },
    33: {
        "name": "Component E (stream 9)",
        "unit": "mol%",
        "isa_tag": "AI-111",
        "category": "composition",
    },
    34: {
        "name": "Component F (stream 9)",
        "unit": "mol%",
        "isa_tag": "AI-112",
        "category": "composition",
    },
    35: {
        "name": "Component G (stream 9)",
        "unit": "mol%",
        "isa_tag": "AI-113",
        "category": "composition",
    },
    36: {
        "name": "Component H (stream 9)",
        "unit": "mol%",
        "isa_tag": "AI-114",
        "category": "composition",
    },
    # Composition measurements (stream 11 - product)
    37: {
        "name": "Component D (stream 11)",
        "unit": "mol%",
        "isa_tag": "AI-115",
        "category": "composition",
    },
    38: {
        "name": "Component E (stream 11)",
        "unit": "mol%",
        "isa_tag": "AI-116",
        "category": "composition",
    },
    39: {
        "name": "Component F (stream 11)",
        "unit": "mol%",
        "isa_tag": "AI-117",
        "category": "composition",
    },
    40: {
        "name": "Component G (stream 11)",
        "unit": "mol%",
        "isa_tag": "AI-118",
        "category": "composition",
    },
    41: {
        "name": "Component H (stream 11)",
        "unit": "mol%",
        "isa_tag": "AI-119",
        "category": "composition",
    },
}

TEP_XMV_VARIABLES: dict[int, dict[str, str]] = {
    1: {
        "name": "D Feed Flow (stream 2)",
        "unit": "%",
        "isa_tag": "FCV-102",
        "category": "valve_position",
    },
    2: {
        "name": "E Feed Flow (stream 3)",
        "unit": "%",
        "isa_tag": "FCV-103",
        "category": "valve_position",
    },
    3: {
        "name": "A Feed Flow (stream 1)",
        "unit": "%",
        "isa_tag": "FCV-101",
        "category": "valve_position",
    },
    4: {
        "name": "A and C Feed Flow (stream 4)",
        "unit": "%",
        "isa_tag": "FCV-104",
        "category": "valve_position",
    },
    5: {
        "name": "Compressor Recycle Valve",
        "unit": "%",
        "isa_tag": "FCV-105",
        "category": "valve_position",
    },
    6: {
        "name": "Purge Valve (stream 9)",
        "unit": "%",
        "isa_tag": "FCV-109",
        "category": "valve_position",
    },
    7: {
        "name": "Separator Pot Liquid Flow (stream 10)",
        "unit": "%",
        "isa_tag": "FCV-110",
        "category": "valve_position",
    },
    8: {
        "name": "Stripper Liquid Product Flow (stream 11)",
        "unit": "%",
        "isa_tag": "FCV-111",
        "category": "valve_position",
    },
    9: {
        "name": "Stripper Steam Valve",
        "unit": "%",
        "isa_tag": "FCV-112",
        "category": "valve_position",
    },
    10: {
        "name": "Reactor Cooling Water Flow",
        "unit": "%",
        "isa_tag": "TCV-101",
        "category": "valve_position",
    },
    11: {
        "name": "Condenser Cooling Water Flow",
        "unit": "%",
        "isa_tag": "TCV-102",
        "category": "valve_position",
    },
}


@dataclass
class TEPVariable:
    """Single TEP variable with its metadata.

    Attributes:
        index: Variable index (1-based)
        var_type: Variable type ('XMEAS' or 'XMV')
        name: Variable description
        unit: Engineering unit
        isa_tag: ISA-style tag name
        category: Semantic category for IRDI mapping
        irdi: Associated IRDI
    """

    index: int
    var_type: str
    name: str
    unit: str
    isa_tag: str
    category: str
    irdi: str

    @property
    def original_name(self) -> str:
        """Return original TEP variable name (e.g., XMEAS_1)."""
        return f"{self.var_type}_{self.index}"

    def to_tag_sample(self, features: dict[str, float] | None = None) -> TagSample:
        """Convert to TagSample for ML training.

        Args:
            features: Optional feature values for this variable

        Returns:
            TagSample instance
        """
        feature_dict: dict[str, str] = {
            "var_type": self.var_type,
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


class TEPDataset:
    """Tennessee Eastman Process dataset loader.

    Loads TEP simulation data and provides mappings from process variables
    to ISA-style tags and IEC 61987 IRDIs.

    Attributes:
        data_dir: Path to data directory
        variables: List of all TEP variables with metadata
        data: Loaded data array (if data is loaded)

    Example:
        >>> dataset = TEPDataset(data_dir=Path("data/tep"))
        >>> variables = dataset.get_variables()
        >>> for var in variables[:5]:
        ...     print(f"{var.isa_tag}: {var.name} -> {var.irdi}")
        FI-101: A Feed (stream 1) -> 0173-1#01-AAA001#001
        ...
    """

    def __init__(self, data_dir: Path | str | None = None) -> None:
        """Initialize TEP dataset.

        Args:
            data_dir: Path to directory containing TEP data files.
                      If None, only metadata is available.
        """
        self.data_dir = Path(data_dir) if data_dir else None
        self.variables: list[TEPVariable] = self._build_variables()
        self.data: NDArray[np.floating[Any]] | None = None
        self._fault_labels: NDArray[np.floating[Any]] | None = None

    def _build_variables(self) -> list[TEPVariable]:
        """Build list of all TEP variables with metadata.

        Returns:
            List of TEPVariable instances
        """
        variables: list[TEPVariable] = []

        # Add XMEAS variables
        for idx, info in TEP_XMEAS_VARIABLES.items():
            irdi = SEED_IRDIS.get(info["category"], SEED_IRDIS["process_value"])
            variables.append(
                TEPVariable(
                    index=idx,
                    var_type="XMEAS",
                    name=info["name"],
                    unit=info["unit"],
                    isa_tag=info["isa_tag"],
                    category=info["category"],
                    irdi=irdi,
                )
            )

        # Add XMV variables
        for idx, info in TEP_XMV_VARIABLES.items():
            irdi = SEED_IRDIS.get(info["category"], SEED_IRDIS["valve_position"])
            variables.append(
                TEPVariable(
                    index=idx,
                    var_type="XMV",
                    name=info["name"],
                    unit=info["unit"],
                    isa_tag=info["isa_tag"],
                    category=info["category"],
                    irdi=irdi,
                )
            )

        return variables

    def get_variables(self) -> list[TEPVariable]:
        """Get all TEP variables.

        Returns:
            List of TEPVariable instances
        """
        return self.variables.copy()

    def get_variable_by_tag(self, tag: str) -> TEPVariable | None:
        """Find a variable by its ISA tag name.

        Args:
            tag: ISA-style tag name (e.g., 'FI-101')

        Returns:
            TEPVariable if found, None otherwise
        """
        for var in self.variables:
            if var.isa_tag == tag:
                return var
        return None

    def get_variable_by_index(self, var_type: str, index: int) -> TEPVariable | None:
        """Find a variable by its type and index.

        Args:
            var_type: Variable type ('XMEAS' or 'XMV')
            index: Variable index (1-based)

        Returns:
            TEPVariable if found, None otherwise
        """
        for var in self.variables:
            if var.var_type == var_type and var.index == index:
                return var
        return None

    def get_tag_samples(self) -> list[TagSample]:
        """Get all variables as TagSample instances.

        Returns:
            List of TagSample instances for ML training
        """
        return [var.to_tag_sample() for var in self.variables]

    def get_tag_samples_iter(self) -> Iterator[TagSample]:
        """Iterate over variables as TagSample instances.

        Yields:
            TagSample instances
        """
        for var in self.variables:
            yield var.to_tag_sample()

    def get_split(
        self,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        seed: int = 42,
    ) -> DatasetSplit:
        """Get train/val/test split of TEP variables.

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

    def load_data(self, filename: str = "TEP_FaultFree_Training.csv") -> NDArray[np.floating[Any]]:
        """Load TEP data from a CSV file.

        Args:
            filename: Name of the data file

        Returns:
            NumPy array of shape (samples, variables)

        Raises:
            FileNotFoundError: If data file is not found
            ValueError: If data_dir is not set
        """
        if self.data_dir is None:
            raise ValueError("data_dir must be set to load data")

        filepath = self.data_dir / filename
        if not filepath.exists():
            raise FileNotFoundError(f"TEP data file not found: {filepath}")

        try:
            import pandas as pd  # type: ignore[import-untyped]

            df = pd.read_csv(filepath)
            self.data = df.values
        except ImportError:
            # Fallback to numpy
            self.data = np.loadtxt(filepath, delimiter=",", skiprows=1)

        assert self.data is not None  # For type narrowing
        return self.data

    def load_mat_data(self, filename: str = "TEP.mat") -> NDArray[np.floating[Any]]:
        """Load TEP data from a MATLAB .mat file.

        Args:
            filename: Name of the MAT file

        Returns:
            NumPy array of shape (samples, variables)

        Raises:
            FileNotFoundError: If data file is not found
            ValueError: If data_dir is not set
        """
        if self.data_dir is None:
            raise ValueError("data_dir must be set to load data")

        filepath = self.data_dir / filename
        if not filepath.exists():
            raise FileNotFoundError(f"TEP MAT file not found: {filepath}")

        try:
            from scipy.io import loadmat

            mat_data = loadmat(filepath)
            # Common key names in TEP MAT files
            for key in ["TEP", "data", "X", "simout"]:
                if key in mat_data:
                    self.data = mat_data[key]
                    break
            else:
                # Use first non-metadata key
                data_keys = [k for k in mat_data if not k.startswith("_")]
                if data_keys:
                    self.data = mat_data[data_keys[0]]
                else:
                    raise ValueError(f"Could not find data in MAT file: {filepath}")
        except ImportError as e:
            raise ImportError("scipy is required to load MAT files") from e

        assert self.data is not None  # For type narrowing
        return self.data

    def get_xmeas_variables(self) -> list[TEPVariable]:
        """Get only measurement variables (XMEAS).

        Returns:
            List of XMEAS TEPVariable instances
        """
        return [v for v in self.variables if v.var_type == "XMEAS"]

    def get_xmv_variables(self) -> list[TEPVariable]:
        """Get only manipulated variables (XMV).

        Returns:
            List of XMV TEPVariable instances
        """
        return [v for v in self.variables if v.var_type == "XMV"]

    def get_variables_by_category(self, category: str) -> list[TEPVariable]:
        """Get variables by semantic category.

        Args:
            category: Semantic category (e.g., 'flow_rate', 'temperature')

        Returns:
            List of matching TEPVariable instances
        """
        return [v for v in self.variables if v.category == category]

    def get_irdi_mapping(self) -> dict[str, str]:
        """Get mapping from ISA tags to IRDIs.

        Returns:
            Dictionary mapping ISA tag names to IRDIs
        """
        return {v.isa_tag: v.irdi for v in self.variables}

    @property
    def num_variables(self) -> int:
        """Return total number of variables."""
        return len(self.variables)

    @property
    def num_xmeas(self) -> int:
        """Return number of measurement variables."""
        return len(TEP_XMEAS_VARIABLES)

    @property
    def num_xmv(self) -> int:
        """Return number of manipulated variables."""
        return len(TEP_XMV_VARIABLES)

    def __repr__(self) -> str:
        """Return string representation."""
        return (
            f"TEPDataset(variables={self.num_variables}, "
            f"xmeas={self.num_xmeas}, xmv={self.num_xmv}, "
            f"data_loaded={self.data is not None})"
        )
