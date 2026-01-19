"""Dataset loaders for ML model training.

This module provides dataset loaders for industrial process data used to
train the CharCNN semantic tag mapping model:

- **TEPDataset**: Tennessee Eastman Process simulation data
- **CMAPSSDataset**: NASA turbofan engine degradation data
- **SyntheticTagGenerator**: Generate unlimited labeled training data

Example:
    >>> from noa_swarm.ml.datasets import SyntheticTagGenerator, TEPDataset
    >>>
    >>> # Generate synthetic training data
    >>> generator = SyntheticTagGenerator(seed=42)
    >>> split = generator.generate_split(total=1000, train_ratio=0.8)
    >>> print(f"Train: {len(split.train)}, Val: {len(split.val)}, Test: {len(split.test)}")
    Train: 800, Val: 100, Test: 100
    >>>
    >>> # Load TEP variables
    >>> tep = TEPDataset()
    >>> variables = tep.get_tag_samples()
    >>> print(f"TEP has {len(variables)} variables")
    TEP has 52 variables
"""

from noa_swarm.ml.datasets.cmapss import (
    CMAPSS_OPERATIONAL_SETTINGS,
    CMAPSS_SENSORS,
    CMAPSS_SUBSETS,
    CMAPSSDataset,
    CMAPSSSensor,
)
from noa_swarm.ml.datasets.synth_tags import (
    ENGINEERING_UNITS,
    SEED_IRDIS,
    TAG_PATTERN_TO_IRDI,
    DatasetSplit,
    SyntheticTagGenerator,
    TagSample,
)
from noa_swarm.ml.datasets.tep import (
    TEP_XMEAS_VARIABLES,
    TEP_XMV_VARIABLES,
    TEPDataset,
    TEPVariable,
)

__all__ = [
    # Synthetic generator
    "DatasetSplit",
    "ENGINEERING_UNITS",
    "SEED_IRDIS",
    "SyntheticTagGenerator",
    "TAG_PATTERN_TO_IRDI",
    "TagSample",
    # TEP dataset
    "TEP_XMEAS_VARIABLES",
    "TEP_XMV_VARIABLES",
    "TEPDataset",
    "TEPVariable",
    # C-MAPSS dataset
    "CMAPSS_OPERATIONAL_SETTINGS",
    "CMAPSS_SENSORS",
    "CMAPSS_SUBSETS",
    "CMAPSSDataset",
    "CMAPSSSensor",
]
