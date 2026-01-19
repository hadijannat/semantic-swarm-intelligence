"""Machine Learning module for NOA Semantic Swarm Mapper.

This module provides:
- Dataset loaders for training ML models (TEP, C-MAPSS)
- Synthetic tag generation with ISA patterns
- CharCNN model for semantic tag mapping
- Training and inference utilities
"""

from noa_swarm.ml.datasets import (
    CMAPSSDataset,
    DatasetSplit,
    SyntheticTagGenerator,
    TagSample,
    TEPDataset,
)

__all__ = [
    "CMAPSSDataset",
    "DatasetSplit",
    "SyntheticTagGenerator",
    "TagSample",
    "TEPDataset",
]
