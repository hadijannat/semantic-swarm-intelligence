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
from noa_swarm.ml.models import CharCNN, CharacterTokenizer
from noa_swarm.ml.training import (
    compute_accuracy,
    compute_confusion_matrix,
    compute_macro_f1,
    compute_top_k_accuracy,
)
from noa_swarm.ml.serving import (
    FusionInferenceConfig,
    FusionInferenceEngine,
    InferenceConfig,
    RuleBasedInferenceEngine,
    build_fusion_engine_from_env,
)

__all__ = [
    # Datasets
    "CMAPSSDataset",
    "DatasetSplit",
    "SyntheticTagGenerator",
    "TagSample",
    "TEPDataset",
    # Models
    "CharCNN",
    "CharacterTokenizer",
    # Evaluation
    "compute_accuracy",
    "compute_confusion_matrix",
    "compute_macro_f1",
    "compute_top_k_accuracy",
    # Serving
    "InferenceConfig",
    "RuleBasedInferenceEngine",
    "FusionInferenceConfig",
    "FusionInferenceEngine",
    "build_fusion_engine_from_env",
]
