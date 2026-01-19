"""Training utilities for ML models.

This module provides:
- Local training script for CharCNN
- Evaluation metrics and utilities
"""

from noa_swarm.ml.training.eval import (
    compute_accuracy,
    compute_confusion_matrix,
    compute_macro_f1,
    compute_top_k_accuracy,
)

__all__ = [
    "compute_accuracy",
    "compute_confusion_matrix",
    "compute_macro_f1",
    "compute_top_k_accuracy",
]
