"""Training utilities for ML models.

This module provides:
- Local training script for CharCNN
- Evaluation metrics and utilities
- Calibration utilities (temperature scaling, ECE)
"""

from noa_swarm.ml.training.calibration import (
    CalibrationResult,
    ReliabilityDiagramData,
    TemperatureScaler,
    calibrate_fusion_model,
    calibrate_model,
    compute_brier_score,
    compute_calibration_metrics,
    compute_ece,
    compute_reliability_diagram,
    plot_reliability_diagram,
)
from noa_swarm.ml.training.eval import (
    compute_accuracy,
    compute_confusion_matrix,
    compute_macro_f1,
    compute_top_k_accuracy,
)

__all__ = [
    # Evaluation metrics
    "compute_accuracy",
    "compute_confusion_matrix",
    "compute_macro_f1",
    "compute_top_k_accuracy",
    # Calibration
    "TemperatureScaler",
    "CalibrationResult",
    "ReliabilityDiagramData",
    "compute_ece",
    "compute_brier_score",
    "compute_reliability_diagram",
    "compute_calibration_metrics",
    "calibrate_model",
    "calibrate_fusion_model",
    "plot_reliability_diagram",
]
