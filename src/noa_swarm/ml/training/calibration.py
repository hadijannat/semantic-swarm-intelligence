"""Temperature scaling and calibration utilities.

This module provides tools for calibrating neural network predictions
using temperature scaling, and computing calibration metrics like
Expected Calibration Error (ECE).

Well-calibrated models are essential for swarm consensus where
predictions are weighted by confidence scores. Target ECE <= 0.05.

Key components:
- TemperatureScaler: Learn optimal temperature on validation set
- compute_ece: Expected Calibration Error
- compute_brier_score: Brier score for probabilistic predictions
- compute_reliability_diagram: Data for reliability diagram plotting
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import LBFGS

if TYPE_CHECKING:
    from collections.abc import Callable

    from torch.utils.data import DataLoader

    from noa_swarm.ml.models.fusion import FusionModel


@dataclass
class CalibrationResult:
    """Result from temperature scaling calibration.

    Attributes:
        temperature: Optimal temperature value
        ece_before: ECE before calibration
        ece_after: ECE after calibration
        nll_before: NLL before calibration
        nll_after: NLL after calibration
        num_samples: Number of samples used for calibration
    """

    temperature: float
    ece_before: float
    ece_after: float
    nll_before: float
    nll_after: float
    num_samples: int


class TemperatureScaler(nn.Module):
    """Temperature scaling for model calibration.

    Temperature scaling divides logits by a learned temperature parameter
    to produce well-calibrated probability estimates. This is a simple
    post-hoc calibration method that works well in practice.

    The temperature is optimized to minimize negative log-likelihood
    on a held-out validation set.

    Attributes:
        temperature: The learned temperature parameter

    Example:
        >>> scaler = TemperatureScaler()
        >>> # Fit on validation data
        >>> scaler.fit(val_logits, val_labels)
        >>> # Apply to new logits
        >>> calibrated_logits = scaler(test_logits)
        >>> calibrated_probs = F.softmax(calibrated_logits, dim=-1)
    """

    def __init__(self, initial_temperature: float = 1.0) -> None:
        """Initialize the TemperatureScaler.

        Args:
            initial_temperature: Initial temperature value (default 1.0)
        """
        super().__init__()
        # Use log-space to ensure temperature is always positive
        self._log_temperature = nn.Parameter(
            torch.tensor(np.log(initial_temperature), dtype=torch.float32)
        )

    @property
    def temperature(self) -> torch.Tensor:
        """Get the temperature value."""
        return torch.exp(self._log_temperature)

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        """Apply temperature scaling to logits.

        Args:
            logits: Input logits of shape (batch, num_classes)

        Returns:
            Temperature-scaled logits
        """
        return logits / self.temperature

    def fit(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        max_iter: int = 50,
        lr: float = 0.01,
        verbose: bool = False,
    ) -> CalibrationResult:
        """Fit the temperature parameter using validation data.

        Optimizes temperature to minimize negative log-likelihood.

        Args:
            logits: Validation logits of shape (num_samples, num_classes)
            labels: True labels of shape (num_samples,)
            max_iter: Maximum optimization iterations
            lr: Learning rate for LBFGS optimizer
            verbose: Whether to print progress

        Returns:
            CalibrationResult with calibration metrics
        """
        # Compute metrics before calibration
        with torch.no_grad():
            probs_before = F.softmax(logits, dim=-1)
            nll_before = F.cross_entropy(logits, labels).item()
            ece_before = compute_ece(probs_before, labels)

        # Reset temperature to 1.0 before fitting
        with torch.no_grad():
            self._log_temperature.fill_(0.0)

        # Optimize temperature using LBFGS
        optimizer = LBFGS([self._log_temperature], lr=lr, max_iter=max_iter)

        def closure() -> torch.Tensor:
            optimizer.zero_grad()
            scaled_logits = self.forward(logits)
            loss = F.cross_entropy(scaled_logits, labels)
            loss.backward(retain_graph=True)  # type: ignore[no-untyped-call]
            return loss

        # Note: LBFGS.step() expects Callable[[], Tensor] but mypy infers stricter type
        optimizer.step(closure)  # type: ignore[no-untyped-call]

        # Compute metrics after calibration
        with torch.no_grad():
            scaled_logits = self.forward(logits)
            probs_after = F.softmax(scaled_logits, dim=-1)
            nll_after = F.cross_entropy(scaled_logits, labels).item()
            ece_after = compute_ece(probs_after, labels)

        if verbose:
            print(f"Temperature: {self.temperature.item():.4f}")
            print(f"ECE: {ece_before:.4f} -> {ece_after:.4f}")
            print(f"NLL: {nll_before:.4f} -> {nll_after:.4f}")

        return CalibrationResult(
            temperature=self.temperature.item(),
            ece_before=ece_before,
            ece_after=ece_after,
            nll_before=nll_before,
            nll_after=nll_after,
            num_samples=logits.size(0),
        )

    def set_temperature(self, temperature: float) -> None:
        """Manually set the temperature value.

        Args:
            temperature: New temperature value
        """
        with torch.no_grad():
            self._log_temperature.fill_(np.log(temperature))


def compute_ece(
    probs: torch.Tensor | np.ndarray[Any, np.dtype[Any]],
    labels: torch.Tensor | np.ndarray[Any, np.dtype[Any]],
    n_bins: int = 15,
) -> float:
    """Compute Expected Calibration Error (ECE).

    ECE measures how well the predicted probabilities match the actual
    accuracy. It bins predictions by confidence and computes the weighted
    average of |accuracy - confidence| in each bin.

    Args:
        probs: Predicted probabilities of shape (num_samples, num_classes)
        labels: True labels of shape (num_samples,)
        n_bins: Number of bins for calibration (default 15)

    Returns:
        ECE value (0.0 = perfectly calibrated, 1.0 = worst calibration)

    Example:
        >>> probs = torch.softmax(logits, dim=-1)
        >>> ece = compute_ece(probs, labels)
        >>> print(f"ECE: {ece:.4f}")
    """
    # Convert to numpy for easier binning
    if isinstance(probs, torch.Tensor):
        probs = probs.detach().cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.detach().cpu().numpy()

    # Get predicted classes and confidences
    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    accuracies = predictions == labels

    # Create bins
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    # Compute ECE
    ece = 0.0
    total_samples = len(confidences)

    if total_samples == 0:
        return 0.0

    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers, strict=False):
        # Find samples in this bin
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = np.mean(in_bin)

        if prop_in_bin > 0:
            # Compute accuracy and average confidence in bin
            accuracy_in_bin = np.mean(accuracies[in_bin])
            avg_confidence_in_bin = np.mean(confidences[in_bin])

            # Add weighted absolute difference
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

    return float(ece)


def compute_brier_score(
    probs: torch.Tensor | np.ndarray[Any, np.dtype[Any]],
    labels: torch.Tensor | np.ndarray[Any, np.dtype[Any]],
) -> float:
    """Compute Brier score for probabilistic predictions.

    Brier score measures the mean squared difference between predicted
    probabilities and actual outcomes. Lower is better.

    Args:
        probs: Predicted probabilities of shape (num_samples, num_classes)
        labels: True labels of shape (num_samples,)

    Returns:
        Brier score (0.0 = perfect, 1.0 = worst)

    Example:
        >>> probs = torch.softmax(logits, dim=-1)
        >>> brier = compute_brier_score(probs, labels)
    """
    # Convert to numpy
    if isinstance(probs, torch.Tensor):
        probs = probs.detach().cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.detach().cpu().numpy()

    num_samples = probs.shape[0]
    num_classes = probs.shape[1]

    if num_samples == 0:
        return 0.0

    # Create one-hot encoding of labels
    one_hot = np.zeros((num_samples, num_classes))
    one_hot[np.arange(num_samples), labels] = 1

    # Compute Brier score
    brier = np.mean(np.sum((probs - one_hot) ** 2, axis=1))

    return float(brier)


@dataclass
class ReliabilityDiagramData:
    """Data for plotting a reliability diagram.

    Attributes:
        bin_confidences: Average confidence in each bin
        bin_accuracies: Accuracy in each bin
        bin_counts: Number of samples in each bin
        bin_edges: Bin edge values
        ece: Expected Calibration Error
    """

    bin_confidences: list[float]
    bin_accuracies: list[float]
    bin_counts: list[int]
    bin_edges: list[float]
    ece: float


def compute_reliability_diagram(
    probs: torch.Tensor | np.ndarray[Any, np.dtype[Any]],
    labels: torch.Tensor | np.ndarray[Any, np.dtype[Any]],
    n_bins: int = 15,
) -> ReliabilityDiagramData:
    """Compute data for a reliability diagram.

    A reliability diagram plots accuracy vs. confidence. For a perfectly
    calibrated model, the accuracy should equal confidence (diagonal line).

    Args:
        probs: Predicted probabilities of shape (num_samples, num_classes)
        labels: True labels of shape (num_samples,)
        n_bins: Number of bins (default 15)

    Returns:
        ReliabilityDiagramData with binned statistics

    Example:
        >>> data = compute_reliability_diagram(probs, labels)
        >>> # Plot with matplotlib:
        >>> plt.bar(data.bin_confidences, data.bin_accuracies)
        >>> plt.plot([0, 1], [0, 1], 'k--')  # Perfect calibration line
    """
    # Convert to numpy
    if isinstance(probs, torch.Tensor):
        probs = probs.detach().cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.detach().cpu().numpy()

    # Get predicted classes and confidences
    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    accuracies = predictions == labels

    # Create bins
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    bin_confidences: list[float] = []
    bin_accuracies: list[float] = []
    bin_counts: list[int] = []

    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers, strict=False):
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        count = np.sum(in_bin)
        bin_counts.append(int(count))

        if count > 0:
            bin_confidences.append(float(np.mean(confidences[in_bin])))
            bin_accuracies.append(float(np.mean(accuracies[in_bin])))
        else:
            # Use bin center for empty bins
            bin_center = (bin_lower + bin_upper) / 2
            bin_confidences.append(float(bin_center))
            bin_accuracies.append(0.0)

    # Compute ECE
    ece = compute_ece(probs, labels, n_bins=n_bins)

    return ReliabilityDiagramData(
        bin_confidences=bin_confidences,
        bin_accuracies=bin_accuracies,
        bin_counts=bin_counts,
        bin_edges=bin_boundaries.tolist(),
        ece=ece,
    )


def calibrate_model(
    model: nn.Module,
    val_dataloader: DataLoader[Any],
    device: torch.device | str = "cpu",
    forward_fn: Callable[[nn.Module, torch.Tensor], torch.Tensor] | None = None,
    max_iter: int = 50,
    verbose: bool = False,
) -> tuple[TemperatureScaler, CalibrationResult]:
    """Calibrate a model using temperature scaling on validation data.

    This function collects all validation predictions, then fits a
    temperature scaler to minimize NLL.

    Args:
        model: The model to calibrate (should output logits)
        val_dataloader: Validation data loader
        device: Device to run inference on
        forward_fn: Optional custom forward function. If None, assumes
                    model(x) returns logits directly. If provided, should
                    take (model, batch) and return logits tensor.
        max_iter: Maximum iterations for temperature optimization
        verbose: Whether to print progress

    Returns:
        Tuple of (TemperatureScaler, CalibrationResult)

    Example:
        >>> # For a simple model
        >>> scaler, result = calibrate_model(model, val_loader)
        >>> print(f"Optimal temperature: {result.temperature:.4f}")
        >>> print(f"ECE improved: {result.ece_before:.4f} -> {result.ece_after:.4f}")
    """
    # Set model to inference mode
    training_mode = model.training
    model.train(False)

    all_logits: list[torch.Tensor] = []
    all_labels: list[torch.Tensor] = []

    with torch.no_grad():
        for batch in val_dataloader:
            # Handle different batch formats
            if isinstance(batch, tuple | list):
                inputs = batch[0].to(device)
                labels: torch.Tensor | None = batch[1].to(device)
            else:
                inputs = batch.to(device)
                labels = None

            # Get logits
            if forward_fn is not None:
                logits = forward_fn(model, inputs)
            else:
                output = model(inputs)
                if isinstance(output, dict):
                    # Assume 'property_class' is the main output
                    logits_value = output.get("property_class") or output.get("logits")
                    if logits_value is None:
                        raise ValueError("Model output dict is missing logits")
                    logits = cast(torch.Tensor, logits_value)
                else:
                    logits = cast(torch.Tensor, output)

            all_logits.append(logits.cpu())
            if labels is not None:
                all_labels.append(labels.cpu())

    # Restore training mode
    model.train(training_mode)

    # Concatenate all predictions
    all_logits_tensor = torch.cat(all_logits, dim=0)
    if all_labels:
        all_labels_tensor = torch.cat(all_labels, dim=0)
    else:
        # No labels collected - this shouldn't happen in normal use
        raise ValueError("No labels collected from dataloader. Ensure batches contain labels.")

    # Fit temperature scaler
    scaler = TemperatureScaler()
    result = scaler.fit(all_logits_tensor, all_labels_tensor, max_iter=max_iter, verbose=verbose)

    return scaler, result


def calibrate_fusion_model(
    fusion_model: FusionModel,
    val_dataloader: DataLoader[Any],
    device: torch.device | str = "cpu",
    calibrate_property: bool = True,
    calibrate_signal: bool = True,
    max_iter: int = 50,
    verbose: bool = False,
) -> dict[str, CalibrationResult]:
    """Calibrate a FusionModel's temperature parameters.

    This function directly updates the FusionModel's temperature
    parameters based on validation data.

    Args:
        fusion_model: The FusionModel to calibrate
        val_dataloader: Validation data loader yielding
                        (charcnn_output, gnn_output, property_label, signal_label)
        device: Device to run inference on
        calibrate_property: Whether to calibrate property temperature
        calibrate_signal: Whether to calibrate signal temperature
        max_iter: Maximum iterations for optimization
        verbose: Whether to print progress

    Returns:
        Dictionary with CalibrationResult for 'property' and/or 'signal'
    """
    # Set model to inference mode
    training_mode = fusion_model.training
    fusion_model.train(False)

    # Collect all predictions
    all_property_logits: list[torch.Tensor] = []
    all_signal_logits: list[torch.Tensor] = []
    all_property_labels: list[torch.Tensor] = []
    all_signal_labels: list[torch.Tensor] = []

    with torch.no_grad():
        for batch in val_dataloader:
            charcnn_out, gnn_out, prop_labels, sig_labels = batch

            # Move to device
            charcnn_out = {k: v.to(device) for k, v in charcnn_out.items()}
            if gnn_out is not None:
                gnn_out = {k: v.to(device) for k, v in gnn_out.items()}

            # Get uncalibrated logits
            outputs = fusion_model(charcnn_out, gnn_out, return_uncalibrated=True)

            all_property_logits.append(outputs["property_class_uncalibrated"].cpu())
            all_signal_logits.append(outputs["signal_role_uncalibrated"].cpu())
            all_property_labels.append(prop_labels)
            all_signal_labels.append(sig_labels)

    # Restore training mode
    fusion_model.train(training_mode)

    # Concatenate
    property_logits = torch.cat(all_property_logits, dim=0)
    signal_logits = torch.cat(all_signal_logits, dim=0)
    property_labels = torch.cat(all_property_labels, dim=0)
    signal_labels = torch.cat(all_signal_labels, dim=0)

    results: dict[str, CalibrationResult] = {}

    # Calibrate property temperature
    if calibrate_property:
        scaler = TemperatureScaler()
        result = scaler.fit(property_logits, property_labels, max_iter=max_iter, verbose=verbose)
        fusion_model.set_temperature(temperature_property=result.temperature)
        results["property"] = result
        if verbose:
            print(f"Property temperature set to: {result.temperature:.4f}")

    # Calibrate signal temperature
    if calibrate_signal:
        scaler = TemperatureScaler()
        result = scaler.fit(signal_logits, signal_labels, max_iter=max_iter, verbose=verbose)
        fusion_model.set_temperature(temperature_signal=result.temperature)
        results["signal"] = result
        if verbose:
            print(f"Signal temperature set to: {result.temperature:.4f}")

    return results


def compute_calibration_metrics(
    probs: torch.Tensor | np.ndarray[Any, np.dtype[Any]],
    labels: torch.Tensor | np.ndarray[Any, np.dtype[Any]],
    n_bins: int = 15,
) -> dict[str, float]:
    """Compute multiple calibration metrics at once.

    Args:
        probs: Predicted probabilities of shape (num_samples, num_classes)
        labels: True labels of shape (num_samples,)
        n_bins: Number of bins for ECE computation

    Returns:
        Dictionary with:
            - 'ece': Expected Calibration Error
            - 'brier': Brier score
            - 'accuracy': Classification accuracy
            - 'avg_confidence': Average confidence
            - 'calibration_gap': |avg_confidence - accuracy|
    """
    # Convert to numpy
    if isinstance(probs, torch.Tensor):
        probs = probs.detach().cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.detach().cpu().numpy()

    # Get predictions and confidences
    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)

    accuracy = float(np.mean(predictions == labels))
    avg_confidence = float(np.mean(confidences))

    return {
        "ece": compute_ece(probs, labels, n_bins=n_bins),
        "brier": compute_brier_score(probs, labels),
        "accuracy": accuracy,
        "avg_confidence": avg_confidence,
        "calibration_gap": abs(avg_confidence - accuracy),
    }


def plot_reliability_diagram(
    probs: torch.Tensor | np.ndarray[Any, np.dtype[Any]],
    labels: torch.Tensor | np.ndarray[Any, np.dtype[Any]],
    n_bins: int = 15,
    title: str = "Reliability Diagram",
    save_path: str | None = None,
) -> None:
    """Plot a reliability diagram using matplotlib.

    Args:
        probs: Predicted probabilities of shape (num_samples, num_classes)
        labels: True labels of shape (num_samples,)
        n_bins: Number of bins
        title: Plot title
        save_path: Optional path to save the figure

    Note:
        Requires matplotlib to be installed.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError as err:
        raise ImportError(
            "matplotlib is required for plotting. Install with: pip install matplotlib"
        ) from err

    data = compute_reliability_diagram(probs, labels, n_bins=n_bins)

    fig, ax = plt.subplots(figsize=(8, 6))

    # Bar chart of accuracies
    bin_width = 1.0 / n_bins
    ax.bar(
        data.bin_confidences,
        data.bin_accuracies,
        width=bin_width * 0.8,
        alpha=0.7,
        color="steelblue",
        edgecolor="black",
        label="Accuracy",
    )

    # Perfect calibration line
    ax.plot([0, 1], [0, 1], "k--", linewidth=2, label="Perfect calibration")

    # Gap visualization (optional)
    for conf, acc in zip(data.bin_confidences, data.bin_accuracies, strict=False):
        ax.plot([conf, conf], [acc, conf], color="red", alpha=0.5, linewidth=1)

    ax.set_xlabel("Confidence", fontsize=12)
    ax.set_ylabel("Accuracy", fontsize=12)
    ax.set_title(f"{title}\nECE: {data.ece:.4f}", fontsize=14)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    plt.close(fig)
