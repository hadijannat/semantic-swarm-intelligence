"""Evaluation metrics for CharCNN model.

This module provides evaluation functions for assessing model performance:
- Accuracy (overall and per-class)
- Macro F1 score
- Confusion matrix
- Top-k accuracy for retrieval tasks
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch

if TYPE_CHECKING:
    from collections.abc import Sequence


def compute_accuracy(
    predictions: torch.Tensor | np.ndarray,
    targets: torch.Tensor | np.ndarray,
) -> float:
    """Compute overall classification accuracy.

    Args:
        predictions: Predicted class indices (N,)
        targets: Ground truth class indices (N,)

    Returns:
        Accuracy as a float in [0, 1]

    Example:
        >>> preds = torch.tensor([0, 1, 2, 0])
        >>> targets = torch.tensor([0, 1, 1, 0])
        >>> compute_accuracy(preds, targets)
        0.75
    """
    if isinstance(predictions, np.ndarray):
        predictions = torch.from_numpy(predictions)
    if isinstance(targets, np.ndarray):
        targets = torch.from_numpy(targets)

    predictions = predictions.long()
    targets = targets.long()

    if predictions.numel() == 0:
        return 0.0

    correct = (predictions == targets).sum().item()
    total = targets.numel()
    return correct / total


def compute_per_class_accuracy(
    predictions: torch.Tensor | np.ndarray,
    targets: torch.Tensor | np.ndarray,
    num_classes: int | None = None,
) -> dict[int, float]:
    """Compute per-class accuracy.

    Args:
        predictions: Predicted class indices (N,)
        targets: Ground truth class indices (N,)
        num_classes: Number of classes (inferred from targets if None)

    Returns:
        Dictionary mapping class index to accuracy

    Example:
        >>> preds = torch.tensor([0, 1, 2, 0, 1, 2])
        >>> targets = torch.tensor([0, 1, 1, 0, 0, 2])
        >>> compute_per_class_accuracy(preds, targets, num_classes=3)
        {0: 1.0, 1: 0.5, 2: 0.5}
    """
    if isinstance(predictions, np.ndarray):
        predictions = torch.from_numpy(predictions)
    if isinstance(targets, np.ndarray):
        targets = torch.from_numpy(targets)

    predictions = predictions.long()
    targets = targets.long()

    if num_classes is None:
        num_classes = int(targets.max().item()) + 1

    per_class_acc: dict[int, float] = {}
    for c in range(num_classes):
        mask = targets == c
        if mask.sum() > 0:
            correct = ((predictions == c) & mask).sum().item()
            total = mask.sum().item()
            per_class_acc[c] = correct / total
        else:
            per_class_acc[c] = 0.0

    return per_class_acc


def compute_macro_f1(
    predictions: torch.Tensor | np.ndarray,
    targets: torch.Tensor | np.ndarray,
    num_classes: int | None = None,
) -> float:
    """Compute macro-averaged F1 score.

    Macro F1 is the unweighted mean of F1 scores for each class.
    F1 = 2 * (precision * recall) / (precision + recall)

    Args:
        predictions: Predicted class indices (N,)
        targets: Ground truth class indices (N,)
        num_classes: Number of classes (inferred from targets if None)

    Returns:
        Macro F1 score as a float in [0, 1]

    Example:
        >>> preds = torch.tensor([0, 1, 2, 0, 1, 2])
        >>> targets = torch.tensor([0, 1, 1, 0, 0, 2])
        >>> f1 = compute_macro_f1(preds, targets, num_classes=3)
        >>> 0 <= f1 <= 1
        True
    """
    if isinstance(predictions, np.ndarray):
        predictions = torch.from_numpy(predictions)
    if isinstance(targets, np.ndarray):
        targets = torch.from_numpy(targets)

    predictions = predictions.long()
    targets = targets.long()

    if num_classes is None:
        num_classes = max(int(targets.max().item()) + 1, int(predictions.max().item()) + 1)

    f1_scores: list[float] = []

    for c in range(num_classes):
        # True positives: predicted c and target is c
        tp = ((predictions == c) & (targets == c)).sum().item()
        # False positives: predicted c but target is not c
        fp = ((predictions == c) & (targets != c)).sum().item()
        # False negatives: not predicted c but target is c
        fn = ((predictions != c) & (targets == c)).sum().item()

        # Precision and recall
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

        # F1 score
        if precision + recall > 0:
            f1 = 2 * (precision * recall) / (precision + recall)
        else:
            f1 = 0.0

        f1_scores.append(f1)

    return float(np.mean(f1_scores)) if f1_scores else 0.0


def compute_confusion_matrix(
    predictions: torch.Tensor | np.ndarray,
    targets: torch.Tensor | np.ndarray,
    num_classes: int | None = None,
) -> np.ndarray:
    """Compute confusion matrix.

    Args:
        predictions: Predicted class indices (N,)
        targets: Ground truth class indices (N,)
        num_classes: Number of classes (inferred from data if None)

    Returns:
        Confusion matrix of shape (num_classes, num_classes)
        where cm[i, j] = count of samples with true label i, predicted as j

    Example:
        >>> preds = torch.tensor([0, 1, 2, 0, 1, 2])
        >>> targets = torch.tensor([0, 1, 1, 0, 0, 2])
        >>> cm = compute_confusion_matrix(preds, targets, num_classes=3)
        >>> cm.shape
        (3, 3)
    """
    if isinstance(predictions, np.ndarray):
        predictions = torch.from_numpy(predictions)
    if isinstance(targets, np.ndarray):
        targets = torch.from_numpy(targets)

    predictions = predictions.long()
    targets = targets.long()

    if num_classes is None:
        num_classes = max(int(targets.max().item()) + 1, int(predictions.max().item()) + 1)

    cm = np.zeros((num_classes, num_classes), dtype=np.int64)

    for pred, target in zip(predictions.tolist(), targets.tolist(), strict=False):
        cm[target, pred] += 1

    return cm


def compute_top_k_accuracy(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    k: int = 5,
    metric: str = "cosine",
) -> float:
    """Compute top-k accuracy for embedding retrieval.

    For each sample, find the k nearest neighbors by embedding similarity.
    Accuracy is the fraction of samples where the correct label is among
    the k nearest neighbors' labels.

    Args:
        embeddings: Embedding vectors (N, D)
        labels: Ground truth labels (N,)
        k: Number of nearest neighbors to consider
        metric: Distance metric ("cosine" or "euclidean")

    Returns:
        Top-k accuracy as a float in [0, 1]

    Example:
        >>> embeddings = torch.randn(10, 128)
        >>> labels = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3, 4, 4])
        >>> acc = compute_top_k_accuracy(embeddings, labels, k=3)
        >>> 0 <= acc <= 1
        True
    """
    if embeddings.ndim != 2:
        raise ValueError(f"embeddings must be 2D, got shape {embeddings.shape}")
    if labels.ndim != 1:
        raise ValueError(f"labels must be 1D, got shape {labels.shape}")
    if embeddings.size(0) != labels.size(0):
        raise ValueError(
            f"embeddings and labels must have same length, "
            f"got {embeddings.size(0)} and {labels.size(0)}"
        )

    n = embeddings.size(0)
    if n <= 1:
        return 0.0

    # Clamp k to valid range (exclude self)
    k = min(k, n - 1)

    # Compute pairwise distances/similarities
    if metric == "cosine":
        # Normalize embeddings
        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        # Cosine similarity (higher is closer)
        similarity = torch.mm(embeddings, embeddings.t())
        # Convert to distance (lower is closer)
        distances = 1 - similarity
    elif metric == "euclidean":
        # Euclidean distance
        diff = embeddings.unsqueeze(0) - embeddings.unsqueeze(1)
        distances = torch.sqrt((diff**2).sum(dim=-1))
    else:
        raise ValueError(f"Unknown metric: {metric}")

    # Set self-distance to infinity to exclude self from neighbors
    distances.fill_diagonal_(float("inf"))

    # Find k nearest neighbors
    _, top_k_indices = torch.topk(distances, k, dim=1, largest=False)

    # Get labels of nearest neighbors
    neighbor_labels = labels[top_k_indices]  # (N, k)

    # Check if true label is among neighbor labels
    labels_expanded = labels.unsqueeze(1).expand_as(neighbor_labels)
    hits = (neighbor_labels == labels_expanded).any(dim=1)

    return hits.float().mean().item()


def compute_retrieval_metrics(
    query_embeddings: torch.Tensor,
    query_labels: torch.Tensor,
    gallery_embeddings: torch.Tensor,
    gallery_labels: torch.Tensor,
    k_values: Sequence[int] = (1, 5, 10),
    metric: str = "cosine",
) -> dict[str, float]:
    """Compute retrieval metrics with separate query and gallery sets.

    Args:
        query_embeddings: Query embedding vectors (Q, D)
        query_labels: Query labels (Q,)
        gallery_embeddings: Gallery embedding vectors (G, D)
        gallery_labels: Gallery labels (G,)
        k_values: K values for top-k accuracy
        metric: Distance metric ("cosine" or "euclidean")

    Returns:
        Dictionary with metrics: "top_1_acc", "top_5_acc", etc., and "mAP"
    """
    if metric == "cosine":
        # Normalize
        query_embeddings = torch.nn.functional.normalize(query_embeddings, p=2, dim=1)
        gallery_embeddings = torch.nn.functional.normalize(gallery_embeddings, p=2, dim=1)
        # Similarity
        similarity = torch.mm(query_embeddings, gallery_embeddings.t())
        distances = 1 - similarity
    elif metric == "euclidean":
        diff = query_embeddings.unsqueeze(1) - gallery_embeddings.unsqueeze(0)
        distances = torch.sqrt((diff**2).sum(dim=-1))
    else:
        raise ValueError(f"Unknown metric: {metric}")

    # Sort gallery by distance for each query
    _, sorted_indices = torch.sort(distances, dim=1)

    metrics: dict[str, float] = {}

    # Compute top-k accuracy for each k
    for k in k_values:
        top_k_indices = sorted_indices[:, :k]
        top_k_labels = gallery_labels[top_k_indices]
        query_labels_expanded = query_labels.unsqueeze(1).expand_as(top_k_labels)
        hits = (top_k_labels == query_labels_expanded).any(dim=1)
        metrics[f"top_{k}_acc"] = hits.float().mean().item()

    # Mean Average Precision (mAP)
    n_queries = query_embeddings.size(0)
    aps: list[float] = []

    for i in range(n_queries):
        query_label = query_labels[i]
        sorted_labels = gallery_labels[sorted_indices[i]]
        relevant = (sorted_labels == query_label).float()
        n_relevant = relevant.sum().item()

        if n_relevant > 0:
            # Cumulative sum of relevant items
            cumsum = torch.cumsum(relevant, dim=0)
            # Precision at each position
            precision_at_k = cumsum / torch.arange(1, len(relevant) + 1, dtype=torch.float)
            # Average precision
            ap = (precision_at_k * relevant).sum().item() / n_relevant
            aps.append(ap)
        else:
            aps.append(0.0)

    metrics["mAP"] = float(np.mean(aps))

    return metrics


def evaluate_model(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,  # type: ignore[type-arg]
    device: torch.device | str = "cpu",
) -> dict[str, float]:
    """Evaluate model on a dataloader.

    Args:
        model: The CharCNN model to evaluate
        dataloader: DataLoader yielding (inputs, property_labels, signal_labels) tuples
        device: Device to run evaluation on

    Returns:
        Dictionary with evaluation metrics
    """
    model.eval()
    device = torch.device(device) if isinstance(device, str) else device

    all_property_preds: list[torch.Tensor] = []
    all_property_targets: list[torch.Tensor] = []
    all_signal_preds: list[torch.Tensor] = []
    all_signal_targets: list[torch.Tensor] = []
    all_embeddings: list[torch.Tensor] = []

    with torch.no_grad():
        for batch in dataloader:
            inputs, property_targets, signal_targets = batch
            inputs = inputs.to(device)

            outputs = model(inputs, return_embeddings=True)

            property_preds = outputs["property_class"].argmax(dim=-1)
            signal_preds = outputs["signal_role"].argmax(dim=-1)

            all_property_preds.append(property_preds.cpu())
            all_property_targets.append(property_targets)
            all_signal_preds.append(signal_preds.cpu())
            all_signal_targets.append(signal_targets)
            all_embeddings.append(outputs["irdi_embedding"].cpu())

    # Concatenate all predictions
    property_preds = torch.cat(all_property_preds)
    property_targets = torch.cat(all_property_targets)
    signal_preds = torch.cat(all_signal_preds)
    signal_targets = torch.cat(all_signal_targets)
    embeddings = torch.cat(all_embeddings)

    # Compute metrics
    metrics = {
        "property_accuracy": compute_accuracy(property_preds, property_targets),
        "property_macro_f1": compute_macro_f1(property_preds, property_targets),
        "signal_accuracy": compute_accuracy(signal_preds, signal_targets),
        "signal_macro_f1": compute_macro_f1(signal_preds, signal_targets),
        "embedding_top_5_acc": compute_top_k_accuracy(
            embeddings, property_targets, k=5, metric="cosine"
        ),
    }

    return metrics
