#!/usr/bin/env python3
"""Train baseline CharCNN model with fixed random seed.

This script trains the CharCNN model on synthetic tag data for
reproducible baseline results.

Usage:
    python scripts/train_baseline.py --seed 42 --epochs 10
    python scripts/train_baseline.py --seed 42 --epochs 10 --output models/baseline.pt
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from noa_swarm.ml.models.charcnn import CharCNN, CharCNNConfig
from noa_swarm.ml.datasets.synth_tags import SyntheticTagGenerator
from noa_swarm.common.logging import get_logger

logger = get_logger(__name__)


def set_seed(seed: int) -> None:
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def create_dataset(
    generator: SyntheticTagGenerator,
    num_samples: int,
    max_seq_length: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Create training dataset from synthetic tags."""
    tags = generator.generate(num_samples)

    # Encode tag names
    encoded_tags = []
    labels = []

    for tag in tags:
        encoded = CharCNN.encode_tag_name(tag.tag_name, max_seq_length)
        encoded_tags.append(encoded)
        # Use instrument type as label (simplified)
        label = hash(tag.tag_name.split("-")[0]) % 10
        labels.append(label)

    return torch.tensor(encoded_tags), torch.tensor(labels)


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
) -> float:
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    num_batches = 0

    for batch_x, batch_y in dataloader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)

        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    return total_loss / max(num_batches, 1)


def compute_metrics(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> dict[str, float]:
    """Compute accuracy and F1 score."""
    model.set_mode_for_testing()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch_x, batch_y in dataloader:
            batch_x = batch_x.to(device)
            outputs = model(batch_x)
            preds = outputs.argmax(dim=1).cpu()
            all_preds.extend(preds.tolist())
            all_labels.extend(batch_y.tolist())

    # Calculate accuracy
    correct = sum(p == l for p, l in zip(all_preds, all_labels))
    accuracy = correct / len(all_labels) if all_labels else 0.0

    # Calculate macro F1 (simplified)
    from collections import Counter

    label_counts = Counter(all_labels)
    pred_counts = Counter(all_preds)

    # Per-class F1
    f1_scores = []
    for label in set(all_labels) | set(all_preds):
        tp = sum(1 for p, l in zip(all_preds, all_labels) if p == l == label)
        fp = pred_counts.get(label, 0) - tp
        fn = label_counts.get(label, 0) - tp

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        f1_scores.append(f1)

    macro_f1 = sum(f1_scores) / len(f1_scores) if f1_scores else 0.0

    return {"accuracy": accuracy, "macro_f1": macro_f1}


def main() -> None:
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train baseline CharCNN model")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--train-samples", type=int, default=5000, help="Training samples")
    parser.add_argument("--val-samples", type=int, default=1000, help="Validation samples")
    parser.add_argument("--output", type=str, default="models/baseline.pt", help="Output path")
    args = parser.parse_args()

    # Set seed for reproducibility
    logger.info("Setting random seed", seed=args.seed)
    set_seed(args.seed)

    # Device selection
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device", device=str(device))

    # Model configuration
    config = CharCNNConfig(
        vocab_size=70,
        embedding_dim=64,
        num_filters=256,
        kernel_sizes=[3, 5, 7],
        num_classes=10,
        max_seq_length=100,
        dropout=0.5,
    )

    # Create model
    model = CharCNN(config)
    model.to(device)
    logger.info("Created CharCNN model", params=sum(p.numel() for p in model.parameters()))

    # Create synthetic data generator
    generator = SyntheticTagGenerator(seed=args.seed)

    # Create datasets
    logger.info("Generating training data", samples=args.train_samples)
    train_x, train_y = create_dataset(generator, args.train_samples, config.max_seq_length)
    train_dataset = TensorDataset(train_x, train_y)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    logger.info("Generating validation data", samples=args.val_samples)
    val_x, val_y = create_dataset(generator, args.val_samples, config.max_seq_length)
    val_dataset = TensorDataset(val_x, val_y)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)

    # Training setup
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Training loop
    best_f1 = 0.0
    for epoch in range(args.epochs):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        metrics = compute_metrics(model, val_loader, device)

        logger.info(
            "Epoch completed",
            epoch=epoch + 1,
            train_loss=f"{train_loss:.4f}",
            val_accuracy=f"{metrics['accuracy']:.4f}",
            val_f1=f"{metrics['macro_f1']:.4f}",
        )

        if metrics["macro_f1"] > best_f1:
            best_f1 = metrics["macro_f1"]

    # Save model
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "config": config.__dict__,
            "state_dict": model.state_dict(),
            "seed": args.seed,
            "best_f1": best_f1,
        },
        output_path,
    )
    logger.info("Model saved", path=str(output_path), best_f1=f"{best_f1:.4f}")

    # Print final results
    print(f"\n{'=' * 50}")
    print("Training Complete")
    print(f"{'=' * 50}")
    print(f"Seed: {args.seed}")
    print(f"Best Macro F1: {best_f1:.4f}")
    print(f"Model saved to: {output_path}")


if __name__ == "__main__":
    main()
