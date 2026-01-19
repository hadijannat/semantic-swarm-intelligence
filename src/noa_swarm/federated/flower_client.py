"""FedProx Flower client for federated learning with proximal term regularization.

This module implements a Flower client that uses the FedProx algorithm for
federated learning. FedProx adds a proximal term to the local loss function
that penalizes deviation from the global model, which helps with heterogeneous
data across clients.

The proximal term is:
    L_local = L_task + (mu/2) * ||w - w_global||^2

Where:
- L_task is the standard task loss (e.g., cross-entropy)
- mu is the proximal term weight (typically 0.01-1.0)
- w is the local model weights
- w_global is the global model weights from the server

References:
    Li, T., Sahu, A. K., Zaheer, M., Sanjabi, M., Talwalkar, A., & Smith, V. (2020).
    Federated optimization in heterogeneous networks. Proceedings of Machine Learning
    and Systems, 2, 429-450.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import flwr as fl
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from noa_swarm.common.logging import get_logger

if TYPE_CHECKING:
    from collections.abc import Sequence

logger = get_logger(__name__)


@dataclass
class FedProxConfig:
    """Configuration for FedProx federated learning client.

    Attributes:
        client_id: Unique identifier for this client.
        mu: Proximal term weight. Controls how much the local model is
            regularized towards the global model. Higher values mean
            stronger regularization. Default is 0.01.
        local_epochs: Number of local training epochs per federated round.
            Default is 3.
        batch_size: Batch size for local training. Default is 32.
        learning_rate: Learning rate for local optimizer. Default is 0.001.
    """

    client_id: str
    mu: float = 0.01
    local_epochs: int = 3
    batch_size: int = 32
    learning_rate: float = 0.001


class FedProxClient(fl.client.NumPyClient):
    """Flower client implementing FedProx algorithm.

    This client extends Flower's NumPyClient to implement federated learning
    with a proximal term that keeps local models close to the global model.
    This is particularly useful when clients have heterogeneous (non-IID) data.

    The client supports:
    - Standard supervised training with labeled data
    - Semi-supervised training with pseudo-labeled data from consensus
    - Configurable proximal term weight (mu)

    Attributes:
        config: FedProxConfig with client settings.
        model: PyTorch model to train.
        train_data: Tuple of (features, labels) tensors for training.
        val_data: Tuple of (features, labels) tensors for validation.
        pseudo_data: Optional tuple of (features, pseudo_labels) for
            semi-supervised learning.
        pseudo_weight: Weight applied to pseudo-label loss (default 0.5).
        device: Device to run training on (CPU or CUDA).

    Example:
        >>> config = FedProxConfig(client_id="client-1", mu=0.1)
        >>> model = MyModel()
        >>> train_data = (train_x, train_y)
        >>> val_data = (val_x, val_y)
        >>> client = FedProxClient(config, model, train_data, val_data)
        >>> # In Flower federation:
        >>> fl.client.start_numpy_client(server_address="...", client=client)
    """

    def __init__(
        self,
        config: FedProxConfig,
        model: nn.Module,
        train_data: tuple[torch.Tensor, torch.Tensor],
        val_data: tuple[torch.Tensor, torch.Tensor],
        pseudo_data: tuple[torch.Tensor, torch.Tensor] | None = None,
        pseudo_weight: float = 0.5,
    ) -> None:
        """Initialize the FedProx client.

        Args:
            config: Configuration for the client.
            model: PyTorch model to train.
            train_data: Tuple of (features, labels) tensors for training.
            val_data: Tuple of (features, labels) tensors for validation.
            pseudo_data: Optional tuple of (features, pseudo_labels) for
                semi-supervised learning with high-confidence consensus labels.
            pseudo_weight: Weight applied to pseudo-label loss. Should be
                less than 1.0 since pseudo-labels are less reliable than
                true labels. Default is 0.5.
        """
        super().__init__()
        self.config = config
        self.model = model
        self.train_data = train_data
        self.val_data = val_data
        self.pseudo_data = pseudo_data
        self.pseudo_weight = pseudo_weight

        # Select device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        logger.info(
            "Initialized FedProxClient",
            client_id=config.client_id,
            mu=config.mu,
            train_samples=len(train_data[0]),
            val_samples=len(val_data[0]),
            pseudo_samples=len(pseudo_data[0]) if pseudo_data else 0,
            device=str(self.device),
        )

    def get_parameters(self, config: dict[str, Any]) -> list[np.ndarray]:
        """Extract model parameters as a list of NumPy arrays.

        This method is called by the Flower server to retrieve the current
        model weights for aggregation.

        Args:
            config: Configuration dictionary from the server (unused).

        Returns:
            List of NumPy arrays containing model parameters.
        """
        return [
            param.detach().cpu().numpy()
            for param in self.model.parameters()
        ]

    def set_parameters(self, parameters: Sequence[np.ndarray]) -> None:
        """Update model parameters from a list of NumPy arrays.

        This method is called to set the model weights, typically with
        the global model weights received from the Flower server.

        Args:
            parameters: List of NumPy arrays containing model parameters.
        """
        params_dict = zip(self.model.parameters(), parameters, strict=True)
        for param, new_value in params_dict:
            param.data = torch.tensor(
                new_value,
                dtype=param.dtype,
                device=param.device,
            )

    def _proximal_loss(self, global_params: Sequence[np.ndarray]) -> torch.Tensor:
        """Compute the proximal term ||w - w_global||^2.

        This penalizes the local model for deviating too far from the
        global model, which helps maintain convergence with heterogeneous
        data across clients.

        Args:
            global_params: Global model parameters as NumPy arrays.

        Returns:
            Scalar tensor containing the proximal term value.
        """
        prox_term = torch.tensor(0.0, device=self.device)

        for local_param, global_param in zip(
            self.model.parameters(), global_params, strict=True
        ):
            global_tensor = torch.tensor(
                global_param,
                dtype=local_param.dtype,
                device=self.device,
            )
            prox_term = prox_term + torch.sum((local_param - global_tensor) ** 2)

        return prox_term

    def fit(
        self,
        parameters: Sequence[np.ndarray],
        config: dict[str, Any],
    ) -> tuple[list[np.ndarray], int, dict[str, float]]:
        """Train the model on local data with FedProx regularization.

        This method performs local training while adding a proximal term
        to the loss function that keeps the local model close to the
        global model.

        Args:
            parameters: Global model parameters from the server.
            config: Configuration dictionary from the server (unused).

        Returns:
            Tuple of:
            - Updated model parameters as list of NumPy arrays
            - Number of training examples used
            - Dictionary with training metrics (e.g., {"loss": 0.5})
        """
        # Store global params for proximal term
        global_params = [np.copy(p) for p in parameters]

        # Set model to global parameters
        self.set_parameters(parameters)

        # Handle empty training data
        num_train = len(self.train_data[0])
        num_pseudo = len(self.pseudo_data[0]) if self.pseudo_data else 0
        total_examples = num_train + num_pseudo

        if num_train == 0 and num_pseudo == 0:
            logger.warning("No training data available", client_id=self.config.client_id)
            return self.get_parameters(config={}), 0, {"loss": 0.0}

        # Create data loaders
        train_loader = self._create_dataloader(
            self.train_data[0], self.train_data[1], shuffle=True
        ) if num_train > 0 else None

        pseudo_loader = self._create_dataloader(
            self.pseudo_data[0], self.pseudo_data[1], shuffle=True
        ) if self.pseudo_data and num_pseudo > 0 else None

        # Set up optimizer and loss
        optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=self.config.learning_rate,
        )
        criterion = nn.CrossEntropyLoss()

        # Training loop
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        for epoch in range(self.config.local_epochs):
            epoch_loss = 0.0
            epoch_batches = 0

            # Train on labeled data
            if train_loader is not None:
                for batch_x, batch_y in train_loader:
                    batch_x = batch_x.to(self.device)
                    batch_y = batch_y.to(self.device)

                    optimizer.zero_grad()

                    # Forward pass
                    outputs = self.model(batch_x)
                    task_loss = criterion(outputs, batch_y)

                    # Add proximal term: (mu/2) * ||w - w_global||^2
                    prox_loss = (self.config.mu / 2.0) * self._proximal_loss(global_params)
                    loss = task_loss + prox_loss

                    # Backward pass
                    loss.backward()
                    optimizer.step()

                    epoch_loss += task_loss.item()
                    epoch_batches += 1

            # Train on pseudo-labeled data
            if pseudo_loader is not None:
                for batch_x, batch_y in pseudo_loader:
                    batch_x = batch_x.to(self.device)
                    batch_y = batch_y.to(self.device)

                    optimizer.zero_grad()

                    # Forward pass
                    outputs = self.model(batch_x)
                    task_loss = criterion(outputs, batch_y) * self.pseudo_weight

                    # Add proximal term
                    prox_loss = (self.config.mu / 2.0) * self._proximal_loss(global_params)
                    loss = task_loss + prox_loss

                    # Backward pass
                    loss.backward()
                    optimizer.step()

                    epoch_loss += task_loss.item()
                    epoch_batches += 1

            if epoch_batches > 0:
                total_loss += epoch_loss
                num_batches += epoch_batches

        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0

        logger.info(
            "Local training completed",
            client_id=self.config.client_id,
            epochs=self.config.local_epochs,
            avg_loss=avg_loss,
            train_samples=num_train,
            pseudo_samples=num_pseudo,
        )

        return self.get_parameters(config={}), total_examples, {"loss": avg_loss}

    def run_validation(
        self,
        parameters: Sequence[np.ndarray],
        config: dict[str, Any],
    ) -> tuple[float, int, dict[str, float]]:
        """Run validation on local validation data.

        Args:
            parameters: Model parameters to use for validation.
            config: Configuration dictionary from the server (unused).

        Returns:
            Tuple of:
            - Validation loss
            - Number of validation examples
            - Dictionary with validation metrics (e.g., {"accuracy": 0.9})
        """
        # Set model parameters
        self.set_parameters(parameters)

        # Handle empty validation data
        num_val = len(self.val_data[0])
        if num_val == 0:
            logger.warning(
                "No validation data available",
                client_id=self.config.client_id,
            )
            return 0.0, 0, {"accuracy": 0.0}

        # Create data loader
        val_loader = self._create_dataloader(
            self.val_data[0], self.val_data[1], shuffle=False
        )

        # Validation
        self.model.eval()
        criterion = nn.CrossEntropyLoss()

        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)

                outputs = self.model(batch_x)
                loss = criterion(outputs, batch_y)

                total_loss += loss.item() * len(batch_x)
                predictions = torch.argmax(outputs, dim=1)
                correct += (predictions == batch_y).sum().item()
                total += len(batch_x)

        avg_loss = total_loss / total if total > 0 else 0.0
        accuracy = correct / total if total > 0 else 0.0

        logger.info(
            "Validation completed",
            client_id=self.config.client_id,
            loss=avg_loss,
            accuracy=accuracy,
            val_samples=num_val,
        )

        return avg_loss, num_val, {"accuracy": accuracy}

    # Flower requires a method named 'evaluate' - this is the standard interface
    def evaluate(
        self,
        parameters: Sequence[np.ndarray],
        config: dict[str, Any],
    ) -> tuple[float, int, dict[str, float]]:
        """Validate the model on local validation data (Flower interface).

        This method is part of the Flower NumPyClient interface and is called
        by the Flower server to assess model performance on local data.

        Args:
            parameters: Model parameters to validate.
            config: Configuration dictionary from the server (unused).

        Returns:
            Tuple of:
            - Validation loss
            - Number of validation examples
            - Dictionary with validation metrics (e.g., {"accuracy": 0.9})
        """
        return self.run_validation(parameters, config)

    def _create_dataloader(
        self,
        features: torch.Tensor,
        labels: torch.Tensor,
        shuffle: bool = True,
    ) -> DataLoader:
        """Create a DataLoader from tensors.

        Args:
            features: Feature tensor.
            labels: Label tensor.
            shuffle: Whether to shuffle the data.

        Returns:
            PyTorch DataLoader.
        """
        dataset = TensorDataset(features, labels)
        return DataLoader(
            dataset,
            batch_size=min(self.config.batch_size, len(features)),
            shuffle=shuffle,
            drop_last=False,
        )
