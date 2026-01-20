"""FedProx Flower server for federated learning coordination.

This module implements a Flower server that uses the FedProx algorithm for
federated learning. The server coordinates training rounds, aggregates model
updates using weighted averaging, and tracks model versions.

FedProx server-side aggregation uses FedAvg-style weighted averaging where
each client's contribution is weighted by its number of training examples.
The proximal term regularization is handled client-side.

Key components:
- **FedProxServerConfig**: Configuration for the FL server
- **FedProxStrategy**: Aggregation strategy with model versioning
- **create_fedprox_server**: Factory function to create server components

Example usage:
    >>> from noa_swarm.federated.flower_server import (
    ...     FedProxServerConfig,
    ...     create_fedprox_server,
    ... )
    >>> config = FedProxServerConfig(num_rounds=10, min_fit_clients=3)
    >>> server_config, strategy = create_fedprox_server(config)
    >>> # Use with fl.server.start_server()

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
from flwr.common import (
    FitRes,
    EvaluateRes,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.server import ServerConfig
from flwr.server.strategy import FedAvg

from noa_swarm.common.logging import get_logger

if TYPE_CHECKING:
    from collections.abc import Sequence

    from flwr.server.client_proxy import ClientProxy

logger = get_logger(__name__)


@dataclass
class FedProxServerConfig:
    """Configuration for FedProx federated learning server.

    Attributes:
        num_rounds: Number of federated learning rounds. Default is 10.
        min_fit_clients: Minimum number of clients required for training.
            Default is 2.
        min_evaluate_clients: Minimum number of clients required for evaluation.
            Default is 2.
        min_available_clients: Minimum number of available clients to start a round.
            Default is 2.
        fraction_fit: Fraction of clients to sample for training (0.0, 1.0].
            Default is 1.0 (all available).
        fraction_evaluate: Fraction of clients to sample for evaluation (0.0, 1.0].
            Default is 1.0 (all available).
        server_address: Address to bind the server to. Default is "0.0.0.0:8080".
    """

    num_rounds: int = 10
    min_fit_clients: int = 2
    min_evaluate_clients: int = 2
    min_available_clients: int = 2
    fraction_fit: float = 1.0
    fraction_evaluate: float = 1.0
    server_address: str = "0.0.0.0:8080"

    def __post_init__(self) -> None:
        """Validate configuration parameters."""
        if self.num_rounds < 1:
            raise ValueError(f"num_rounds must be >= 1, got {self.num_rounds}")
        if self.min_fit_clients < 1:
            raise ValueError(
                f"min_fit_clients must be >= 1, got {self.min_fit_clients}"
            )
        if self.min_evaluate_clients < 1:
            raise ValueError(
                f"min_evaluate_clients must be >= 1, got {self.min_evaluate_clients}"
            )
        if self.min_available_clients < 1:
            raise ValueError(
                f"min_available_clients must be >= 1, got {self.min_available_clients}"
            )
        if not (0.0 < self.fraction_fit <= 1.0):
            raise ValueError(
                f"fraction_fit must be in (0.0, 1.0], got {self.fraction_fit}"
            )
        if not (0.0 < self.fraction_evaluate <= 1.0):
            raise ValueError(
                f"fraction_evaluate must be in (0.0, 1.0], got {self.fraction_evaluate}"
            )
        if not self.server_address:
            raise ValueError("server_address cannot be empty")


class FedProxStrategy(FedAvg):
    """Flower strategy implementing FedProx aggregation.

    This strategy extends FedAvg to add model versioning and round tracking.
    The actual FedProx proximal term regularization is handled client-side;
    server-side aggregation uses weighted averaging based on client sample counts.

    Attributes:
        current_round: Current training round number.
        model_version: Version number incremented after each aggregation.
        initial_parameters: Optional initial model parameters.
    """

    def __init__(
        self,
        *,
        min_fit_clients: int = 2,
        min_evaluate_clients: int = 2,
        min_available_clients: int = 2,
        fraction_fit: float = 1.0,
        fraction_evaluate: float = 1.0,
        initial_parameters: Parameters | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the FedProx strategy.

        Args:
            min_fit_clients: Minimum number of clients for training.
            min_evaluate_clients: Minimum number of clients for evaluation.
            min_available_clients: Minimum available clients to start round.
            fraction_fit: Fraction of clients to sample for training.
            fraction_evaluate: Fraction of clients to sample for evaluation.
            initial_parameters: Optional initial model parameters.
            **kwargs: Additional arguments passed to FedAvg.
        """
        super().__init__(
            min_fit_clients=min_fit_clients,
            min_evaluate_clients=min_evaluate_clients,
            min_available_clients=min_available_clients,
            fraction_fit=fraction_fit,
            fraction_evaluate=fraction_evaluate,
            initial_parameters=initial_parameters,
            **kwargs,
        )

        self.current_round: int = 0
        self.model_version: int = 0

        logger.info(
            "Initialized FedProxStrategy",
            min_fit_clients=min_fit_clients,
            min_evaluate_clients=min_evaluate_clients,
            fraction_fit=fraction_fit,
            fraction_evaluate=fraction_evaluate,
        )

    def aggregate_fit(
        self,
        server_round: int,
        results: list[tuple[ClientProxy, FitRes]],
        failures: list[tuple[ClientProxy, FitRes] | BaseException],
    ) -> tuple[Parameters | None, dict[str, Scalar]]:
        """Aggregate training results using weighted averaging.

        This method aggregates model parameters from clients using weighted
        averaging, where weights are the number of training examples per client.

        Args:
            server_round: Current server round number.
            results: List of (client, fit_result) tuples from successful clients.
            failures: List of failed client results or exceptions.

        Returns:
            Tuple of:
            - Aggregated parameters (or None if no results)
            - Dictionary with aggregation metrics
        """
        if not results:
            logger.warning(
                "No results to aggregate",
                server_round=server_round,
                num_failures=len(failures),
            )
            return None, {}

        # Perform weighted aggregation (FedAvg style)
        aggregated_parameters, metrics = super().aggregate_fit(
            server_round, results, failures
        )

        if aggregated_parameters is not None:
            # Update round and version tracking
            self.current_round += 1
            self.model_version += 1

            # Compute aggregation statistics
            total_examples = sum(fit_res.num_examples for _, fit_res in results)
            num_clients = len(results)

            logger.info(
                "Aggregated fit results",
                server_round=server_round,
                current_round=self.current_round,
                model_version=self.model_version,
                num_clients=num_clients,
                total_examples=total_examples,
                num_failures=len(failures),
            )

            # Add metrics
            metrics["num_clients"] = num_clients
            metrics["total_examples"] = total_examples
            metrics["model_version"] = self.model_version

        return aggregated_parameters, metrics

    def aggregate_evaluate(
        self,
        server_round: int,
        results: list[tuple[ClientProxy, EvaluateRes]],
        failures: list[tuple[ClientProxy, EvaluateRes] | BaseException],
    ) -> tuple[float | None, dict[str, Scalar]]:
        """Aggregate evaluation results using weighted averaging.

        Args:
            server_round: Current server round number.
            results: List of (client, eval_result) tuples from successful clients.
            failures: List of failed client results or exceptions.

        Returns:
            Tuple of:
            - Weighted average loss (or None if no results)
            - Dictionary with aggregated metrics
        """
        if not results:
            logger.warning(
                "No evaluation results to aggregate",
                server_round=server_round,
                num_failures=len(failures),
            )
            return None, {}

        # Compute weighted average loss
        total_examples = sum(eval_res.num_examples for _, eval_res in results)
        weighted_loss = sum(
            eval_res.loss * eval_res.num_examples for _, eval_res in results
        )
        loss_aggregated = weighted_loss / total_examples if total_examples > 0 else 0.0

        # Aggregate accuracy if available
        accuracies = [
            eval_res.metrics.get("accuracy", 0.0) * eval_res.num_examples
            for _, eval_res in results
        ]
        weighted_accuracy = sum(accuracies) / total_examples if total_examples > 0 else 0.0

        logger.info(
            "Aggregated evaluation results",
            server_round=server_round,
            num_clients=len(results),
            total_examples=total_examples,
            loss=loss_aggregated,
            accuracy=weighted_accuracy,
            num_failures=len(failures),
        )

        metrics: dict[str, Scalar] = {
            "num_clients": len(results),
            "total_examples": total_examples,
            "accuracy": weighted_accuracy,
        }

        return loss_aggregated, metrics


def create_fedprox_server(
    config: FedProxServerConfig,
    initial_parameters: Sequence[np.ndarray] | None = None,
) -> tuple[ServerConfig, FedProxStrategy]:
    """Create a FedProx server configuration and strategy.

    This factory function creates the Flower server components needed to
    run federated learning with FedProx.

    Args:
        config: Server configuration.
        initial_parameters: Optional initial model parameters as numpy arrays.
            If provided, clients will receive these as the starting parameters.

    Returns:
        Tuple of (ServerConfig, FedProxStrategy) for use with fl.server.start_server().

    Example:
        >>> config = FedProxServerConfig(num_rounds=10, min_fit_clients=3)
        >>> server_config, strategy = create_fedprox_server(config)
        >>> fl.server.start_server(
        ...     server_address=config.server_address,
        ...     config=server_config,
        ...     strategy=strategy,
        ... )
    """
    # Convert numpy arrays to Flower Parameters if provided
    initial_params: Parameters | None = None
    if initial_parameters is not None:
        initial_params = ndarrays_to_parameters(list(initial_parameters))

    # Create strategy
    strategy = FedProxStrategy(
        min_fit_clients=config.min_fit_clients,
        min_evaluate_clients=config.min_evaluate_clients,
        min_available_clients=config.min_available_clients,
        fraction_fit=config.fraction_fit,
        fraction_evaluate=config.fraction_evaluate,
        initial_parameters=initial_params,
    )

    # Create server config
    server_config = ServerConfig(num_rounds=config.num_rounds)

    logger.info(
        "Created FedProx server",
        num_rounds=config.num_rounds,
        server_address=config.server_address,
    )

    return server_config, strategy


def main() -> None:
    """CLI entry point for starting the FedProx Flower server."""
    import os

    num_rounds = int(os.getenv("FL_ROUNDS", "10"))
    min_clients = int(os.getenv("FL_MIN_CLIENTS", "2"))
    server_address = os.getenv("FL_SERVER_ADDRESS", "0.0.0.0:8080")

    config = FedProxServerConfig(
        num_rounds=num_rounds,
        min_fit_clients=min_clients,
        min_evaluate_clients=min_clients,
        min_available_clients=min_clients,
        server_address=server_address,
    )
    server_config, strategy = create_fedprox_server(config)

    logger.info(
        "Starting Flower server",
        address=config.server_address,
        rounds=config.num_rounds,
        min_clients=config.min_fit_clients,
    )
    fl.server.start_server(
        server_address=config.server_address,
        config=server_config,
        strategy=strategy,
    )


if __name__ == "__main__":
    main()
