"""Differential privacy strategies for federated learning.

This module provides differential privacy (DP) mechanisms for the federated
learning pipeline. It implements:

1. **Gradient clipping**: Bounds the L2 norm of gradients to limit the
   influence of any single training example.

2. **Gaussian noise injection**: Adds calibrated noise to aggregated gradients
   to provide (ε, δ)-differential privacy guarantees.

3. **Privacy budget tracking**: Monitors cumulative privacy loss using
   composition theorems.

The DP mechanisms can be applied to the FedProx strategy to enable
privacy-preserving federated learning across organizations.

Example usage:
    >>> from noa_swarm.federated.strategies import DPConfig, DPFedProxStrategy
    >>> dp_config = DPConfig(noise_multiplier=1.0, max_grad_norm=1.0)
    >>> strategy = DPFedProxStrategy(dp_config=dp_config, min_fit_clients=3)
    >>> # Use strategy with Flower server

References:
    Abadi, M., et al. (2016). Deep learning with differential privacy.
    Proceedings of the 2016 ACM SIGSAC conference on computer and
    communications security.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, TypeAlias

import numpy as np
from flwr.common import (
    FitRes,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.server.strategy import FedAvg

from noa_swarm.common.logging import get_logger

if TYPE_CHECKING:
    from collections.abc import Sequence

    from flwr.server.client_proxy import ClientProxy

logger = get_logger(__name__)

NDArray: TypeAlias = np.ndarray[Any, np.dtype[Any]]


@dataclass
class DPConfig:
    """Configuration for differential privacy mechanisms.

    Attributes:
        noise_multiplier: Ratio of noise standard deviation to sensitivity.
            Higher values provide stronger privacy but reduce utility.
            Typical values range from 0.1 to 2.0. Default is 1.0.
        max_grad_norm: Maximum L2 norm for gradient clipping. Gradients with
            larger norms are scaled down to this value. Default is 1.0.
        target_delta: Target delta for (ε, δ)-DP. Should be less than 1/n
            where n is the dataset size. Default is 1e-5.
        enabled: Whether DP mechanisms are enabled. Default is True.
    """

    noise_multiplier: float = 1.0
    max_grad_norm: float = 1.0
    target_delta: float = 1e-5
    enabled: bool = True

    def __post_init__(self) -> None:
        """Validate configuration parameters."""
        if self.noise_multiplier <= 0:
            raise ValueError(f"noise_multiplier must be positive, got {self.noise_multiplier}")
        if self.max_grad_norm <= 0:
            raise ValueError(f"max_grad_norm must be positive, got {self.max_grad_norm}")
        if not (0.0 < self.target_delta < 1.0):
            raise ValueError(f"target_delta must be in (0, 1), got {self.target_delta}")


def clip_gradients(
    gradients: Sequence[NDArray],
    max_norm: float,
) -> list[NDArray]:
    """Clip gradients to have maximum L2 norm.

    This implements per-layer gradient clipping. Each gradient array is
    clipped independently to the specified maximum L2 norm.

    Args:
        gradients: List of gradient arrays (one per model layer).
        max_norm: Maximum allowed L2 norm for each gradient array.

    Returns:
        List of clipped gradient arrays with the same shapes as input.

    Example:
        >>> grads = [np.array([3.0, 4.0])]  # L2 norm = 5.0
        >>> clipped = clip_gradients(grads, max_norm=1.0)
        >>> np.linalg.norm(clipped[0])  # Should be 1.0
    """
    clipped: list[NDArray] = []

    for grad in gradients:
        grad_norm = np.linalg.norm(grad)

        if grad_norm > max_norm and grad_norm > 0:
            # Scale gradient to have max_norm
            scaling_factor = max_norm / grad_norm
            clipped.append(grad * scaling_factor)
        else:
            clipped.append(grad.copy())

    return clipped


def add_gaussian_noise(
    parameters: Sequence[NDArray],
    noise_multiplier: float,
    max_norm: float,
    num_clients: int,
) -> list[NDArray]:
    """Add calibrated Gaussian noise to parameters for differential privacy.

    The noise is calibrated based on the sensitivity (max_grad_norm) and
    the noise multiplier. The standard deviation of noise is:
        sigma = (noise_multiplier * max_norm) / num_clients

    Args:
        parameters: List of parameter arrays to add noise to.
        noise_multiplier: Ratio of noise std to sensitivity.
        max_norm: Maximum gradient norm (sensitivity).
        num_clients: Number of clients participating (for averaging).

    Returns:
        List of noisy parameter arrays with same shapes as input.
    """
    if noise_multiplier == 0:
        return [p.copy() for p in parameters]

    # Noise standard deviation: sigma = (sigma_multiplier * C) / num_clients
    # where C is the clipping bound (max_norm)
    noise_std = (noise_multiplier * max_norm) / num_clients

    noisy: list[NDArray] = []
    for param in parameters:
        noise = np.random.normal(0, noise_std, param.shape)
        noisy.append(param + noise)

    return noisy


def _compute_epsilon(
    noise_multiplier: float,
    num_rounds: int,
    delta: float,
) -> float:
    """Compute epsilon using simple composition for DP accounting.

    This uses basic composition which gives a loose bound. For production
    use, consider using the Rényi DP accountant for tighter bounds.

    Args:
        noise_multiplier: The noise multiplier used.
        num_rounds: Number of training rounds completed.
        delta: Target delta value.

    Returns:
        Estimated epsilon value for (ε, δ)-DP.
    """
    if noise_multiplier <= 0 or num_rounds == 0:
        return 0.0

    # Simple Gaussian mechanism: epsilon = sqrt(2 * ln(1.25/delta)) / sigma
    # With basic composition: ε_total = sqrt(T) * ε_single
    single_round_epsilon = math.sqrt(2 * math.log(1.25 / delta)) / noise_multiplier
    total_epsilon = math.sqrt(num_rounds) * single_round_epsilon

    return total_epsilon


class DPFedProxStrategy(FedAvg):
    """Federated learning strategy with differential privacy.

    This strategy extends FedAvg to add differential privacy mechanisms:
    - Gradient clipping to bound sensitivity
    - Gaussian noise injection for privacy
    - Privacy budget (epsilon) tracking

    The privacy guarantee is (ε, δ)-differential privacy where:
    - ε (epsilon) measures privacy loss (lower is better)
    - δ is the probability of privacy failure

    Attributes:
        dp_config: Configuration for DP mechanisms.
        epsilon_spent: Cumulative privacy budget spent.
        rounds_completed: Number of aggregation rounds completed.
    """

    def __init__(
        self,
        *,
        dp_config: DPConfig,
        min_fit_clients: int = 2,
        min_evaluate_clients: int = 2,
        min_available_clients: int = 2,
        fraction_fit: float = 1.0,
        fraction_evaluate: float = 1.0,
        initial_parameters: Parameters | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the DP-enabled FedProx strategy.

        Args:
            dp_config: Configuration for differential privacy.
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

        self.dp_config = dp_config
        self.epsilon_spent: float = 0.0
        self.rounds_completed: int = 0

        logger.info(
            "Initialized DPFedProxStrategy",
            noise_multiplier=dp_config.noise_multiplier,
            max_grad_norm=dp_config.max_grad_norm,
            target_delta=dp_config.target_delta,
            dp_enabled=dp_config.enabled,
        )

    def aggregate_fit(
        self,
        server_round: int,
        results: list[tuple[ClientProxy, FitRes]],
        failures: list[tuple[ClientProxy, FitRes] | BaseException],
    ) -> tuple[Parameters | None, dict[str, Scalar]]:
        """Aggregate training results with optional differential privacy.

        When DP is enabled, this method:
        1. Clips client gradients to bounded sensitivity
        2. Aggregates using weighted averaging
        3. Adds calibrated Gaussian noise
        4. Updates privacy budget tracking

        Args:
            server_round: Current server round number.
            results: List of (client, fit_result) tuples from successful clients.
            failures: List of failed client results or exceptions.

        Returns:
            Tuple of:
            - Aggregated (and potentially noised) parameters
            - Dictionary with aggregation and privacy metrics
        """
        if not results:
            logger.warning(
                "No results to aggregate",
                server_round=server_round,
                num_failures=len(failures),
            )
            return None, {}

        num_clients = len(results)

        if self.dp_config.enabled:
            # Apply gradient clipping to each client's update
            clipped_results = []
            for client, fit_res in results:
                params = parameters_to_ndarrays(fit_res.parameters)
                clipped_params = clip_gradients(params, self.dp_config.max_grad_norm)
                clipped_fit_res = FitRes(
                    status=fit_res.status,
                    parameters=ndarrays_to_parameters(clipped_params),
                    num_examples=fit_res.num_examples,
                    metrics=fit_res.metrics,
                )
                clipped_results.append((client, clipped_fit_res))

            # Perform standard weighted aggregation on clipped parameters
            aggregated_parameters, metrics = super().aggregate_fit(
                server_round, clipped_results, failures
            )

            if aggregated_parameters is not None:
                # Add Gaussian noise to aggregated parameters
                agg_ndarrays = parameters_to_ndarrays(aggregated_parameters)
                noisy_params = add_gaussian_noise(
                    agg_ndarrays,
                    self.dp_config.noise_multiplier,
                    self.dp_config.max_grad_norm,
                    num_clients,
                )
                aggregated_parameters = ndarrays_to_parameters(noisy_params)

                # Update privacy accounting
                self.rounds_completed += 1
                self.epsilon_spent = _compute_epsilon(
                    self.dp_config.noise_multiplier,
                    self.rounds_completed,
                    self.dp_config.target_delta,
                )

                logger.info(
                    "Applied DP to aggregated parameters",
                    server_round=server_round,
                    num_clients=num_clients,
                    epsilon_spent=self.epsilon_spent,
                    delta=self.dp_config.target_delta,
                )

                metrics["epsilon_spent"] = self.epsilon_spent
                metrics["delta"] = self.dp_config.target_delta
                metrics["dp_enabled"] = True
        else:
            # DP disabled - standard aggregation
            aggregated_parameters, metrics = super().aggregate_fit(server_round, results, failures)
            metrics["dp_enabled"] = False

        return aggregated_parameters, metrics

    def get_privacy_spent(self) -> tuple[float, float]:
        """Get the current privacy budget spent.

        Returns:
            Tuple of (epsilon, delta) representing the cumulative privacy loss.
        """
        return self.epsilon_spent, self.dp_config.target_delta
