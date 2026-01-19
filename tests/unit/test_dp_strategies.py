"""Unit tests for differential privacy strategies in federated learning."""

from __future__ import annotations

from dataclasses import fields
from typing import TYPE_CHECKING
from unittest.mock import MagicMock

import numpy as np
import pytest

from noa_swarm.federated.strategies import (
    DPConfig,
    clip_gradients,
    add_gaussian_noise,
    DPFedProxStrategy,
)

if TYPE_CHECKING:
    pass


class TestDPConfig:
    """Tests for DPConfig dataclass."""

    def test_default_values(self) -> None:
        """Test DPConfig has correct default values."""
        config = DPConfig()

        assert config.noise_multiplier == 1.0
        assert config.max_grad_norm == 1.0
        assert config.target_delta == 1e-5
        assert config.enabled is True

    def test_custom_values(self) -> None:
        """Test DPConfig accepts custom values."""
        config = DPConfig(
            noise_multiplier=0.5,
            max_grad_norm=2.0,
            target_delta=1e-6,
            enabled=False,
        )

        assert config.noise_multiplier == 0.5
        assert config.max_grad_norm == 2.0
        assert config.target_delta == 1e-6
        assert config.enabled is False

    def test_is_dataclass(self) -> None:
        """Test that DPConfig is a proper dataclass."""
        config = DPConfig()
        field_names = [f.name for f in fields(config)]
        assert "noise_multiplier" in field_names
        assert "max_grad_norm" in field_names
        assert "target_delta" in field_names
        assert "enabled" in field_names

    def test_invalid_noise_multiplier_raises(self) -> None:
        """Test that noise_multiplier <= 0 raises ValueError."""
        with pytest.raises(ValueError, match="noise_multiplier must be positive"):
            DPConfig(noise_multiplier=0.0)
        with pytest.raises(ValueError, match="noise_multiplier must be positive"):
            DPConfig(noise_multiplier=-0.1)

    def test_invalid_max_grad_norm_raises(self) -> None:
        """Test that max_grad_norm <= 0 raises ValueError."""
        with pytest.raises(ValueError, match="max_grad_norm must be positive"):
            DPConfig(max_grad_norm=0.0)
        with pytest.raises(ValueError, match="max_grad_norm must be positive"):
            DPConfig(max_grad_norm=-1.0)

    def test_invalid_target_delta_raises(self) -> None:
        """Test that target_delta outside (0, 1) raises ValueError."""
        with pytest.raises(ValueError, match="target_delta must be in"):
            DPConfig(target_delta=0.0)
        with pytest.raises(ValueError, match="target_delta must be in"):
            DPConfig(target_delta=1.0)
        with pytest.raises(ValueError, match="target_delta must be in"):
            DPConfig(target_delta=1.5)


class TestGradientClipping:
    """Tests for gradient clipping functions."""

    def test_clip_gradients_no_change_when_within_norm(self) -> None:
        """Test that gradients within norm are not changed."""
        gradients = [np.array([0.3, 0.4])]  # L2 norm = 0.5
        max_norm = 1.0

        clipped = clip_gradients(gradients, max_norm)

        np.testing.assert_array_almost_equal(clipped[0], gradients[0])

    def test_clip_gradients_clips_when_exceeds_norm(self) -> None:
        """Test that gradients exceeding norm are clipped."""
        gradients = [np.array([3.0, 4.0])]  # L2 norm = 5.0
        max_norm = 1.0

        clipped = clip_gradients(gradients, max_norm)

        # After clipping: [3/5, 4/5] = [0.6, 0.8]
        expected = np.array([0.6, 0.8])
        np.testing.assert_array_almost_equal(clipped[0], expected)

    def test_clip_gradients_preserves_direction(self) -> None:
        """Test that clipping preserves gradient direction."""
        gradients = [np.array([6.0, 8.0])]  # L2 norm = 10.0
        max_norm = 2.0

        clipped = clip_gradients(gradients, max_norm)

        # Direction should be preserved (ratio should be same)
        original_ratio = gradients[0][0] / gradients[0][1]
        clipped_ratio = clipped[0][0] / clipped[0][1]
        assert original_ratio == pytest.approx(clipped_ratio)

        # Norm should be max_norm
        clipped_norm = np.linalg.norm(clipped[0])
        assert clipped_norm == pytest.approx(max_norm)

    def test_clip_gradients_handles_multiple_arrays(self) -> None:
        """Test clipping works with multiple gradient arrays."""
        gradients = [
            np.array([3.0, 4.0]),   # norm = 5
            np.array([0.1, 0.1]),   # norm = 0.14
        ]
        max_norm = 1.0

        clipped = clip_gradients(gradients, max_norm)

        # First array should be clipped
        assert np.linalg.norm(clipped[0]) == pytest.approx(1.0)
        # Second array unchanged (within norm)
        np.testing.assert_array_almost_equal(clipped[1], gradients[1])

    def test_clip_gradients_handles_zero_gradients(self) -> None:
        """Test clipping handles zero gradients gracefully."""
        gradients = [np.array([0.0, 0.0])]
        max_norm = 1.0

        clipped = clip_gradients(gradients, max_norm)

        np.testing.assert_array_almost_equal(clipped[0], gradients[0])


class TestGaussianNoise:
    """Tests for Gaussian noise injection."""

    def test_add_gaussian_noise_adds_noise(self) -> None:
        """Test that noise is actually added to parameters."""
        np.random.seed(42)
        params = [np.array([1.0, 2.0, 3.0])]
        noise_multiplier = 1.0
        max_norm = 1.0
        num_clients = 10

        noisy = add_gaussian_noise(params, noise_multiplier, max_norm, num_clients)

        # Parameters should be different after noise
        assert not np.allclose(noisy[0], params[0])

    def test_add_gaussian_noise_zero_multiplier(self) -> None:
        """Test that zero noise multiplier adds no noise."""
        params = [np.array([1.0, 2.0, 3.0])]
        noise_multiplier = 0.0
        max_norm = 1.0
        num_clients = 10

        noisy = add_gaussian_noise(params, noise_multiplier, max_norm, num_clients)

        np.testing.assert_array_almost_equal(noisy[0], params[0])

    def test_add_gaussian_noise_scales_with_multiplier(self) -> None:
        """Test that noise scales with noise multiplier."""
        np.random.seed(42)
        params = [np.zeros(1000)]  # Large array for stable statistics
        max_norm = 1.0
        num_clients = 10

        # Low noise
        low_noisy = add_gaussian_noise(params.copy(), 0.1, max_norm, num_clients)
        low_std = np.std(low_noisy[0])

        # High noise
        np.random.seed(42)
        high_noisy = add_gaussian_noise(params.copy(), 1.0, max_norm, num_clients)
        high_std = np.std(high_noisy[0])

        # Higher multiplier should give higher standard deviation
        assert high_std > low_std

    def test_add_gaussian_noise_scales_with_clients(self) -> None:
        """Test that noise scales inversely with number of clients."""
        np.random.seed(42)
        params = [np.zeros(1000)]
        noise_multiplier = 1.0
        max_norm = 1.0

        # Few clients (more noise per client needed)
        few_noisy = add_gaussian_noise(params.copy(), noise_multiplier, max_norm, 2)
        few_std = np.std(few_noisy[0])

        # Many clients (less noise per client needed)
        np.random.seed(42)
        many_noisy = add_gaussian_noise(params.copy(), noise_multiplier, max_norm, 100)
        many_std = np.std(many_noisy[0])

        # More clients should result in less noise (noise divided by num_clients)
        assert many_std < few_std


class TestDPFedProxStrategy:
    """Tests for DPFedProxStrategy class."""

    @pytest.fixture
    def dp_config(self) -> DPConfig:
        """Create a DPConfig for testing."""
        return DPConfig(
            noise_multiplier=0.1,
            max_grad_norm=1.0,
            target_delta=1e-5,
            enabled=True,
        )

    @pytest.fixture
    def strategy(self, dp_config: DPConfig) -> DPFedProxStrategy:
        """Create a DPFedProxStrategy for testing."""
        return DPFedProxStrategy(
            dp_config=dp_config,
            min_fit_clients=1,
            min_evaluate_clients=1,
            min_available_clients=1,
        )

    def test_initialization(self, strategy: DPFedProxStrategy) -> None:
        """Test DPFedProxStrategy initializes correctly."""
        assert strategy is not None
        assert strategy.dp_config.noise_multiplier == 0.1
        assert strategy.dp_config.max_grad_norm == 1.0
        assert strategy.dp_config.enabled is True

    def test_privacy_budget_tracking(self, strategy: DPFedProxStrategy) -> None:
        """Test that strategy tracks privacy budget (epsilon)."""
        assert hasattr(strategy, "epsilon_spent")
        assert strategy.epsilon_spent == 0.0

    def test_aggregate_fit_with_dp_enabled(
        self, strategy: DPFedProxStrategy
    ) -> None:
        """Test aggregate_fit applies DP when enabled."""
        from flwr.common import (
            FitRes,
            Status,
            Code,
            ndarrays_to_parameters,
        )
        from flwr.server.client_proxy import ClientProxy

        np.random.seed(42)
        mock_client = MagicMock(spec=ClientProxy)
        mock_client.cid = "client-1"
        params = ndarrays_to_parameters([np.array([1.0, 2.0])])
        fit_res = FitRes(
            status=Status(code=Code.OK, message=""),
            parameters=params,
            num_examples=10,
            metrics={},
        )

        results = [(mock_client, fit_res)]
        failures = []

        aggregated, metrics = strategy.aggregate_fit(1, results, failures)

        assert aggregated is not None
        # Epsilon should be updated after aggregation
        assert strategy.epsilon_spent > 0.0

    def test_aggregate_fit_with_dp_disabled(self) -> None:
        """Test aggregate_fit skips DP when disabled."""
        dp_config = DPConfig(enabled=False)
        strategy = DPFedProxStrategy(
            dp_config=dp_config,
            min_fit_clients=1,
            min_evaluate_clients=1,
            min_available_clients=1,
        )

        from flwr.common import (
            FitRes,
            Status,
            Code,
            ndarrays_to_parameters,
        )
        from flwr.server.client_proxy import ClientProxy

        mock_client = MagicMock(spec=ClientProxy)
        params = ndarrays_to_parameters([np.array([1.0, 2.0])])
        fit_res = FitRes(
            status=Status(code=Code.OK, message=""),
            parameters=params,
            num_examples=10,
            metrics={},
        )

        results = [(mock_client, fit_res)]
        failures = []

        aggregated, metrics = strategy.aggregate_fit(1, results, failures)

        assert aggregated is not None
        # Epsilon should not change when DP is disabled
        assert strategy.epsilon_spent == 0.0

    def test_get_privacy_spent(self, strategy: DPFedProxStrategy) -> None:
        """Test get_privacy_spent returns epsilon and delta."""
        epsilon, delta = strategy.get_privacy_spent()

        assert isinstance(epsilon, float)
        assert isinstance(delta, float)
        assert epsilon >= 0.0
        assert delta == strategy.dp_config.target_delta
