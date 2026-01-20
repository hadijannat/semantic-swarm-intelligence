"""Unit tests for FedProx Flower client implementation."""

from __future__ import annotations

import copy
from dataclasses import fields

import numpy as np
import pytest
import torch
import torch.nn as nn

from noa_swarm.federated.flower_client import (
    FedProxClient,
    FedProxConfig,
)


# Simple test model for unit tests
class SimpleModel(nn.Module):
    """Simple model for testing federated learning client."""

    def __init__(self, input_dim: int = 10, hidden_dim: int = 16, output_dim: int = 3) -> None:
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


@pytest.fixture
def simple_model() -> SimpleModel:
    """Create a simple model for testing."""
    return SimpleModel()


@pytest.fixture
def train_data() -> tuple[torch.Tensor, torch.Tensor]:
    """Create simple training data."""
    torch.manual_seed(42)
    x = torch.randn(100, 10)
    y = torch.randint(0, 3, (100,))
    return x, y


@pytest.fixture
def val_data() -> tuple[torch.Tensor, torch.Tensor]:
    """Create simple validation data."""
    torch.manual_seed(123)
    x = torch.randn(20, 10)
    y = torch.randint(0, 3, (20,))
    return x, y


class TestFedProxConfig:
    """Tests for FedProxConfig dataclass."""

    def test_default_values(self) -> None:
        """Test FedProxConfig has correct default values."""
        config = FedProxConfig(client_id="test-client")

        assert config.client_id == "test-client"
        assert config.mu == 0.01
        assert config.local_epochs == 3
        assert config.batch_size == 32
        assert config.learning_rate == 0.001

    def test_custom_values(self) -> None:
        """Test FedProxConfig accepts custom values."""
        config = FedProxConfig(
            client_id="custom-client",
            mu=0.1,
            local_epochs=5,
            batch_size=64,
            learning_rate=0.01,
        )

        assert config.client_id == "custom-client"
        assert config.mu == 0.1
        assert config.local_epochs == 5
        assert config.batch_size == 64
        assert config.learning_rate == 0.01

    def test_is_dataclass(self) -> None:
        """Test that FedProxConfig is a proper dataclass."""
        config = FedProxConfig(client_id="test")
        # Should have __dataclass_fields__ attribute
        field_names = [f.name for f in fields(config)]
        assert "client_id" in field_names
        assert "mu" in field_names
        assert "local_epochs" in field_names
        assert "batch_size" in field_names
        assert "learning_rate" in field_names

    def test_client_id_required(self) -> None:
        """Test that client_id is a required field."""
        # This should raise TypeError because client_id has no default
        with pytest.raises(TypeError):
            FedProxConfig()  # type: ignore[call-arg]

    def test_empty_client_id_raises(self) -> None:
        """Test that empty client_id raises ValueError."""
        with pytest.raises(ValueError, match="client_id cannot be empty"):
            FedProxConfig(client_id="")

    def test_negative_mu_raises(self) -> None:
        """Test that negative mu raises ValueError."""
        with pytest.raises(ValueError, match="mu must be non-negative"):
            FedProxConfig(client_id="test", mu=-0.1)

    def test_zero_local_epochs_raises(self) -> None:
        """Test that local_epochs < 1 raises ValueError."""
        with pytest.raises(ValueError, match="local_epochs must be >= 1"):
            FedProxConfig(client_id="test", local_epochs=0)

    def test_zero_batch_size_raises(self) -> None:
        """Test that batch_size < 1 raises ValueError."""
        with pytest.raises(ValueError, match="batch_size must be >= 1"):
            FedProxConfig(client_id="test", batch_size=0)

    def test_zero_learning_rate_raises(self) -> None:
        """Test that learning_rate <= 0 raises ValueError."""
        with pytest.raises(ValueError, match="learning_rate must be positive"):
            FedProxConfig(client_id="test", learning_rate=0.0)

    def test_negative_learning_rate_raises(self) -> None:
        """Test that negative learning_rate raises ValueError."""
        with pytest.raises(ValueError, match="learning_rate must be positive"):
            FedProxConfig(client_id="test", learning_rate=-0.01)

    def test_valid_edge_values(self) -> None:
        """Test that valid edge values are accepted."""
        # mu=0 is valid (disables proximal term)
        config = FedProxConfig(client_id="test", mu=0.0)
        assert config.mu == 0.0

        # local_epochs=1 is valid
        config = FedProxConfig(client_id="test", local_epochs=1)
        assert config.local_epochs == 1

        # batch_size=1 is valid
        config = FedProxConfig(client_id="test", batch_size=1)
        assert config.batch_size == 1

        # Very small learning rate is valid
        config = FedProxConfig(client_id="test", learning_rate=1e-10)
        assert config.learning_rate == 1e-10


class TestFedProxClient:
    """Tests for FedProxClient class."""

    @pytest.fixture
    def config(self) -> FedProxConfig:
        """Create a FedProxConfig for testing."""
        return FedProxConfig(
            client_id="test-client",
            mu=0.01,
            local_epochs=2,
            batch_size=16,
            learning_rate=0.01,
        )

    @pytest.fixture
    def client(
        self,
        config: FedProxConfig,
        simple_model: SimpleModel,
        train_data: tuple[torch.Tensor, torch.Tensor],
        val_data: tuple[torch.Tensor, torch.Tensor],
    ) -> FedProxClient:
        """Create a FedProxClient for testing."""
        return FedProxClient(
            config=config,
            model=simple_model,
            train_data=train_data,
            val_data=val_data,
        )

    def test_initialization(
        self,
        config: FedProxConfig,
        simple_model: SimpleModel,
        train_data: tuple[torch.Tensor, torch.Tensor],
        val_data: tuple[torch.Tensor, torch.Tensor],
    ) -> None:
        """Test FedProxClient initialization."""
        client = FedProxClient(
            config=config,
            model=simple_model,
            train_data=train_data,
            val_data=val_data,
        )

        assert client.config == config
        assert client.model is simple_model
        assert client.train_data == train_data
        assert client.val_data == val_data

    def test_get_parameters(self, client: FedProxClient) -> None:
        """Test get_parameters returns list of numpy arrays."""
        params = client.get_parameters(config={})

        assert isinstance(params, list)
        assert len(params) > 0
        for param in params:
            assert isinstance(param, np.ndarray)

    def test_get_parameters_shape(
        self, client: FedProxClient, simple_model: SimpleModel
    ) -> None:
        """Test get_parameters returns correct shapes."""
        params = client.get_parameters(config={})

        # SimpleModel has fc1 (weight, bias) and fc2 (weight, bias)
        expected_shapes = [
            p.detach().cpu().numpy().shape for p in simple_model.parameters()
        ]

        assert len(params) == len(expected_shapes)
        for param, expected_shape in zip(params, expected_shapes, strict=True):
            assert param.shape == expected_shape

    def test_set_parameters(self, client: FedProxClient) -> None:
        """Test set_parameters updates model weights."""
        # Get original parameters
        original_params = client.get_parameters(config={})

        # Create new parameters with different values
        new_params = [np.ones_like(p) * 0.5 for p in original_params]

        # Set new parameters
        client.set_parameters(new_params)

        # Get updated parameters
        updated_params = client.get_parameters(config={})

        # Verify parameters were updated
        for updated, new in zip(updated_params, new_params, strict=True):
            np.testing.assert_array_almost_equal(updated, new)

    def test_set_parameters_roundtrip(self, client: FedProxClient) -> None:
        """Test get/set parameters roundtrip preserves values."""
        original_params = client.get_parameters(config={})

        # Modify parameters
        modified_params = [p * 2.0 for p in original_params]
        client.set_parameters(modified_params)

        # Get them back
        retrieved_params = client.get_parameters(config={})

        for orig, modified, retrieved in zip(
            original_params, modified_params, retrieved_params, strict=True
        ):
            # Verify they were actually changed
            assert not np.allclose(orig, retrieved)
            # Verify they match what we set
            np.testing.assert_array_almost_equal(modified, retrieved)

    def test_fit_returns_tuple(self, client: FedProxClient) -> None:
        """Test fit returns (parameters, num_examples, metrics)."""
        initial_params = client.get_parameters(config={})

        result = client.fit(initial_params, config={})

        assert isinstance(result, tuple)
        assert len(result) == 3

        params, num_examples, metrics = result
        assert isinstance(params, list)
        assert isinstance(num_examples, int)
        assert isinstance(metrics, dict)

    def test_fit_returns_num_examples(
        self,
        client: FedProxClient,
        train_data: tuple[torch.Tensor, torch.Tensor],
    ) -> None:
        """Test fit returns correct number of training examples."""
        initial_params = client.get_parameters(config={})

        _, num_examples, _ = client.fit(initial_params, config={})

        assert num_examples == len(train_data[0])

    def test_fit_returns_loss_metric(self, client: FedProxClient) -> None:
        """Test fit returns loss in metrics."""
        initial_params = client.get_parameters(config={})

        _, _, metrics = client.fit(initial_params, config={})

        assert "loss" in metrics
        assert isinstance(metrics["loss"], float)
        assert metrics["loss"] > 0

    def test_fit_updates_model(self, client: FedProxClient) -> None:
        """Test fit updates model parameters."""
        initial_params = client.get_parameters(config={})
        initial_params_copy = [p.copy() for p in initial_params]

        new_params, _, _ = client.fit(initial_params, config={})

        # At least some parameters should have changed
        changed = any(
            not np.allclose(new, initial)
            for new, initial in zip(new_params, initial_params_copy, strict=True)
        )
        assert changed, "Model parameters should change after training"

    def test_evaluate_returns_tuple(self, client: FedProxClient) -> None:
        """Test evaluate returns (loss, num_examples, metrics)."""
        params = client.get_parameters(config={})

        result = client.evaluate(params, config={})

        assert isinstance(result, tuple)
        assert len(result) == 3

        loss, num_examples, metrics = result
        assert isinstance(loss, float)
        assert isinstance(num_examples, int)
        assert isinstance(metrics, dict)

    def test_evaluate_returns_num_val_examples(
        self,
        client: FedProxClient,
        val_data: tuple[torch.Tensor, torch.Tensor],
    ) -> None:
        """Test evaluate returns correct number of validation examples."""
        params = client.get_parameters(config={})

        _, num_examples, _ = client.evaluate(params, config={})

        assert num_examples == len(val_data[0])

    def test_evaluate_returns_accuracy(self, client: FedProxClient) -> None:
        """Test evaluate returns accuracy in metrics."""
        params = client.get_parameters(config={})

        _, _, metrics = client.evaluate(params, config={})

        assert "accuracy" in metrics
        assert 0.0 <= metrics["accuracy"] <= 1.0

    def test_evaluate_sets_model_params(self, client: FedProxClient) -> None:
        """Test evaluate sets model parameters before evaluation."""
        # Get original parameters
        original_params = client.get_parameters(config={})

        # Create different parameters
        new_params = [np.zeros_like(p) for p in original_params]

        # Evaluate with new parameters
        client.evaluate(new_params, config={})

        # Verify model now has new parameters
        current_params = client.get_parameters(config={})
        for current, new in zip(current_params, new_params, strict=True):
            np.testing.assert_array_almost_equal(current, new)


class TestFedProxProximalTerm:
    """Tests for FedProx proximal term computation."""

    @pytest.fixture
    def config(self) -> FedProxConfig:
        """Create a FedProxConfig with specific mu for testing."""
        return FedProxConfig(
            client_id="test-client",
            mu=1.0,  # Use mu=1.0 for easier testing
            local_epochs=1,
            batch_size=16,
            learning_rate=0.01,
        )

    @pytest.fixture
    def client(
        self,
        config: FedProxConfig,
        simple_model: SimpleModel,
        train_data: tuple[torch.Tensor, torch.Tensor],
        val_data: tuple[torch.Tensor, torch.Tensor],
    ) -> FedProxClient:
        """Create a FedProxClient for testing."""
        return FedProxClient(
            config=config,
            model=simple_model,
            train_data=train_data,
            val_data=val_data,
        )

    def test_proximal_loss_zero_when_same(self, client: FedProxClient) -> None:
        """Test proximal loss is zero when local and global params are the same."""
        params = client.get_parameters(config={})

        # Set model to these parameters
        client.set_parameters(params)

        # Convert to tensors (as fit() does internally)
        global_tensors = [
            torch.tensor(p, dtype=param.dtype, device=client.device)
            for p, param in zip(params, client.model.parameters(), strict=True)
        ]

        # Compute proximal loss comparing to itself
        prox_loss = client._proximal_loss(global_tensors)

        assert prox_loss.item() == pytest.approx(0.0, abs=1e-6)

    def test_proximal_loss_positive_when_different(
        self, client: FedProxClient
    ) -> None:
        """Test proximal loss is positive when local and global params differ."""
        global_params = client.get_parameters(config={})

        # Modify model parameters
        new_params = [p + 1.0 for p in global_params]
        client.set_parameters(new_params)

        # Convert to tensors (as fit() does internally)
        global_tensors = [
            torch.tensor(p, dtype=param.dtype, device=client.device)
            for p, param in zip(global_params, client.model.parameters(), strict=True)
        ]

        # Compute proximal loss
        prox_loss = client._proximal_loss(global_tensors)

        assert prox_loss.item() > 0

    def test_proximal_loss_scales_with_difference(
        self, client: FedProxClient
    ) -> None:
        """Test proximal loss scales with parameter difference."""
        global_params = client.get_parameters(config={})

        # Convert to tensors once (as fit() does internally)
        global_tensors = [
            torch.tensor(p, dtype=param.dtype, device=client.device)
            for p, param in zip(global_params, client.model.parameters(), strict=True)
        ]

        # Small difference
        small_diff_params = [p + 0.1 for p in global_params]
        client.set_parameters(small_diff_params)
        small_loss = client._proximal_loss(global_tensors)

        # Large difference
        large_diff_params = [p + 1.0 for p in global_params]
        client.set_parameters(large_diff_params)
        large_loss = client._proximal_loss(global_tensors)

        assert large_loss.item() > small_loss.item()

    def test_fit_uses_proximal_term(self, simple_model: SimpleModel) -> None:
        """Test that fit with mu > 0 uses proximal term (parameters stay closer to global)."""
        torch.manual_seed(42)
        x = torch.randn(50, 10)
        y = torch.randint(0, 3, (50,))
        train_data = (x, y)
        val_data = (x[:10], y[:10])

        # Clone model for comparison
        model_with_prox = copy.deepcopy(simple_model)
        model_without_prox = copy.deepcopy(simple_model)

        # Client with proximal term (mu=1.0)
        config_with_prox = FedProxConfig(
            client_id="with-prox",
            mu=1.0,
            local_epochs=5,
            batch_size=16,
            learning_rate=0.1,
        )
        client_with_prox = FedProxClient(
            config=config_with_prox,
            model=model_with_prox,
            train_data=train_data,
            val_data=val_data,
        )

        # Client without proximal term (mu=0.0)
        config_without_prox = FedProxConfig(
            client_id="without-prox",
            mu=0.0,
            local_epochs=5,
            batch_size=16,
            learning_rate=0.1,
        )
        client_without_prox = FedProxClient(
            config=config_without_prox,
            model=model_without_prox,
            train_data=train_data,
            val_data=val_data,
        )

        # Get initial params
        initial_params = client_with_prox.get_parameters(config={})

        # Train both
        params_with_prox, _, _ = client_with_prox.fit(initial_params, config={})
        params_without_prox, _, _ = client_without_prox.fit(initial_params, config={})

        # Calculate distance from initial params
        dist_with_prox = sum(
            np.sum((new - orig) ** 2)
            for new, orig in zip(params_with_prox, initial_params, strict=True)
        )
        dist_without_prox = sum(
            np.sum((new - orig) ** 2)
            for new, orig in zip(params_without_prox, initial_params, strict=True)
        )

        # With proximal term, parameters should stay closer to initial
        assert dist_with_prox < dist_without_prox


class TestFedProxPseudoLabels:
    """Tests for pseudo-label support in FedProx client."""

    @pytest.fixture
    def config(self) -> FedProxConfig:
        """Create a FedProxConfig for testing."""
        return FedProxConfig(
            client_id="test-client",
            mu=0.01,
            local_epochs=2,
            batch_size=16,
            learning_rate=0.01,
        )

    def test_accepts_pseudo_labels(
        self,
        config: FedProxConfig,
        simple_model: SimpleModel,
        train_data: tuple[torch.Tensor, torch.Tensor],
        val_data: tuple[torch.Tensor, torch.Tensor],
    ) -> None:
        """Test client accepts pseudo-labeled data."""
        torch.manual_seed(42)
        pseudo_x = torch.randn(30, 10)
        pseudo_y = torch.randint(0, 3, (30,))
        pseudo_data = (pseudo_x, pseudo_y)

        client = FedProxClient(
            config=config,
            model=simple_model,
            train_data=train_data,
            val_data=val_data,
            pseudo_data=pseudo_data,
        )

        assert client.pseudo_data is not None
        assert len(client.pseudo_data[0]) == 30

    def test_pseudo_label_weight_default(
        self,
        config: FedProxConfig,
        simple_model: SimpleModel,
        train_data: tuple[torch.Tensor, torch.Tensor],
        val_data: tuple[torch.Tensor, torch.Tensor],
    ) -> None:
        """Test default pseudo-label weight is 0.5."""
        pseudo_data = (torch.randn(10, 10), torch.randint(0, 3, (10,)))

        client = FedProxClient(
            config=config,
            model=simple_model,
            train_data=train_data,
            val_data=val_data,
            pseudo_data=pseudo_data,
        )

        assert client.pseudo_weight == 0.5

    def test_custom_pseudo_label_weight(
        self,
        config: FedProxConfig,
        simple_model: SimpleModel,
        train_data: tuple[torch.Tensor, torch.Tensor],
        val_data: tuple[torch.Tensor, torch.Tensor],
    ) -> None:
        """Test custom pseudo-label weight."""
        pseudo_data = (torch.randn(10, 10), torch.randint(0, 3, (10,)))

        client = FedProxClient(
            config=config,
            model=simple_model,
            train_data=train_data,
            val_data=val_data,
            pseudo_data=pseudo_data,
            pseudo_weight=0.3,
        )

        assert client.pseudo_weight == 0.3

    def test_fit_with_pseudo_labels(
        self,
        config: FedProxConfig,
        simple_model: SimpleModel,
        train_data: tuple[torch.Tensor, torch.Tensor],
        val_data: tuple[torch.Tensor, torch.Tensor],
    ) -> None:
        """Test fit works with pseudo-labeled data."""
        torch.manual_seed(42)
        pseudo_data = (torch.randn(30, 10), torch.randint(0, 3, (30,)))

        client = FedProxClient(
            config=config,
            model=simple_model,
            train_data=train_data,
            val_data=val_data,
            pseudo_data=pseudo_data,
        )

        initial_params = client.get_parameters(config={})
        params, num_examples, metrics = client.fit(initial_params, config={})

        # Should return total examples (train + pseudo)
        assert num_examples == len(train_data[0]) + len(pseudo_data[0])
        assert "loss" in metrics


class TestFedProxEdgeCases:
    """Tests for edge cases and error handling."""

    def test_empty_train_data(
        self,
        simple_model: SimpleModel,
        val_data: tuple[torch.Tensor, torch.Tensor],
    ) -> None:
        """Test handling of empty training data."""
        config = FedProxConfig(client_id="test")
        empty_train = (torch.empty(0, 10), torch.empty(0, dtype=torch.long))

        client = FedProxClient(
            config=config,
            model=simple_model,
            train_data=empty_train,
            val_data=val_data,
        )

        params = client.get_parameters(config={})
        new_params, num_examples, metrics = client.fit(params, config={})

        assert num_examples == 0
        # Should return same params when no training data
        for new, orig in zip(new_params, params, strict=True):
            np.testing.assert_array_almost_equal(new, orig)

    def test_empty_val_data(
        self,
        simple_model: SimpleModel,
        train_data: tuple[torch.Tensor, torch.Tensor],
    ) -> None:
        """Test handling of empty validation data."""
        config = FedProxConfig(client_id="test")
        empty_val = (torch.empty(0, 10), torch.empty(0, dtype=torch.long))

        client = FedProxClient(
            config=config,
            model=simple_model,
            train_data=train_data,
            val_data=empty_val,
        )

        params = client.get_parameters(config={})
        loss, num_examples, metrics = client.evaluate(params, config={})

        assert num_examples == 0
        assert loss == 0.0
        assert metrics["accuracy"] == 0.0

    def test_single_sample_train(
        self,
        simple_model: SimpleModel,
        val_data: tuple[torch.Tensor, torch.Tensor],
    ) -> None:
        """Test training with single sample."""
        config = FedProxConfig(client_id="test", batch_size=1, local_epochs=1)
        single_train = (torch.randn(1, 10), torch.tensor([0]))

        client = FedProxClient(
            config=config,
            model=simple_model,
            train_data=single_train,
            val_data=val_data,
        )

        params = client.get_parameters(config={})
        new_params, num_examples, metrics = client.fit(params, config={})

        assert num_examples == 1
        assert "loss" in metrics

    def test_batch_size_larger_than_data(
        self,
        simple_model: SimpleModel,
        val_data: tuple[torch.Tensor, torch.Tensor],
    ) -> None:
        """Test when batch_size is larger than dataset."""
        config = FedProxConfig(client_id="test", batch_size=100, local_epochs=1)
        small_train = (torch.randn(10, 10), torch.randint(0, 3, (10,)))

        client = FedProxClient(
            config=config,
            model=simple_model,
            train_data=small_train,
            val_data=val_data,
        )

        params = client.get_parameters(config={})
        new_params, num_examples, metrics = client.fit(params, config={})

        assert num_examples == 10
        assert "loss" in metrics
