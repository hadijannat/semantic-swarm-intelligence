"""Unit tests for FedProx Flower server implementation."""

from __future__ import annotations

from dataclasses import fields
from unittest.mock import MagicMock

import numpy as np
import pytest

from noa_swarm.federated.flower_server import (
    FedProxServerConfig,
    FedProxStrategy,
    create_fedprox_server,
)


class TestFedProxServerConfig:
    """Tests for FedProxServerConfig dataclass."""

    def test_default_values(self) -> None:
        """Test FedProxServerConfig has correct default values."""
        config = FedProxServerConfig()

        assert config.num_rounds == 10
        assert config.min_fit_clients == 2
        assert config.min_evaluate_clients == 2
        assert config.min_available_clients == 2
        assert config.fraction_fit == 1.0
        assert config.fraction_evaluate == 1.0
        assert config.server_address == "0.0.0.0:8080"

    def test_custom_values(self) -> None:
        """Test FedProxServerConfig accepts custom values."""
        config = FedProxServerConfig(
            num_rounds=20,
            min_fit_clients=5,
            min_evaluate_clients=3,
            min_available_clients=10,
            fraction_fit=0.5,
            fraction_evaluate=0.8,
            server_address="localhost:9090",
        )

        assert config.num_rounds == 20
        assert config.min_fit_clients == 5
        assert config.min_evaluate_clients == 3
        assert config.min_available_clients == 10
        assert config.fraction_fit == 0.5
        assert config.fraction_evaluate == 0.8
        assert config.server_address == "localhost:9090"

    def test_is_dataclass(self) -> None:
        """Test that FedProxServerConfig is a proper dataclass."""
        config = FedProxServerConfig()
        field_names = [f.name for f in fields(config)]
        assert "num_rounds" in field_names
        assert "min_fit_clients" in field_names
        assert "server_address" in field_names

    def test_invalid_num_rounds_raises(self) -> None:
        """Test that num_rounds < 1 raises ValueError."""
        with pytest.raises(ValueError, match="num_rounds must be >= 1"):
            FedProxServerConfig(num_rounds=0)

    def test_invalid_min_fit_clients_raises(self) -> None:
        """Test that min_fit_clients < 1 raises ValueError."""
        with pytest.raises(ValueError, match="min_fit_clients must be >= 1"):
            FedProxServerConfig(min_fit_clients=0)

    def test_invalid_min_evaluate_clients_raises(self) -> None:
        """Test that min_evaluate_clients < 1 raises ValueError."""
        with pytest.raises(ValueError, match="min_evaluate_clients must be >= 1"):
            FedProxServerConfig(min_evaluate_clients=0)

    def test_invalid_min_available_clients_raises(self) -> None:
        """Test that min_available_clients < 1 raises ValueError."""
        with pytest.raises(ValueError, match="min_available_clients must be >= 1"):
            FedProxServerConfig(min_available_clients=0)

    def test_invalid_fraction_fit_raises(self) -> None:
        """Test that fraction_fit outside (0, 1] raises ValueError."""
        with pytest.raises(ValueError, match="fraction_fit must be in"):
            FedProxServerConfig(fraction_fit=0.0)
        with pytest.raises(ValueError, match="fraction_fit must be in"):
            FedProxServerConfig(fraction_fit=1.1)

    def test_invalid_fraction_evaluate_raises(self) -> None:
        """Test that fraction_evaluate outside (0, 1] raises ValueError."""
        with pytest.raises(ValueError, match="fraction_evaluate must be in"):
            FedProxServerConfig(fraction_evaluate=0.0)
        with pytest.raises(ValueError, match="fraction_evaluate must be in"):
            FedProxServerConfig(fraction_evaluate=1.5)

    def test_empty_server_address_raises(self) -> None:
        """Test that empty server_address raises ValueError."""
        with pytest.raises(ValueError, match="server_address cannot be empty"):
            FedProxServerConfig(server_address="")


class TestFedProxStrategy:
    """Tests for FedProxStrategy class."""

    @pytest.fixture
    def strategy(self) -> FedProxStrategy:
        """Create a FedProxStrategy for testing."""
        return FedProxStrategy(
            min_fit_clients=2,
            min_evaluate_clients=2,
            min_available_clients=2,
        )

    def test_initialization(self, strategy: FedProxStrategy) -> None:
        """Test FedProxStrategy initializes correctly."""
        assert strategy is not None
        assert strategy.min_fit_clients == 2
        assert strategy.min_evaluate_clients == 2
        assert strategy.min_available_clients == 2

    def test_has_aggregate_fit_method(self, strategy: FedProxStrategy) -> None:
        """Test FedProxStrategy has aggregate_fit method."""
        assert hasattr(strategy, "aggregate_fit")
        assert callable(strategy.aggregate_fit)

    def test_has_aggregate_evaluate_method(self, strategy: FedProxStrategy) -> None:
        """Test FedProxStrategy has aggregate_evaluate method."""
        assert hasattr(strategy, "aggregate_evaluate")
        assert callable(strategy.aggregate_evaluate)

    def test_current_round_starts_at_zero(self, strategy: FedProxStrategy) -> None:
        """Test current_round tracking starts at 0."""
        assert strategy.current_round == 0

    def test_model_version_tracking(self, strategy: FedProxStrategy) -> None:
        """Test that strategy tracks model version."""
        assert hasattr(strategy, "model_version")
        assert strategy.model_version == 0


class TestFedProxAggregation:
    """Tests for FedProx aggregation behavior."""

    @pytest.fixture
    def strategy(self) -> FedProxStrategy:
        """Create a FedProxStrategy for testing."""
        return FedProxStrategy(
            min_fit_clients=1,
            min_evaluate_clients=1,
            min_available_clients=1,
        )

    def test_aggregate_fit_weighted_average(self, strategy: FedProxStrategy) -> None:
        """Test that aggregate_fit computes weighted average correctly."""
        # Create mock client results with different weights
        # Client 1: params = [1.0, 2.0], num_examples = 10
        # Client 2: params = [3.0, 4.0], num_examples = 30
        # Expected: weighted_avg = (10*[1,2] + 30*[3,4]) / 40 = [2.5, 3.5]
        from flwr.common import (
            Code,
            FitRes,
            Status,
            ndarrays_to_parameters,
        )
        from flwr.server.client_proxy import ClientProxy

        # Create mock client proxies
        mock_client1 = MagicMock(spec=ClientProxy)
        mock_client1.cid = "client-1"
        mock_client2 = MagicMock(spec=ClientProxy)
        mock_client2.cid = "client-2"

        # Create parameters
        params1 = ndarrays_to_parameters([np.array([1.0, 2.0])])
        params2 = ndarrays_to_parameters([np.array([3.0, 4.0])])

        # Create fit results
        fit_res1 = FitRes(
            status=Status(code=Code.OK, message=""),
            parameters=params1,
            num_examples=10,
            metrics={},
        )
        fit_res2 = FitRes(
            status=Status(code=Code.OK, message=""),
            parameters=params2,
            num_examples=30,
            metrics={},
        )

        results = [(mock_client1, fit_res1), (mock_client2, fit_res2)]
        failures = []

        # Aggregate
        aggregated, metrics = strategy.aggregate_fit(1, results, failures)

        assert aggregated is not None
        from flwr.common import parameters_to_ndarrays

        agg_params = parameters_to_ndarrays(aggregated)

        # Check weighted average
        expected = np.array([2.5, 3.5])
        np.testing.assert_array_almost_equal(agg_params[0], expected)

    def test_aggregate_fit_increments_round(self, strategy: FedProxStrategy) -> None:
        """Test that aggregate_fit increments current_round."""
        from flwr.common import (
            Code,
            FitRes,
            Status,
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

        initial_round = strategy.current_round
        strategy.aggregate_fit(1, results, failures)

        assert strategy.current_round == initial_round + 1

    def test_aggregate_fit_increments_model_version(self, strategy: FedProxStrategy) -> None:
        """Test that aggregate_fit increments model_version."""
        from flwr.common import (
            Code,
            FitRes,
            Status,
            ndarrays_to_parameters,
        )
        from flwr.server.client_proxy import ClientProxy

        mock_client = MagicMock(spec=ClientProxy)
        params = ndarrays_to_parameters([np.array([1.0])])
        fit_res = FitRes(
            status=Status(code=Code.OK, message=""),
            parameters=params,
            num_examples=5,
            metrics={},
        )

        results = [(mock_client, fit_res)]
        failures = []

        initial_version = strategy.model_version
        strategy.aggregate_fit(1, results, failures)

        assert strategy.model_version == initial_version + 1

    def test_aggregate_fit_handles_empty_results(self, strategy: FedProxStrategy) -> None:
        """Test that aggregate_fit handles empty results gracefully."""
        results = []
        failures = []

        aggregated, metrics = strategy.aggregate_fit(1, results, failures)

        assert aggregated is None

    def test_aggregate_evaluate_computes_weighted_loss(self, strategy: FedProxStrategy) -> None:
        """Test that aggregate_evaluate computes weighted loss correctly."""
        from flwr.common import (
            Code,
            EvaluateRes,
            Status,
        )
        from flwr.server.client_proxy import ClientProxy

        mock_client1 = MagicMock(spec=ClientProxy)
        mock_client2 = MagicMock(spec=ClientProxy)

        # Client 1: loss=0.5, num_examples=10
        # Client 2: loss=0.3, num_examples=30
        # Expected: weighted_avg = (10*0.5 + 30*0.3) / 40 = 0.35
        eval_res1 = EvaluateRes(
            status=Status(code=Code.OK, message=""),
            loss=0.5,
            num_examples=10,
            metrics={"accuracy": 0.8},
        )
        eval_res2 = EvaluateRes(
            status=Status(code=Code.OK, message=""),
            loss=0.3,
            num_examples=30,
            metrics={"accuracy": 0.9},
        )

        results = [(mock_client1, eval_res1), (mock_client2, eval_res2)]
        failures = []

        loss_agg, metrics = strategy.aggregate_evaluate(1, results, failures)

        assert loss_agg is not None
        assert loss_agg == pytest.approx(0.35, abs=1e-6)


class TestServerFactory:
    """Tests for server factory function."""

    def test_create_fedprox_server_returns_server_config(self) -> None:
        """Test that create_fedprox_server returns a ServerConfig."""
        from flwr.server import ServerConfig

        config = FedProxServerConfig(num_rounds=5)
        server_config, strategy = create_fedprox_server(config)

        assert isinstance(server_config, ServerConfig)
        assert server_config.num_rounds == 5

    def test_create_fedprox_server_returns_strategy(self) -> None:
        """Test that create_fedprox_server returns a FedProxStrategy."""
        config = FedProxServerConfig()
        server_config, strategy = create_fedprox_server(config)

        assert isinstance(strategy, FedProxStrategy)

    def test_create_fedprox_server_applies_config(self) -> None:
        """Test that create_fedprox_server applies config to strategy."""
        config = FedProxServerConfig(
            min_fit_clients=5,
            min_evaluate_clients=3,
            min_available_clients=8,
            fraction_fit=0.7,
            fraction_evaluate=0.5,
        )
        server_config, strategy = create_fedprox_server(config)

        assert strategy.min_fit_clients == 5
        assert strategy.min_evaluate_clients == 3
        assert strategy.min_available_clients == 8
        assert strategy.fraction_fit == 0.7
        assert strategy.fraction_evaluate == 0.5

    def test_create_fedprox_server_with_initial_parameters(self) -> None:
        """Test create_fedprox_server with initial model parameters."""
        config = FedProxServerConfig()
        initial_params = [np.array([1.0, 2.0, 3.0])]

        server_config, strategy = create_fedprox_server(config, initial_parameters=initial_params)

        assert strategy.initial_parameters is not None
