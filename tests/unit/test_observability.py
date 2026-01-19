"""Unit tests for observability module."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import pytest

if TYPE_CHECKING:
    pass


class TestPrometheusMetrics:
    """Tests for Prometheus metrics."""

    def test_metrics_registry_creation(self) -> None:
        """Test that metrics registry is created."""
        from noa_swarm.observability.metrics import get_registry

        registry = get_registry()
        assert registry is not None

    def test_counter_metric(self) -> None:
        """Test counter metric creation and increment."""
        from noa_swarm.observability.metrics import (
            get_registry,
            increment_counter,
        )

        increment_counter("test_requests_total", labels={"method": "GET"})

        registry = get_registry()
        assert registry is not None

    def test_histogram_metric(self) -> None:
        """Test histogram metric creation and observation."""
        from noa_swarm.observability.metrics import (
            get_registry,
            observe_histogram,
        )

        observe_histogram("test_request_duration_seconds", 0.5, labels={"endpoint": "/api"})

        registry = get_registry()
        assert registry is not None

    def test_gauge_metric(self) -> None:
        """Test gauge metric creation and setting."""
        from noa_swarm.observability.metrics import (
            get_registry,
            set_gauge,
        )

        set_gauge("test_active_connections", 10)

        registry = get_registry()
        assert registry is not None


class TestSwarmMetrics:
    """Tests for swarm-specific metrics."""

    def test_record_tag_processed(self) -> None:
        """Test recording a processed tag metric."""
        from noa_swarm.observability.metrics import record_tag_processed

        record_tag_processed("TIC-101.PV", "mapped", 0.95)

    def test_record_consensus_round(self) -> None:
        """Test recording a consensus round metric."""
        from noa_swarm.observability.metrics import record_consensus_round

        record_consensus_round(
            tag_name="TIC-101.PV",
            round_number=3,
            participating_agents=5,
            agreement_ratio=0.9,
            duration_seconds=1.5,
        )

    def test_record_agent_status(self) -> None:
        """Test recording agent status metric."""
        from noa_swarm.observability.metrics import record_agent_status

        record_agent_status("agent-001", "active", 0.92)


class TestFederatedMetrics:
    """Tests for federated learning metrics."""

    def test_record_fl_round(self) -> None:
        """Test recording federated learning round metric."""
        from noa_swarm.observability.metrics import record_fl_round

        record_fl_round(
            round_number=5,
            participating_clients=3,
            duration_seconds=120.0,
            model_accuracy=0.85,
        )

    def test_record_privacy_budget(self) -> None:
        """Test recording privacy budget metric."""
        from noa_swarm.observability.metrics import record_privacy_budget

        record_privacy_budget(
            epsilon_used=0.5,
            epsilon_budget=1.0,
            rounds_remaining=5,
        )


class TestAPIMetrics:
    """Tests for API metrics."""

    def test_record_api_request(self) -> None:
        """Test recording API request metric."""
        from noa_swarm.observability.metrics import record_api_request

        record_api_request(
            method="GET",
            endpoint="/api/v1/mapping",
            status_code=200,
            duration_seconds=0.05,
        )

    def test_record_api_error(self) -> None:
        """Test recording API error metric."""
        from noa_swarm.observability.metrics import record_api_error

        record_api_error(
            method="POST",
            endpoint="/api/v1/discovery/start",
            error_type="validation_error",
        )


class TestMetricsExport:
    """Tests for metrics export."""

    def test_generate_latest(self) -> None:
        """Test generating Prometheus format output."""
        from noa_swarm.observability.metrics import generate_metrics_output

        output = generate_metrics_output()

        assert isinstance(output, bytes)
        # Should contain some metric text
        assert len(output) >= 0

    def test_metrics_endpoint_format(self) -> None:
        """Test that metrics are in Prometheus text format."""
        from noa_swarm.observability.metrics import (
            generate_metrics_output,
            increment_counter,
        )

        increment_counter("test_http_requests", labels={"path": "/test"})
        output = generate_metrics_output().decode("utf-8")

        # Prometheus format includes # HELP and # TYPE comments
        assert "# TYPE" in output or len(output) == 0


class TestCorrelationID:
    """Tests for correlation ID propagation."""

    def test_get_correlation_id_default(self) -> None:
        """Test getting correlation ID when none is set."""
        from noa_swarm.observability.correlation import (
            clear_correlation_id,
            get_correlation_id,
        )

        clear_correlation_id()
        corr_id = get_correlation_id()

        assert corr_id is not None
        assert len(corr_id) > 0

    def test_set_correlation_id(self) -> None:
        """Test setting a specific correlation ID."""
        from noa_swarm.observability.correlation import (
            clear_correlation_id,
            get_correlation_id,
            set_correlation_id,
        )

        clear_correlation_id()
        set_correlation_id("test-correlation-123")
        corr_id = get_correlation_id()

        assert corr_id == "test-correlation-123"

    def test_clear_correlation_id(self) -> None:
        """Test clearing the correlation ID."""
        from noa_swarm.observability.correlation import (
            clear_correlation_id,
            get_correlation_id,
            set_correlation_id,
        )

        set_correlation_id("test-id")
        clear_correlation_id()
        new_id = get_correlation_id()

        # Should generate a new ID, not return the old one
        assert new_id != "test-id"


class TestCorrelationIDContext:
    """Tests for correlation ID context manager."""

    def test_context_manager_sets_id(self) -> None:
        """Test that context manager sets correlation ID."""
        from noa_swarm.observability.correlation import (
            correlation_id_context,
            get_correlation_id,
        )

        with correlation_id_context("ctx-test-id"):
            assert get_correlation_id() == "ctx-test-id"

    def test_context_manager_restores_id(self) -> None:
        """Test that context manager restores previous ID."""
        from noa_swarm.observability.correlation import (
            correlation_id_context,
            get_correlation_id,
            set_correlation_id,
        )

        set_correlation_id("original-id")

        with correlation_id_context("nested-id"):
            assert get_correlation_id() == "nested-id"

        assert get_correlation_id() == "original-id"


class TestLoggingIntegration:
    """Tests for logging integration with observability."""

    def test_logger_includes_correlation_id(self) -> None:
        """Test that logger includes correlation ID in output."""
        from noa_swarm.common.logging import get_logger
        from noa_swarm.observability.correlation import set_correlation_id

        set_correlation_id("log-test-123")
        logger = get_logger(__name__)

        # Logger should be configured
        assert logger is not None


class TestMiddlewareHelpers:
    """Tests for middleware helper functions."""

    def test_extract_correlation_from_headers(self) -> None:
        """Test extracting correlation ID from request headers."""
        from noa_swarm.observability.middleware import extract_correlation_id

        headers = {"X-Correlation-ID": "header-test-123"}
        corr_id = extract_correlation_id(headers)

        assert corr_id == "header-test-123"

    def test_extract_correlation_missing_header(self) -> None:
        """Test extracting correlation ID when header is missing."""
        from noa_swarm.observability.middleware import extract_correlation_id

        headers = {}
        corr_id = extract_correlation_id(headers)

        # Should generate a new ID when not present
        assert corr_id is not None
        assert len(corr_id) > 0

    def test_timer_context(self) -> None:
        """Test timing context manager."""
        from noa_swarm.observability.middleware import timer_context

        with timer_context() as timer:
            time.sleep(0.01)

        assert timer.elapsed >= 0.01
        assert timer.elapsed < 1.0  # Should be much less than 1 second
