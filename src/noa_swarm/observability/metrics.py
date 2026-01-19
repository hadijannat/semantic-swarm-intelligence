"""Prometheus metrics for NOA Semantic Swarm Mapper.

This module provides metrics collection and export for:
- Swarm coordination (tags processed, consensus rounds)
- Federated learning (rounds, privacy budget)
- API requests (latency, errors)
- Agent status and reliability
"""

from __future__ import annotations

from typing import Any

from prometheus_client import (
    REGISTRY,
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)

from noa_swarm.common.logging import get_logger

logger = get_logger(__name__)

# Use a custom registry to avoid conflicts in tests
_registry: CollectorRegistry = REGISTRY

# =============================================================================
# Generic Metrics
# =============================================================================

# Counters (monotonically increasing values)
_counters: dict[str, Counter] = {}

# Histograms (distributions of values)
_histograms: dict[str, Histogram] = {}

# Gauges (values that can go up and down)
_gauges: dict[str, Gauge] = {}


def get_registry() -> CollectorRegistry:
    """Get the metrics registry.

    Returns:
        The Prometheus collector registry.
    """
    return _registry


def _get_or_create_counter(name: str, description: str, labels: list[str]) -> Counter:
    """Get or create a counter metric."""
    if name not in _counters:
        _counters[name] = Counter(
            name,
            description,
            labels,
            registry=_registry,
        )
    return _counters[name]


def _get_or_create_histogram(
    name: str,
    description: str,
    labels: list[str],
    buckets: tuple[float, ...] | None = None,
) -> Histogram:
    """Get or create a histogram metric."""
    if name not in _histograms:
        kwargs: dict[str, Any] = {
            "registry": _registry,
        }
        if buckets:
            kwargs["buckets"] = buckets
        _histograms[name] = Histogram(
            name,
            description,
            labels,
            **kwargs,
        )
    return _histograms[name]


def _get_or_create_gauge(name: str, description: str, labels: list[str] | None = None) -> Gauge:
    """Get or create a gauge metric."""
    if name not in _gauges:
        _gauges[name] = Gauge(
            name,
            description,
            labels or [],
            registry=_registry,
        )
    return _gauges[name]


def increment_counter(name: str, labels: dict[str, str] | None = None) -> None:
    """Increment a counter metric.

    Args:
        name: Name of the counter.
        labels: Optional label values.
    """
    label_names = list(labels.keys()) if labels else []
    counter = _get_or_create_counter(name, f"Counter: {name}", label_names)
    if labels:
        counter.labels(**labels).inc()
    else:
        counter.inc()


def observe_histogram(
    name: str,
    value: float,
    labels: dict[str, str] | None = None,
) -> None:
    """Observe a value in a histogram metric.

    Args:
        name: Name of the histogram.
        value: Value to observe.
        labels: Optional label values.
    """
    label_names = list(labels.keys()) if labels else []
    histogram = _get_or_create_histogram(name, f"Histogram: {name}", label_names)
    if labels:
        histogram.labels(**labels).observe(value)
    else:
        histogram.observe(value)


def set_gauge(name: str, value: float, labels: dict[str, str] | None = None) -> None:
    """Set a gauge metric value.

    Args:
        name: Name of the gauge.
        value: Value to set.
        labels: Optional label values.
    """
    label_names = list(labels.keys()) if labels else []
    gauge = _get_or_create_gauge(name, f"Gauge: {name}", label_names or None)
    if labels:
        gauge.labels(**labels).set(value)
    else:
        gauge.set(value)


def generate_metrics_output() -> bytes:
    """Generate Prometheus text format output.

    Returns:
        Metrics in Prometheus text format.
    """
    return generate_latest(_registry)


# =============================================================================
# Swarm Metrics
# =============================================================================

# Tag processing metrics
_tags_processed = Counter(
    "noa_tags_processed_total",
    "Total number of tags processed",
    ["status"],
    registry=_registry,
)

_tag_confidence = Histogram(
    "noa_tag_confidence",
    "Confidence scores for tag mappings",
    ["status"],
    buckets=(0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.99, 1.0),
    registry=_registry,
)

# Consensus metrics
_consensus_rounds = Counter(
    "noa_consensus_rounds_total",
    "Total number of consensus rounds",
    registry=_registry,
)

_consensus_duration = Histogram(
    "noa_consensus_duration_seconds",
    "Duration of consensus rounds",
    buckets=(0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0),
    registry=_registry,
)

_consensus_agreement = Histogram(
    "noa_consensus_agreement_ratio",
    "Agreement ratio in consensus rounds",
    buckets=(0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0),
    registry=_registry,
)

_consensus_participants = Histogram(
    "noa_consensus_participants",
    "Number of participants in consensus rounds",
    buckets=(1, 2, 3, 5, 10, 20, 50),
    registry=_registry,
)

# Agent metrics
_agent_status = Gauge(
    "noa_agent_active",
    "Whether agent is active (1) or inactive (0)",
    ["agent_id"],
    registry=_registry,
)

_agent_reliability = Gauge(
    "noa_agent_reliability",
    "Agent reliability score",
    ["agent_id"],
    registry=_registry,
)


def record_tag_processed(tag_name: str, status: str, confidence: float) -> None:
    """Record a processed tag metric.

    Args:
        tag_name: Name of the tag.
        status: Processing status (mapped, pending, rejected, etc.).
        confidence: Confidence score (0-1).
    """
    _tags_processed.labels(status=status).inc()
    _tag_confidence.labels(status=status).observe(confidence)
    logger.debug("Tag processed", tag_name=tag_name, status=status, confidence=confidence)


def record_consensus_round(
    tag_name: str,
    round_number: int,
    participating_agents: int,
    agreement_ratio: float,
    duration_seconds: float,
) -> None:
    """Record a consensus round metric.

    Args:
        tag_name: Name of the tag.
        round_number: Round number.
        participating_agents: Number of participating agents.
        agreement_ratio: Agreement ratio (0-1).
        duration_seconds: Duration of the round.
    """
    _consensus_rounds.inc()
    _consensus_duration.observe(duration_seconds)
    _consensus_agreement.observe(agreement_ratio)
    _consensus_participants.observe(participating_agents)
    logger.debug(
        "Consensus round completed",
        tag_name=tag_name,
        round=round_number,
        participants=participating_agents,
        agreement=agreement_ratio,
        duration=duration_seconds,
    )


def record_agent_status(agent_id: str, status: str, reliability: float) -> None:
    """Record agent status metric.

    Args:
        agent_id: ID of the agent.
        status: Agent status (active, inactive, etc.).
        reliability: Reliability score (0-1).
    """
    _agent_status.labels(agent_id=agent_id).set(1 if status == "active" else 0)
    _agent_reliability.labels(agent_id=agent_id).set(reliability)
    logger.debug("Agent status updated", agent_id=agent_id, status=status, reliability=reliability)


# =============================================================================
# Federated Learning Metrics
# =============================================================================

_fl_rounds = Counter(
    "noa_fl_rounds_total",
    "Total number of federated learning rounds",
    registry=_registry,
)

_fl_round_duration = Histogram(
    "noa_fl_round_duration_seconds",
    "Duration of federated learning rounds",
    buckets=(10, 30, 60, 120, 300, 600, 1800),
    registry=_registry,
)

_fl_clients = Histogram(
    "noa_fl_clients_participating",
    "Number of clients participating in FL rounds",
    buckets=(1, 2, 3, 5, 10, 20, 50),
    registry=_registry,
)

_fl_accuracy = Gauge(
    "noa_fl_model_accuracy",
    "Current federated model accuracy",
    registry=_registry,
)

_fl_privacy_used = Gauge(
    "noa_fl_privacy_epsilon_used",
    "Epsilon budget used for differential privacy",
    registry=_registry,
)

_fl_privacy_budget = Gauge(
    "noa_fl_privacy_epsilon_budget",
    "Total epsilon budget for differential privacy",
    registry=_registry,
)

_fl_privacy_rounds_remaining = Gauge(
    "noa_fl_privacy_rounds_remaining",
    "Rounds remaining before privacy budget exhausted",
    registry=_registry,
)


def record_fl_round(
    round_number: int,
    participating_clients: int,
    duration_seconds: float,
    model_accuracy: float,
) -> None:
    """Record federated learning round metric.

    Args:
        round_number: Round number.
        participating_clients: Number of participating clients.
        duration_seconds: Duration of the round.
        model_accuracy: Model accuracy after the round.
    """
    _fl_rounds.inc()
    _fl_round_duration.observe(duration_seconds)
    _fl_clients.observe(participating_clients)
    _fl_accuracy.set(model_accuracy)
    logger.debug(
        "FL round completed",
        round=round_number,
        clients=participating_clients,
        duration=duration_seconds,
        accuracy=model_accuracy,
    )


def record_privacy_budget(
    epsilon_used: float,
    epsilon_budget: float,
    rounds_remaining: int,
) -> None:
    """Record privacy budget metric.

    Args:
        epsilon_used: Epsilon budget used so far.
        epsilon_budget: Total epsilon budget.
        rounds_remaining: Estimated rounds remaining.
    """
    _fl_privacy_used.set(epsilon_used)
    _fl_privacy_budget.set(epsilon_budget)
    _fl_privacy_rounds_remaining.set(rounds_remaining)
    logger.debug(
        "Privacy budget updated",
        epsilon_used=epsilon_used,
        epsilon_budget=epsilon_budget,
        rounds_remaining=rounds_remaining,
    )


# =============================================================================
# API Metrics
# =============================================================================

_api_requests = Counter(
    "noa_api_requests_total",
    "Total number of API requests",
    ["method", "endpoint", "status_code"],
    registry=_registry,
)

_api_request_duration = Histogram(
    "noa_api_request_duration_seconds",
    "Duration of API requests",
    ["method", "endpoint"],
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5),
    registry=_registry,
)

_api_errors = Counter(
    "noa_api_errors_total",
    "Total number of API errors",
    ["method", "endpoint", "error_type"],
    registry=_registry,
)


def record_api_request(
    method: str,
    endpoint: str,
    status_code: int,
    duration_seconds: float,
) -> None:
    """Record API request metric.

    Args:
        method: HTTP method.
        endpoint: API endpoint path.
        status_code: Response status code.
        duration_seconds: Request duration.
    """
    _api_requests.labels(
        method=method,
        endpoint=endpoint,
        status_code=str(status_code),
    ).inc()
    _api_request_duration.labels(
        method=method,
        endpoint=endpoint,
    ).observe(duration_seconds)


def record_api_error(
    method: str,
    endpoint: str,
    error_type: str,
) -> None:
    """Record API error metric.

    Args:
        method: HTTP method.
        endpoint: API endpoint path.
        error_type: Type of error.
    """
    _api_errors.labels(
        method=method,
        endpoint=endpoint,
        error_type=error_type,
    ).inc()
    logger.warning(
        "API error recorded",
        method=method,
        endpoint=endpoint,
        error_type=error_type,
    )
