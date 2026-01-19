"""Observability module for NOA Semantic Swarm Mapper.

This module provides:
- Prometheus metrics for monitoring
- Correlation ID propagation for distributed tracing
- Middleware helpers for FastAPI integration
"""

from noa_swarm.observability.correlation import (
    clear_correlation_id,
    correlation_id_context,
    get_correlation_id,
    set_correlation_id,
)
from noa_swarm.observability.metrics import (
    generate_metrics_output,
    get_registry,
    increment_counter,
    observe_histogram,
    record_agent_status,
    record_api_error,
    record_api_request,
    record_consensus_round,
    record_fl_round,
    record_privacy_budget,
    record_tag_processed,
    set_gauge,
)
from noa_swarm.observability.middleware import (
    extract_correlation_id,
    timer_context,
)

__all__ = [
    # Correlation
    "clear_correlation_id",
    "correlation_id_context",
    "get_correlation_id",
    "set_correlation_id",
    # Metrics
    "generate_metrics_output",
    "get_registry",
    "increment_counter",
    "observe_histogram",
    "record_agent_status",
    "record_api_error",
    "record_api_request",
    "record_consensus_round",
    "record_fl_round",
    "record_privacy_budget",
    "record_tag_processed",
    "set_gauge",
    # Middleware
    "extract_correlation_id",
    "timer_context",
]
