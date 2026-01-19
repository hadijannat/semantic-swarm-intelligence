"""UI module for NOA Semantic Swarm Mapper.

This module provides the Gradio-based dashboard for:
- Monitoring tag discovery and mapping progress
- Reviewing and approving mappings
- Monitoring swarm agent status
- Exporting AAS packages
- Managing federated learning
"""

from noa_swarm.ui.gradio_app import (
    APIClient,
    create_dashboard,
    format_agents,
    format_candidates,
    format_consensus,
    format_metrics,
    format_tags_table,
    format_training_status,
    get_export_formats,
    validate_export_config,
)

__all__ = [
    "APIClient",
    "create_dashboard",
    "format_agents",
    "format_candidates",
    "format_consensus",
    "format_metrics",
    "format_tags_table",
    "format_training_status",
    "get_export_formats",
    "validate_export_config",
]
