"""Federated learning components for NOA Swarm.

This package contains federated learning implementations including:
- FedProx client for local training with proximal term regularization
- Integration with Flower (flwr) federated learning framework
"""

from noa_swarm.federated.flower_client import FedProxClient, FedProxConfig

__all__ = [
    "FedProxClient",
    "FedProxConfig",
]
