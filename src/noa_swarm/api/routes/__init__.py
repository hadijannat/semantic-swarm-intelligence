"""API route modules.

This package contains the route handlers for each API domain:
- discovery: OPC UA tag discovery
- mapping: Tag-to-IRDI mapping
- aas: Asset Administration Shell export
- swarm: Swarm coordination
- federated: Federated learning
"""

from noa_swarm.api.routes import aas, discovery, federated, mapping, swarm

__all__ = [
    "discovery",
    "mapping",
    "aas",
    "swarm",
    "federated",
]
