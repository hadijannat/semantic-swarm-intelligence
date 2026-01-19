"""FastAPI application for NOA Semantic Swarm Mapper.

This module provides the REST API for the semantic mapping system,
including endpoints for:

- **Discovery**: OPC UA tag discovery and browsing
- **Mapping**: Tag-to-IRDI mapping operations
- **AAS**: Asset Administration Shell export
- **Swarm**: Swarm coordination and consensus
- **Federated**: Federated learning operations

Example usage:
    >>> import uvicorn
    >>> from noa_swarm.api import app
    >>> uvicorn.run(app, host="0.0.0.0", port=8000)
"""

from noa_swarm.api.main import app, get_api_info

__all__ = [
    "app",
    "get_api_info",
]
