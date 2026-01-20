"""FastAPI dependency providers for shared services."""

from __future__ import annotations

from typing import TYPE_CHECKING

from noa_swarm.services.state import get_state

if TYPE_CHECKING:
    from noa_swarm.services import (
        AASService,
        DiscoveryService,
        FederatedService,
        MappingService,
        SwarmService,
    )


def get_discovery_service() -> DiscoveryService:
    return get_state().discovery


def get_mapping_service() -> MappingService:
    return get_state().mapping


def get_aas_service() -> AASService:
    return get_state().aas


def get_swarm_service() -> SwarmService:
    return get_state().swarm


def get_federated_service() -> FederatedService:
    return get_state().federated
