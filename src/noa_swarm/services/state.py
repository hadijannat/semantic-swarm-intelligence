"""Application service registry and shared state."""

from __future__ import annotations

from dataclasses import dataclass

from noa_swarm.dictionaries import ProviderRegistry, SeedDictionaryProvider
from noa_swarm.services.aas import AASService
from noa_swarm.services.discovery import DiscoveryService
from noa_swarm.services.federated import FederatedService
from noa_swarm.services.mapping import MappingService
from noa_swarm.services.swarm import SwarmService
from noa_swarm.storage import (
    Database,
    DatabaseConfig,
    InMemoryConsensusRepository,
    InMemoryMappingRepository,
    InMemoryTagRepository,
    SqlAlchemyConsensusRepository,
    SqlAlchemyMappingRepository,
    SqlAlchemyTagRepository,
)
from noa_swarm.storage.base import ConsensusRepository, MappingRepository, TagRepository


@dataclass
class AppState:
    """Shared application state for API services."""

    tags: TagRepository
    mappings: MappingRepository
    consensus: ConsensusRepository
    dictionaries: ProviderRegistry
    discovery: DiscoveryService
    mapping: MappingService
    aas: AASService
    swarm: SwarmService
    federated: FederatedService
    database: Database | None = None


_STATE: AppState | None = None


def _create_dictionary_registry() -> ProviderRegistry:
    registry = ProviderRegistry()
    registry.register(SeedDictionaryProvider())
    return registry


def get_state() -> AppState:
    """Return the shared AppState singleton."""
    global _STATE
    if _STATE is None:
        import os

        registry = _create_dictionary_registry()
        database_url = os.getenv("DATABASE_URL")

        database: Database | None = None
        if database_url:
            database = Database(DatabaseConfig(url=database_url))
            tag_repo = SqlAlchemyTagRepository(database.sessionmaker)
            mapping_repo = SqlAlchemyMappingRepository(database.sessionmaker)
            consensus_repo = SqlAlchemyConsensusRepository(database.sessionmaker)
        else:
            tag_repo = InMemoryTagRepository()
            mapping_repo = InMemoryMappingRepository()
            consensus_repo = InMemoryConsensusRepository()

        discovery = DiscoveryService(tag_repo, mapping_repo)
        mapping = MappingService(mapping_repo, tag_repo, registry)
        aas = AASService(mapping_repo)
        swarm = SwarmService(consensus_repo)
        federated = FederatedService()

        _STATE = AppState(
            tags=tag_repo,
            mappings=mapping_repo,
            consensus=consensus_repo,
            dictionaries=registry,
            discovery=discovery,
            mapping=mapping,
            aas=aas,
            swarm=swarm,
            federated=federated,
            database=database,
        )
    return _STATE
