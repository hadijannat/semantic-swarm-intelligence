"""Abstract protocol for dictionary providers.

This module defines the interface for industrial semantic dictionary providers
such as IEC 61987 CDD and eCl@ss. All dictionary providers must implement
the DictionaryProvider protocol to be used interchangeably in the system.

Key components:
- **DictionaryConcept**: A concept/property from a semantic dictionary
- **SearchResult**: A search match with relevance score
- **HierarchyNode**: A node in the concept hierarchy tree
- **DictionaryProvider**: Abstract protocol for providers
- **ProviderRegistry**: Registry for managing multiple providers

Example usage:
    >>> from noa_swarm.dictionaries.base import DictionaryProvider, ProviderRegistry
    >>>
    >>> class MyProvider(DictionaryProvider):
    ...     # Implement required methods
    ...     pass
    >>>
    >>> registry = ProviderRegistry()
    >>> registry.register(MyProvider())
    >>> provider = registry.get("my_provider")
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from noa_swarm.common.logging import get_logger

if TYPE_CHECKING:
    pass

logger = get_logger(__name__)


@dataclass
class DictionaryConcept:
    """A concept from a semantic dictionary.

    Represents a property, class, or value from an industrial dictionary
    like IEC 61987 CDD or eCl@ss.

    Attributes:
        irdi: International Registration Data Identifier. Unique identifier
            following ISO 29002-5 format (e.g., "0173-1#01-ABA234#001").
        preferred_name: The primary name for this concept.
        definition: Full textual definition of the concept.
        unit: Unit of measure if applicable (e.g., "°C", "bar", "m/s").
        data_type: Expected data type (e.g., "float", "integer", "string").
        alternate_names: Alternative names or synonyms for the concept.
        source: The dictionary source (e.g., "IEC CDD", "eCl@ss").
        version: Version of the dictionary entry.
    """

    irdi: str
    preferred_name: str
    definition: str | None = None
    unit: str | None = None
    data_type: str | None = None
    alternate_names: list[str] = field(default_factory=list)
    source: str | None = None
    version: str | None = None

    def __post_init__(self) -> None:
        """Validate concept fields."""
        if not self.irdi:
            raise ValueError("irdi cannot be empty")
        if not self.preferred_name:
            raise ValueError("preferred_name cannot be empty")


@dataclass
class SearchResult:
    """A search result from a dictionary query.

    Attributes:
        concept: The matched dictionary concept.
        score: Relevance score between 0.0 and 1.0 (1.0 = perfect match).
        match_type: Type of match (e.g., "exact", "prefix", "fuzzy").
        highlight: Optional highlighted text showing match location.
    """

    concept: DictionaryConcept
    score: float = 1.0
    match_type: str | None = None
    highlight: str | None = None

    def __post_init__(self) -> None:
        """Validate search result fields."""
        if not (0.0 <= self.score <= 1.0):
            raise ValueError(f"score must be between 0.0 and 1.0, got {self.score}")


@dataclass
class HierarchyNode:
    """A node in the dictionary concept hierarchy.

    Represents a concept's position in the taxonomy tree, including
    its parent and children relationships.

    Attributes:
        irdi: IRDI of this concept.
        preferred_name: Name of this concept.
        parent_irdi: IRDI of the parent concept (None if root).
        child_irdis: List of IRDIs for direct child concepts.
        depth: Depth in the hierarchy tree (0 = root).
    """

    irdi: str
    preferred_name: str
    parent_irdi: str | None = None
    child_irdis: list[str] = field(default_factory=list)
    depth: int = 0

    def __post_init__(self) -> None:
        """Validate hierarchy node fields."""
        if not self.irdi:
            raise ValueError("irdi cannot be empty")
        if not self.preferred_name:
            raise ValueError("preferred_name cannot be empty")
        if self.depth < 0:
            raise ValueError(f"depth must be non-negative, got {self.depth}")


class DictionaryProvider(ABC):
    """Abstract protocol for semantic dictionary providers.

    This protocol defines the interface that all dictionary providers must
    implement. Providers give access to industrial semantic dictionaries
    like IEC 61987 CDD (Common Data Dictionary) or eCl@ss.

    Implementations should handle:
    - Caching for performance
    - Error handling for network failures
    - Rate limiting for API-based providers

    Example implementation:
        >>> class IECCDDProvider(DictionaryProvider):
        ...     @property
        ...     def name(self) -> str:
        ...         return "iec_cdd"
        ...
        ...     async def lookup(self, irdi: str) -> DictionaryConcept | None:
        ...         # Fetch from IEC CDD API
        ...         pass
        ...
        ...     # ... implement other methods
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the unique name of this provider.

        Returns:
            Provider name (e.g., "iec_cdd", "eclass", "seed").
        """
        ...

    @abstractmethod
    async def lookup(self, irdi: str) -> DictionaryConcept | None:
        """Look up a concept by its IRDI.

        Args:
            irdi: The International Registration Data Identifier.

        Returns:
            The concept if found, None otherwise.
        """
        ...

    @abstractmethod
    async def search(
        self,
        query: str,
        max_results: int = 10,
    ) -> list[SearchResult]:
        """Search for concepts matching a query.

        Args:
            query: Search query string.
            max_results: Maximum number of results to return.

        Returns:
            List of search results ordered by relevance score.
        """
        ...

    @abstractmethod
    async def get_hierarchy(
        self,
        irdi: str,
        depth: int = 1,
    ) -> HierarchyNode | None:
        """Get the hierarchy information for a concept.

        Args:
            irdi: The IRDI of the concept.
            depth: How many levels of children to include (0 = none).

        Returns:
            Hierarchy node with parent and children, None if not found.
        """
        ...

    @abstractmethod
    async def is_available(self) -> bool:
        """Check if the provider is currently available.

        Returns:
            True if the provider can service requests, False otherwise.
        """
        ...


class ProviderRegistry:
    """Registry for managing dictionary providers.

    This class maintains a collection of dictionary providers and allows
    looking them up by name. It also provides convenience methods for
    querying multiple providers.

    Example:
        >>> registry = ProviderRegistry()
        >>> registry.register(IECCDDProvider())
        >>> registry.register(EClassProvider())
        >>> provider = registry.get("iec_cdd")
        >>> concept = await provider.lookup("0173-1#01-ABA234#001")
    """

    def __init__(self) -> None:
        """Initialize an empty provider registry."""
        self._providers: dict[str, DictionaryProvider] = {}

    @property
    def providers(self) -> dict[str, DictionaryProvider]:
        """Return the dictionary of registered providers."""
        return self._providers

    def register(self, provider: DictionaryProvider) -> None:
        """Register a dictionary provider.

        Args:
            provider: The provider to register.

        Raises:
            ValueError: If a provider with the same name is already registered.
        """
        name = provider.name
        if name in self._providers:
            raise ValueError(f"Provider '{name}' is already registered")

        self._providers[name] = provider
        logger.info("Registered dictionary provider", provider_name=name)

    def unregister(self, name: str) -> bool:
        """Unregister a dictionary provider.

        Args:
            name: Name of the provider to unregister.

        Returns:
            True if the provider was removed, False if not found.
        """
        if name in self._providers:
            del self._providers[name]
            logger.info("Unregistered dictionary provider", provider_name=name)
            return True
        return False

    def get(self, name: str) -> DictionaryProvider | None:
        """Get a provider by name.

        Args:
            name: Name of the provider.

        Returns:
            The provider if found, None otherwise.
        """
        return self._providers.get(name)

    def list_names(self) -> list[str]:
        """List names of all registered providers.

        Returns:
            List of provider names.
        """
        return list(self._providers.keys())

    async def lookup_any(self, irdi: str) -> DictionaryConcept | None:
        """Look up a concept in any available provider.

        Tries each provider in order until a match is found.

        Args:
            irdi: The IRDI to look up.

        Returns:
            The first matching concept, or None if not found.
        """
        for name, provider in self._providers.items():
            try:
                if await provider.is_available():
                    concept = await provider.lookup(irdi)
                    if concept is not None:
                        logger.debug(
                            "Found concept",
                            irdi=irdi,
                            provider=name,
                        )
                        return concept
            except Exception as e:
                logger.warning(
                    "Provider lookup failed",
                    provider=name,
                    irdi=irdi,
                    error=str(e),
                )
        return None

    async def search_all(
        self,
        query: str,
        max_results: int = 10,
    ) -> list[SearchResult]:
        """Search all available providers and merge results.

        Combines results from all providers, sorted by score.

        Args:
            query: Search query string.
            max_results: Maximum total results to return.

        Returns:
            Combined and sorted list of search results.
        """
        all_results: list[SearchResult] = []

        for name, provider in self._providers.items():
            try:
                if await provider.is_available():
                    results = await provider.search(query, max_results)
                    all_results.extend(results)
            except Exception as e:
                logger.warning(
                    "Provider search failed",
                    provider=name,
                    query=query,
                    error=str(e),
                )

        # Sort by score descending and limit results
        all_results.sort(key=lambda r: r.score, reverse=True)
        return all_results[:max_results]
