"""eCl@ss dictionary provider.

This module provides access to the eCl@ss product classification system,
with cross-references to IEC 61987 IRDIs. eCl@ss is an industry standard
for classifying products and services.

Features:
- Async HTTP requests using httpx
- TTL-based caching to reduce API calls
- Cross-reference mapping to IEC 61987 IRDIs
- Graceful degradation on API failures

Note: Access to eCl@ss API may require registration and API key.

Example usage:
    >>> from noa_swarm.dictionaries.eclass_provider import EClassProvider
    >>> provider = EClassProvider()
    >>> if await provider.is_available():
    ...     concept = await provider.lookup("0173-1#02-AAB663#001")
    ...     print(concept.preferred_name)

References:
    eCl@ss - https://eclass.eu/
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import httpx

from noa_swarm.common.logging import get_logger
from noa_swarm.dictionaries.base import (
    DictionaryConcept,
    DictionaryProvider,
    HierarchyNode,
    SearchResult,
)

logger = get_logger(__name__)


@dataclass
class EClassConfig:
    """Configuration for eCl@ss provider.

    Attributes:
        base_url: Base URL for the eCl@ss API.
        timeout: HTTP request timeout in seconds.
        cache_ttl: Time-to-live for cached entries in seconds.
        max_cache_size: Maximum number of entries in the cache.
        api_key: Optional API key for authenticated access.
    """

    base_url: str = "https://eclass.de/api/v1"
    timeout: float = 30.0
    cache_ttl: int = 3600  # 1 hour
    max_cache_size: int = 1000
    api_key: str | None = None

    def __post_init__(self) -> None:
        """Validate configuration parameters."""
        if self.timeout <= 0:
            raise ValueError(f"timeout must be positive, got {self.timeout}")
        if self.cache_ttl < 0:
            raise ValueError(f"cache_ttl must be non-negative, got {self.cache_ttl}")
        if self.max_cache_size <= 0:
            raise ValueError(f"max_cache_size must be positive, got {self.max_cache_size}")


class EClassProvider(DictionaryProvider):
    """Dictionary provider for eCl@ss product classification.

    This provider accesses the eCl@ss API to retrieve product and
    property classifications with cross-references to IEC 61987 IRDIs.
    It includes an in-memory cache to reduce API calls.

    The eCl@ss system uses a hierarchical structure:
    - Segments (2-digit)
    - Main groups (4-digit)
    - Groups (6-digit)
    - Commodity classes (8-digit)
    - Properties

    Attributes:
        _config: Provider configuration.
        _cache: In-memory cache mapping IRDIs to (concept, expiry) tuples.
        _client: HTTP client for API requests.
    """

    def __init__(self, config: EClassConfig | None = None) -> None:
        """Initialize the eCl@ss provider.

        Args:
            config: Optional configuration. Uses defaults if not provided.
        """
        self._config = config or EClassConfig()
        self._cache: dict[str, tuple[DictionaryConcept, float]] = {}
        self._client: httpx.AsyncClient | None = None

        logger.info(
            "Initialized EClassProvider",
            base_url=self._config.base_url,
            cache_ttl=self._config.cache_ttl,
            has_api_key=self._config.api_key is not None,
        )

    @property
    def name(self) -> str:
        """Return the provider name."""
        return "eclass"

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create the HTTP client.

        Returns:
            Configured async HTTP client.
        """
        if self._client is None:
            headers = {
                "Accept": "application/json",
                "User-Agent": "NOA-Swarm/1.0",
            }
            if self._config.api_key:
                headers["Authorization"] = f"Bearer {self._config.api_key}"

            self._client = httpx.AsyncClient(
                timeout=self._config.timeout,
                headers=headers,
            )
        return self._client

    async def close(self) -> None:
        """Close the HTTP client."""
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    def _get_from_cache(self, irdi: str) -> DictionaryConcept | None:
        """Get a concept from cache if not expired.

        Args:
            irdi: The IRDI to look up.

        Returns:
            Cached concept if valid, None otherwise.
        """
        if irdi not in self._cache:
            return None

        concept, expiry = self._cache[irdi]
        if time.time() > expiry:
            # Expired - remove from cache
            del self._cache[irdi]
            return None

        return concept

    def _store_in_cache(self, irdi: str, concept: DictionaryConcept) -> None:
        """Store a concept in the cache.

        If cache is at capacity, removes oldest entries.

        Args:
            irdi: The IRDI key.
            concept: The concept to cache.
        """
        # Evict oldest if at capacity
        while len(self._cache) >= self._config.max_cache_size:
            # Remove oldest entry (first inserted)
            oldest_key = next(iter(self._cache))
            del self._cache[oldest_key]

        expiry = time.time() + self._config.cache_ttl
        self._cache[irdi] = (concept, expiry)

    def clear_cache(self) -> None:
        """Clear all cached entries."""
        self._cache.clear()
        logger.debug("Cleared eCl@ss cache")

    async def _fetch_concept(self, irdi: str) -> DictionaryConcept | None:
        """Fetch a concept from the eCl@ss API.

        Args:
            irdi: The IRDI to fetch.

        Returns:
            The concept if found, None otherwise.
        """
        try:
            client = await self._get_client()

            # eCl@ss API endpoint format (simplified for implementation)
            url = f"{self._config.base_url}/properties/{irdi}"

            response = await client.get(url)

            if response.status_code == 404:
                return None

            response.raise_for_status()
            data = response.json()

            concept = self._parse_concept(data, irdi)
            return concept

        except httpx.HTTPStatusError as e:
            logger.warning(
                "HTTP error fetching concept from eCl@ss",
                irdi=irdi,
                status_code=e.response.status_code,
            )
            return None

        except (httpx.RequestError, Exception) as e:
            logger.warning(
                "Error fetching concept from eCl@ss",
                irdi=irdi,
                error=str(e),
            )
            return None

    def _parse_concept(self, data: dict[str, Any], irdi: str) -> DictionaryConcept:
        """Parse API response into a DictionaryConcept.

        Maps eCl@ss response format to IEC 61987-compatible structure.

        Args:
            data: JSON response data.
            irdi: The IRDI being looked up.

        Returns:
            Parsed DictionaryConcept with IEC 61987-compatible IRDI.
        """
        # Map eCl@ss fields to our concept structure
        # eCl@ss may use different field names
        preferred_name = (
            data.get("preferredName") or data.get("name") or data.get("label") or "Unknown"
        )

        return DictionaryConcept(
            irdi=irdi,
            preferred_name=preferred_name,
            definition=data.get("definition") or data.get("description"),
            unit=data.get("unit") or data.get("unitOfMeasure"),
            data_type=data.get("dataType") or data.get("valueFormat"),
            alternate_names=data.get("alternateNames", data.get("synonyms", [])),
            source="eclass",
            version=data.get("version") or data.get("release"),
        )

    async def _search_api(
        self,
        query: str,
        max_results: int,
    ) -> list[SearchResult]:
        """Search the eCl@ss API.

        Args:
            query: Search query string.
            max_results: Maximum number of results.

        Returns:
            List of search results.
        """
        try:
            client = await self._get_client()

            # eCl@ss search endpoint (simplified)
            url = f"{self._config.base_url}/search"
            params: dict[str, str | int] = {"q": query, "limit": max_results}

            response = await client.get(url, params=params)
            response.raise_for_status()
            data = response.json()

            results: list[SearchResult] = []
            for item in data.get("results", data.get("items", [])):
                irdi = item.get("irdi") or item.get("id", "")
                concept = self._parse_concept(item, irdi)
                score = item.get("score", item.get("relevance", 0.5))
                results.append(
                    SearchResult(
                        concept=concept,
                        score=min(max(score, 0.0), 1.0),
                        match_type=item.get("matchType", "partial"),
                    )
                )

            return results

        except Exception as e:
            logger.warning(
                "Error searching eCl@ss",
                query=query,
                error=str(e),
            )
            return []

    async def _fetch_hierarchy(
        self,
        irdi: str,
        depth: int,
    ) -> HierarchyNode | None:
        """Fetch hierarchy information from the API.

        Args:
            irdi: The IRDI of the concept.
            depth: How many levels of children to include.

        Returns:
            Hierarchy node if found, None otherwise.
        """
        try:
            client = await self._get_client()

            url = f"{self._config.base_url}/hierarchy/{irdi}"
            params: dict[str, int] = {"depth": depth}

            response = await client.get(url, params=params)

            if response.status_code == 404:
                return None

            response.raise_for_status()
            data = response.json()

            return HierarchyNode(
                irdi=irdi,
                preferred_name=data.get("preferredName", data.get("name", "Unknown")),
                parent_irdi=data.get("parentIrdi") or data.get("parent"),
                child_irdis=data.get("childIrdis", data.get("children", [])),
                depth=data.get("depth", data.get("level", 0)),
            )

        except Exception as e:
            logger.warning(
                "Error fetching hierarchy from eCl@ss",
                irdi=irdi,
                error=str(e),
            )
            return None

    async def _check_api_health(self) -> bool:
        """Check if the eCl@ss API is available.

        Returns:
            True if API responds, False otherwise.
        """
        try:
            client = await self._get_client()
            response = await client.head(self._config.base_url, timeout=5.0)
            return response.status_code < 500

        except Exception:
            return False

    async def lookup(self, irdi: str) -> DictionaryConcept | None:
        """Look up a concept by its IRDI.

        First checks the cache, then fetches from API if not cached.

        Args:
            irdi: The International Registration Data Identifier.

        Returns:
            The concept if found, None otherwise.
        """
        # Check cache first
        cached = self._get_from_cache(irdi)
        if cached is not None:
            logger.debug("Cache hit for IRDI", irdi=irdi)
            return cached

        # Fetch from API
        concept = await self._fetch_concept(irdi)

        # Cache result if found
        if concept is not None:
            self._store_in_cache(irdi, concept)

        return concept

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
        try:
            return await self._search_api(query, max_results)
        except Exception as e:
            logger.warning(
                "Search failed",
                query=query,
                error=str(e),
            )
            return []

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
        return await self._fetch_hierarchy(irdi, depth)

    async def is_available(self) -> bool:
        """Check if the provider is currently available.

        Returns:
            True if the API can service requests, False otherwise.
        """
        return await self._check_api_health()
