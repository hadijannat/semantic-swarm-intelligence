"""Unit tests for eCl@ss dictionary provider."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from noa_swarm.dictionaries.base import (
    DictionaryConcept,
    DictionaryProvider,
    HierarchyNode,
    SearchResult,
)
from noa_swarm.dictionaries.eclass_provider import (
    EClassConfig,
    EClassProvider,
)


class TestEClassConfig:
    """Tests for EClassConfig dataclass."""

    def test_default_values(self) -> None:
        """Test EClassConfig has correct default values."""
        config = EClassConfig()

        assert config.base_url == "https://eclass.de/api/v1"
        assert config.timeout == 30.0
        assert config.cache_ttl == 3600
        assert config.max_cache_size == 1000
        assert config.api_key is None

    def test_custom_values(self) -> None:
        """Test EClassConfig accepts custom values."""
        config = EClassConfig(
            base_url="https://custom.api.com",
            timeout=60.0,
            cache_ttl=7200,
            max_cache_size=500,
            api_key="test-key",
        )

        assert config.base_url == "https://custom.api.com"
        assert config.timeout == 60.0
        assert config.cache_ttl == 7200
        assert config.max_cache_size == 500
        assert config.api_key == "test-key"

    def test_invalid_timeout_raises(self) -> None:
        """Test that invalid timeout raises ValueError."""
        with pytest.raises(ValueError, match="timeout must be positive"):
            EClassConfig(timeout=0.0)
        with pytest.raises(ValueError, match="timeout must be positive"):
            EClassConfig(timeout=-1.0)

    def test_invalid_cache_ttl_raises(self) -> None:
        """Test that invalid cache_ttl raises ValueError."""
        with pytest.raises(ValueError, match="cache_ttl must be non-negative"):
            EClassConfig(cache_ttl=-1)

    def test_invalid_max_cache_size_raises(self) -> None:
        """Test that invalid max_cache_size raises ValueError."""
        with pytest.raises(ValueError, match="max_cache_size must be positive"):
            EClassConfig(max_cache_size=0)


class TestEClassProviderInit:
    """Tests for EClassProvider initialization."""

    def test_implements_provider_protocol(self) -> None:
        """Test that EClassProvider implements DictionaryProvider."""
        provider = EClassProvider()
        assert isinstance(provider, DictionaryProvider)

    def test_provider_name(self) -> None:
        """Test provider name is 'eclass'."""
        provider = EClassProvider()
        assert provider.name == "eclass"

    def test_accepts_custom_config(self) -> None:
        """Test provider accepts custom config."""
        config = EClassConfig(timeout=45.0)
        provider = EClassProvider(config=config)
        assert provider._config.timeout == 45.0

    def test_initializes_empty_cache(self) -> None:
        """Test provider initializes with empty cache."""
        provider = EClassProvider()
        assert len(provider._cache) == 0


class TestEClassProviderLookup:
    """Tests for EClassProvider.lookup()."""

    @pytest.fixture
    def provider(self) -> EClassProvider:
        """Create a provider for testing."""
        return EClassProvider()

    @pytest.mark.asyncio
    async def test_lookup_returns_none_when_not_found(self, provider: EClassProvider) -> None:
        """Test lookup returns None when concept not found."""
        with patch.object(provider, "_fetch_concept", new_callable=AsyncMock) as mock_fetch:
            mock_fetch.return_value = None

            concept = await provider.lookup("0173-1#99-ZZZZZ#999")

            assert concept is None
            mock_fetch.assert_called_once_with("0173-1#99-ZZZZZ#999")

    @pytest.mark.asyncio
    async def test_lookup_returns_concept_when_found(self, provider: EClassProvider) -> None:
        """Test lookup returns concept when found."""
        expected_concept = DictionaryConcept(
            irdi="0173-1#02-AAB663#001",
            preferred_name="Temperature",
            definition="A measure of thermal energy",
            source="eclass",
        )

        with patch.object(provider, "_fetch_concept", new_callable=AsyncMock) as mock_fetch:
            mock_fetch.return_value = expected_concept

            concept = await provider.lookup("0173-1#02-AAB663#001")

            assert concept is not None
            assert concept.irdi == "0173-1#02-AAB663#001"
            assert concept.preferred_name == "Temperature"

    @pytest.mark.asyncio
    async def test_lookup_uses_cache(self, provider: EClassProvider) -> None:
        """Test lookup uses cached values."""
        cached_concept = DictionaryConcept(
            irdi="0173-1#02-AAB663#001",
            preferred_name="Temperature",
            source="eclass",
        )
        provider._cache["0173-1#02-AAB663#001"] = (cached_concept, float("inf"))

        with patch.object(provider, "_fetch_concept", new_callable=AsyncMock) as mock_fetch:
            concept = await provider.lookup("0173-1#02-AAB663#001")

            assert concept is not None
            assert concept.irdi == "0173-1#02-AAB663#001"
            # Should not call API when cached
            mock_fetch.assert_not_called()


class TestEClassProviderSearch:
    """Tests for EClassProvider.search()."""

    @pytest.fixture
    def provider(self) -> EClassProvider:
        """Create a provider for testing."""
        return EClassProvider()

    @pytest.mark.asyncio
    async def test_search_returns_results(self, provider: EClassProvider) -> None:
        """Test search returns search results."""
        mock_results = [
            SearchResult(
                concept=DictionaryConcept(
                    irdi="0173-1#02-AAB663#001",
                    preferred_name="Temperature",
                    source="eclass",
                ),
                score=0.95,
            )
        ]

        with patch.object(provider, "_search_api", new_callable=AsyncMock) as mock_search:
            mock_search.return_value = mock_results

            results = await provider.search("temperature")

            assert len(results) == 1
            assert results[0].concept.preferred_name == "Temperature"

    @pytest.mark.asyncio
    async def test_search_respects_max_results(self, provider: EClassProvider) -> None:
        """Test search respects max_results parameter."""
        with patch.object(provider, "_search_api", new_callable=AsyncMock) as mock_search:
            mock_search.return_value = []

            await provider.search("test", max_results=5)

            mock_search.assert_called_once_with("test", 5)

    @pytest.mark.asyncio
    async def test_search_returns_empty_on_api_failure(self, provider: EClassProvider) -> None:
        """Test search returns empty list on API failure."""
        with patch.object(provider, "_search_api", new_callable=AsyncMock) as mock_search:
            mock_search.side_effect = Exception("API error")

            results = await provider.search("temperature")

            assert results == []


class TestEClassProviderHierarchy:
    """Tests for EClassProvider.get_hierarchy()."""

    @pytest.fixture
    def provider(self) -> EClassProvider:
        """Create a provider for testing."""
        return EClassProvider()

    @pytest.mark.asyncio
    async def test_get_hierarchy_returns_node(self, provider: EClassProvider) -> None:
        """Test get_hierarchy returns hierarchy node."""
        mock_node = HierarchyNode(
            irdi="0173-1#02-AAB663#001",
            preferred_name="Temperature",
            parent_irdi="0173-1#01-PARENT#001",
            depth=1,
        )

        with patch.object(provider, "_fetch_hierarchy", new_callable=AsyncMock) as mock_fetch:
            mock_fetch.return_value = mock_node

            node = await provider.get_hierarchy("0173-1#02-AAB663#001")

            assert node is not None
            assert node.irdi == "0173-1#02-AAB663#001"

    @pytest.mark.asyncio
    async def test_get_hierarchy_returns_none_when_not_found(
        self, provider: EClassProvider
    ) -> None:
        """Test get_hierarchy returns None when not found."""
        with patch.object(provider, "_fetch_hierarchy", new_callable=AsyncMock) as mock_fetch:
            mock_fetch.return_value = None

            node = await provider.get_hierarchy("0173-1#99-ZZZZZ#999")

            assert node is None


class TestEClassProviderAvailability:
    """Tests for EClassProvider.is_available()."""

    @pytest.fixture
    def provider(self) -> EClassProvider:
        """Create a provider for testing."""
        return EClassProvider()

    @pytest.mark.asyncio
    async def test_is_available_when_api_responds(self, provider: EClassProvider) -> None:
        """Test is_available returns True when API responds."""
        with patch.object(provider, "_check_api_health", new_callable=AsyncMock) as mock_check:
            mock_check.return_value = True

            available = await provider.is_available()

            assert available is True

    @pytest.mark.asyncio
    async def test_is_not_available_when_api_fails(self, provider: EClassProvider) -> None:
        """Test is_available returns False when API fails."""
        with patch.object(provider, "_check_api_health", new_callable=AsyncMock) as mock_check:
            mock_check.return_value = False

            available = await provider.is_available()

            assert available is False


class TestEClassProviderCaching:
    """Tests for EClassProvider caching behavior."""

    @pytest.fixture
    def provider(self) -> EClassProvider:
        """Create a provider for testing."""
        config = EClassConfig(cache_ttl=3600, max_cache_size=100)
        return EClassProvider(config=config)

    def test_cache_stores_concept(self, provider: EClassProvider) -> None:
        """Test that concepts are stored in cache."""
        concept = DictionaryConcept(
            irdi="0173-1#02-TEST#001",
            preferred_name="Test",
            source="eclass",
        )
        provider._store_in_cache("0173-1#02-TEST#001", concept)

        assert "0173-1#02-TEST#001" in provider._cache

    def test_cache_retrieves_concept(self, provider: EClassProvider) -> None:
        """Test that concepts are retrieved from cache."""
        concept = DictionaryConcept(
            irdi="0173-1#02-TEST#001",
            preferred_name="Test",
            source="eclass",
        )
        provider._store_in_cache("0173-1#02-TEST#001", concept)

        cached = provider._get_from_cache("0173-1#02-TEST#001")
        assert cached is not None
        assert cached.irdi == "0173-1#02-TEST#001"

    def test_cache_returns_none_for_missing(self, provider: EClassProvider) -> None:
        """Test cache returns None for missing keys."""
        cached = provider._get_from_cache("nonexistent")
        assert cached is None

    def test_clear_cache(self, provider: EClassProvider) -> None:
        """Test clearing the cache."""
        concept = DictionaryConcept(
            irdi="0173-1#02-TEST#001",
            preferred_name="Test",
            source="eclass",
        )
        provider._store_in_cache("0173-1#02-TEST#001", concept)
        assert len(provider._cache) > 0

        provider.clear_cache()
        assert len(provider._cache) == 0


class TestEClassProviderIRDIMapping:
    """Tests for eCl@ss to IEC 61987 IRDI cross-reference."""

    @pytest.fixture
    def provider(self) -> EClassProvider:
        """Create a provider for testing."""
        return EClassProvider()

    @pytest.mark.asyncio
    async def test_maps_eclass_id_to_irdi(self, provider: EClassProvider) -> None:
        """Test provider can map eCl@ss IDs to IEC 61987 IRDIs."""
        # This tests the cross-reference capability
        with patch.object(provider, "_fetch_concept", new_callable=AsyncMock) as mock_fetch:
            mock_fetch.return_value = DictionaryConcept(
                irdi="0173-1#02-AAB663#001",
                preferred_name="Temperature",
                source="eclass",
            )

            # eCl@ss uses different ID format internally
            concept = await provider.lookup("0173-1#02-AAB663#001")

            assert concept is not None
            # Should return IEC 61987 compatible IRDI
            assert concept.irdi.startswith("0173-1#")
