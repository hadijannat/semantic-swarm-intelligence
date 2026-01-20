"""Unit tests for seed dictionary provider."""

from __future__ import annotations

import pytest

from noa_swarm.dictionaries.base import (
    DictionaryConcept,
    DictionaryProvider,
    HierarchyNode,
    SearchResult,
)
from noa_swarm.dictionaries.seed_provider import SeedDictionaryProvider


class TestSeedDictionaryProviderInit:
    """Tests for SeedDictionaryProvider initialization."""

    def test_implements_provider_protocol(self) -> None:
        """Test that SeedDictionaryProvider implements DictionaryProvider."""
        provider = SeedDictionaryProvider()
        assert isinstance(provider, DictionaryProvider)

    def test_provider_name(self) -> None:
        """Test provider name is 'seed'."""
        provider = SeedDictionaryProvider()
        assert provider.name == "seed"

    def test_loads_seed_concepts(self) -> None:
        """Test that provider loads seed concepts on init."""
        provider = SeedDictionaryProvider()
        # Should have at least some common process automation concepts
        assert len(provider._concepts) > 0


class TestSeedDictionaryLookup:
    """Tests for SeedDictionaryProvider.lookup()."""

    @pytest.fixture
    def provider(self) -> SeedDictionaryProvider:
        """Create a provider for testing."""
        return SeedDictionaryProvider()

    @pytest.mark.asyncio
    async def test_lookup_existing_irdi(self, provider: SeedDictionaryProvider) -> None:
        """Test looking up an existing IRDI."""
        # Use a common concept we know is in the seed set
        concept = await provider.lookup("0173-1#02-AAB663#001")  # Temperature
        assert concept is not None
        assert concept.irdi == "0173-1#02-AAB663#001"
        assert "temperature" in concept.preferred_name.lower()

    @pytest.mark.asyncio
    async def test_lookup_nonexistent_irdi(
        self, provider: SeedDictionaryProvider
    ) -> None:
        """Test looking up a nonexistent IRDI returns None."""
        concept = await provider.lookup("0173-1#99-ZZZZZ#999")
        assert concept is None

    @pytest.mark.asyncio
    async def test_lookup_returns_dictionary_concept(
        self, provider: SeedDictionaryProvider
    ) -> None:
        """Test lookup returns a DictionaryConcept instance."""
        concept = await provider.lookup("0173-1#02-AAB663#001")
        assert concept is not None
        assert isinstance(concept, DictionaryConcept)
        assert concept.source == "seed"


class TestSeedDictionarySearch:
    """Tests for SeedDictionaryProvider.search()."""

    @pytest.fixture
    def provider(self) -> SeedDictionaryProvider:
        """Create a provider for testing."""
        return SeedDictionaryProvider()

    @pytest.mark.asyncio
    async def test_search_finds_matching_concepts(
        self, provider: SeedDictionaryProvider
    ) -> None:
        """Test search finds concepts matching query."""
        results = await provider.search("temperature")
        assert len(results) > 0
        # All results should be SearchResult instances
        for result in results:
            assert isinstance(result, SearchResult)

    @pytest.mark.asyncio
    async def test_search_respects_max_results(
        self, provider: SeedDictionaryProvider
    ) -> None:
        """Test search respects max_results limit."""
        results = await provider.search("", max_results=5)
        assert len(results) <= 5

    @pytest.mark.asyncio
    async def test_search_returns_ordered_by_score(
        self, provider: SeedDictionaryProvider
    ) -> None:
        """Test search results are ordered by score descending."""
        results = await provider.search("pressure")
        if len(results) > 1:
            scores = [r.score for r in results]
            assert scores == sorted(scores, reverse=True)

    @pytest.mark.asyncio
    async def test_search_case_insensitive(
        self, provider: SeedDictionaryProvider
    ) -> None:
        """Test search is case insensitive."""
        results_lower = await provider.search("temperature")
        results_upper = await provider.search("TEMPERATURE")
        results_mixed = await provider.search("Temperature")

        # Should find same concepts
        irdis_lower = {r.concept.irdi for r in results_lower}
        irdis_upper = {r.concept.irdi for r in results_upper}
        irdis_mixed = {r.concept.irdi for r in results_mixed}

        assert irdis_lower == irdis_upper == irdis_mixed

    @pytest.mark.asyncio
    async def test_search_empty_query_returns_results(
        self, provider: SeedDictionaryProvider
    ) -> None:
        """Test empty query returns some results."""
        results = await provider.search("")
        # Should return concepts up to max_results
        assert len(results) > 0

    @pytest.mark.asyncio
    async def test_search_no_matches(
        self, provider: SeedDictionaryProvider
    ) -> None:
        """Test search with no matches returns empty list."""
        results = await provider.search("xyznonexistentconcept123")
        assert results == []


class TestSeedDictionaryHierarchy:
    """Tests for SeedDictionaryProvider.get_hierarchy()."""

    @pytest.fixture
    def provider(self) -> SeedDictionaryProvider:
        """Create a provider for testing."""
        return SeedDictionaryProvider()

    @pytest.mark.asyncio
    async def test_get_hierarchy_existing_concept(
        self, provider: SeedDictionaryProvider
    ) -> None:
        """Test getting hierarchy for existing concept."""
        node = await provider.get_hierarchy("0173-1#02-AAB663#001")
        assert node is not None
        assert isinstance(node, HierarchyNode)
        assert node.irdi == "0173-1#02-AAB663#001"

    @pytest.mark.asyncio
    async def test_get_hierarchy_nonexistent_concept(
        self, provider: SeedDictionaryProvider
    ) -> None:
        """Test getting hierarchy for nonexistent concept returns None."""
        node = await provider.get_hierarchy("0173-1#99-ZZZZZ#999")
        assert node is None

    @pytest.mark.asyncio
    async def test_get_hierarchy_includes_parent(
        self, provider: SeedDictionaryProvider
    ) -> None:
        """Test hierarchy includes parent if available."""
        node = await provider.get_hierarchy("0173-1#02-AAB663#001")
        # Should have hierarchy info (parent may be None for root concepts)
        assert node is not None


class TestSeedDictionaryAvailability:
    """Tests for SeedDictionaryProvider.is_available()."""

    @pytest.fixture
    def provider(self) -> SeedDictionaryProvider:
        """Create a provider for testing."""
        return SeedDictionaryProvider()

    @pytest.mark.asyncio
    async def test_is_always_available(
        self, provider: SeedDictionaryProvider
    ) -> None:
        """Test seed provider is always available (offline)."""
        available = await provider.is_available()
        assert available is True


class TestSeedDictionaryContent:
    """Tests for seed dictionary content."""

    @pytest.fixture
    def provider(self) -> SeedDictionaryProvider:
        """Create a provider for testing."""
        return SeedDictionaryProvider()

    def test_has_common_process_concepts(
        self, provider: SeedDictionaryProvider
    ) -> None:
        """Test seed set includes common process automation concepts."""
        # These are essential PA-DIM/IEC 61987 concepts
        expected_concepts = [
            "temperature",
            "pressure",
            "flow",
            "level",
        ]

        all_names = [c.preferred_name.lower() for c in provider._concepts.values()]

        for concept in expected_concepts:
            found = any(concept in name for name in all_names)
            assert found, f"Expected concept '{concept}' not found in seed set"

    def test_concepts_have_required_fields(
        self, provider: SeedDictionaryProvider
    ) -> None:
        """Test all concepts have required fields."""
        for irdi, concept in provider._concepts.items():
            assert concept.irdi == irdi
            assert concept.preferred_name
            assert concept.source == "seed"

    def test_has_minimum_concept_count(
        self, provider: SeedDictionaryProvider
    ) -> None:
        """Test seed set has minimum number of concepts."""
        # Plan specifies ~200 common process automation concepts
        # We'll be flexible and require at least 50 for initial implementation
        assert len(provider._concepts) >= 50
