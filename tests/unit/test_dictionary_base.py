"""Unit tests for dictionary provider protocol."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from noa_swarm.dictionaries.base import (
    DictionaryProvider,
    DictionaryConcept,
    SearchResult,
    HierarchyNode,
)

if TYPE_CHECKING:
    pass


class TestDictionaryConcept:
    """Tests for DictionaryConcept dataclass."""

    def test_create_concept(self) -> None:
        """Test creating a dictionary concept."""
        concept = DictionaryConcept(
            irdi="0173-1#01-ABA234#001",
            preferred_name="Temperature",
            definition="A measure of thermal energy",
            unit="°C",
            data_type="float",
        )

        assert concept.irdi == "0173-1#01-ABA234#001"
        assert concept.preferred_name == "Temperature"
        assert concept.definition == "A measure of thermal energy"
        assert concept.unit == "°C"
        assert concept.data_type == "float"

    def test_concept_optional_fields(self) -> None:
        """Test concept with minimal required fields."""
        concept = DictionaryConcept(
            irdi="0173-1#01-XYZ#001",
            preferred_name="Pressure",
        )

        assert concept.irdi == "0173-1#01-XYZ#001"
        assert concept.preferred_name == "Pressure"
        assert concept.definition is None
        assert concept.unit is None
        assert concept.data_type is None

    def test_concept_empty_irdi_raises(self) -> None:
        """Test that empty IRDI raises ValueError."""
        with pytest.raises(ValueError, match="irdi cannot be empty"):
            DictionaryConcept(irdi="", preferred_name="Test")

    def test_concept_empty_name_raises(self) -> None:
        """Test that empty preferred_name raises ValueError."""
        with pytest.raises(ValueError, match="preferred_name cannot be empty"):
            DictionaryConcept(irdi="0173-1#01-TEST#001", preferred_name="")

    def test_concept_alternate_names(self) -> None:
        """Test concept with alternate names."""
        concept = DictionaryConcept(
            irdi="0173-1#01-ABC#001",
            preferred_name="Flow Rate",
            alternate_names=["Volume Flow", "Volumetric Flow Rate"],
        )

        assert len(concept.alternate_names) == 2
        assert "Volume Flow" in concept.alternate_names


class TestSearchResult:
    """Tests for SearchResult dataclass."""

    def test_create_search_result(self) -> None:
        """Test creating a search result."""
        concept = DictionaryConcept(
            irdi="0173-1#01-ABC#001",
            preferred_name="Test Concept",
        )
        result = SearchResult(
            concept=concept,
            score=0.95,
            match_type="exact",
        )

        assert result.concept.irdi == "0173-1#01-ABC#001"
        assert result.score == 0.95
        assert result.match_type == "exact"

    def test_search_result_default_score(self) -> None:
        """Test search result with default score."""
        concept = DictionaryConcept(
            irdi="0173-1#01-DEF#001",
            preferred_name="Another Concept",
        )
        result = SearchResult(concept=concept)

        assert result.score == 1.0
        assert result.match_type is None

    def test_search_result_invalid_score_raises(self) -> None:
        """Test that score outside [0, 1] raises ValueError."""
        concept = DictionaryConcept(
            irdi="0173-1#01-GHI#001",
            preferred_name="Invalid Score",
        )
        with pytest.raises(ValueError, match="score must be between"):
            SearchResult(concept=concept, score=1.5)
        with pytest.raises(ValueError, match="score must be between"):
            SearchResult(concept=concept, score=-0.1)


class TestHierarchyNode:
    """Tests for HierarchyNode dataclass."""

    def test_create_hierarchy_node(self) -> None:
        """Test creating a hierarchy node."""
        node = HierarchyNode(
            irdi="0173-1#01-ABC#001",
            preferred_name="Process Variable",
            parent_irdi="0173-1#01-PARENT#001",
            child_irdis=["0173-1#01-CHILD1#001", "0173-1#01-CHILD2#001"],
            depth=2,
        )

        assert node.irdi == "0173-1#01-ABC#001"
        assert node.preferred_name == "Process Variable"
        assert node.parent_irdi == "0173-1#01-PARENT#001"
        assert len(node.child_irdis) == 2
        assert node.depth == 2

    def test_hierarchy_node_root(self) -> None:
        """Test hierarchy node at root level."""
        node = HierarchyNode(
            irdi="0173-1#01-ROOT#001",
            preferred_name="Root Concept",
            depth=0,
        )

        assert node.parent_irdi is None
        assert node.child_irdis == []
        assert node.depth == 0


class TestDictionaryProvider:
    """Tests for DictionaryProvider protocol."""

    def test_protocol_defines_required_methods(self) -> None:
        """Test that DictionaryProvider defines required methods."""
        from typing import get_type_hints

        # These should be defined as abstract methods
        assert hasattr(DictionaryProvider, "lookup")
        assert hasattr(DictionaryProvider, "search")
        assert hasattr(DictionaryProvider, "get_hierarchy")
        assert hasattr(DictionaryProvider, "is_available")

    def test_concrete_implementation(self) -> None:
        """Test that a concrete implementation can be created."""
        class MockProvider(DictionaryProvider):
            """Mock implementation for testing."""

            @property
            def name(self) -> str:
                return "mock"

            async def lookup(self, irdi: str) -> DictionaryConcept | None:
                if irdi == "test-irdi":
                    return DictionaryConcept(
                        irdi=irdi,
                        preferred_name="Test Concept",
                    )
                return None

            async def search(
                self,
                query: str,
                max_results: int = 10,
            ) -> list[SearchResult]:
                return []

            async def get_hierarchy(
                self,
                irdi: str,
                depth: int = 1,
            ) -> HierarchyNode | None:
                return None

            async def is_available(self) -> bool:
                return True

        provider = MockProvider()
        assert provider.name == "mock"

    def test_provider_has_name_property(self) -> None:
        """Test that provider has a name property."""
        assert hasattr(DictionaryProvider, "name")


class TestProviderRegistry:
    """Tests for provider registry functionality."""

    def test_registry_can_add_provider(self) -> None:
        """Test adding a provider to registry."""
        from noa_swarm.dictionaries.base import ProviderRegistry, DictionaryProvider

        registry = ProviderRegistry()

        class TestProvider(DictionaryProvider):
            @property
            def name(self) -> str:
                return "test"

            async def lookup(self, irdi: str) -> DictionaryConcept | None:
                return None

            async def search(
                self,
                query: str,
                max_results: int = 10,
            ) -> list[SearchResult]:
                return []

            async def get_hierarchy(
                self,
                irdi: str,
                depth: int = 1,
            ) -> HierarchyNode | None:
                return None

            async def is_available(self) -> bool:
                return True

        provider = TestProvider()
        registry.register(provider)

        assert "test" in registry.providers
        assert registry.get("test") is provider

    def test_registry_get_nonexistent_returns_none(self) -> None:
        """Test getting nonexistent provider returns None."""
        from noa_swarm.dictionaries.base import ProviderRegistry

        registry = ProviderRegistry()
        assert registry.get("nonexistent") is None

    def test_registry_list_providers(self) -> None:
        """Test listing registered providers."""
        from noa_swarm.dictionaries.base import ProviderRegistry, DictionaryProvider

        registry = ProviderRegistry()

        class Provider1(DictionaryProvider):
            @property
            def name(self) -> str:
                return "provider1"

            async def lookup(self, irdi: str) -> DictionaryConcept | None:
                return None

            async def search(
                self,
                query: str,
                max_results: int = 10,
            ) -> list[SearchResult]:
                return []

            async def get_hierarchy(
                self,
                irdi: str,
                depth: int = 1,
            ) -> HierarchyNode | None:
                return None

            async def is_available(self) -> bool:
                return True

        class Provider2(DictionaryProvider):
            @property
            def name(self) -> str:
                return "provider2"

            async def lookup(self, irdi: str) -> DictionaryConcept | None:
                return None

            async def search(
                self,
                query: str,
                max_results: int = 10,
            ) -> list[SearchResult]:
                return []

            async def get_hierarchy(
                self,
                irdi: str,
                depth: int = 1,
            ) -> HierarchyNode | None:
                return None

            async def is_available(self) -> bool:
                return True

        registry.register(Provider1())
        registry.register(Provider2())

        names = registry.list_names()
        assert len(names) == 2
        assert "provider1" in names
        assert "provider2" in names
