"""Dictionary providers for semantic mapping.

This module provides access to industrial semantic dictionaries like
IEC 61987 CDD and eCl@ss for mapping process automation tags to
standardized property definitions.

Available providers:
- **SeedDictionaryProvider**: Local curated subset for offline use
- **IECCDDProvider**: IEC 61987 Common Data Dictionary API
- **EClassProvider**: eCl@ss product classification API

Example usage:
    >>> from noa_swarm.dictionaries import (
    ...     ProviderRegistry,
    ...     SeedDictionaryProvider,
    ...     IECCDDProvider,
    ... )
    >>> registry = ProviderRegistry()
    >>> registry.register(SeedDictionaryProvider())
    >>> registry.register(IECCDDProvider())
    >>> concept = await registry.lookup_any("0173-1#02-AAB663#001")
"""

from noa_swarm.dictionaries.base import (
    DictionaryConcept,
    DictionaryProvider,
    HierarchyNode,
    ProviderRegistry,
    SearchResult,
)
from noa_swarm.dictionaries.eclass_provider import EClassConfig, EClassProvider
from noa_swarm.dictionaries.iec_cdd_provider import IECCDDConfig, IECCDDProvider
from noa_swarm.dictionaries.seed_provider import SeedDictionaryProvider

__all__ = [
    # Base types
    "DictionaryConcept",
    "DictionaryProvider",
    "HierarchyNode",
    "ProviderRegistry",
    "SearchResult",
    # Providers
    "SeedDictionaryProvider",
    "IECCDDProvider",
    "IECCDDConfig",
    "EClassProvider",
    "EClassConfig",
]
