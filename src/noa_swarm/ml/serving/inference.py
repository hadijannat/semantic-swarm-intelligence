"""Inference utilities for tag-to-IRDI mapping."""

from __future__ import annotations

import re
from dataclasses import dataclass

from noa_swarm.common.schemas import Candidate, Hypothesis, TagRecord, utc_now
from noa_swarm.dictionaries import ProviderRegistry, SeedDictionaryProvider
from noa_swarm.ml.datasets.synth_tags import TAG_PATTERN_TO_IRDI


@dataclass(frozen=True)
class InferenceConfig:
    """Configuration for inference engines."""

    agent_id: str
    top_k: int = 5
    rule_confidence: float = 0.9
    min_search_score: float = 0.5


class RuleBasedInferenceEngine:
    """Rule-based inference engine using tag patterns and dictionary search."""

    def __init__(
        self,
        config: InferenceConfig,
        registry: ProviderRegistry | None = None,
    ) -> None:
        self._config = config
        self._registry = registry or self._create_default_registry()

    async def infer(self, tags: list[TagRecord]) -> list[Hypothesis]:
        """Generate hypotheses for a list of tags."""
        hypotheses: list[Hypothesis] = []
        for tag in tags:
            candidates = await self._infer_candidates(tag)
            if not candidates:
                continue
            hypotheses.append(
                Hypothesis(
                    tag_id=tag.tag_id,
                    candidates=candidates,
                    agent_id=self._config.agent_id,
                    model_version="rules-v1",
                    created_at=utc_now(),
                )
            )
        return hypotheses

    async def _infer_candidates(self, tag: TagRecord) -> list[Candidate]:
        """Infer candidate IRDIs for a single tag."""
        tag_name = self._choose_tag_name(tag)
        prefix = self._extract_prefix(tag_name)

        candidates: list[Candidate] = []

        # Rule-based mapping by prefix (ISA-style)
        if prefix and prefix in TAG_PATTERN_TO_IRDI:
            irdi = TAG_PATTERN_TO_IRDI[prefix]
            candidates.append(
                Candidate(
                    irdi=irdi,
                    confidence=self._config.rule_confidence,
                    source_model="rules",
                    reasoning=f"Matched ISA prefix '{prefix}'",
                )
            )

        # Fall back to dictionary search using tag tokens
        query = self._build_query(tag_name)
        if query:
            results = await self._registry.search_all(query, max_results=self._config.top_k)
            for result in results:
                if result.score < self._config.min_search_score:
                    continue
                candidates.append(
                    Candidate(
                        irdi=result.concept.irdi,
                        confidence=min(1.0, result.score),
                        source_model=f"dict:{result.concept.source or 'seed'}",
                        reasoning=f"Matched '{result.concept.preferred_name}' via query '{query}'",
                    )
                )

        # Deduplicate by IRDI and keep top-k
        deduped: dict[str, Candidate] = {}
        for candidate in candidates:
            if candidate.irdi in deduped:
                if candidate.confidence > deduped[candidate.irdi].confidence:
                    deduped[candidate.irdi] = candidate
            else:
                deduped[candidate.irdi] = candidate

        ordered = sorted(deduped.values(), key=lambda c: c.confidence, reverse=True)
        return ordered[: self._config.top_k]

    @staticmethod
    def _choose_tag_name(tag: TagRecord) -> str:
        """Choose the best tag string for inference."""
        if tag.display_name:
            return tag.display_name
        return tag.browse_name or tag.full_path

    @staticmethod
    def _extract_prefix(tag_name: str) -> str | None:
        """Extract the ISA-style prefix from a tag name."""
        if not tag_name:
            return None

        # Take the first token before separators
        token = re.split(r"[./\\-_ ]+", tag_name.strip(), maxsplit=1)[0]
        match = re.match(r"[A-Za-z]+", token)
        if not match:
            return None
        return match.group(0).upper()

    @staticmethod
    def _build_query(tag_name: str) -> str:
        """Build a dictionary search query from a tag name."""
        tokens = re.split(r"[^A-Za-z]+", tag_name)
        filtered = [token.lower() for token in tokens if token]
        return " ".join(filtered[:4])

    @staticmethod
    def _create_default_registry() -> ProviderRegistry:
        registry = ProviderRegistry()
        registry.register(SeedDictionaryProvider())
        return registry
