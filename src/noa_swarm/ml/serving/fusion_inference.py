"""Fusion-based inference engine for tag-to-IRDI mapping."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, cast

import torch

from noa_swarm.common.config import Settings, get_settings
from noa_swarm.common.logging import get_logger
from noa_swarm.common.schemas import Candidate, Hypothesis, TagRecord, utc_now
from noa_swarm.dictionaries import ProviderRegistry, SeedDictionaryProvider
from noa_swarm.ml.datasets.synth_tags import SEED_IRDIS
from noa_swarm.ml.models.charcnn import (
    PROPERTY_CLASSES,
    SIGNAL_ROLES,
    CharacterTokenizer,
    CharCNN,
    CharCNNConfig,
)
from noa_swarm.ml.models.fusion import FusionConfig, FusionModel, IRDIEntry, IRDIRetriever
from noa_swarm.ml.models.gnn import (
    TagGraphGNN,
    TagGraphGNNConfig,
    add_self_loops,
    build_hierarchy_edges,
)
from noa_swarm.ml.serving.inference import InferenceConfig, RuleBasedInferenceEngine

if TYPE_CHECKING:
    from pathlib import Path

logger = get_logger(__name__)


DEFAULT_CHARCNN_FILES = ("charcnn.pt", "charcnn.ckpt", "best_model.pt")
DEFAULT_GNN_FILES = ("gnn.pt", "gnn.ckpt")
DEFAULT_FUSION_FILES = ("fusion.pt", "fusion.ckpt")
DEFAULT_CALIBRATION_FILES = ("calibration.json", "temperature.json")


@dataclass(frozen=True)
class FusionInferenceConfig:
    """Configuration for fusion-based inference."""

    agent_id: str
    top_k: int = 5
    min_confidence: float = 0.0
    device: str = "auto"
    use_gnn: bool = True
    allow_untrained: bool = False
    model_version: str = "fusion-v1"
    retriever_metric: Literal["cosine", "euclidean"] = "cosine"
    rule_fallback: bool = True
    rule_confidence: float = 0.9
    model_path: Path | None = None
    charcnn_checkpoint: Path | None = None
    gnn_checkpoint: Path | None = None
    fusion_checkpoint: Path | None = None
    calibration_path: Path | None = None

    @classmethod
    def from_settings(cls, settings: Settings, agent_id: str) -> FusionInferenceConfig:
        """Build config from application settings."""
        ml = settings.ml
        device = ml.device
        if not ml.use_gpu:
            device = "cpu"

        return cls(
            agent_id=agent_id,
            top_k=ml.top_k_candidates,
            min_confidence=ml.confidence_threshold,
            device=device,
            use_gnn=ml.use_gnn,
            allow_untrained=ml.allow_untrained,
            model_path=ml.model_path,
            charcnn_checkpoint=ml.charcnn_checkpoint,
            gnn_checkpoint=ml.gnn_checkpoint,
            fusion_checkpoint=ml.fusion_checkpoint,
            calibration_path=ml.calibration_path,
        )


class FusionInferenceEngine:
    """Inference engine using CharCNN + optional GNN + fusion calibration."""

    def __init__(
        self,
        config: FusionInferenceConfig,
        registry: ProviderRegistry | None = None,
    ) -> None:
        self._config = config
        self._registry = registry or self._create_default_registry()
        self._device = self._resolve_device(config.device)

        self._charcnn: CharCNN | None = None
        self._tokenizer: CharacterTokenizer | None = None
        self._gnn: TagGraphGNN | None = None
        self._fusion: FusionModel | None = None
        self._retriever: IRDIRetriever | None = None
        self._retriever_ready = False

        self._rule_engine: RuleBasedInferenceEngine | None = None
        if config.rule_fallback:
            self._rule_engine = RuleBasedInferenceEngine(
                InferenceConfig(
                    agent_id=config.agent_id,
                    top_k=config.top_k,
                    rule_confidence=config.rule_confidence,
                )
            )

        self._ready = self._load_models()

    @property
    def is_ready(self) -> bool:
        """Return True if the model-based inference engine is ready."""
        return self._ready

    def _resolve_device(self, device: str) -> torch.device:
        if device == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            if torch.backends.mps.is_available():
                return torch.device("mps")
            return torch.device("cpu")
        return torch.device(device)

    def _load_models(self) -> bool:
        charcnn_path = self._resolve_artifact_path(
            self._config.charcnn_checkpoint, self._config.model_path, DEFAULT_CHARCNN_FILES
        )
        gnn_path = self._resolve_artifact_path(
            self._config.gnn_checkpoint, self._config.model_path, DEFAULT_GNN_FILES
        )
        fusion_path = self._resolve_artifact_path(
            self._config.fusion_checkpoint, self._config.model_path, DEFAULT_FUSION_FILES
        )
        calibration_path = self._resolve_artifact_path(
            self._config.calibration_path,
            self._config.model_path,
            DEFAULT_CALIBRATION_FILES,
        )

        self._charcnn = self._load_charcnn(charcnn_path)
        if self._charcnn is None:
            return False

        self._tokenizer = CharacterTokenizer(
            alphabet=self._charcnn.config.alphabet,
            max_length=self._charcnn.config.max_seq_length,
            case_sensitive=False,
        )

        self._gnn = self._load_gnn(gnn_path) if self._config.use_gnn else None
        if self._gnn is not None:
            expected_dim = self._gnn.config.input_dim
            actual_dim = self._charcnn.config.irdi_embedding_dim
            if expected_dim != actual_dim:
                logger.warning(
                    "GNN input dimension mismatch; disabling GNN",
                    expected_dim=expected_dim,
                    actual_dim=actual_dim,
                )
                self._gnn = None

        self._fusion = self._load_fusion(fusion_path, self._charcnn, self._gnn)
        if self._fusion is None:
            return False

        if calibration_path and calibration_path.exists():
            self._load_calibration(calibration_path, self._fusion)

        return True

    def _resolve_artifact_path(
        self,
        explicit: Path | None,
        base_path: Path | None,
        candidates: tuple[str, ...],
    ) -> Path | None:
        if explicit is not None:
            return explicit
        if base_path is None:
            return None
        for name in candidates:
            candidate = base_path / name
            if candidate.exists():
                return candidate
        return None

    def _load_charcnn(self, checkpoint_path: Path | None) -> CharCNN | None:
        if checkpoint_path is None or not checkpoint_path.exists():
            if self._config.allow_untrained:
                model = CharCNN(CharCNNConfig())
                model.to(self._device)
                model.eval()
                logger.warning("Using untrained CharCNN model for inference")
                return model
            logger.info("CharCNN checkpoint not found; skipping model inference")
            return None

        checkpoint = torch.load(checkpoint_path, map_location=self._device, weights_only=False)
        model_config = checkpoint.get("model_config", CharCNNConfig())
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        model = CharCNN(model_config)
        model.load_state_dict(state_dict)
        model.to(self._device)
        model.eval()
        logger.info("Loaded CharCNN checkpoint", path=str(checkpoint_path))
        return model

    def _load_gnn(self, checkpoint_path: Path | None) -> TagGraphGNN | None:
        if checkpoint_path is None or not checkpoint_path.exists():
            if self._config.allow_untrained:
                model = TagGraphGNN(TagGraphGNNConfig())
                model.to(self._device)
                model.eval()
                logger.warning("Using untrained GNN model for inference")
                return model
            logger.info("GNN checkpoint not found; running without GNN")
            return None

        checkpoint = torch.load(checkpoint_path, map_location=self._device, weights_only=False)
        model_config = checkpoint.get("model_config", TagGraphGNNConfig())
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        model = TagGraphGNN(model_config)
        model.load_state_dict(state_dict)
        model.to(self._device)
        model.eval()
        logger.info("Loaded GNN checkpoint", path=str(checkpoint_path))
        return model

    def _load_fusion(
        self,
        checkpoint_path: Path | None,
        charcnn: CharCNN,
        gnn: TagGraphGNN | None,
    ) -> FusionModel | None:
        fusion_config = FusionConfig(
            num_property_classes=charcnn.config.num_property_classes,
            num_signal_roles=charcnn.config.num_signal_roles,
            charcnn_embed_dim=charcnn.config.irdi_embedding_dim,
            gnn_embed_dim=gnn.config.output_dim if gnn is not None else 0,
            use_gnn=gnn is not None,
        )

        model = FusionModel(fusion_config)
        if checkpoint_path is not None and checkpoint_path.exists():
            checkpoint = torch.load(checkpoint_path, map_location=self._device, weights_only=False)
            state_dict = checkpoint.get("model_state_dict", checkpoint)
            try:
                model.load_state_dict(state_dict)
                logger.info("Loaded fusion checkpoint", path=str(checkpoint_path))
            except RuntimeError as exc:
                logger.warning("Fusion checkpoint incompatible; using default weights", error=str(exc))

        model.to(self._device)
        model.eval()
        return model

    def _load_calibration(self, path: Path, fusion: FusionModel) -> None:
        try:
            payload = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("Failed to read calibration file", path=str(path), error=str(exc))
            return

        temperature_property = payload.get("temperature_property")
        temperature_signal = payload.get("temperature_signal")

        if temperature_property is None and "temperature" in payload:
            temperature_property = payload["temperature"]
        if temperature_signal is None and "temperature" in payload:
            temperature_signal = payload["temperature"]

        if temperature_property is not None or temperature_signal is not None:
            fusion.set_temperature(
                temperature_property=temperature_property,
                temperature_signal=temperature_signal,
            )
            logger.info("Loaded calibration temperatures", path=str(path))

    async def _ensure_retriever(self) -> None:
        if self._retriever_ready or self._charcnn is None or self._tokenizer is None:
            return

        concepts = await self._registry.list_concepts()
        if not concepts:
            self._retriever_ready = True
            return

        names = [concept.preferred_name for concept in concepts]
        embeddings = self._encode_strings(names)

        retriever = IRDIRetriever(metric=self._config.retriever_metric, device=self._device)
        entries = []
        for concept, embedding in zip(concepts, embeddings, strict=False):
            entries.append(
                IRDIEntry(
                    irdi=concept.irdi,
                    embedding=embedding,
                    metadata={"name": concept.preferred_name},
                )
            )
        retriever.add_entries(entries)
        retriever.build_index()

        self._retriever = retriever
        self._retriever_ready = True

    def _encode_strings(self, texts: list[str]) -> torch.Tensor:
        if self._charcnn is None or self._tokenizer is None:
            raise RuntimeError("CharCNN model not initialized")

        batch = self._tokenizer.encode_batch(texts).to(self._device)
        with torch.no_grad():
            outputs = self._charcnn(batch, return_embeddings=True)
        embedding = cast(torch.Tensor, outputs["irdi_embedding"])
        return embedding.detach().cpu()

    async def infer(self, tags: list[TagRecord]) -> list[Hypothesis]:
        """Infer semantic mappings for a list of tags."""
        if not tags:
            return []
        if not self._ready or self._charcnn is None or self._tokenizer is None or self._fusion is None:
            raise RuntimeError("Fusion inference engine is not ready")

        await self._ensure_retriever()

        tag_names = [self._choose_tag_name(tag) for tag in tags]
        batch = self._tokenizer.encode_batch(tag_names).to(self._device)

        with torch.no_grad():
            charcnn_output = self._charcnn(batch, return_embeddings=True)

            gnn_output = None
            if self._gnn is not None and self._config.use_gnn:
                node_features = charcnn_output["irdi_embedding"].detach()
                edge_index, _ = build_hierarchy_edges(tags)
                edge_index = add_self_loops(edge_index, len(tags))
                edge_index = edge_index.to(self._device)
                node_embeddings, _ = self._gnn(
                    node_features,
                    edge_index,
                    return_graph_embedding=False,
                )
                gnn_output = {"embedding": node_embeddings}

            fused = self._fusion.predict(charcnn_output, gnn_output)

        property_probs = fused["property_probs"].cpu()
        signal_probs = fused["signal_probs"].cpu()
        embeddings = charcnn_output["irdi_embedding"].detach().cpu()

        retriever_results = None
        if self._retriever is not None:
            retriever_results = self._retriever.retrieve_batch(
                embeddings, top_k=self._config.top_k
            )

        hypotheses: list[Hypothesis] = []
        for idx, tag in enumerate(tags):
            candidates = self._build_candidates(
                property_probs=property_probs[idx],
                signal_probs=signal_probs[idx],
                retriever_hits=None if retriever_results is None else retriever_results[idx],
            )

            if not candidates and self._rule_engine is not None:
                fallback = await self._rule_engine.infer([tag])
                if fallback:
                    candidates = fallback[0].candidates

            if not candidates:
                continue

            hypotheses.append(
                Hypothesis(
                    tag_id=tag.tag_id,
                    candidates=candidates,
                    agent_id=self._config.agent_id,
                    model_version=self._config.model_version,
                    created_at=utc_now(),
                )
            )

        return hypotheses

    def _build_candidates(
        self,
        property_probs: torch.Tensor,
        signal_probs: torch.Tensor,
        retriever_hits: list[tuple[str, float, dict[str, str]]] | None,
    ) -> list[Candidate]:
        candidates: dict[str, Candidate] = {}

        prop_values, prop_indices = torch.topk(
            property_probs, min(self._config.top_k, property_probs.size(0))
        )
        role_idx = int(torch.argmax(signal_probs).item())
        role_name = SIGNAL_ROLES[role_idx]
        role_conf = float(signal_probs[role_idx].item())

        for prop_idx, prop_conf in zip(prop_indices.tolist(), prop_values.tolist(), strict=False):
            prop_name = PROPERTY_CLASSES[prop_idx]
            irdi = SEED_IRDIS.get(prop_name)
            if irdi is None:
                continue
            confidence = float(prop_conf)
            if confidence < self._config.min_confidence:
                continue
            reasoning = f"property={prop_name} ({confidence:.2f}), role={role_name} ({role_conf:.2f})"
            candidates[irdi] = Candidate(
                irdi=irdi,
                confidence=confidence,
                source_model=self._config.model_version,
                reasoning=reasoning,
            )

        if retriever_hits:
            for irdi, score, meta in retriever_hits:
                confidence = self._normalize_similarity(score)
                if confidence < self._config.min_confidence:
                    continue
                label = meta.get("name") if meta else None
                reasoning = f"embedding_match={label}" if label else "embedding_match"
                existing = candidates.get(irdi)
                if existing is None or confidence > existing.confidence:
                    candidates[irdi] = Candidate(
                        irdi=irdi,
                        confidence=confidence,
                        source_model=f"{self._config.model_version}-retriever",
                        reasoning=reasoning,
                    )

        ordered = sorted(candidates.values(), key=lambda c: c.confidence, reverse=True)
        return ordered[: self._config.top_k]

    def _normalize_similarity(self, score: float) -> float:
        if self._config.retriever_metric == "cosine":
            return max(0.0, min(1.0, (score + 1.0) / 2.0))
        return max(0.0, min(1.0, score))

    @staticmethod
    def _choose_tag_name(tag: TagRecord) -> str:
        if tag.display_name:
            return tag.display_name
        return tag.browse_name or tag.full_path

    @staticmethod
    def _create_default_registry() -> ProviderRegistry:
        registry = ProviderRegistry()
        registry.register(SeedDictionaryProvider())
        return registry


def build_fusion_engine_from_env(agent_id: str) -> FusionInferenceEngine:
    """Create a fusion inference engine using environment-backed settings."""
    settings = get_settings()
    config = FusionInferenceConfig.from_settings(settings, agent_id=agent_id)
    return FusionInferenceEngine(config)
