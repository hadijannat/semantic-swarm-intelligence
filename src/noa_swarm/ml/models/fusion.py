"""Late fusion model for combining CharCNN and GNN outputs.

This module implements a FusionModel that combines predictions from:
- CharCNN: Character-level CNN for tag name classification
- GNN: Graph Neural Network for structural relationships (optional)

The fusion uses learnable weights and temperature scaling for calibration,
ensuring well-calibrated confidence scores essential for swarm consensus
where votes are weighted by confidence.

Architecture:
    CharCNN logits + GNN logits (optional)
    -> Late Fusion (learnable weights alpha, beta)
    -> Temperature Scaling (calibration)
    -> Calibrated probabilities
    -> Top-K IRDI retrieval via embedding similarity
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal, NotRequired, TypedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

if TYPE_CHECKING:
    from collections.abc import Sequence


class FusionOutput(TypedDict):
    """Typed output for fusion forward pass."""

    property_class: torch.Tensor
    signal_role: torch.Tensor
    fused_embedding: torch.Tensor
    property_probs: torch.Tensor
    signal_probs: torch.Tensor
    fusion_weights: dict[str, torch.Tensor]
    temperatures: dict[str, torch.Tensor]
    property_class_uncalibrated: NotRequired[torch.Tensor]
    signal_role_uncalibrated: NotRequired[torch.Tensor]


@dataclass
class FusionConfig:
    """Configuration for FusionModel.

    Attributes:
        num_property_classes: Number of property class outputs
        num_signal_roles: Number of signal role outputs
        charcnn_embed_dim: CharCNN embedding dimension
        gnn_embed_dim: GNN embedding dimension (optional)
        initial_charcnn_weight: Initial weight for CharCNN (default 0.5)
        initial_gnn_weight: Initial weight for GNN (default 0.5)
        initial_temperature: Initial temperature for calibration (default 1.0)
        use_gnn: Whether GNN is available/used
    """

    num_property_classes: int = 26
    num_signal_roles: int = 12
    charcnn_embed_dim: int = 128
    gnn_embed_dim: int = 128
    initial_charcnn_weight: float = 0.5
    initial_gnn_weight: float = 0.5
    initial_temperature: float = 1.0
    use_gnn: bool = True


class FusionModel(nn.Module):
    """Late fusion model for combining CharCNN and GNN outputs.

    Combines logits from CharCNN and GNN using learnable weights,
    then applies temperature scaling for calibration. Supports
    CharCNN-only mode when GNN is not available.

    Attributes:
        config: Model configuration
        alpha: Learnable weight for CharCNN (property class)
        beta: Learnable weight for GNN (property class)
        alpha_signal: Learnable weight for CharCNN (signal role)
        beta_signal: Learnable weight for GNN (signal role)
        temperature_property: Temperature for property class calibration
        temperature_signal: Temperature for signal role calibration

    Example:
        >>> config = FusionConfig(use_gnn=True)
        >>> model = FusionModel(config)
        >>> charcnn_out = {
        ...     "property_class": torch.randn(2, 26),
        ...     "signal_role": torch.randn(2, 12),
        ...     "irdi_embedding": torch.randn(2, 128),
        ... }
        >>> gnn_out = {"logits": torch.randn(2, 26), "embedding": torch.randn(2, 128)}
        >>> fused = model(charcnn_out, gnn_out)
    """

    def __init__(self, config: FusionConfig | None = None) -> None:
        """Initialize the FusionModel.

        Args:
            config: Model configuration. If None, uses default config.
        """
        super().__init__()
        self.config = config or FusionConfig()

        # Learnable fusion weights for property class
        # Using log-space for stability (softmax will be applied)
        self._alpha_logit = nn.Parameter(
            torch.tensor(self._to_logit(self.config.initial_charcnn_weight))
        )
        self._beta_logit = nn.Parameter(
            torch.tensor(self._to_logit(self.config.initial_gnn_weight))
        )

        # Learnable fusion weights for signal role
        self._alpha_signal_logit = nn.Parameter(
            torch.tensor(self._to_logit(self.config.initial_charcnn_weight))
        )
        self._beta_signal_logit = nn.Parameter(
            torch.tensor(self._to_logit(self.config.initial_gnn_weight))
        )

        # Temperature parameters for calibration (must be positive)
        # Use log-space to ensure positivity
        self._log_temperature_property = nn.Parameter(
            torch.tensor(0.0)  # log(1.0) = 0
        )
        self._log_temperature_signal = nn.Parameter(
            torch.tensor(0.0)  # log(1.0) = 0
        )

        # Set initial temperature
        if self.config.initial_temperature != 1.0:
            with torch.no_grad():
                self._log_temperature_property.fill_(
                    torch.log(torch.tensor(self.config.initial_temperature))
                )
                self._log_temperature_signal.fill_(
                    torch.log(torch.tensor(self.config.initial_temperature))
                )

        # Embedding projection layers (optional, for combining embeddings)
        total_embed_dim = self.config.charcnn_embed_dim
        if self.config.use_gnn:
            total_embed_dim += self.config.gnn_embed_dim

        self._fused_embed_dim = total_embed_dim

    @staticmethod
    def _to_logit(weight: float) -> float:
        """Convert a weight (0-1) to logit space."""
        # Clamp to avoid numerical issues
        weight = max(0.01, min(0.99, weight))
        return float(torch.log(torch.tensor(weight / (1 - weight))))

    @property
    def alpha(self) -> torch.Tensor:
        """Get the normalized CharCNN weight for property class."""
        weights = F.softmax(torch.stack([self._alpha_logit, self._beta_logit]), dim=0)
        return weights[0]

    @property
    def beta(self) -> torch.Tensor:
        """Get the normalized GNN weight for property class."""
        weights = F.softmax(torch.stack([self._alpha_logit, self._beta_logit]), dim=0)
        return weights[1]

    @property
    def alpha_signal(self) -> torch.Tensor:
        """Get the normalized CharCNN weight for signal role."""
        weights = F.softmax(torch.stack([self._alpha_signal_logit, self._beta_signal_logit]), dim=0)
        return weights[0]

    @property
    def beta_signal(self) -> torch.Tensor:
        """Get the normalized GNN weight for signal role."""
        weights = F.softmax(torch.stack([self._alpha_signal_logit, self._beta_signal_logit]), dim=0)
        return weights[1]

    @property
    def temperature_property(self) -> torch.Tensor:
        """Get the temperature for property class calibration."""
        return torch.exp(self._log_temperature_property)

    @property
    def temperature_signal(self) -> torch.Tensor:
        """Get the temperature for signal role calibration."""
        return torch.exp(self._log_temperature_signal)

    @property
    def fused_embed_dim(self) -> int:
        """Get the dimension of fused embeddings."""
        return self._fused_embed_dim

    def forward(
        self,
        charcnn_output: dict[str, torch.Tensor],
        gnn_output: dict[str, torch.Tensor] | None = None,
        return_uncalibrated: bool = False,
    ) -> FusionOutput:
        """Forward pass through the fusion model.

        Args:
            charcnn_output: Dictionary with keys:
                - "property_class": Logits (batch, num_property_classes)
                - "signal_role": Logits (batch, num_signal_roles)
                - "irdi_embedding": Embedding (batch, embed_dim)
            gnn_output: Optional dictionary with keys:
                - "property_logits": Logits (batch, num_property_classes)
                - "signal_logits": Logits (batch, num_signal_roles) - optional
                - "embedding": Embedding (batch, gnn_embed_dim)
            return_uncalibrated: If True, also return uncalibrated logits

        Returns:
            Dictionary with keys:
                - "property_class": Calibrated logits (batch, num_property_classes)
                - "signal_role": Calibrated logits (batch, num_signal_roles)
                - "fused_embedding": Combined embedding (batch, fused_dim)
                - "property_probs": Calibrated probabilities
                - "signal_probs": Calibrated probabilities
                - "fusion_weights": Dictionary of current fusion weights
                - "temperatures": Dictionary of current temperatures
                Optionally (if return_uncalibrated=True):
                - "property_class_uncalibrated": Fused but uncalibrated logits
                - "signal_role_uncalibrated": Fused but uncalibrated logits
        """
        # Extract CharCNN outputs
        charcnn_property = charcnn_output["property_class"]
        charcnn_signal = charcnn_output["signal_role"]
        charcnn_embed = charcnn_output["irdi_embedding"]

        # Fuse logits
        if self.config.use_gnn and gnn_output is not None:
            # GNN is available - use weighted combination
            gnn_property = gnn_output.get("property_logits")
            gnn_signal = gnn_output.get("signal_logits")
            gnn_embed = gnn_output.get("embedding")

            # Property class fusion
            if gnn_property is not None:
                fused_property = self.alpha * charcnn_property + self.beta * gnn_property
            else:
                fused_property = charcnn_property

            # Signal role fusion
            if gnn_signal is not None:
                fused_signal = self.alpha_signal * charcnn_signal + self.beta_signal * gnn_signal
            else:
                fused_signal = charcnn_signal

            # Embedding concatenation
            if gnn_embed is not None:
                fused_embed = torch.cat([charcnn_embed, gnn_embed], dim=-1)
            else:
                fused_embed = charcnn_embed
        else:
            # CharCNN-only mode
            fused_property = charcnn_property
            fused_signal = charcnn_signal
            fused_embed = charcnn_embed

        # Store uncalibrated if requested
        uncalibrated_property = fused_property
        uncalibrated_signal = fused_signal

        # Apply temperature scaling for calibration
        calibrated_property = fused_property / self.temperature_property
        calibrated_signal = fused_signal / self.temperature_signal

        # Compute calibrated probabilities
        property_probs = F.softmax(calibrated_property, dim=-1)
        signal_probs = F.softmax(calibrated_signal, dim=-1)

        result: FusionOutput = {
            "property_class": calibrated_property,
            "signal_role": calibrated_signal,
            "fused_embedding": fused_embed,
            "property_probs": property_probs,
            "signal_probs": signal_probs,
            "fusion_weights": {
                "alpha": self.alpha.detach(),
                "beta": self.beta.detach(),
                "alpha_signal": self.alpha_signal.detach(),
                "beta_signal": self.beta_signal.detach(),
            },
            "temperatures": {
                "property": self.temperature_property.detach(),
                "signal": self.temperature_signal.detach(),
            },
        }

        if return_uncalibrated:
            result["property_class_uncalibrated"] = uncalibrated_property
            result["signal_role_uncalibrated"] = uncalibrated_signal

        return result

    def predict(
        self,
        charcnn_output: dict[str, torch.Tensor],
        gnn_output: dict[str, torch.Tensor] | None = None,
    ) -> dict[str, torch.Tensor]:
        """Make predictions with calibrated probabilities.

        Args:
            charcnn_output: CharCNN model output dictionary
            gnn_output: Optional GNN model output dictionary

        Returns:
            Dictionary with predicted classes and probabilities
        """
        with torch.no_grad():
            outputs = self.forward(charcnn_output, gnn_output)

            return {
                "property_class": torch.argmax(outputs["property_probs"], dim=-1),
                "property_probs": outputs["property_probs"],
                "property_confidence": outputs["property_probs"].max(dim=-1).values,
                "signal_role": torch.argmax(outputs["signal_probs"], dim=-1),
                "signal_probs": outputs["signal_probs"],
                "signal_confidence": outputs["signal_probs"].max(dim=-1).values,
                "fused_embedding": outputs["fused_embedding"],
            }

    def set_temperature(
        self,
        temperature_property: float | None = None,
        temperature_signal: float | None = None,
    ) -> None:
        """Set the temperature parameters (useful after calibration).

        Args:
            temperature_property: New temperature for property class
            temperature_signal: New temperature for signal role
        """
        with torch.no_grad():
            if temperature_property is not None:
                self._log_temperature_property.fill_(torch.log(torch.tensor(temperature_property)))
            if temperature_signal is not None:
                self._log_temperature_signal.fill_(torch.log(torch.tensor(temperature_signal)))

    def freeze_fusion_weights(self) -> None:
        """Freeze fusion weights (useful when only calibrating temperature)."""
        self._alpha_logit.requires_grad = False
        self._beta_logit.requires_grad = False
        self._alpha_signal_logit.requires_grad = False
        self._beta_signal_logit.requires_grad = False

    def unfreeze_fusion_weights(self) -> None:
        """Unfreeze fusion weights."""
        self._alpha_logit.requires_grad = True
        self._beta_logit.requires_grad = True
        self._alpha_signal_logit.requires_grad = True
        self._beta_signal_logit.requires_grad = True

    def freeze_temperature(self) -> None:
        """Freeze temperature parameters."""
        self._log_temperature_property.requires_grad = False
        self._log_temperature_signal.requires_grad = False

    def unfreeze_temperature(self) -> None:
        """Unfreeze temperature parameters."""
        self._log_temperature_property.requires_grad = True
        self._log_temperature_signal.requires_grad = True


@dataclass
class IRDIEntry:
    """An entry in the IRDI dictionary.

    Attributes:
        irdi: The IRDI identifier string
        embedding: The embedding vector for this IRDI
        metadata: Optional metadata dictionary
    """

    irdi: str
    embedding: torch.Tensor
    metadata: dict[str, str] = field(default_factory=dict)


class IRDIRetriever:
    """Top-K IRDI retrieval using embedding similarity.

    Maintains an index of IRDI embeddings and supports efficient
    retrieval of the most similar IRDIs given a query embedding.

    Supports both cosine similarity and euclidean distance metrics.

    Attributes:
        entries: List of IRDI entries
        embeddings: Stacked embedding matrix for efficient computation
        metric: Similarity metric ('cosine' or 'euclidean')

    Example:
        >>> retriever = IRDIRetriever(metric='cosine')
        >>> retriever.add_entry(IRDIEntry(irdi="0173-1#02-AAA123#001", embedding=emb1))
        >>> retriever.add_entry(IRDIEntry(irdi="0173-1#02-AAA456#001", embedding=emb2))
        >>> retriever.build_index()
        >>> results = retriever.retrieve(query_embedding, top_k=5)
    """

    def __init__(
        self,
        metric: Literal["cosine", "euclidean"] = "cosine",
        device: torch.device | str | None = None,
    ) -> None:
        """Initialize the IRDIRetriever.

        Args:
            metric: Similarity metric to use ('cosine' or 'euclidean')
            device: Device to store embeddings on
        """
        if metric not in ("cosine", "euclidean"):
            raise ValueError(f"Unknown metric: {metric}. Use 'cosine' or 'euclidean'.")

        self.metric = metric
        self.device = device if device is not None else torch.device("cpu")
        self.entries: list[IRDIEntry] = []
        self._embeddings: torch.Tensor | None = None
        self._index_built = False

    def add_entry(self, entry: IRDIEntry) -> None:
        """Add an IRDI entry to the retriever.

        Args:
            entry: IRDIEntry to add

        Note:
            After adding entries, call build_index() to enable retrieval.
        """
        self.entries.append(entry)
        self._index_built = False

    def add_entries(self, entries: Sequence[IRDIEntry]) -> None:
        """Add multiple IRDI entries to the retriever.

        Args:
            entries: Sequence of IRDIEntry objects to add
        """
        self.entries.extend(entries)
        self._index_built = False

    def add_from_dict(
        self,
        irdi_embeddings: dict[str, torch.Tensor],
        metadata: dict[str, dict[str, str]] | None = None,
    ) -> None:
        """Add entries from a dictionary of IRDI to embedding mappings.

        Args:
            irdi_embeddings: Dictionary mapping IRDI strings to embeddings
            metadata: Optional dictionary mapping IRDI strings to metadata
        """
        metadata = metadata or {}
        for irdi, embedding in irdi_embeddings.items():
            entry = IRDIEntry(
                irdi=irdi,
                embedding=embedding,
                metadata=metadata.get(irdi, {}),
            )
            self.entries.append(entry)
        self._index_built = False

    def build_index(self) -> None:
        """Build the embedding index for efficient retrieval.

        Must be called after adding entries and before retrieval.
        """
        if not self.entries:
            self._embeddings = torch.zeros((0, 1), device=self.device)
            self._index_built = True
            return

        # Stack embeddings into a matrix
        embeddings = torch.stack([e.embedding for e in self.entries])
        self._embeddings = embeddings.to(self.device)

        # Normalize for cosine similarity
        if self.metric == "cosine":
            self._embeddings = F.normalize(self._embeddings, p=2, dim=-1)

        self._index_built = True

    def retrieve(
        self,
        query_embedding: torch.Tensor,
        top_k: int = 5,
    ) -> list[tuple[str, float, dict[str, str]]]:
        """Retrieve the top-K most similar IRDIs.

        Args:
            query_embedding: Query embedding vector (embed_dim,) or (1, embed_dim)
            top_k: Number of results to return

        Returns:
            List of tuples (irdi, similarity_score, metadata) sorted by similarity

        Raises:
            RuntimeError: If index has not been built
        """
        if not self._index_built:
            raise RuntimeError("Index not built. Call build_index() first.")

        if self._embeddings is None or self._embeddings.size(0) == 0:
            return []

        # Ensure query is 1D
        if query_embedding.dim() == 2:
            query_embedding = query_embedding.squeeze(0)

        query = query_embedding.to(self.device)

        # Compute similarity/distance
        if self.metric == "cosine":
            # Normalize query for cosine similarity
            query = F.normalize(query.unsqueeze(0), p=2, dim=-1).squeeze(0)
            # Cosine similarity = dot product of normalized vectors
            similarities = torch.mv(self._embeddings, query)
            # Higher is better for cosine similarity
            top_k_values, top_k_indices = torch.topk(similarities, min(top_k, len(self.entries)))
        else:
            # Euclidean distance
            distances = torch.norm(self._embeddings - query.unsqueeze(0), p=2, dim=-1)
            # Lower is better for euclidean distance
            top_k_values, top_k_indices = torch.topk(
                distances, min(top_k, len(self.entries)), largest=False
            )
            # Convert distances to similarity scores (1 / (1 + distance))
            top_k_values = 1.0 / (1.0 + top_k_values)

        # Build results
        results: list[tuple[str, float, dict[str, str]]] = []
        for idx, score in zip(top_k_indices.tolist(), top_k_values.tolist(), strict=False):
            entry = self.entries[idx]
            results.append((entry.irdi, score, entry.metadata))

        return results

    def retrieve_batch(
        self,
        query_embeddings: torch.Tensor,
        top_k: int = 5,
    ) -> list[list[tuple[str, float, dict[str, str]]]]:
        """Retrieve top-K IRDIs for a batch of queries.

        Args:
            query_embeddings: Batch of query embeddings (batch, embed_dim)
            top_k: Number of results per query

        Returns:
            List of results for each query
        """
        if not self._index_built:
            raise RuntimeError("Index not built. Call build_index() first.")

        if self._embeddings is None or self._embeddings.size(0) == 0:
            return [[] for _ in range(query_embeddings.size(0))]

        queries = query_embeddings.to(self.device)
        batch_size = queries.size(0)

        if self.metric == "cosine":
            # Normalize queries
            queries = F.normalize(queries, p=2, dim=-1)
            # Compute all similarities at once: (batch, num_entries)
            similarities = torch.mm(queries, self._embeddings.t())
            # Get top-k for each query
            top_k_values, top_k_indices = torch.topk(
                similarities, min(top_k, len(self.entries)), dim=-1
            )
        else:
            # Compute pairwise distances: (batch, num_entries)
            distances = torch.cdist(queries, self._embeddings, p=2)
            # Get top-k smallest distances
            top_k_values, top_k_indices = torch.topk(
                distances, min(top_k, len(self.entries)), dim=-1, largest=False
            )
            # Convert to similarity
            top_k_values = 1.0 / (1.0 + top_k_values)

        # Build results for each query
        all_results: list[list[tuple[str, float, dict[str, str]]]] = []
        for b in range(batch_size):
            results: list[tuple[str, float, dict[str, str]]] = []
            for idx, score in zip(
                top_k_indices[b].tolist(), top_k_values[b].tolist(), strict=False
            ):
                entry = self.entries[idx]
                results.append((entry.irdi, score, entry.metadata))
            all_results.append(results)

        return all_results

    def get_irdi_embedding(self, irdi: str) -> torch.Tensor | None:
        """Get the embedding for a specific IRDI.

        Args:
            irdi: The IRDI identifier

        Returns:
            The embedding tensor, or None if not found
        """
        for entry in self.entries:
            if entry.irdi == irdi:
                return entry.embedding
        return None

    def __len__(self) -> int:
        """Return the number of entries in the retriever."""
        return len(self.entries)

    @property
    def embed_dim(self) -> int:
        """Get the embedding dimension."""
        if self.entries:
            return self.entries[0].embedding.size(-1)
        return 0

    def clear(self) -> None:
        """Clear all entries and reset the index."""
        self.entries.clear()
        self._embeddings = None
        self._index_built = False
