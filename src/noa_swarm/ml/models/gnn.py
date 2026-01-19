"""Graph Neural Network for semantic tag mapping.

This module implements a GraphSAGE-based GNN architecture that learns from
structural relationships between industrial tags (OPC UA hierarchy and
time-series correlation). The model produces node embeddings and/or
graph-level embeddings for classification.

Architecture:
    Input Node Features + Edge Index
    -> GraphSAGE Layer 1 (aggregator: mean/max/LSTM)
    -> ReLU + Dropout
    -> GraphSAGE Layer 2
    -> ReLU + Dropout
    -> [Optional: Global Mean Pooling for graph-level output]
    -> Output: node embeddings and/or graph embedding

The GNN is designed to be fused with CharCNN outputs for final prediction.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, Dataset
from torch_geometric.nn import SAGEConv, global_mean_pool

if TYPE_CHECKING:
    from collections.abc import Sequence

    from noa_swarm.common.schemas import TagRecord


# Aggregator types supported by GraphSAGE
AggregatorType = Literal["mean", "max", "lstm"]


@dataclass
class TagGraphGNNConfig:
    """Configuration for TagGraphGNN model.

    Attributes:
        input_dim: Dimension of input node features
        hidden_dim: Hidden layer dimension
        output_dim: Output embedding dimension
        num_layers: Number of GraphSAGE layers
        aggregator: Aggregation method ('mean', 'max', 'lstm')
        dropout: Dropout probability
        normalize: Whether to L2-normalize output embeddings
        use_batch_norm: Whether to use batch normalization
    """

    input_dim: int = 128
    hidden_dim: int = 128
    output_dim: int = 128
    num_layers: int = 2
    aggregator: AggregatorType = "mean"
    dropout: float = 0.5
    normalize: bool = True
    use_batch_norm: bool = True


class TagGraphGNN(nn.Module):
    """GraphSAGE-based GNN for industrial tag embeddings.

    This model takes node features (e.g., from CharCNN) and edge relationships
    (from OPC UA hierarchy and/or time-series correlation) to produce
    refined node embeddings that capture structural relationships.

    Supports both node-level and graph-level predictions.

    Attributes:
        config: Model configuration
        convs: List of SAGEConv layers
        batch_norms: List of batch normalization layers (if enabled)
        dropout: Dropout layer

    Example:
        >>> config = TagGraphGNNConfig(input_dim=128, hidden_dim=64, output_dim=64)
        >>> model = TagGraphGNN(config)
        >>> x = torch.randn(10, 128)  # 10 nodes, 128 features each
        >>> edge_index = torch.tensor([[0, 1, 2], [1, 2, 3]])  # edges
        >>> node_emb, graph_emb = model(x, edge_index)
    """

    def __init__(self, config: TagGraphGNNConfig | None = None) -> None:
        """Initialize the TagGraphGNN model.

        Args:
            config: Model configuration. If None, uses default config.
        """
        super().__init__()
        self.config = config or TagGraphGNNConfig()

        # Build GraphSAGE layers
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList() if self.config.use_batch_norm else None

        # First layer: input_dim -> hidden_dim
        self.convs.append(
            SAGEConv(
                in_channels=self.config.input_dim,
                out_channels=self.config.hidden_dim,
                aggr=self._get_aggr_type(),
            )
        )
        if self.config.use_batch_norm:
            self.batch_norms.append(nn.BatchNorm1d(self.config.hidden_dim))

        # Middle layers: hidden_dim -> hidden_dim
        for _ in range(self.config.num_layers - 2):
            self.convs.append(
                SAGEConv(
                    in_channels=self.config.hidden_dim,
                    out_channels=self.config.hidden_dim,
                    aggr=self._get_aggr_type(),
                )
            )
            if self.config.use_batch_norm:
                self.batch_norms.append(nn.BatchNorm1d(self.config.hidden_dim))

        # Last layer: hidden_dim -> output_dim
        if self.config.num_layers > 1:
            self.convs.append(
                SAGEConv(
                    in_channels=self.config.hidden_dim,
                    out_channels=self.config.output_dim,
                    aggr=self._get_aggr_type(),
                )
            )
            # No batch norm after last layer typically

        self.dropout = nn.Dropout(self.config.dropout)

        # Initialize weights
        self._init_weights()

    def _get_aggr_type(self) -> str:
        """Get PyG aggregation type string.

        Note: PyG SAGEConv uses 'mean' or 'max'. LSTM aggregator
        requires custom implementation.
        """
        if self.config.aggregator == "lstm":
            # PyG SAGEConv doesn't directly support LSTM aggregator,
            # fall back to mean for now (could extend with custom layer)
            return "mean"
        return self.config.aggregator

    def _init_weights(self) -> None:
        """Initialize model weights."""
        for conv in self.convs:
            if hasattr(conv, "lin_l"):
                nn.init.xavier_uniform_(conv.lin_l.weight)
                if conv.lin_l.bias is not None:
                    nn.init.zeros_(conv.lin_l.bias)
            if hasattr(conv, "lin_r"):
                nn.init.xavier_uniform_(conv.lin_r.weight)
                if conv.lin_r.bias is not None:
                    nn.init.zeros_(conv.lin_r.bias)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor | None = None,
        return_graph_embedding: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Forward pass through the GNN.

        Args:
            x: Node features of shape (num_nodes, input_dim)
            edge_index: Edge index in COO format of shape (2, num_edges)
            batch: Batch tensor for graph-level pooling (num_nodes,).
                   If None and return_graph_embedding=True, assumes single graph.
            return_graph_embedding: Whether to compute graph-level embedding

        Returns:
            Tuple of:
                - node_embeddings: Node-level embeddings (num_nodes, output_dim)
                - graph_embedding: Graph-level embedding (batch_size, output_dim)
                                   or None if return_graph_embedding=False
        """
        # Pass through GraphSAGE layers
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            if self.batch_norms is not None and i < len(self.batch_norms):
                x = self.batch_norms[i](x)
            x = F.relu(x)
            x = self.dropout(x)

        # Last layer (no ReLU/dropout after)
        x = self.convs[-1](x, edge_index)

        # Optionally normalize node embeddings
        if self.config.normalize:
            x = F.normalize(x, p=2, dim=-1)

        node_embeddings = x

        # Compute graph-level embedding via global mean pooling
        graph_embedding = None
        if return_graph_embedding:
            if batch is None:
                # Assume single graph - all nodes belong to batch 0
                batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
            graph_embedding = global_mean_pool(x, batch)

            if self.config.normalize:
                graph_embedding = F.normalize(graph_embedding, p=2, dim=-1)

        return node_embeddings, graph_embedding

    def get_node_embeddings(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        """Get only node-level embeddings (no graph pooling).

        Args:
            x: Node features of shape (num_nodes, input_dim)
            edge_index: Edge index in COO format

        Returns:
            Node embeddings of shape (num_nodes, output_dim)
        """
        node_emb, _ = self.forward(x, edge_index, return_graph_embedding=False)
        return node_emb

    def get_graph_embedding(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Get only graph-level embedding.

        Args:
            x: Node features of shape (num_nodes, input_dim)
            edge_index: Edge index in COO format
            batch: Optional batch tensor

        Returns:
            Graph embedding of shape (batch_size, output_dim)
        """
        _, graph_emb = self.forward(x, edge_index, batch=batch, return_graph_embedding=True)
        assert graph_emb is not None
        return graph_emb


# =============================================================================
# Graph Construction Utilities
# =============================================================================


def build_hierarchy_edges(
    tag_records: Sequence[TagRecord],
    node_id_to_idx: dict[str, int] | None = None,
) -> tuple[torch.Tensor, dict[str, int]]:
    """Build edges from OPC UA parent-child hierarchy relationships.

    Creates directed edges from parent nodes to child nodes based on
    the parent_path attribute of TagRecord objects.

    Args:
        tag_records: Sequence of TagRecord objects
        node_id_to_idx: Optional pre-computed mapping from tag_id to node index.
                        If None, will be computed from tag_records.

    Returns:
        Tuple of:
            - edge_index: Tensor of shape (2, num_edges) in COO format
            - node_id_to_idx: Mapping from tag_id to node index

    Example:
        >>> # Tags with hierarchy: root/area/tag1, root/area/tag2
        >>> edges, mapping = build_hierarchy_edges(tags)
        >>> # Returns edges connecting parent nodes to children
    """
    # Build node index mapping if not provided
    if node_id_to_idx is None:
        node_id_to_idx = {tag.tag_id: i for i, tag in enumerate(tag_records)}

    # Build path-to-tags mapping for parent lookup
    # Key: full path, Value: list of tag indices at that path
    path_to_tags: dict[str, list[int]] = {}
    for tag in tag_records:
        path = tag.full_path
        idx = node_id_to_idx.get(tag.tag_id)
        if idx is not None:
            if path not in path_to_tags:
                path_to_tags[path] = []
            path_to_tags[path].append(idx)

    # Build edges: parent -> child
    src_nodes: list[int] = []
    dst_nodes: list[int] = []

    for tag in tag_records:
        child_idx = node_id_to_idx.get(tag.tag_id)
        if child_idx is None:
            continue

        # Find parent path
        if tag.parent_path:
            parent_path = "/".join(tag.parent_path)
            # Look for tags that match the parent path
            parent_indices = path_to_tags.get(parent_path, [])
            for parent_idx in parent_indices:
                if parent_idx != child_idx:
                    # Edge from parent to child
                    src_nodes.append(parent_idx)
                    dst_nodes.append(child_idx)

    if not src_nodes:
        # Return empty edge tensor
        edge_index = torch.zeros((2, 0), dtype=torch.long)
    else:
        edge_index = torch.tensor([src_nodes, dst_nodes], dtype=torch.long)

    return edge_index, node_id_to_idx


def build_correlation_edges(
    time_series_data: np.ndarray | torch.Tensor,
    threshold: float = 0.7,
    method: Literal["pearson", "spearman"] = "pearson",
) -> torch.Tensor:
    """Build edges from time-series correlation between tags.

    Creates edges between tags whose time-series values are highly correlated
    (above the threshold). This captures operational relationships that may
    not be reflected in the OPC UA hierarchy.

    Args:
        time_series_data: Array of shape (num_tags, num_timesteps) containing
                          time-series values for each tag
        threshold: Minimum absolute correlation to create an edge (0.0 to 1.0)
        method: Correlation method ('pearson' or 'spearman')

    Returns:
        edge_index: Tensor of shape (2, num_edges) in COO format

    Example:
        >>> data = np.random.randn(10, 100)  # 10 tags, 100 timesteps
        >>> edges = build_correlation_edges(data, threshold=0.5)
    """
    if isinstance(time_series_data, torch.Tensor):
        time_series_data = time_series_data.numpy()

    num_tags = time_series_data.shape[0]

    # Handle degenerate cases
    if num_tags < 2 or time_series_data.shape[1] < 2:
        return torch.zeros((2, 0), dtype=torch.long)

    # Compute correlation matrix
    if method == "pearson":
        # Center the data
        centered = time_series_data - time_series_data.mean(axis=1, keepdims=True)
        # Compute norms
        norms = np.linalg.norm(centered, axis=1, keepdims=True)
        # Avoid division by zero
        norms = np.where(norms == 0, 1, norms)
        normalized = centered / norms
        # Correlation matrix
        corr_matrix = normalized @ normalized.T
    elif method == "spearman":
        # Rank-based correlation
        from scipy.stats import spearmanr

        corr_matrix, _ = spearmanr(time_series_data, axis=1)
        if num_tags == 2:
            # spearmanr returns scalar for 2 variables
            corr_matrix = np.array([[1.0, corr_matrix], [corr_matrix, 1.0]])
    else:
        raise ValueError(f"Unknown correlation method: {method}")

    # Find edges above threshold (excluding self-loops)
    abs_corr = np.abs(corr_matrix)
    np.fill_diagonal(abs_corr, 0)  # No self-loops

    # Get pairs above threshold
    src_nodes, dst_nodes = np.where(abs_corr >= threshold)

    # Remove duplicates (keep only upper triangle for undirected)
    mask = src_nodes < dst_nodes
    src_nodes = src_nodes[mask]
    dst_nodes = dst_nodes[mask]

    # Make bidirectional (for GNN message passing)
    all_src = np.concatenate([src_nodes, dst_nodes])
    all_dst = np.concatenate([dst_nodes, src_nodes])

    if len(all_src) == 0:
        return torch.zeros((2, 0), dtype=torch.long)

    # Stack as numpy array first, then convert to tensor (avoids slow list-of-arrays warning)
    edge_array = np.array([all_src, all_dst])
    edge_index = torch.from_numpy(edge_array).long()
    return edge_index


def combine_edge_sources(
    *edge_indices: torch.Tensor,
    remove_duplicates: bool = True,
) -> torch.Tensor:
    """Combine edges from multiple sources into a single edge index.

    Merges edge indices from hierarchy, correlation, or other sources.
    Optionally removes duplicate edges.

    Args:
        *edge_indices: Variable number of edge_index tensors (2, num_edges)
        remove_duplicates: Whether to remove duplicate edges

    Returns:
        Combined edge_index tensor of shape (2, total_edges)

    Example:
        >>> hierarchy_edges = build_hierarchy_edges(tags)
        >>> correlation_edges = build_correlation_edges(data)
        >>> all_edges = combine_edge_sources(hierarchy_edges, correlation_edges)
    """
    # Filter out empty tensors
    non_empty = [e for e in edge_indices if e.numel() > 0]

    if not non_empty:
        return torch.zeros((2, 0), dtype=torch.long)

    # Concatenate all edges
    combined = torch.cat(non_empty, dim=1)

    if remove_duplicates and combined.size(1) > 0:
        # Convert to set of tuples for deduplication
        edges_set: set[tuple[int, int]] = set()
        for i in range(combined.size(1)):
            src, dst = combined[0, i].item(), combined[1, i].item()
            edges_set.add((src, dst))

        # Convert back to tensor
        if edges_set:
            edges_list = list(edges_set)
            combined = torch.tensor(
                [[e[0] for e in edges_list], [e[1] for e in edges_list]],
                dtype=torch.long,
            )
        else:
            combined = torch.zeros((2, 0), dtype=torch.long)

    return combined


def add_self_loops(edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
    """Add self-loops to an edge index.

    Self-loops are important for GNNs to incorporate a node's own features
    during message passing.

    Args:
        edge_index: Edge index tensor of shape (2, num_edges)
        num_nodes: Total number of nodes in the graph

    Returns:
        Edge index with self-loops added
    """
    # Create self-loop edges
    self_loops = torch.arange(num_nodes, dtype=torch.long)
    self_loop_edges = torch.stack([self_loops, self_loops], dim=0)

    # Combine with existing edges
    if edge_index.numel() == 0:
        return self_loop_edges

    return torch.cat([edge_index, self_loop_edges], dim=1)


# =============================================================================
# Dataset Class
# =============================================================================


class TagGraphDataset(Dataset):
    """PyTorch Geometric dataset for industrial tag graphs.

    This dataset creates graph structures from TagRecord objects, including:
    - Node features (from CharCNN embeddings or other sources)
    - Edge connectivity (from hierarchy and/or correlation)
    - Labels for classification tasks

    Each item in the dataset represents a graph (which could be a single
    connected component or the full tag hierarchy).

    Attributes:
        tag_records: List of TagRecord objects
        node_features: Pre-computed node features tensor
        edge_index: Graph connectivity
        labels: Classification labels (optional)
        transform: Optional transform function

    Example:
        >>> dataset = TagGraphDataset(
        ...     tag_records=tags,
        ...     node_features=charcnn_embeddings,
        ...     labels=property_labels
        ... )
        >>> data = dataset[0]  # Returns PyG Data object
    """

    def __init__(
        self,
        tag_records: Sequence[TagRecord],
        node_features: torch.Tensor | np.ndarray,
        edge_index: torch.Tensor | None = None,
        labels: torch.Tensor | np.ndarray | None = None,
        time_series_data: np.ndarray | None = None,
        correlation_threshold: float = 0.7,
        include_hierarchy_edges: bool = True,
        include_correlation_edges: bool = True,
        add_self_loops: bool = True,
        transform: Any | None = None,
    ) -> None:
        """Initialize the TagGraphDataset.

        Args:
            tag_records: Sequence of TagRecord objects
            node_features: Pre-computed node features (num_nodes, feature_dim)
            edge_index: Pre-computed edge index. If None, computed from
                        hierarchy and/or correlation.
            labels: Optional labels for each node (num_nodes,)
            time_series_data: Optional time series for correlation edges
                              (num_nodes, num_timesteps)
            correlation_threshold: Threshold for correlation-based edges
            include_hierarchy_edges: Whether to include hierarchy-based edges
            include_correlation_edges: Whether to include correlation-based edges
            add_self_loops: Whether to add self-loops to the graph
            transform: Optional PyG transform function
        """
        super().__init__(transform=transform)

        self.tag_records = list(tag_records)
        self.num_nodes = len(self.tag_records)

        # Convert node features to tensor
        if isinstance(node_features, np.ndarray):
            node_features = torch.from_numpy(node_features).float()
        self._node_features = node_features

        # Build node index mapping
        self._node_id_to_idx = {tag.tag_id: i for i, tag in enumerate(self.tag_records)}

        # Build or use provided edge index
        if edge_index is not None:
            self._edge_index = edge_index
        else:
            edge_sources: list[torch.Tensor] = []

            if include_hierarchy_edges:
                hierarchy_edges, _ = build_hierarchy_edges(
                    self.tag_records, self._node_id_to_idx
                )
                edge_sources.append(hierarchy_edges)

            if include_correlation_edges and time_series_data is not None:
                correlation_edges = build_correlation_edges(
                    time_series_data, threshold=correlation_threshold
                )
                edge_sources.append(correlation_edges)

            self._edge_index = combine_edge_sources(*edge_sources)

        # Add self-loops if requested
        if add_self_loops:
            self._edge_index = globals()["add_self_loops"](self._edge_index, self.num_nodes)

        # Convert labels to tensor
        if labels is not None:
            if isinstance(labels, np.ndarray):
                labels = torch.from_numpy(labels).long()
            self._labels = labels
        else:
            self._labels = None

        # We represent the entire graph as a single Data object
        # For batching multiple graphs, see PyG's DataLoader

    def len(self) -> int:
        """Return the number of graphs in the dataset.

        For this implementation, we have a single graph containing all tags.
        """
        return 1

    def get(self, idx: int) -> Data:
        """Get a graph by index.

        Args:
            idx: Graph index (only 0 is valid for single-graph dataset)

        Returns:
            PyG Data object with x, edge_index, and optionally y
        """
        if idx != 0:
            raise IndexError(f"Index {idx} out of range for dataset with 1 graph")

        data = Data(
            x=self._node_features,
            edge_index=self._edge_index,
            num_nodes=self.num_nodes,
        )

        if self._labels is not None:
            data.y = self._labels

        return data

    @property
    def node_features(self) -> torch.Tensor:
        """Get the node features tensor."""
        return self._node_features

    @property
    def edge_index(self) -> torch.Tensor:
        """Get the edge index tensor."""
        return self._edge_index

    @property
    def labels(self) -> torch.Tensor | None:
        """Get the labels tensor."""
        return self._labels

    @property
    def node_id_to_idx(self) -> dict[str, int]:
        """Get the mapping from tag_id to node index."""
        return self._node_id_to_idx

    def get_tag_by_idx(self, idx: int) -> TagRecord:
        """Get a TagRecord by its node index.

        Args:
            idx: Node index

        Returns:
            TagRecord at that index
        """
        return self.tag_records[idx]


class MultiGraphTagDataset(Dataset):
    """PyTorch Geometric dataset with multiple graphs.

    This dataset creates separate graphs for different subsets of tags
    (e.g., by server, by area, or by time window). Useful for batch
    training with PyG's DataLoader.

    Each item is a separate Data object representing one graph.
    """

    def __init__(
        self,
        graphs: Sequence[Data],
        transform: Any | None = None,
    ) -> None:
        """Initialize the MultiGraphTagDataset.

        Args:
            graphs: Sequence of PyG Data objects
            transform: Optional PyG transform function
        """
        super().__init__(transform=transform)
        self._graphs = list(graphs)

    def len(self) -> int:
        """Return the number of graphs."""
        return len(self._graphs)

    def get(self, idx: int) -> Data:
        """Get a graph by index."""
        return self._graphs[idx]

    @classmethod
    def from_tag_records_by_server(
        cls,
        tag_records: Sequence[TagRecord],
        node_features: torch.Tensor | np.ndarray,
        labels: torch.Tensor | np.ndarray | None = None,
        **kwargs: Any,
    ) -> MultiGraphTagDataset:
        """Create a dataset with one graph per source server.

        Args:
            tag_records: Sequence of TagRecord objects
            node_features: Node features for all tags (num_total_tags, feature_dim)
            labels: Labels for all tags (num_total_tags,)
            **kwargs: Additional arguments passed to TagGraphDataset

        Returns:
            MultiGraphTagDataset with one graph per server
        """
        if isinstance(node_features, np.ndarray):
            node_features = torch.from_numpy(node_features).float()
        if labels is not None and isinstance(labels, np.ndarray):
            labels = torch.from_numpy(labels).long()

        # Group tags by server
        server_to_tags: dict[str, list[tuple[int, TagRecord]]] = {}
        for i, tag in enumerate(tag_records):
            server = tag.source_server
            if server not in server_to_tags:
                server_to_tags[server] = []
            server_to_tags[server].append((i, tag))

        graphs: list[Data] = []
        for server, indexed_tags in server_to_tags.items():
            indices = [i for i, _ in indexed_tags]
            tags = [tag for _, tag in indexed_tags]

            # Extract features and labels for this subset
            subset_features = node_features[indices]
            subset_labels = labels[indices] if labels is not None else None

            # Create dataset for this server (single graph)
            server_dataset = TagGraphDataset(
                tag_records=tags,
                node_features=subset_features,
                labels=subset_labels,
                **kwargs,
            )

            graphs.append(server_dataset.get(0))

        return cls(graphs)


def create_node_features_from_embeddings(
    embeddings: torch.Tensor,
    statistical_features: torch.Tensor | None = None,
    metadata_features: torch.Tensor | None = None,
) -> torch.Tensor:
    """Combine multiple feature sources into node features.

    Concatenates CharCNN embeddings with optional statistical and metadata
    features to create comprehensive node feature vectors.

    Args:
        embeddings: Base embeddings (e.g., from CharCNN) of shape (num_nodes, embed_dim)
        statistical_features: Optional statistical features (mean, std, range)
                              of shape (num_nodes, stat_dim)
        metadata_features: Optional metadata features (data type encoding, etc.)
                           of shape (num_nodes, meta_dim)

    Returns:
        Combined features of shape (num_nodes, total_dim)
    """
    features = [embeddings]

    if statistical_features is not None:
        features.append(statistical_features)

    if metadata_features is not None:
        features.append(metadata_features)

    return torch.cat(features, dim=-1)


def compute_statistical_features(
    time_series_data: np.ndarray | torch.Tensor,
) -> torch.Tensor:
    """Compute statistical features from time-series data.

    Args:
        time_series_data: Array of shape (num_tags, num_timesteps)

    Returns:
        Statistical features of shape (num_tags, 5) containing:
            - mean
            - std
            - min
            - max
            - range (max - min)
    """
    if isinstance(time_series_data, torch.Tensor):
        data = time_series_data
        mean = data.mean(dim=1, keepdim=True)
        std = data.std(dim=1, keepdim=True)
        min_val = data.min(dim=1, keepdim=True).values
        max_val = data.max(dim=1, keepdim=True).values
        range_val = max_val - min_val
    else:
        data = time_series_data
        mean = data.mean(axis=1, keepdims=True)
        std = data.std(axis=1, keepdims=True)
        min_val = data.min(axis=1, keepdims=True)
        max_val = data.max(axis=1, keepdims=True)
        range_val = max_val - min_val

        # Convert to tensor
        mean = torch.from_numpy(mean).float()
        std = torch.from_numpy(std).float()
        min_val = torch.from_numpy(min_val).float()
        max_val = torch.from_numpy(max_val).float()
        range_val = torch.from_numpy(range_val).float()

    return torch.cat([mean, std, min_val, max_val, range_val], dim=1)
