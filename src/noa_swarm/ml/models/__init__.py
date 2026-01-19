"""Neural network models for semantic tag mapping.

This module provides:
- CharCNN: Character-level CNN for tag name classification
- CharacterTokenizer: Convert tag strings to character indices
- TagGraphGNN: GraphSAGE-based GNN for structural relationships
- TagGraphDataset: PyTorch Geometric dataset for tag graphs
- Graph construction utilities for building edges from hierarchy and correlation
"""

from noa_swarm.ml.models.charcnn import CharCNN, CharacterTokenizer
from noa_swarm.ml.models.gnn import (
    MultiGraphTagDataset,
    TagGraphDataset,
    TagGraphGNN,
    TagGraphGNNConfig,
    add_self_loops,
    build_correlation_edges,
    build_hierarchy_edges,
    combine_edge_sources,
    compute_statistical_features,
    create_node_features_from_embeddings,
)

__all__ = [
    # CharCNN
    "CharCNN",
    "CharacterTokenizer",
    # GNN
    "TagGraphGNN",
    "TagGraphGNNConfig",
    "TagGraphDataset",
    "MultiGraphTagDataset",
    # Graph construction utilities
    "build_hierarchy_edges",
    "build_correlation_edges",
    "combine_edge_sources",
    "add_self_loops",
    "create_node_features_from_embeddings",
    "compute_statistical_features",
]
