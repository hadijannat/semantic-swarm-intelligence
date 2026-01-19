"""Unit tests for GNN model and graph construction utilities."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from torch_geometric.data import Data

from noa_swarm.common.schemas import TagRecord
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


class TestTagGraphGNNConfig:
    """Tests for TagGraphGNNConfig dataclass."""

    def test_default_config(self) -> None:
        """Test default configuration values."""
        config = TagGraphGNNConfig()
        assert config.input_dim == 128
        assert config.hidden_dim == 128
        assert config.output_dim == 128
        assert config.num_layers == 2
        assert config.aggregator == "mean"
        assert config.dropout == 0.5
        assert config.normalize is True
        assert config.use_batch_norm is True

    def test_custom_config(self) -> None:
        """Test custom configuration values."""
        config = TagGraphGNNConfig(
            input_dim=64,
            hidden_dim=256,
            output_dim=32,
            num_layers=3,
            aggregator="max",
            dropout=0.3,
            normalize=False,
            use_batch_norm=False,
        )
        assert config.input_dim == 64
        assert config.hidden_dim == 256
        assert config.output_dim == 32
        assert config.num_layers == 3
        assert config.aggregator == "max"
        assert config.dropout == 0.3
        assert config.normalize is False
        assert config.use_batch_norm is False


class TestTagGraphGNN:
    """Tests for TagGraphGNN model."""

    @pytest.fixture
    def model(self) -> TagGraphGNN:
        """Create a TagGraphGNN model for testing."""
        config = TagGraphGNNConfig(
            input_dim=64,
            hidden_dim=32,
            output_dim=32,
            num_layers=2,
            dropout=0.1,
        )
        return TagGraphGNN(config)

    @pytest.fixture
    def sample_graph(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Create sample graph data for testing."""
        # 10 nodes with 64 features each
        x = torch.randn(10, 64)
        # Simple chain graph: 0->1->2->...->9
        edge_index = torch.tensor(
            [[0, 1, 2, 3, 4, 5, 6, 7, 8], [1, 2, 3, 4, 5, 6, 7, 8, 9]],
            dtype=torch.long,
        )
        return x, edge_index

    def test_model_init(self, model: TagGraphGNN) -> None:
        """Test model initialization."""
        assert model.config.input_dim == 64
        assert model.config.hidden_dim == 32
        assert model.config.output_dim == 32
        assert len(model.convs) == 2

    def test_forward_pass(
        self, model: TagGraphGNN, sample_graph: tuple[torch.Tensor, torch.Tensor]
    ) -> None:
        """Test forward pass produces correct output shapes."""
        x, edge_index = sample_graph

        node_emb, graph_emb = model(x, edge_index)

        assert node_emb.shape == (10, 32)  # (num_nodes, output_dim)
        assert graph_emb is not None
        assert graph_emb.shape == (1, 32)  # (1 graph, output_dim)

    def test_forward_no_graph_embedding(
        self, model: TagGraphGNN, sample_graph: tuple[torch.Tensor, torch.Tensor]
    ) -> None:
        """Test forward pass without graph-level embedding."""
        x, edge_index = sample_graph

        node_emb, graph_emb = model(x, edge_index, return_graph_embedding=False)

        assert node_emb.shape == (10, 32)
        assert graph_emb is None

    def test_forward_with_batch(self, model: TagGraphGNN) -> None:
        """Test forward pass with batch tensor for multiple graphs."""
        # Two graphs: first has 5 nodes, second has 5 nodes
        x = torch.randn(10, 64)
        edge_index = torch.tensor(
            [[0, 1, 2, 5, 6, 7], [1, 2, 3, 6, 7, 8]], dtype=torch.long
        )
        batch = torch.tensor([0, 0, 0, 0, 0, 1, 1, 1, 1, 1], dtype=torch.long)

        node_emb, graph_emb = model(x, edge_index, batch=batch)

        assert node_emb.shape == (10, 32)
        assert graph_emb is not None
        assert graph_emb.shape == (2, 32)  # 2 graphs

    def test_get_node_embeddings(
        self, model: TagGraphGNN, sample_graph: tuple[torch.Tensor, torch.Tensor]
    ) -> None:
        """Test get_node_embeddings method."""
        x, edge_index = sample_graph

        node_emb = model.get_node_embeddings(x, edge_index)

        assert node_emb.shape == (10, 32)

    def test_get_graph_embedding(
        self, model: TagGraphGNN, sample_graph: tuple[torch.Tensor, torch.Tensor]
    ) -> None:
        """Test get_graph_embedding method."""
        x, edge_index = sample_graph

        graph_emb = model.get_graph_embedding(x, edge_index)

        assert graph_emb.shape == (1, 32)

    def test_normalization(
        self, sample_graph: tuple[torch.Tensor, torch.Tensor]
    ) -> None:
        """Test that embeddings are L2 normalized when config.normalize=True."""
        config = TagGraphGNNConfig(
            input_dim=64,
            hidden_dim=32,
            output_dim=32,
            normalize=True,
        )
        model = TagGraphGNN(config)
        x, edge_index = sample_graph

        node_emb, graph_emb = model(x, edge_index)

        # Check node embeddings are normalized
        norms = torch.norm(node_emb, p=2, dim=-1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)

        # Check graph embedding is normalized
        assert graph_emb is not None
        graph_norm = torch.norm(graph_emb, p=2, dim=-1)
        assert torch.allclose(graph_norm, torch.ones_like(graph_norm), atol=1e-5)

    def test_no_normalization(
        self, sample_graph: tuple[torch.Tensor, torch.Tensor]
    ) -> None:
        """Test that embeddings are not normalized when config.normalize=False."""
        config = TagGraphGNNConfig(
            input_dim=64,
            hidden_dim=32,
            output_dim=32,
            normalize=False,
        )
        model = TagGraphGNN(config)
        x, edge_index = sample_graph

        node_emb, _ = model(x, edge_index)

        # Embeddings should generally not have unit norm
        norms = torch.norm(node_emb, p=2, dim=-1)
        # At least some should differ from 1.0
        assert not torch.allclose(norms, torch.ones_like(norms), atol=1e-2)

    def test_gradient_flow(
        self, model: TagGraphGNN, sample_graph: tuple[torch.Tensor, torch.Tensor]
    ) -> None:
        """Test that gradients flow through the model."""
        x, edge_index = sample_graph

        node_emb, graph_emb = model(x, edge_index)
        assert graph_emb is not None

        # Compute a simple loss and backprop
        loss = node_emb.sum() + graph_emb.sum()
        loss.backward()

        # Check gradients exist on conv layers
        for conv in model.convs:
            if hasattr(conv, "lin_l") and conv.lin_l.weight.grad is not None:
                assert conv.lin_l.weight.grad is not None

    def test_different_aggregators(
        self, sample_graph: tuple[torch.Tensor, torch.Tensor]
    ) -> None:
        """Test different aggregation methods."""
        x, edge_index = sample_graph

        for aggr in ["mean", "max"]:
            config = TagGraphGNNConfig(
                input_dim=64,
                hidden_dim=32,
                output_dim=32,
                aggregator=aggr,  # type: ignore
            )
            model = TagGraphGNN(config)
            node_emb, graph_emb = model(x, edge_index)

            assert node_emb.shape == (10, 32)
            assert graph_emb is not None

    def test_multi_layer_config(
        self, sample_graph: tuple[torch.Tensor, torch.Tensor]
    ) -> None:
        """Test GNN with more than 2 layers."""
        config = TagGraphGNNConfig(
            input_dim=64,
            hidden_dim=32,
            output_dim=16,
            num_layers=4,
        )
        model = TagGraphGNN(config)
        x, edge_index = sample_graph

        node_emb, graph_emb = model(x, edge_index)

        assert node_emb.shape == (10, 16)
        assert len(model.convs) == 4

    def test_single_layer_config(
        self, sample_graph: tuple[torch.Tensor, torch.Tensor]
    ) -> None:
        """Test GNN with single layer."""
        config = TagGraphGNNConfig(
            input_dim=64,
            hidden_dim=32,
            output_dim=32,
            num_layers=1,
        )
        model = TagGraphGNN(config)
        x, edge_index = sample_graph

        node_emb, graph_emb = model(x, edge_index)

        assert node_emb.shape == (10, 32)
        assert len(model.convs) == 1

    def test_empty_graph(self, model: TagGraphGNN) -> None:
        """Test forward pass with empty edge index (isolated nodes)."""
        x = torch.randn(5, 64)
        edge_index = torch.zeros((2, 0), dtype=torch.long)

        node_emb, graph_emb = model(x, edge_index)

        assert node_emb.shape == (5, 32)
        assert graph_emb is not None
        assert graph_emb.shape == (1, 32)


class TestBuildHierarchyEdges:
    """Tests for build_hierarchy_edges function."""

    @pytest.fixture
    def sample_tags(self) -> list[TagRecord]:
        """Create sample TagRecord objects with hierarchy."""
        return [
            TagRecord(
                node_id="ns=2;s=Root",
                browse_name="Root",
                parent_path=[],
                source_server="opc.tcp://localhost:4840",
            ),
            TagRecord(
                node_id="ns=2;s=Area1",
                browse_name="Area1",
                parent_path=["Root"],
                source_server="opc.tcp://localhost:4840",
            ),
            TagRecord(
                node_id="ns=2;s=FIC-101",
                browse_name="FIC-101",
                parent_path=["Root", "Area1"],
                source_server="opc.tcp://localhost:4840",
            ),
            TagRecord(
                node_id="ns=2;s=TIC-101",
                browse_name="TIC-101",
                parent_path=["Root", "Area1"],
                source_server="opc.tcp://localhost:4840",
            ),
        ]

    def test_build_hierarchy_edges(self, sample_tags: list[TagRecord]) -> None:
        """Test building edges from hierarchy."""
        edge_index, node_id_to_idx = build_hierarchy_edges(sample_tags)

        assert isinstance(edge_index, torch.Tensor)
        assert edge_index.shape[0] == 2
        assert len(node_id_to_idx) == 4

    def test_empty_tags(self) -> None:
        """Test with empty tag list."""
        edge_index, node_id_to_idx = build_hierarchy_edges([])

        assert edge_index.shape == (2, 0)
        assert len(node_id_to_idx) == 0

    def test_single_tag_no_parent(self) -> None:
        """Test with single tag without parent."""
        tags = [
            TagRecord(
                node_id="ns=2;s=Tag1",
                browse_name="Tag1",
                parent_path=[],
                source_server="opc.tcp://localhost:4840",
            )
        ]
        edge_index, node_id_to_idx = build_hierarchy_edges(tags)

        assert edge_index.shape == (2, 0)  # No edges
        assert len(node_id_to_idx) == 1

    def test_provided_mapping(self, sample_tags: list[TagRecord]) -> None:
        """Test with pre-provided node_id_to_idx mapping."""
        custom_mapping = {tag.tag_id: i * 2 for i, tag in enumerate(sample_tags)}
        edge_index, returned_mapping = build_hierarchy_edges(sample_tags, custom_mapping)

        assert returned_mapping == custom_mapping


class TestBuildCorrelationEdges:
    """Tests for build_correlation_edges function."""

    def test_high_correlation(self) -> None:
        """Test edges created for highly correlated time series."""
        # Create two perfectly correlated series
        np.random.seed(42)
        base = np.random.randn(100)
        data = np.array([base, base, np.random.randn(100)])  # First two are identical

        edge_index = build_correlation_edges(data, threshold=0.9)

        # Should have edges between nodes 0 and 1 (bidirectional)
        assert edge_index.shape[0] == 2
        assert edge_index.shape[1] >= 2  # At least one bidirectional edge

    def test_no_correlation(self) -> None:
        """Test no edges for uncorrelated time series."""
        np.random.seed(42)
        # Completely independent random series
        data = np.random.randn(5, 1000)

        edge_index = build_correlation_edges(data, threshold=0.9)

        # Should have very few or no edges (random series unlikely to correlate >0.9)
        assert edge_index.shape[0] == 2

    def test_threshold_effect(self) -> None:
        """Test that lower threshold produces more edges."""
        np.random.seed(42)
        data = np.random.randn(5, 100)

        edges_high = build_correlation_edges(data, threshold=0.9)
        edges_low = build_correlation_edges(data, threshold=0.1)

        # Lower threshold should produce more or equal edges
        assert edges_low.shape[1] >= edges_high.shape[1]

    def test_torch_tensor_input(self) -> None:
        """Test with torch tensor input."""
        data = torch.randn(5, 100)

        edge_index = build_correlation_edges(data, threshold=0.5)

        assert isinstance(edge_index, torch.Tensor)
        assert edge_index.shape[0] == 2

    def test_single_tag(self) -> None:
        """Test with single tag (no edges possible)."""
        data = np.random.randn(1, 100)

        edge_index = build_correlation_edges(data, threshold=0.5)

        assert edge_index.shape == (2, 0)

    def test_two_timesteps(self) -> None:
        """Test with minimal timesteps."""
        data = np.array([[1, 2], [1, 2], [2, 1]])  # 3 tags, 2 timesteps

        edge_index = build_correlation_edges(data, threshold=0.5)

        assert edge_index.shape[0] == 2


class TestCombineEdgeSources:
    """Tests for combine_edge_sources function."""

    def test_combine_two_sources(self) -> None:
        """Test combining two edge sources."""
        edges1 = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
        edges2 = torch.tensor([[2, 3], [3, 4]], dtype=torch.long)

        combined = combine_edge_sources(edges1, edges2)

        assert combined.shape[0] == 2
        assert combined.shape[1] == 4

    def test_remove_duplicates(self) -> None:
        """Test duplicate removal."""
        edges1 = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
        edges2 = torch.tensor([[0, 2], [1, 3]], dtype=torch.long)  # (0,1) is duplicate

        combined = combine_edge_sources(edges1, edges2, remove_duplicates=True)

        # Should have 3 unique edges: (0,1), (1,2), (2,3)
        assert combined.shape[0] == 2
        assert combined.shape[1] == 3

    def test_keep_duplicates(self) -> None:
        """Test keeping duplicates."""
        edges1 = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
        edges2 = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)  # Duplicates

        combined = combine_edge_sources(edges1, edges2, remove_duplicates=False)

        assert combined.shape[1] == 4  # All edges kept

    def test_empty_sources(self) -> None:
        """Test with empty edge sources."""
        edges1 = torch.zeros((2, 0), dtype=torch.long)
        edges2 = torch.zeros((2, 0), dtype=torch.long)

        combined = combine_edge_sources(edges1, edges2)

        assert combined.shape == (2, 0)

    def test_single_source(self) -> None:
        """Test with single edge source."""
        edges = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)

        combined = combine_edge_sources(edges)

        assert torch.equal(combined, edges)


class TestAddSelfLoops:
    """Tests for add_self_loops function."""

    def test_add_self_loops(self) -> None:
        """Test adding self loops to existing edges."""
        edge_index = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)

        result = add_self_loops(edge_index, num_nodes=3)

        assert result.shape[0] == 2
        assert result.shape[1] == 5  # 2 original + 3 self-loops

    def test_add_self_loops_empty(self) -> None:
        """Test adding self loops to empty edge index."""
        edge_index = torch.zeros((2, 0), dtype=torch.long)

        result = add_self_loops(edge_index, num_nodes=3)

        assert result.shape == (2, 3)  # 3 self-loops only


class TestTagGraphDataset:
    """Tests for TagGraphDataset class."""

    @pytest.fixture
    def sample_tags(self) -> list[TagRecord]:
        """Create sample TagRecord objects."""
        return [
            TagRecord(
                node_id="ns=2;s=Tag1",
                browse_name="Tag1",
                parent_path=[],
                source_server="opc.tcp://localhost:4840",
            ),
            TagRecord(
                node_id="ns=2;s=Tag2",
                browse_name="Tag2",
                parent_path=["Tag1"],
                source_server="opc.tcp://localhost:4840",
            ),
            TagRecord(
                node_id="ns=2;s=Tag3",
                browse_name="Tag3",
                parent_path=["Tag1"],
                source_server="opc.tcp://localhost:4840",
            ),
        ]

    @pytest.fixture
    def sample_features(self) -> torch.Tensor:
        """Create sample node features."""
        return torch.randn(3, 64)

    def test_dataset_creation(
        self, sample_tags: list[TagRecord], sample_features: torch.Tensor
    ) -> None:
        """Test dataset creation."""
        dataset = TagGraphDataset(
            tag_records=sample_tags,
            node_features=sample_features,
        )

        assert len(dataset) == 1  # Single graph
        assert dataset.num_nodes == 3

    def test_dataset_get(
        self, sample_tags: list[TagRecord], sample_features: torch.Tensor
    ) -> None:
        """Test getting a graph from the dataset."""
        dataset = TagGraphDataset(
            tag_records=sample_tags,
            node_features=sample_features,
        )

        data = dataset[0]

        assert isinstance(data, Data)
        assert data.x.shape == (3, 64)
        assert data.edge_index is not None
        assert data.num_nodes == 3

    def test_dataset_with_labels(
        self, sample_tags: list[TagRecord], sample_features: torch.Tensor
    ) -> None:
        """Test dataset with labels."""
        labels = torch.tensor([0, 1, 2])
        dataset = TagGraphDataset(
            tag_records=sample_tags,
            node_features=sample_features,
            labels=labels,
        )

        data = dataset[0]

        assert data.y is not None
        assert torch.equal(data.y, labels)

    def test_dataset_with_correlation_edges(
        self, sample_tags: list[TagRecord], sample_features: torch.Tensor
    ) -> None:
        """Test dataset with correlation-based edges."""
        time_series = np.random.randn(3, 100)
        dataset = TagGraphDataset(
            tag_records=sample_tags,
            node_features=sample_features,
            time_series_data=time_series,
            include_correlation_edges=True,
            include_hierarchy_edges=False,
        )

        data = dataset[0]

        assert data.edge_index is not None

    def test_dataset_no_self_loops(
        self, sample_tags: list[TagRecord], sample_features: torch.Tensor
    ) -> None:
        """Test dataset without self-loops."""
        dataset_with = TagGraphDataset(
            tag_records=sample_tags,
            node_features=sample_features,
            add_self_loops=True,
        )
        dataset_without = TagGraphDataset(
            tag_records=sample_tags,
            node_features=sample_features,
            add_self_loops=False,
        )

        # Dataset with self-loops should have more edges
        assert dataset_with.edge_index.shape[1] >= dataset_without.edge_index.shape[1]

    def test_dataset_properties(
        self, sample_tags: list[TagRecord], sample_features: torch.Tensor
    ) -> None:
        """Test dataset property accessors."""
        labels = torch.tensor([0, 1, 2])
        dataset = TagGraphDataset(
            tag_records=sample_tags,
            node_features=sample_features,
            labels=labels,
        )

        assert torch.equal(dataset.node_features, sample_features)
        assert dataset.labels is not None
        assert len(dataset.node_id_to_idx) == 3

    def test_get_tag_by_idx(
        self, sample_tags: list[TagRecord], sample_features: torch.Tensor
    ) -> None:
        """Test getting a tag by index."""
        dataset = TagGraphDataset(
            tag_records=sample_tags,
            node_features=sample_features,
        )

        tag = dataset.get_tag_by_idx(0)

        assert tag == sample_tags[0]

    def test_numpy_features(
        self, sample_tags: list[TagRecord]
    ) -> None:
        """Test dataset with numpy array features."""
        features = np.random.randn(3, 64).astype(np.float32)
        labels = np.array([0, 1, 2])

        dataset = TagGraphDataset(
            tag_records=sample_tags,
            node_features=features,
            labels=labels,
        )

        assert isinstance(dataset.node_features, torch.Tensor)
        assert isinstance(dataset.labels, torch.Tensor)

    def test_invalid_index(
        self, sample_tags: list[TagRecord], sample_features: torch.Tensor
    ) -> None:
        """Test accessing invalid index raises error."""
        dataset = TagGraphDataset(
            tag_records=sample_tags,
            node_features=sample_features,
        )

        with pytest.raises(IndexError):
            dataset[1]


class TestMultiGraphTagDataset:
    """Tests for MultiGraphTagDataset class."""

    def test_creation_from_graphs(self) -> None:
        """Test creating dataset from list of graphs."""
        graphs = [
            Data(x=torch.randn(5, 32), edge_index=torch.zeros((2, 0), dtype=torch.long)),
            Data(x=torch.randn(3, 32), edge_index=torch.zeros((2, 0), dtype=torch.long)),
        ]

        dataset = MultiGraphTagDataset(graphs)

        assert len(dataset) == 2

    def test_get_graph(self) -> None:
        """Test getting a graph from the dataset."""
        graphs = [
            Data(x=torch.randn(5, 32), edge_index=torch.zeros((2, 0), dtype=torch.long)),
            Data(x=torch.randn(3, 32), edge_index=torch.zeros((2, 0), dtype=torch.long)),
        ]

        dataset = MultiGraphTagDataset(graphs)

        data = dataset[0]
        assert data.x.shape == (5, 32)

        data = dataset[1]
        assert data.x.shape == (3, 32)

    def test_from_tag_records_by_server(self) -> None:
        """Test creating dataset grouped by server."""
        tags = [
            TagRecord(
                node_id="ns=2;s=Tag1",
                browse_name="Tag1",
                source_server="opc.tcp://server1:4840",
            ),
            TagRecord(
                node_id="ns=2;s=Tag2",
                browse_name="Tag2",
                source_server="opc.tcp://server1:4840",
            ),
            TagRecord(
                node_id="ns=2;s=Tag3",
                browse_name="Tag3",
                source_server="opc.tcp://server2:4840",
            ),
        ]
        features = torch.randn(3, 32)
        labels = torch.tensor([0, 1, 2])

        dataset = MultiGraphTagDataset.from_tag_records_by_server(
            tag_records=tags,
            node_features=features,
            labels=labels,
        )

        assert len(dataset) == 2  # Two servers


class TestComputeStatisticalFeatures:
    """Tests for compute_statistical_features function."""

    def test_numpy_input(self) -> None:
        """Test with numpy array input."""
        data = np.array([[1, 2, 3, 4, 5], [10, 20, 30, 40, 50]])

        features = compute_statistical_features(data)

        assert features.shape == (2, 5)  # mean, std, min, max, range

    def test_torch_input(self) -> None:
        """Test with torch tensor input."""
        data = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0], [10.0, 20.0, 30.0, 40.0, 50.0]])

        features = compute_statistical_features(data)

        assert features.shape == (2, 5)

    def test_feature_values(self) -> None:
        """Test computed feature values."""
        data = np.array([[1.0, 2.0, 3.0, 4.0, 5.0]])

        features = compute_statistical_features(data)

        # mean=3, std~=1.41, min=1, max=5, range=4
        assert abs(features[0, 0].item() - 3.0) < 0.01  # mean
        assert abs(features[0, 2].item() - 1.0) < 0.01  # min
        assert abs(features[0, 3].item() - 5.0) < 0.01  # max
        assert abs(features[0, 4].item() - 4.0) < 0.01  # range


class TestCreateNodeFeaturesFromEmbeddings:
    """Tests for create_node_features_from_embeddings function."""

    def test_embeddings_only(self) -> None:
        """Test with embeddings only."""
        embeddings = torch.randn(10, 64)

        features = create_node_features_from_embeddings(embeddings)

        assert torch.equal(features, embeddings)

    def test_with_statistical_features(self) -> None:
        """Test with statistical features."""
        embeddings = torch.randn(10, 64)
        stats = torch.randn(10, 5)

        features = create_node_features_from_embeddings(
            embeddings, statistical_features=stats
        )

        assert features.shape == (10, 69)

    def test_with_all_features(self) -> None:
        """Test with all feature types."""
        embeddings = torch.randn(10, 64)
        stats = torch.randn(10, 5)
        metadata = torch.randn(10, 8)

        features = create_node_features_from_embeddings(
            embeddings, statistical_features=stats, metadata_features=metadata
        )

        assert features.shape == (10, 77)


class TestGNNWithDataset:
    """Integration tests for GNN model with TagGraphDataset."""

    def test_gnn_forward_with_dataset(self) -> None:
        """Test GNN forward pass using TagGraphDataset."""
        # Create sample data
        tags = [
            TagRecord(
                node_id=f"ns=2;s=Tag{i}",
                browse_name=f"Tag{i}",
                source_server="opc.tcp://localhost:4840",
            )
            for i in range(10)
        ]
        features = torch.randn(10, 64)
        labels = torch.randint(0, 5, (10,))

        # Create dataset
        dataset = TagGraphDataset(
            tag_records=tags,
            node_features=features,
            labels=labels,
        )

        # Create model
        config = TagGraphGNNConfig(
            input_dim=64,
            hidden_dim=32,
            output_dim=32,
        )
        model = TagGraphGNN(config)

        # Get data and run forward pass
        data = dataset[0]
        node_emb, graph_emb = model(data.x, data.edge_index)

        assert node_emb.shape == (10, 32)
        assert graph_emb is not None
        assert graph_emb.shape == (1, 32)

    def test_gnn_training_step(self) -> None:
        """Test a simple training step with GNN."""
        # Create sample data
        tags = [
            TagRecord(
                node_id=f"ns=2;s=Tag{i}",
                browse_name=f"Tag{i}",
                source_server="opc.tcp://localhost:4840",
            )
            for i in range(10)
        ]
        features = torch.randn(10, 64)
        labels = torch.randint(0, 5, (10,))

        # Create dataset
        dataset = TagGraphDataset(
            tag_records=tags,
            node_features=features,
            labels=labels,
        )

        # Create model with node-level classifier
        config = TagGraphGNNConfig(
            input_dim=64,
            hidden_dim=32,
            output_dim=5,  # 5 classes
            normalize=False,  # Don't normalize for classification
        )
        model = TagGraphGNN(config)

        # Get data
        data = dataset[0]
        node_emb, _ = model(data.x, data.edge_index, return_graph_embedding=False)

        # Compute loss
        loss = torch.nn.functional.cross_entropy(node_emb, data.y)
        loss.backward()

        # Check gradients exist
        for conv in model.convs:
            if hasattr(conv, "lin_l") and conv.lin_l.weight.grad is not None:
                assert conv.lin_l.weight.grad is not None
