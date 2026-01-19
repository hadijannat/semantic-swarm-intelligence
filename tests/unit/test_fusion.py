"""Unit tests for fusion model and calibration utilities."""

from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from noa_swarm.ml.models.fusion import (
    FusionConfig,
    FusionModel,
    IRDIEntry,
    IRDIRetriever,
)
from noa_swarm.ml.training.calibration import (
    CalibrationResult,
    ReliabilityDiagramData,
    TemperatureScaler,
    compute_brier_score,
    compute_calibration_metrics,
    compute_ece,
    compute_reliability_diagram,
)


class TestFusionConfig:
    """Tests for FusionConfig dataclass."""

    def test_default_config(self) -> None:
        """Test default configuration values."""
        config = FusionConfig()
        assert config.num_property_classes == 26
        assert config.num_signal_roles == 12
        assert config.charcnn_embed_dim == 128
        assert config.gnn_embed_dim == 128
        assert config.initial_charcnn_weight == 0.5
        assert config.initial_gnn_weight == 0.5
        assert config.initial_temperature == 1.0
        assert config.use_gnn is True

    def test_custom_config(self) -> None:
        """Test custom configuration values."""
        config = FusionConfig(
            num_property_classes=10,
            num_signal_roles=5,
            charcnn_embed_dim=64,
            gnn_embed_dim=32,
            initial_charcnn_weight=0.7,
            initial_gnn_weight=0.3,
            initial_temperature=1.5,
            use_gnn=False,
        )
        assert config.num_property_classes == 10
        assert config.num_signal_roles == 5
        assert config.charcnn_embed_dim == 64
        assert config.gnn_embed_dim == 32
        assert config.initial_charcnn_weight == 0.7
        assert config.initial_gnn_weight == 0.3
        assert config.initial_temperature == 1.5
        assert config.use_gnn is False


class TestFusionModel:
    """Tests for FusionModel class."""

    @pytest.fixture
    def config(self) -> FusionConfig:
        """Create a FusionConfig for testing."""
        return FusionConfig(
            num_property_classes=10,
            num_signal_roles=5,
            charcnn_embed_dim=64,
            gnn_embed_dim=32,
            use_gnn=True,
        )

    @pytest.fixture
    def model(self, config: FusionConfig) -> FusionModel:
        """Create a FusionModel for testing."""
        return FusionModel(config)

    @pytest.fixture
    def charcnn_output(self, config: FusionConfig) -> dict[str, torch.Tensor]:
        """Create sample CharCNN output."""
        batch_size = 4
        return {
            "property_class": torch.randn(batch_size, config.num_property_classes),
            "signal_role": torch.randn(batch_size, config.num_signal_roles),
            "irdi_embedding": torch.randn(batch_size, config.charcnn_embed_dim),
        }

    @pytest.fixture
    def gnn_output(self, config: FusionConfig) -> dict[str, torch.Tensor]:
        """Create sample GNN output."""
        batch_size = 4
        return {
            "property_logits": torch.randn(batch_size, config.num_property_classes),
            "signal_logits": torch.randn(batch_size, config.num_signal_roles),
            "embedding": torch.randn(batch_size, config.gnn_embed_dim),
        }

    def test_model_init(self, model: FusionModel, config: FusionConfig) -> None:
        """Test model initialization."""
        assert model.config == config
        # Fusion weights should sum to 1
        assert torch.allclose(model.alpha + model.beta, torch.tensor(1.0), atol=1e-5)
        assert torch.allclose(model.alpha_signal + model.beta_signal, torch.tensor(1.0), atol=1e-5)
        # Temperature should be close to initial
        assert torch.allclose(model.temperature_property, torch.tensor(1.0), atol=1e-5)
        assert torch.allclose(model.temperature_signal, torch.tensor(1.0), atol=1e-5)

    def test_forward_with_gnn(
        self,
        model: FusionModel,
        config: FusionConfig,
        charcnn_output: dict[str, torch.Tensor],
        gnn_output: dict[str, torch.Tensor],
    ) -> None:
        """Test forward pass with GNN output."""
        outputs = model(charcnn_output, gnn_output)

        assert "property_class" in outputs
        assert "signal_role" in outputs
        assert "fused_embedding" in outputs
        assert "property_probs" in outputs
        assert "signal_probs" in outputs
        assert "fusion_weights" in outputs
        assert "temperatures" in outputs

        # Check shapes
        batch_size = 4
        assert outputs["property_class"].shape == (batch_size, config.num_property_classes)
        assert outputs["signal_role"].shape == (batch_size, config.num_signal_roles)
        assert outputs["fused_embedding"].shape == (batch_size, config.charcnn_embed_dim + config.gnn_embed_dim)

        # Check probabilities sum to 1
        assert torch.allclose(outputs["property_probs"].sum(dim=-1), torch.ones(batch_size), atol=1e-5)
        assert torch.allclose(outputs["signal_probs"].sum(dim=-1), torch.ones(batch_size), atol=1e-5)

    def test_forward_without_gnn(
        self,
        config: FusionConfig,
        charcnn_output: dict[str, torch.Tensor],
    ) -> None:
        """Test forward pass without GNN (CharCNN-only mode)."""
        config_no_gnn = FusionConfig(
            num_property_classes=config.num_property_classes,
            num_signal_roles=config.num_signal_roles,
            charcnn_embed_dim=config.charcnn_embed_dim,
            use_gnn=False,
        )
        model = FusionModel(config_no_gnn)
        outputs = model(charcnn_output, None)

        assert outputs["property_class"].shape == (4, config.num_property_classes)
        assert outputs["fused_embedding"].shape == (4, config.charcnn_embed_dim)

    def test_forward_return_uncalibrated(
        self,
        model: FusionModel,
        charcnn_output: dict[str, torch.Tensor],
        gnn_output: dict[str, torch.Tensor],
    ) -> None:
        """Test forward pass returning uncalibrated logits."""
        outputs = model(charcnn_output, gnn_output, return_uncalibrated=True)

        assert "property_class_uncalibrated" in outputs
        assert "signal_role_uncalibrated" in outputs

    def test_predict(
        self,
        model: FusionModel,
        config: FusionConfig,
        charcnn_output: dict[str, torch.Tensor],
        gnn_output: dict[str, torch.Tensor],
    ) -> None:
        """Test prediction mode."""
        preds = model.predict(charcnn_output, gnn_output)

        assert "property_class" in preds
        assert "property_probs" in preds
        assert "property_confidence" in preds
        assert "signal_role" in preds
        assert "signal_probs" in preds
        assert "signal_confidence" in preds
        assert "fused_embedding" in preds

        # Class predictions should be indices
        batch_size = 4
        assert preds["property_class"].shape == (batch_size,)
        assert preds["signal_role"].shape == (batch_size,)

        # Confidences should be between 0 and 1
        assert (preds["property_confidence"] >= 0).all()
        assert (preds["property_confidence"] <= 1).all()

    def test_set_temperature(self, model: FusionModel) -> None:
        """Test setting temperature values."""
        model.set_temperature(temperature_property=2.0, temperature_signal=1.5)

        assert torch.allclose(model.temperature_property, torch.tensor(2.0), atol=1e-4)
        assert torch.allclose(model.temperature_signal, torch.tensor(1.5), atol=1e-4)

    def test_freeze_unfreeze_fusion_weights(self, model: FusionModel) -> None:
        """Test freezing and unfreezing fusion weights."""
        model.freeze_fusion_weights()
        assert not model._alpha_logit.requires_grad
        assert not model._beta_logit.requires_grad

        model.unfreeze_fusion_weights()
        assert model._alpha_logit.requires_grad
        assert model._beta_logit.requires_grad

    def test_freeze_unfreeze_temperature(self, model: FusionModel) -> None:
        """Test freezing and unfreezing temperature parameters."""
        model.freeze_temperature()
        assert not model._log_temperature_property.requires_grad
        assert not model._log_temperature_signal.requires_grad

        model.unfreeze_temperature()
        assert model._log_temperature_property.requires_grad
        assert model._log_temperature_signal.requires_grad

    def test_temperature_effect(
        self,
        model: FusionModel,
        charcnn_output: dict[str, torch.Tensor],
        gnn_output: dict[str, torch.Tensor],
    ) -> None:
        """Test that temperature affects probability sharpness."""
        # Get outputs with default temperature
        outputs_t1 = model(charcnn_output, gnn_output)
        probs_t1 = outputs_t1["property_probs"]

        # Set higher temperature (softer probabilities)
        model.set_temperature(temperature_property=2.0)
        outputs_t2 = model(charcnn_output, gnn_output)
        probs_t2 = outputs_t2["property_probs"]

        # Higher temperature should produce softer (more uniform) probabilities
        # Entropy should be higher with higher temperature
        entropy_t1 = -(probs_t1 * torch.log(probs_t1 + 1e-10)).sum(dim=-1).mean()
        entropy_t2 = -(probs_t2 * torch.log(probs_t2 + 1e-10)).sum(dim=-1).mean()
        assert entropy_t2 > entropy_t1

    def test_fusion_weights_learnable(
        self,
        model: FusionModel,
        config: FusionConfig,
        charcnn_output: dict[str, torch.Tensor],
        gnn_output: dict[str, torch.Tensor],
    ) -> None:
        """Test that fusion weights are learnable."""
        # Create a simple loss and backpropagate
        outputs = model(charcnn_output, gnn_output)
        loss = outputs["property_class"].sum()
        loss.backward()

        # Gradients should exist for fusion weight parameters
        assert model._alpha_logit.grad is not None
        assert model._beta_logit.grad is not None

    def test_fused_embed_dim(self, model: FusionModel, config: FusionConfig) -> None:
        """Test fused embedding dimension property."""
        expected_dim = config.charcnn_embed_dim + config.gnn_embed_dim
        assert model.fused_embed_dim == expected_dim


class TestIRDIRetriever:
    """Tests for IRDIRetriever class."""

    @pytest.fixture
    def sample_entries(self) -> list[IRDIEntry]:
        """Create sample IRDI entries."""
        torch.manual_seed(42)
        return [
            IRDIEntry(
                irdi="0173-1#02-AAA001#001",
                embedding=torch.randn(64),
                metadata={"category": "flow_rate"},
            ),
            IRDIEntry(
                irdi="0173-1#02-AAA002#001",
                embedding=torch.randn(64),
                metadata={"category": "temperature"},
            ),
            IRDIEntry(
                irdi="0173-1#02-AAA003#001",
                embedding=torch.randn(64),
                metadata={"category": "pressure"},
            ),
        ]

    def test_init(self) -> None:
        """Test retriever initialization."""
        retriever = IRDIRetriever(metric="cosine")
        assert retriever.metric == "cosine"
        assert len(retriever) == 0

    def test_init_invalid_metric(self) -> None:
        """Test initialization with invalid metric."""
        with pytest.raises(ValueError, match="Unknown metric"):
            IRDIRetriever(metric="invalid")  # type: ignore

    def test_add_entry(self, sample_entries: list[IRDIEntry]) -> None:
        """Test adding entries."""
        retriever = IRDIRetriever()
        retriever.add_entry(sample_entries[0])
        assert len(retriever) == 1

    def test_add_entries(self, sample_entries: list[IRDIEntry]) -> None:
        """Test adding multiple entries."""
        retriever = IRDIRetriever()
        retriever.add_entries(sample_entries)
        assert len(retriever) == 3

    def test_add_from_dict(self) -> None:
        """Test adding entries from dictionary."""
        retriever = IRDIRetriever()
        irdi_embeddings = {
            "irdi1": torch.randn(64),
            "irdi2": torch.randn(64),
        }
        metadata = {"irdi1": {"category": "flow"}}
        retriever.add_from_dict(irdi_embeddings, metadata)
        assert len(retriever) == 2

    def test_build_index(self, sample_entries: list[IRDIEntry]) -> None:
        """Test building the index."""
        retriever = IRDIRetriever()
        retriever.add_entries(sample_entries)
        retriever.build_index()
        assert retriever._index_built

    def test_retrieve_without_build(self, sample_entries: list[IRDIEntry]) -> None:
        """Test retrieval without building index raises error."""
        retriever = IRDIRetriever()
        retriever.add_entries(sample_entries)
        query = torch.randn(64)

        with pytest.raises(RuntimeError, match="Index not built"):
            retriever.retrieve(query)

    def test_retrieve_cosine(self, sample_entries: list[IRDIEntry]) -> None:
        """Test retrieval with cosine similarity."""
        retriever = IRDIRetriever(metric="cosine")
        retriever.add_entries(sample_entries)
        retriever.build_index()

        # Query with same embedding as first entry (should be most similar)
        query = sample_entries[0].embedding
        results = retriever.retrieve(query, top_k=2)

        assert len(results) == 2
        # First result should be the entry itself (similarity ~1.0)
        assert results[0][0] == "0173-1#02-AAA001#001"
        assert results[0][1] > 0.99  # Very high similarity
        assert results[0][2] == {"category": "flow_rate"}

    def test_retrieve_euclidean(self, sample_entries: list[IRDIEntry]) -> None:
        """Test retrieval with euclidean distance."""
        retriever = IRDIRetriever(metric="euclidean")
        retriever.add_entries(sample_entries)
        retriever.build_index()

        # Query with same embedding as first entry
        query = sample_entries[0].embedding
        results = retriever.retrieve(query, top_k=2)

        assert len(results) == 2
        # First result should be the entry itself (distance ~0, similarity ~1)
        assert results[0][0] == "0173-1#02-AAA001#001"
        assert results[0][1] > 0.99

    def test_retrieve_batch(self, sample_entries: list[IRDIEntry]) -> None:
        """Test batch retrieval."""
        retriever = IRDIRetriever(metric="cosine")
        retriever.add_entries(sample_entries)
        retriever.build_index()

        # Batch of two queries
        queries = torch.stack([sample_entries[0].embedding, sample_entries[1].embedding])
        results = retriever.retrieve_batch(queries, top_k=2)

        assert len(results) == 2
        assert len(results[0]) == 2
        assert len(results[1]) == 2
        # Each query should find itself first
        assert results[0][0][0] == "0173-1#02-AAA001#001"
        assert results[1][0][0] == "0173-1#02-AAA002#001"

    def test_retrieve_2d_query(self, sample_entries: list[IRDIEntry]) -> None:
        """Test retrieval with 2D query (batch dim of 1)."""
        retriever = IRDIRetriever()
        retriever.add_entries(sample_entries)
        retriever.build_index()

        query = sample_entries[0].embedding.unsqueeze(0)
        results = retriever.retrieve(query, top_k=1)
        assert len(results) == 1

    def test_get_irdi_embedding(self, sample_entries: list[IRDIEntry]) -> None:
        """Test getting embedding by IRDI."""
        retriever = IRDIRetriever()
        retriever.add_entries(sample_entries)

        emb = retriever.get_irdi_embedding("0173-1#02-AAA001#001")
        assert emb is not None
        assert torch.equal(emb, sample_entries[0].embedding)

        # Non-existent IRDI
        assert retriever.get_irdi_embedding("nonexistent") is None

    def test_embed_dim(self, sample_entries: list[IRDIEntry]) -> None:
        """Test embed_dim property."""
        retriever = IRDIRetriever()
        assert retriever.embed_dim == 0  # Empty

        retriever.add_entries(sample_entries)
        assert retriever.embed_dim == 64

    def test_clear(self, sample_entries: list[IRDIEntry]) -> None:
        """Test clearing the retriever."""
        retriever = IRDIRetriever()
        retriever.add_entries(sample_entries)
        retriever.build_index()

        retriever.clear()
        assert len(retriever) == 0
        assert not retriever._index_built

    def test_empty_retrieval(self) -> None:
        """Test retrieval with empty index."""
        retriever = IRDIRetriever()
        retriever.build_index()

        query = torch.randn(64)
        results = retriever.retrieve(query, top_k=5)
        assert results == []


class TestTemperatureScaler:
    """Tests for TemperatureScaler class."""

    def test_init(self) -> None:
        """Test scaler initialization."""
        scaler = TemperatureScaler()
        assert torch.allclose(scaler.temperature, torch.tensor(1.0), atol=1e-5)

    def test_init_custom_temperature(self) -> None:
        """Test initialization with custom temperature."""
        scaler = TemperatureScaler(initial_temperature=2.0)
        assert torch.allclose(scaler.temperature, torch.tensor(2.0), atol=1e-4)

    def test_forward(self) -> None:
        """Test forward pass (temperature scaling)."""
        scaler = TemperatureScaler()
        scaler.set_temperature(2.0)

        logits = torch.tensor([[1.0, 2.0, 3.0]])
        scaled = scaler(logits)

        expected = logits / 2.0
        assert torch.allclose(scaled, expected, atol=1e-5)

    def test_fit(self) -> None:
        """Test fitting temperature on validation data."""
        # Create synthetic data where T=1 gives overconfident predictions
        torch.manual_seed(42)
        num_samples = 200
        num_classes = 5

        # Create logits that are too sharp (overconfident)
        logits = torch.randn(num_samples, num_classes) * 3.0
        labels = torch.randint(0, num_classes, (num_samples,))

        scaler = TemperatureScaler()
        result = scaler.fit(logits, labels, max_iter=50)

        assert isinstance(result, CalibrationResult)
        assert result.temperature > 0
        assert result.num_samples == num_samples

    def test_set_temperature(self) -> None:
        """Test manually setting temperature."""
        scaler = TemperatureScaler()
        scaler.set_temperature(1.5)
        assert torch.allclose(scaler.temperature, torch.tensor(1.5), atol=1e-4)


class TestComputeECE:
    """Tests for compute_ece function."""

    def test_perfect_calibration(self) -> None:
        """Test ECE with perfectly calibrated predictions."""
        # Perfect calibration: confidence = accuracy
        # Create predictions where accuracy matches confidence in each bin
        num_samples = 1000
        num_classes = 10
        torch.manual_seed(42)

        # Create uniform predictions (all confidence ~0.1 for 10 classes)
        probs = torch.ones(num_samples, num_classes) / num_classes
        labels = torch.randint(0, num_classes, (num_samples,))

        ece = compute_ece(probs, labels, n_bins=10)
        # For uniform predictions, ECE should be low
        assert ece < 0.1

    def test_overconfident_model(self) -> None:
        """Test ECE with overconfident predictions."""
        # Model is very confident but often wrong
        num_samples = 100
        num_classes = 5

        # High confidence predictions
        probs = torch.zeros(num_samples, num_classes)
        probs[:, 0] = 0.95  # Always predict class 0 with 95% confidence
        probs[:, 1:] = 0.05 / (num_classes - 1)

        # But actual labels are random
        torch.manual_seed(42)
        labels = torch.randint(0, num_classes, (num_samples,))

        ece = compute_ece(probs, labels)
        # ECE should be high for overconfident wrong predictions
        assert ece > 0.5

    def test_empty_input(self) -> None:
        """Test ECE with empty input."""
        probs = torch.zeros((0, 5))
        labels = torch.zeros((0,), dtype=torch.long)

        ece = compute_ece(probs, labels)
        assert ece == 0.0

    def test_numpy_input(self) -> None:
        """Test ECE with numpy arrays."""
        np.random.seed(42)
        probs = np.random.dirichlet(np.ones(5), size=100)
        labels = np.random.randint(0, 5, size=100)

        ece = compute_ece(probs, labels)
        assert 0.0 <= ece <= 1.0


class TestComputeBrierScore:
    """Tests for compute_brier_score function."""

    def test_perfect_predictions(self) -> None:
        """Test Brier score with perfect predictions."""
        probs = torch.tensor([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ])
        labels = torch.tensor([0, 1, 2])

        brier = compute_brier_score(probs, labels)
        assert brier == 0.0

    def test_worst_predictions(self) -> None:
        """Test Brier score with worst predictions."""
        probs = torch.tensor([
            [0.0, 1.0, 0.0],  # Predicts class 1, actual 0
            [1.0, 0.0, 0.0],  # Predicts class 0, actual 1
            [1.0, 0.0, 0.0],  # Predicts class 0, actual 2
        ])
        labels = torch.tensor([0, 1, 2])

        brier = compute_brier_score(probs, labels)
        # Each sample has Brier = (0-1)^2 + (1-0)^2 + 0 = 2 for 3 classes
        # Average = 2
        assert brier == 2.0

    def test_uniform_predictions(self) -> None:
        """Test Brier score with uniform predictions."""
        probs = torch.ones(10, 5) / 5
        labels = torch.zeros(10, dtype=torch.long)

        brier = compute_brier_score(probs, labels)
        # For uniform p=0.2: (0.2-1)^2 + 4*(0.2-0)^2 = 0.64 + 0.16 = 0.8
        assert abs(brier - 0.8) < 0.01

    def test_empty_input(self) -> None:
        """Test Brier score with empty input."""
        probs = torch.zeros((0, 5))
        labels = torch.zeros((0,), dtype=torch.long)

        brier = compute_brier_score(probs, labels)
        assert brier == 0.0


class TestComputeReliabilityDiagram:
    """Tests for compute_reliability_diagram function."""

    def test_basic(self) -> None:
        """Test basic reliability diagram computation."""
        torch.manual_seed(42)
        probs = torch.softmax(torch.randn(100, 5), dim=-1)
        labels = torch.randint(0, 5, (100,))

        data = compute_reliability_diagram(probs, labels, n_bins=10)

        assert isinstance(data, ReliabilityDiagramData)
        assert len(data.bin_confidences) == 10
        assert len(data.bin_accuracies) == 10
        assert len(data.bin_counts) == 10
        assert len(data.bin_edges) == 11
        assert 0.0 <= data.ece <= 1.0

    def test_bin_counts_sum(self) -> None:
        """Test that bin counts sum to total samples."""
        torch.manual_seed(42)
        num_samples = 100
        probs = torch.softmax(torch.randn(num_samples, 5), dim=-1)
        labels = torch.randint(0, 5, (num_samples,))

        data = compute_reliability_diagram(probs, labels, n_bins=15)

        assert sum(data.bin_counts) == num_samples


class TestComputeCalibrationMetrics:
    """Tests for compute_calibration_metrics function."""

    def test_all_metrics_computed(self) -> None:
        """Test that all metrics are computed."""
        torch.manual_seed(42)
        probs = torch.softmax(torch.randn(100, 5), dim=-1)
        labels = torch.randint(0, 5, (100,))

        metrics = compute_calibration_metrics(probs, labels)

        assert "ece" in metrics
        assert "brier" in metrics
        assert "accuracy" in metrics
        assert "avg_confidence" in metrics
        assert "calibration_gap" in metrics

    def test_metric_ranges(self) -> None:
        """Test that metrics are in valid ranges."""
        torch.manual_seed(42)
        probs = torch.softmax(torch.randn(100, 5), dim=-1)
        labels = torch.randint(0, 5, (100,))

        metrics = compute_calibration_metrics(probs, labels)

        assert 0.0 <= metrics["ece"] <= 1.0
        assert 0.0 <= metrics["brier"] <= 2.0  # Max Brier for C classes
        assert 0.0 <= metrics["accuracy"] <= 1.0
        assert 0.0 <= metrics["avg_confidence"] <= 1.0
        assert metrics["calibration_gap"] >= 0.0


class TestIntegration:
    """Integration tests for fusion and calibration."""

    def test_fusion_with_calibration(self) -> None:
        """Test fusion model with temperature calibration."""
        # Create fusion model
        config = FusionConfig(
            num_property_classes=10,
            num_signal_roles=5,
            charcnn_embed_dim=64,
            gnn_embed_dim=32,
        )
        model = FusionModel(config)

        # Create sample data
        batch_size = 100
        torch.manual_seed(42)

        charcnn_output = {
            "property_class": torch.randn(batch_size, config.num_property_classes) * 3,
            "signal_role": torch.randn(batch_size, config.num_signal_roles) * 3,
            "irdi_embedding": torch.randn(batch_size, config.charcnn_embed_dim),
        }
        gnn_output = {
            "property_logits": torch.randn(batch_size, config.num_property_classes) * 2,
            "signal_logits": torch.randn(batch_size, config.num_signal_roles) * 2,
            "embedding": torch.randn(batch_size, config.gnn_embed_dim),
        }

        # Get fused outputs
        outputs = model(charcnn_output, gnn_output, return_uncalibrated=True)

        # Compute ECE before explicit calibration
        labels = torch.randint(0, config.num_property_classes, (batch_size,))
        ece_before = compute_ece(outputs["property_probs"], labels)

        # Calibrate using TemperatureScaler
        scaler = TemperatureScaler()
        result = scaler.fit(outputs["property_class_uncalibrated"], labels)

        # Apply calibration to fusion model
        model.set_temperature(temperature_property=result.temperature)

        # Get new outputs
        outputs_after = model(charcnn_output, gnn_output)
        ece_after = compute_ece(outputs_after["property_probs"], labels)

        # ECE should typically improve (or at least not get much worse)
        # Note: With random data, improvement isn't guaranteed, but calibration should work
        assert ece_after <= ece_before + 0.1  # Allow small tolerance

    def test_irdi_retrieval_with_fusion_embeddings(self) -> None:
        """Test IRDI retrieval using fusion model embeddings."""
        # Create fusion model
        config = FusionConfig(
            num_property_classes=10,
            num_signal_roles=5,
            charcnn_embed_dim=64,
            gnn_embed_dim=32,
        )
        model = FusionModel(config)

        # Create IRDI retriever
        retriever = IRDIRetriever(metric="cosine")

        # Add some reference IRDIs
        for i in range(5):
            retriever.add_entry(IRDIEntry(
                irdi=f"IRDI-{i}",
                embedding=torch.randn(config.charcnn_embed_dim + config.gnn_embed_dim),
            ))
        retriever.build_index()

        # Generate query embedding from fusion model
        charcnn_output = {
            "property_class": torch.randn(1, config.num_property_classes),
            "signal_role": torch.randn(1, config.num_signal_roles),
            "irdi_embedding": torch.randn(1, config.charcnn_embed_dim),
        }
        gnn_output = {
            "property_logits": torch.randn(1, config.num_property_classes),
            "embedding": torch.randn(1, config.gnn_embed_dim),
        }

        outputs = model(charcnn_output, gnn_output)
        query_embedding = outputs["fused_embedding"].squeeze(0)

        # Retrieve similar IRDIs
        results = retriever.retrieve(query_embedding, top_k=3)

        assert len(results) == 3
        for irdi, score, metadata in results:
            assert irdi.startswith("IRDI-")
            # Cosine similarity ranges from -1 to 1
            assert -1.0 <= score <= 1.0
