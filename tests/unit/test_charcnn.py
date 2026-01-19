"""Unit tests for CharCNN model and related components."""

from __future__ import annotations

import pytest
import torch
import numpy as np

from noa_swarm.ml.models.charcnn import (
    CharacterTokenizer,
    CharCNN,
    CharCNNConfig,
    PROPERTY_CLASSES,
    SIGNAL_ROLES,
    DEFAULT_ALPHABET,
    create_label_mappings,
    get_property_class_from_category,
    get_signal_role_from_prefix,
)
from noa_swarm.ml.training.eval import (
    compute_accuracy,
    compute_per_class_accuracy,
    compute_macro_f1,
    compute_confusion_matrix,
    compute_top_k_accuracy,
)
from noa_swarm.ml.training.train_local import (
    TagDataset,
    TrainingConfig,
    TrainingHistory,
    train_model,
)
from noa_swarm.ml.datasets import SyntheticTagGenerator


class TestCharacterTokenizer:
    """Tests for CharacterTokenizer class."""

    def test_init_default_alphabet(self) -> None:
        """Test tokenizer initialization with default alphabet."""
        tokenizer = CharacterTokenizer()
        assert tokenizer.alphabet == DEFAULT_ALPHABET
        assert tokenizer.vocab_size == len(DEFAULT_ALPHABET) + 2
        assert tokenizer.pad_idx == len(DEFAULT_ALPHABET) + 1
        assert tokenizer.unk_idx == len(DEFAULT_ALPHABET)

    def test_init_custom_alphabet(self) -> None:
        """Test tokenizer initialization with custom alphabet."""
        tokenizer = CharacterTokenizer(alphabet="ABC", max_length=10)
        assert tokenizer.alphabet == "ABC"
        assert tokenizer.vocab_size == 5  # A, B, C, UNK, PAD
        assert tokenizer.max_length == 10

    def test_encode_simple(self) -> None:
        """Test encoding a simple string."""
        tokenizer = CharacterTokenizer(alphabet="ABC", max_length=5)
        encoded = tokenizer.encode("AB")

        # Expected: [0, 1, PAD, PAD, PAD]
        assert encoded.shape == (5,)
        assert encoded[0].item() == 0  # A
        assert encoded[1].item() == 1  # B
        assert encoded[2].item() == tokenizer.pad_idx
        assert encoded[3].item() == tokenizer.pad_idx
        assert encoded[4].item() == tokenizer.pad_idx

    def test_encode_with_unknown_chars(self) -> None:
        """Test encoding with unknown characters."""
        tokenizer = CharacterTokenizer(alphabet="ABC", max_length=5)
        encoded = tokenizer.encode("AXB")

        assert encoded[0].item() == 0  # A
        assert encoded[1].item() == tokenizer.unk_idx  # X is unknown
        assert encoded[2].item() == 1  # B

    def test_encode_truncation(self) -> None:
        """Test that long sequences are truncated."""
        tokenizer = CharacterTokenizer(alphabet="ABC", max_length=3)
        encoded = tokenizer.encode("ABCABC")

        assert encoded.shape == (3,)
        assert encoded.tolist() == [0, 1, 2]  # ABC, truncated

    def test_encode_empty_string(self) -> None:
        """Test encoding an empty string."""
        tokenizer = CharacterTokenizer(alphabet="ABC", max_length=3)
        encoded = tokenizer.encode("")

        assert encoded.shape == (3,)
        assert all(idx == tokenizer.pad_idx for idx in encoded.tolist())

    def test_encode_case_insensitive(self) -> None:
        """Test case-insensitive encoding (default)."""
        tokenizer = CharacterTokenizer(alphabet="ABC", max_length=3, case_sensitive=False)
        upper = tokenizer.encode("ABC")
        lower = tokenizer.encode("abc")

        assert torch.equal(upper, lower)

    def test_encode_case_sensitive(self) -> None:
        """Test case-sensitive encoding."""
        tokenizer = CharacterTokenizer(alphabet="ABC", max_length=3, case_sensitive=True)
        upper = tokenizer.encode("ABC")
        lower = tokenizer.encode("abc")

        # Lowercase should be unknown since alphabet only has uppercase
        assert upper[0].item() == 0  # A
        assert lower[0].item() == tokenizer.unk_idx  # a is unknown

    def test_encode_batch(self) -> None:
        """Test batch encoding."""
        tokenizer = CharacterTokenizer(alphabet="ABC", max_length=3)
        texts = ["AB", "BC", "A"]
        encoded = tokenizer.encode_batch(texts)

        assert encoded.shape == (3, 3)
        assert encoded[0, 0].item() == 0  # A
        assert encoded[1, 0].item() == 1  # B
        assert encoded[2, 0].item() == 0  # A

    def test_decode(self) -> None:
        """Test decoding back to string."""
        tokenizer = CharacterTokenizer(alphabet="ABC", max_length=5)
        original = "ABC"
        encoded = tokenizer.encode(original)
        decoded = tokenizer.decode(encoded)

        assert decoded == original

    def test_decode_batch(self) -> None:
        """Test batch decoding."""
        tokenizer = CharacterTokenizer(alphabet="ABC", max_length=5)
        texts = ["AB", "BC"]
        encoded = tokenizer.encode_batch(texts)
        decoded = tokenizer.decode_batch(encoded)

        assert decoded == texts

    def test_industrial_tags(self) -> None:
        """Test encoding industrial tag names."""
        tokenizer = CharacterTokenizer()

        tags = ["FIC-101", "TIC-2301", "PSH-001", "LCV_103.PV"]
        encoded = tokenizer.encode_batch(tags)

        assert encoded.shape == (4, 32)

        # Verify roundtrip
        decoded = tokenizer.decode_batch(encoded)
        for original, recovered in zip(tags, decoded, strict=False):
            assert recovered == original.upper()  # Should match after uppercase


class TestCharCNN:
    """Tests for CharCNN model."""

    @pytest.fixture
    def model(self) -> CharCNN:
        """Create a CharCNN model for testing."""
        config = CharCNNConfig(
            embedding_dim=32,
            conv1_channels=64,
            conv2_channels=64,
            hidden_dim=64,
            max_seq_length=16,
        )
        return CharCNN(config)

    @pytest.fixture
    def tokenizer(self) -> CharacterTokenizer:
        """Create a tokenizer for testing."""
        return CharacterTokenizer(max_length=16)

    def test_model_init(self, model: CharCNN) -> None:
        """Test model initialization."""
        assert model.config.embedding_dim == 32
        assert model.config.conv1_channels == 64
        assert model.config.hidden_dim == 64

    def test_forward_pass(self, model: CharCNN, tokenizer: CharacterTokenizer) -> None:
        """Test forward pass produces correct output shapes."""
        tags = ["FIC-101", "TIC-2301"]
        x = tokenizer.encode_batch(tags)

        outputs = model(x)

        assert "property_class" in outputs
        assert "signal_role" in outputs
        assert "irdi_embedding" in outputs
        assert "features" in outputs

        assert outputs["property_class"].shape == (2, model.config.num_property_classes)
        assert outputs["signal_role"].shape == (2, model.config.num_signal_roles)
        assert outputs["irdi_embedding"].shape == (2, model.config.irdi_embedding_dim)
        assert outputs["features"].shape == (2, model.config.hidden_dim)

    def test_forward_with_embeddings(
        self, model: CharCNN, tokenizer: CharacterTokenizer
    ) -> None:
        """Test forward pass with embedding normalization."""
        tags = ["FIC-101"]
        x = tokenizer.encode_batch(tags)

        outputs = model(x, return_embeddings=True)

        # Embeddings should be L2 normalized
        embedding = outputs["irdi_embedding"]
        norm = torch.norm(embedding, p=2, dim=-1)
        assert torch.allclose(norm, torch.ones_like(norm), atol=1e-5)

    def test_predict(self, model: CharCNN, tokenizer: CharacterTokenizer) -> None:
        """Test prediction mode."""
        tags = ["FIC-101", "TIC-2301"]
        x = tokenizer.encode_batch(tags)

        preds = model.predict(x)

        assert "property_class" in preds
        assert "property_probs" in preds
        assert "signal_role" in preds
        assert "signal_probs" in preds
        assert "irdi_embedding" in preds

        # Class predictions should be indices
        assert preds["property_class"].shape == (2,)
        assert preds["signal_role"].shape == (2,)

        # Probabilities should sum to 1
        assert torch.allclose(
            preds["property_probs"].sum(dim=-1),
            torch.ones(2),
            atol=1e-5,
        )

    def test_single_sample_forward(
        self, model: CharCNN, tokenizer: CharacterTokenizer
    ) -> None:
        """Test forward pass with single sample."""
        x = tokenizer.encode("FIC-101").unsqueeze(0)  # Add batch dim

        outputs = model(x)

        assert outputs["property_class"].shape == (1, model.config.num_property_classes)

    def test_gradient_flow(self, model: CharCNN, tokenizer: CharacterTokenizer) -> None:
        """Test that gradients flow through the model."""
        tags = ["FIC-101"]
        x = tokenizer.encode_batch(tags)
        property_target = torch.tensor([0])
        signal_target = torch.tensor([0])

        outputs = model(x)
        # Use all three heads so all parameters receive gradients
        loss = torch.nn.functional.cross_entropy(outputs["property_class"], property_target)
        loss += torch.nn.functional.cross_entropy(outputs["signal_role"], signal_target)
        loss += outputs["irdi_embedding"].sum()  # Use embedding output
        loss.backward()

        # Check that at least key parameters have gradients
        # (embedding layer should definitely have gradients)
        assert model.embedding.weight.grad is not None
        assert model.conv1.weight.grad is not None
        assert model.fc.weight.grad is not None
        assert model.property_head.weight.grad is not None

    def test_model_parameter_count(self) -> None:
        """Test that model has reasonable parameter count."""
        config = CharCNNConfig()
        model = CharCNN(config)

        param_count = sum(p.numel() for p in model.parameters())

        # Should have a reasonable number of parameters (not too large for edge deployment)
        assert param_count < 10_000_000  # Less than 10M parameters

    def test_get_class_names(self, model: CharCNN) -> None:
        """Test class name retrieval."""
        assert model.get_property_class_name(0) == PROPERTY_CLASSES[0]
        assert model.get_signal_role_name(0) == SIGNAL_ROLES[0]
        assert model.get_property_class_name(999) == "unknown"
        assert model.get_signal_role_name(999) == "unknown"


class TestLabelMappings:
    """Tests for label mapping functions."""

    def test_create_label_mappings(self) -> None:
        """Test label mapping creation."""
        property_to_idx, role_to_idx = create_label_mappings()

        assert len(property_to_idx) == len(PROPERTY_CLASSES)
        assert len(role_to_idx) == len(SIGNAL_ROLES)

        assert property_to_idx["flow_rate"] == 0
        assert role_to_idx["indicator"] == 0

    def test_get_property_class_from_category(self) -> None:
        """Test category to property class mapping."""
        assert get_property_class_from_category("flow_rate") == 0
        assert get_property_class_from_category("temperature") == PROPERTY_CLASSES.index("temperature")

        # Test variations
        assert get_property_class_from_category("flow_rate_volumetric") == 0

        # Unknown should map to last class
        unknown_idx = get_property_class_from_category("nonexistent_category")
        assert unknown_idx == PROPERTY_CLASSES.index("unknown")

    def test_get_signal_role_from_prefix(self) -> None:
        """Test prefix to signal role mapping."""
        # Indicator (ends with I)
        assert get_signal_role_from_prefix("FI") == SIGNAL_ROLES.index("indicator")

        # Controller (ends with C)
        assert get_signal_role_from_prefix("FIC") == SIGNAL_ROLES.index("controller")
        assert get_signal_role_from_prefix("TIC") == SIGNAL_ROLES.index("controller")

        # Transmitter (ends with T)
        assert get_signal_role_from_prefix("FT") == SIGNAL_ROLES.index("transmitter")

        # Valve (ends with V)
        assert get_signal_role_from_prefix("FCV") == SIGNAL_ROLES.index("valve")

        # Alarm patterns - the function looks for "SH"/"SL" substrings
        assert get_signal_role_from_prefix("TSHH") == SIGNAL_ROLES.index("alarm")
        assert get_signal_role_from_prefix("TSLL") == SIGNAL_ROLES.index("alarm")


class TestEvaluationMetrics:
    """Tests for evaluation metric functions."""

    def test_compute_accuracy_perfect(self) -> None:
        """Test accuracy with perfect predictions."""
        preds = torch.tensor([0, 1, 2, 3])
        targets = torch.tensor([0, 1, 2, 3])

        acc = compute_accuracy(preds, targets)
        assert acc == 1.0

    def test_compute_accuracy_partial(self) -> None:
        """Test accuracy with partial correct predictions."""
        preds = torch.tensor([0, 1, 2, 0])
        targets = torch.tensor([0, 1, 1, 0])

        acc = compute_accuracy(preds, targets)
        assert acc == 0.75

    def test_compute_accuracy_numpy(self) -> None:
        """Test accuracy with numpy arrays."""
        preds = np.array([0, 1, 2, 0])
        targets = np.array([0, 1, 1, 0])

        acc = compute_accuracy(preds, targets)
        assert acc == 0.75

    def test_compute_accuracy_empty(self) -> None:
        """Test accuracy with empty tensors."""
        preds = torch.tensor([])
        targets = torch.tensor([])

        acc = compute_accuracy(preds, targets)
        assert acc == 0.0

    def test_compute_per_class_accuracy(self) -> None:
        """Test per-class accuracy computation."""
        # Create a clear test case:
        # targets: [0, 0, 1, 1, 2]
        # preds:   [0, 0, 1, 0, 2]
        # Class 0: 2 samples, 2 correct -> 1.0
        # Class 1: 2 samples, 1 correct -> 0.5
        # Class 2: 1 sample, 1 correct -> 1.0
        preds = torch.tensor([0, 0, 1, 0, 2])
        targets = torch.tensor([0, 0, 1, 1, 2])

        per_class = compute_per_class_accuracy(preds, targets, num_classes=3)

        assert per_class[0] == 1.0  # 2/2 correct for class 0
        assert per_class[1] == 0.5  # 1/2 correct for class 1
        assert per_class[2] == 1.0  # 1/1 correct for class 2

    def test_compute_macro_f1(self) -> None:
        """Test macro F1 score computation."""
        preds = torch.tensor([0, 1, 2, 0, 1, 2])
        targets = torch.tensor([0, 1, 1, 0, 0, 2])

        f1 = compute_macro_f1(preds, targets, num_classes=3)

        assert 0 <= f1 <= 1

    def test_compute_macro_f1_perfect(self) -> None:
        """Test macro F1 with perfect predictions."""
        preds = torch.tensor([0, 1, 2])
        targets = torch.tensor([0, 1, 2])

        f1 = compute_macro_f1(preds, targets, num_classes=3)
        assert f1 == 1.0

    def test_compute_confusion_matrix(self) -> None:
        """Test confusion matrix computation."""
        preds = torch.tensor([0, 1, 2, 0, 1, 2])
        targets = torch.tensor([0, 1, 1, 0, 0, 2])

        cm = compute_confusion_matrix(preds, targets, num_classes=3)

        assert cm.shape == (3, 3)
        assert cm[0, 0] == 2  # Class 0 correct
        assert cm[1, 1] == 1  # Class 1 correct
        assert cm[1, 2] == 1  # Class 1 predicted as 2
        assert cm[2, 2] == 1  # Class 2 correct

    def test_compute_top_k_accuracy(self) -> None:
        """Test top-k accuracy for embeddings."""
        # Create embeddings where similar samples have same labels
        torch.manual_seed(42)
        embeddings = torch.randn(10, 64)
        labels = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3, 4, 4])

        acc = compute_top_k_accuracy(embeddings, labels, k=3)

        assert 0 <= acc <= 1

    def test_compute_top_k_accuracy_validation(self) -> None:
        """Test top-k accuracy input validation."""
        embeddings_1d = torch.randn(10)  # Invalid shape
        labels = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3, 4, 4])

        with pytest.raises(ValueError, match="2D"):
            compute_top_k_accuracy(embeddings_1d, labels)


class TestTagDataset:
    """Tests for TagDataset class."""

    def test_dataset_creation(self) -> None:
        """Test dataset creation from TagSamples."""
        generator = SyntheticTagGenerator(seed=42)
        samples = generator.generate(10)
        tokenizer = CharacterTokenizer()

        dataset = TagDataset(samples, tokenizer)

        assert len(dataset) == 10

    def test_dataset_getitem(self) -> None:
        """Test dataset item retrieval."""
        generator = SyntheticTagGenerator(seed=42)
        samples = generator.generate(10)
        tokenizer = CharacterTokenizer()

        dataset = TagDataset(samples, tokenizer)

        inputs, property_label, signal_label = dataset[0]

        assert inputs.shape == (32,)  # Default max_length
        assert property_label.ndim == 0  # Scalar
        assert signal_label.ndim == 0  # Scalar

    def test_dataset_labels_valid(self) -> None:
        """Test that dataset labels are valid indices."""
        generator = SyntheticTagGenerator(seed=42)
        samples = generator.generate(100)
        tokenizer = CharacterTokenizer()

        dataset = TagDataset(samples, tokenizer)

        for i in range(len(dataset)):
            _, prop_label, sig_label = dataset[i]
            assert 0 <= prop_label.item() < len(PROPERTY_CLASSES)
            assert 0 <= sig_label.item() < len(SIGNAL_ROLES)


class TestTrainingIntegration:
    """Integration tests for training components."""

    @pytest.mark.slow
    def test_short_training_run(self) -> None:
        """Test a short training run completes without errors."""
        config = TrainingConfig(
            num_samples=100,
            epochs=2,
            batch_size=16,
            save_best=False,  # Don't save checkpoints in test
            seed=42,
        )
        model_config = CharCNNConfig(
            embedding_dim=16,
            conv1_channels=32,
            conv2_channels=32,
            hidden_dim=32,
        )

        model, history = train_model(config, model_config)

        assert len(history.train_loss) == 2
        assert len(history.val_loss) == 2
        assert all(loss > 0 for loss in history.train_loss)

        # Model should be trained
        assert model is not None

    def test_training_history_dataclass(self) -> None:
        """Test TrainingHistory dataclass."""
        history = TrainingHistory()

        history.train_loss.append(1.0)
        history.val_loss.append(0.9)
        history.best_epoch = 1
        history.best_val_loss = 0.9

        assert len(history.train_loss) == 1
        assert history.best_epoch == 1

    def test_model_forward_backward(self) -> None:
        """Test full forward-backward pass."""
        config = CharCNNConfig(
            embedding_dim=16,
            conv1_channels=32,
            conv2_channels=32,
            hidden_dim=32,
        )
        model = CharCNN(config)
        tokenizer = CharacterTokenizer()

        # Generate batch
        generator = SyntheticTagGenerator(seed=42)
        samples = generator.generate(4)

        # Prepare data
        inputs = tokenizer.encode_batch([s.tag_name for s in samples])
        property_targets = torch.tensor([
            get_property_class_from_category(s.features.get("category", "unknown"))
            for s in samples
        ])
        signal_targets = torch.tensor([
            get_signal_role_from_prefix(s.features.get("prefix", ""))
            for s in samples
        ])

        # Forward pass
        outputs = model(inputs)

        # Compute loss
        criterion = torch.nn.CrossEntropyLoss()
        loss = criterion(outputs["property_class"], property_targets)
        loss += criterion(outputs["signal_role"], signal_targets)

        # Also use the embedding output
        loss += outputs["irdi_embedding"].sum()

        # Backward pass
        loss.backward()

        # Check key gradients exist (not all may have gradients if unused)
        assert model.embedding.weight.grad is not None
        assert model.conv1.weight.grad is not None
        assert model.property_head.weight.grad is not None
