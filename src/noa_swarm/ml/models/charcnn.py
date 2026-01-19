"""Character-level CNN for semantic tag mapping.

This module implements a CharCNN architecture that maps industrial tag names
(e.g., "FIC-101") to semantic categories. The model has three output heads:
- property_class: Physical property type (flow, temperature, pressure, etc.)
- signal_role: Functional role (indicator, controller, transmitter, etc.)
- irdi_embedding: Embedding for IRDI retrieval

Architecture:
    Input -> Character Embedding -> Conv1D(7) -> MaxPool -> Conv1D(3) -> MaxPool
    -> GlobalMaxPool -> FC -> [property_class, signal_role, irdi_embedding]
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
import torch.nn.functional as F

if TYPE_CHECKING:
    from collections.abc import Sequence


# Default character alphabet for industrial tags
# Includes uppercase letters, digits, and common separators
DEFAULT_ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_./: "

# Property classes from SEED_IRDIS categories
PROPERTY_CLASSES = [
    "flow_rate",
    "flow_totalizer",
    "temperature",
    "pressure",
    "pressure_differential",
    "level",
    "valve_position",
    "composition",
    "speed",
    "weight",
    "alarm_high",
    "alarm_low",
    "alarm_high_high",
    "alarm_low_low",
    "setpoint",
    "process_value",
    "output_value",
    "switch_status",
    "motor_running",
    "density",
    "viscosity",
    "position",
    "power",
    "energy",
    "vibration",
    "unknown",
]

# Signal roles based on ISA S5.1 function codes
SIGNAL_ROLES = [
    "indicator",      # I - Indicator
    "controller",     # C - Controller
    "transmitter",    # T - Transmitter
    "element",        # E - Element/Sensor
    "recorder",       # R - Recorder
    "switch",         # S - Switch
    "valve",          # V - Valve
    "alarm",          # A - Alarm
    "relay",          # Y - Relay/Solenoid
    "glass",          # G - Sight glass
    "totalizer",      # Q - Quantity/Totalizer
    "unknown",
]


@dataclass
class CharCNNConfig:
    """Configuration for CharCNN model.

    Attributes:
        alphabet: Character alphabet string
        max_seq_length: Maximum sequence length (characters)
        embedding_dim: Character embedding dimension
        conv1_channels: Number of channels in first conv layer
        conv2_channels: Number of channels in second conv layer
        hidden_dim: Hidden layer dimension
        num_property_classes: Number of property class outputs
        num_signal_roles: Number of signal role outputs
        irdi_embedding_dim: Dimension of IRDI embedding output
        dropout: Dropout probability
    """

    alphabet: str = DEFAULT_ALPHABET
    max_seq_length: int = 32
    embedding_dim: int = 64
    conv1_channels: int = 256
    conv2_channels: int = 256
    hidden_dim: int = 256
    num_property_classes: int = field(default_factory=lambda: len(PROPERTY_CLASSES))
    num_signal_roles: int = field(default_factory=lambda: len(SIGNAL_ROLES))
    irdi_embedding_dim: int = 128
    dropout: float = 0.5


class CharacterTokenizer:
    """Tokenizer for converting tag strings to character indices.

    Converts strings to sequences of character indices based on a fixed alphabet.
    Handles padding, truncation, and unknown characters.

    Attributes:
        alphabet: The character alphabet
        max_length: Maximum sequence length
        char_to_idx: Mapping from characters to indices
        pad_idx: Index used for padding
        unk_idx: Index used for unknown characters

    Example:
        >>> tokenizer = CharacterTokenizer(alphabet="ABC", max_length=5)
        >>> tokenizer.encode("AB")
        tensor([0, 1, 3, 3, 3])  # [A, B, <PAD>, <PAD>, <PAD>]
        >>> tokenizer.decode(tokenizer.encode("AB"))
        'AB'
    """

    def __init__(
        self,
        alphabet: str = DEFAULT_ALPHABET,
        max_length: int = 32,
        case_sensitive: bool = False,
    ) -> None:
        """Initialize the tokenizer.

        Args:
            alphabet: String containing all valid characters
            max_length: Maximum sequence length (longer sequences are truncated)
            case_sensitive: If False, convert all input to uppercase
        """
        self.alphabet = alphabet
        self.max_length = max_length
        self.case_sensitive = case_sensitive

        # Build character to index mapping
        # Reserve indices: 0 to len(alphabet)-1 for alphabet chars
        # len(alphabet) = UNK, len(alphabet)+1 = PAD
        self.char_to_idx: dict[str, int] = {c: i for i, c in enumerate(alphabet)}
        self.unk_idx = len(alphabet)
        self.pad_idx = len(alphabet) + 1
        self.vocab_size = len(alphabet) + 2  # +2 for UNK and PAD

        # Build reverse mapping for decoding
        self.idx_to_char: dict[int, str] = {i: c for c, i in self.char_to_idx.items()}
        self.idx_to_char[self.unk_idx] = "<UNK>"
        self.idx_to_char[self.pad_idx] = "<PAD>"

    def encode(self, text: str) -> torch.Tensor:
        """Encode a single string to a tensor of character indices.

        Args:
            text: Input string to encode

        Returns:
            Tensor of shape (max_length,) with character indices
        """
        if not self.case_sensitive:
            text = text.upper()

        # Convert characters to indices
        indices: list[int] = []
        for char in text[: self.max_length]:
            indices.append(self.char_to_idx.get(char, self.unk_idx))

        # Pad to max_length
        while len(indices) < self.max_length:
            indices.append(self.pad_idx)

        return torch.tensor(indices, dtype=torch.long)

    def encode_batch(self, texts: Sequence[str]) -> torch.Tensor:
        """Encode a batch of strings to a tensor.

        Args:
            texts: Sequence of strings to encode

        Returns:
            Tensor of shape (batch_size, max_length)
        """
        return torch.stack([self.encode(text) for text in texts])

    def decode(self, indices: torch.Tensor) -> str:
        """Decode a tensor of indices back to a string.

        Args:
            indices: Tensor of character indices

        Returns:
            Decoded string (without padding)
        """
        chars: list[str] = []
        for idx in indices.tolist():
            if idx == self.pad_idx:
                break
            char = self.idx_to_char.get(idx, "?")
            if char not in ("<UNK>", "<PAD>"):
                chars.append(char)
            elif char == "<UNK>":
                chars.append("?")
        return "".join(chars)

    def decode_batch(self, indices: torch.Tensor) -> list[str]:
        """Decode a batch of index tensors.

        Args:
            indices: Tensor of shape (batch_size, seq_length)

        Returns:
            List of decoded strings
        """
        return [self.decode(row) for row in indices]


class CharCNN(nn.Module):
    """Character-level CNN for semantic tag classification.

    This model takes tag name strings and outputs:
    1. property_class: Physical property classification (logits or probabilities)
    2. signal_role: Signal role classification (logits or probabilities)
    3. irdi_embedding: Dense embedding for IRDI retrieval

    Architecture follows the Zhang et al. (2015) character-level CNN design
    with modifications for multi-head output.

    Attributes:
        config: Model configuration
        embedding: Character embedding layer
        conv1: First convolutional layer (kernel size 7)
        conv2: Second convolutional layer (kernel size 3)
        fc: Fully connected layer
        property_head: Output head for property classification
        signal_head: Output head for signal role classification
        embedding_head: Output head for IRDI embedding

    Example:
        >>> config = CharCNNConfig()
        >>> model = CharCNN(config)
        >>> tokenizer = CharacterTokenizer()
        >>> x = tokenizer.encode_batch(["FIC-101", "TIC-2301"])
        >>> outputs = model(x)
        >>> outputs["property_class"].shape
        torch.Size([2, 26])
    """

    def __init__(self, config: CharCNNConfig | None = None) -> None:
        """Initialize the CharCNN model.

        Args:
            config: Model configuration. If None, uses default config.
        """
        super().__init__()
        self.config = config or CharCNNConfig()

        # Character embedding layer
        # +2 for UNK and PAD tokens
        vocab_size = len(self.config.alphabet) + 2
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=self.config.embedding_dim,
            padding_idx=vocab_size - 1,  # PAD is last index
        )

        # Convolutional layers
        # Conv1D: (batch, embedding_dim, seq_len) -> (batch, conv_channels, new_seq_len)
        self.conv1 = nn.Conv1d(
            in_channels=self.config.embedding_dim,
            out_channels=self.config.conv1_channels,
            kernel_size=7,
            padding=3,  # Same padding
        )
        self.pool1 = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        self.conv2 = nn.Conv1d(
            in_channels=self.config.conv1_channels,
            out_channels=self.config.conv2_channels,
            kernel_size=3,
            padding=1,  # Same padding
        )
        self.pool2 = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        # Global max pooling is applied in forward()

        # Fully connected layer
        self.fc = nn.Linear(self.config.conv2_channels, self.config.hidden_dim)
        self.dropout = nn.Dropout(self.config.dropout)

        # Output heads
        self.property_head = nn.Linear(
            self.config.hidden_dim, self.config.num_property_classes
        )
        self.signal_head = nn.Linear(
            self.config.hidden_dim, self.config.num_signal_roles
        )
        self.embedding_head = nn.Linear(
            self.config.hidden_dim, self.config.irdi_embedding_dim
        )

        # Initialize weights
        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize model weights using Xavier initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Conv1d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0, std=0.1)

    def forward(
        self,
        x: torch.Tensor,
        return_embeddings: bool = False,
    ) -> dict[str, torch.Tensor]:
        """Forward pass through the model.

        Args:
            x: Input tensor of shape (batch_size, seq_length) with character indices
            return_embeddings: If True, normalize irdi_embedding with L2 norm

        Returns:
            Dictionary with keys:
                - "property_class": Logits for property classification (batch, num_classes)
                - "signal_role": Logits for signal role classification (batch, num_roles)
                - "irdi_embedding": Dense embedding (batch, embedding_dim)
                - "features": Intermediate features before heads (batch, hidden_dim)
        """
        # Embedding: (batch, seq_len) -> (batch, seq_len, embed_dim)
        embedded = self.embedding(x)

        # Transpose for Conv1d: (batch, seq_len, embed_dim) -> (batch, embed_dim, seq_len)
        embedded = embedded.transpose(1, 2)

        # Conv block 1
        h = F.relu(self.conv1(embedded))
        h = self.pool1(h)

        # Conv block 2
        h = F.relu(self.conv2(h))
        h = self.pool2(h)

        # Global max pooling: (batch, channels, seq) -> (batch, channels)
        h = F.adaptive_max_pool1d(h, 1).squeeze(-1)

        # Fully connected
        features = F.relu(self.fc(h))
        features = self.dropout(features)

        # Output heads
        property_logits = self.property_head(features)
        signal_logits = self.signal_head(features)
        irdi_emb = self.embedding_head(features)

        # Optionally normalize embedding for retrieval
        if return_embeddings:
            irdi_emb = F.normalize(irdi_emb, p=2, dim=-1)

        return {
            "property_class": property_logits,
            "signal_role": signal_logits,
            "irdi_embedding": irdi_emb,
            "features": features,
        }

    def predict(
        self,
        x: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Make predictions with softmax probabilities.

        Args:
            x: Input tensor of shape (batch_size, seq_length)

        Returns:
            Dictionary with keys:
                - "property_class": Predicted class indices (batch,)
                - "property_probs": Class probabilities (batch, num_classes)
                - "signal_role": Predicted role indices (batch,)
                - "signal_probs": Role probabilities (batch, num_roles)
                - "irdi_embedding": Normalized embedding (batch, embedding_dim)
        """
        with torch.no_grad():
            outputs = self.forward(x, return_embeddings=True)

            property_probs = F.softmax(outputs["property_class"], dim=-1)
            signal_probs = F.softmax(outputs["signal_role"], dim=-1)

            return {
                "property_class": torch.argmax(property_probs, dim=-1),
                "property_probs": property_probs,
                "signal_role": torch.argmax(signal_probs, dim=-1),
                "signal_probs": signal_probs,
                "irdi_embedding": outputs["irdi_embedding"],
            }

    def get_property_class_name(self, idx: int) -> str:
        """Get property class name from index.

        Args:
            idx: Class index

        Returns:
            Property class name
        """
        if 0 <= idx < len(PROPERTY_CLASSES):
            return PROPERTY_CLASSES[idx]
        return "unknown"

    def get_signal_role_name(self, idx: int) -> str:
        """Get signal role name from index.

        Args:
            idx: Role index

        Returns:
            Signal role name
        """
        if 0 <= idx < len(SIGNAL_ROLES):
            return SIGNAL_ROLES[idx]
        return "unknown"


def create_label_mappings() -> tuple[dict[str, int], dict[str, int]]:
    """Create label to index mappings for training.

    Returns:
        Tuple of (property_to_idx, role_to_idx) dictionaries
    """
    property_to_idx = {name: idx for idx, name in enumerate(PROPERTY_CLASSES)}
    role_to_idx = {name: idx for idx, name in enumerate(SIGNAL_ROLES)}
    return property_to_idx, role_to_idx


def get_property_class_from_category(category: str) -> int:
    """Map a category name to property class index.

    Args:
        category: Category name from SyntheticTagGenerator

    Returns:
        Property class index
    """
    # Direct mapping for most categories
    if category in PROPERTY_CLASSES:
        return PROPERTY_CLASSES.index(category)

    # Handle variations
    category_mapping: dict[str, str] = {
        "flow_rate_volumetric": "flow_rate",
        "flow_rate_mass": "flow_rate",
        "temperature_process": "temperature",
        "temperature_ambient": "temperature",
        "temperature_differential": "temperature",
        "pressure_absolute": "pressure",
        "pressure_gauge": "pressure",
        "level_percentage": "level",
        "level_volume": "level",
        "valve_position_percentage": "valve_position",
        "valve_command": "valve_position",
        "valve_feedback": "valve_position",
        "composition_percentage": "composition",
        "composition_ppm": "composition",
        "ph_value": "composition",
        "conductivity": "composition",
        "speed_rpm": "speed",
        "speed_percentage": "speed",
        "weight_mass": "weight",
        "weight_force": "weight",
        "controller_mode": "process_value",
        "motor_current": "motor_running",
        "motor_speed": "motor_running",
        "pump_status": "motor_running",
    }

    mapped = category_mapping.get(category, "unknown")
    return PROPERTY_CLASSES.index(mapped) if mapped in PROPERTY_CLASSES else len(PROPERTY_CLASSES) - 1


def get_signal_role_from_prefix(prefix: str) -> int:
    """Map a tag prefix to signal role index.

    Args:
        prefix: Tag prefix (e.g., "FIC", "TI")

    Returns:
        Signal role index
    """
    # Extract the function code from the prefix
    prefix = prefix.upper()

    # Check for alarm patterns FIRST (SH, SL, SHH, SLL patterns)
    # These are switch/alarm combinations in ISA nomenclature
    if "SH" in prefix or "SL" in prefix:
        return SIGNAL_ROLES.index("alarm")

    # Map based on last character(s) which typically indicate function
    role_mapping: dict[str, str] = {
        "I": "indicator",
        "C": "controller",
        "T": "transmitter",
        "E": "element",
        "R": "recorder",
        "S": "switch",
        "V": "valve",
        "A": "alarm",
        "Y": "relay",
        "G": "glass",
        "Q": "totalizer",
    }

    # Check each character from the end for function codes
    for i in range(len(prefix) - 1, -1, -1):
        char = prefix[i]
        if char in role_mapping:
            role_name = role_mapping[char]
            if role_name in SIGNAL_ROLES:
                return SIGNAL_ROLES.index(role_name)

    return SIGNAL_ROLES.index("unknown")
