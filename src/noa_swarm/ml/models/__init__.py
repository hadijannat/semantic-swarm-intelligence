"""Neural network models for semantic tag mapping.

This module provides:
- CharCNN: Character-level CNN for tag name classification
- CharacterTokenizer: Convert tag strings to character indices
"""

from noa_swarm.ml.models.charcnn import CharCNN, CharacterTokenizer

__all__ = [
    "CharCNN",
    "CharacterTokenizer",
]
