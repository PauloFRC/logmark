from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Type

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

class WindowEmbedder(nn.Module, ABC):
    @property
    @abstractmethod
    def output_dim(self) -> int:
        """Dimensionality of the vector this encoder produces per window"""

    @abstractmethod
    def forward(self, padded_input: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            padded_input: (batch, seq_len) long tensor of event ids
            mask: (batch, seq_len) bool tensor; True marks a real event, False marks a padded position.

        Returns:
            (batch, output_dim) float tensor.
        """

    @staticmethod
    def _check_inputs(padded_input: torch.Tensor, mask: torch.Tensor) -> None:
        if padded_input.shape != mask.shape:
            raise ValueError(
                f"padded_input shape {tuple(padded_input.shape)} does not match mask shape {tuple(mask.shape)}"
            )
        if mask.dtype != torch.bool:
            raise TypeError(f"mask must be torch.bool, got {mask.dtype}")

_WINDOW_EMBEDDER_REGISTRY: Dict[str, Type[WindowEmbedder]] = {}

def register_encoder(name: str):
    """Class decorator that makes an encoder available to build_encoder(name)"""
    def _decorator(cls: Type[WindowEmbedder]) -> Type[WindowEmbedder]:
        if name in _WINDOW_EMBEDDER_REGISTRY:
            raise ValueError(
                f"Encoder name '{name}' is already registered to {_WINDOW_EMBEDDER_REGISTRY[name].__name__}"
            )
        _WINDOW_EMBEDDER_REGISTRY[name] = cls
        return cls

    return _decorator


def build_encoder(name: str, **kwargs) -> WindowEmbedder:
    """Instantiate a registered encoder by name.
    Example:
        build_encoder("mean_pool", vocab_size=500, dim=64)
    """
    try:
        cls = _WINDOW_EMBEDDER_REGISTRY[name]
    except KeyError:
        available = ", ".join(available_encoders()) or "None registered"
        raise ValueError(f"Unknown encoder '{name}'. Available: {available}") from None
    return cls(**kwargs)

def available_encoders() -> List[str]:
    return sorted(_WINDOW_EMBEDDER_REGISTRY)
