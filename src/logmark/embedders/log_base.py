from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, List, Type

import torch
import torch.nn as nn

class LogEmbedder(nn.Module, ABC):
    @property
    @abstractmethod
    def output_dim(self) -> int:
        """Dimensionality of the vector this encoder produces per log"""

    @abstractmethod
    def forward(self, padded_tokens: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            padded_tokens: (batch, max_tokens) long tensor of token/word piece ids
            mask: (batch, max_tokens) bool tensor; True marks a real token, False marks a padded position.

        Returns:
            (batch, output_dim) float tensor representing the embedding of each log
        """

    @staticmethod
    def _check_inputs(padded_tokens: torch.Tensor, mask: torch.Tensor) -> None:
        if padded_tokens.shape != mask.shape:
            raise ValueError(
                f"padded_tokens shape {tuple(padded_tokens.shape)} does not match mask shape {tuple(mask.shape)}"
            )
        if mask.dtype != torch.bool:
            raise TypeError(f"mask must be torch.bool, got {mask.dtype}")

_LOG_EMBEDDER_REGISTRY: Dict[str, Type[LogEmbedder]] = {}

def register_single_encoder(name: str):
    """Class decorator that makes an encoder available to build_single_encoder(name)"""
    def _decorator(cls: Type[LogEmbedder]) -> Type[LogEmbedder]:
        if name in _LOG_EMBEDDER_REGISTRY:
            raise ValueError(
                f"SingleEncoder name '{name}' is already registered to {_LOG_EMBEDDER_REGISTRY[name].__name__}"
            )
        _LOG_EMBEDDER_REGISTRY[name] = cls
        return cls

    return _decorator


def build_single_encoder(name: str, **kwargs) -> LogEmbedder:
    """Instantiate a registered single log encoder by name.
    Example:
        build_single_encoder("tfidf_linear", vocab_size=5000, dim=128)
    """
    try:
        cls = _LOG_EMBEDDER_REGISTRY[name]
    except KeyError:
        available = ", ".join(available_log_embedders()) or "None registered"
        raise ValueError(f"Unknown encoder '{name}'. Available: {available}") from None
    return cls(**kwargs)

def available_log_embedders() -> List[str]:
    return sorted(_LOG_EMBEDDER_REGISTRY)
