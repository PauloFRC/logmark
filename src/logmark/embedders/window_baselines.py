from __future__ import annotations

import warnings
from typing import Literal

import torch
import torch.nn as nn

from .window_base import WindowEmbedder, register_encoder
from .utils import clamp_oov

@register_encoder("count")
class CountVectorEncoder(WindowEmbedder):
    """
    Bag-of-words: each window becomes a vector of how many times each event id occurred
    """
    def __init__(self, vocab_size: int, normalize: Literal["none", "l1"] = "none"):
        super().__init__()
        self.vocab_size = vocab_size
        self.normalize = normalize

    @property
    def output_dim(self) -> int:
        return self.vocab_size

    def forward(self, padded_input: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        self._check_inputs(padded_input, mask)
        batch_size = padded_input.shape[0]
        safe_input = clamp_oov(padded_input, mask, self.vocab_size, self.__class__.__name__)

        batch_coords, seq_coords = mask.nonzero(as_tuple=True)
        if batch_coords.numel() == 0:
            return torch.zeros(batch_size, self.vocab_size, device=padded_input.device)

        token_coords = safe_input[batch_coords, seq_coords]
        indices = torch.stack([batch_coords, token_coords])
        values = torch.ones_like(batch_coords, dtype=torch.float32)

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Sparse invariant checks are implicitly disabled")
            sparse_counts = torch.sparse_coo_tensor(
                indices,
                values,
                size=(batch_size, self.vocab_size),
                device=padded_input.device,
                check_invariants=False,
            )
        counts = sparse_counts.coalesce().to_dense()

        if self.normalize == "l1":
            totals = counts.sum(dim=1, keepdim=True).clamp(min=1)
            counts = counts / totals
        return counts

class PoolingEmbeddingEncoder(WindowEmbedder):
    """Embed each event id, then reduce the window via mean/max/sum pooling."""
    def __init__(self, vocab_size: int, dim: int, aggregation: Literal["mean", "max", "sum"]):
        super().__init__()
        self.vocab_size = vocab_size
        self._dim = dim
        self.aggregation = aggregation
        self.embedding = nn.Embedding(vocab_size, dim)

    @property
    def output_dim(self) -> int:
        return self._dim

    def forward(self, padded_input: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        self._check_inputs(padded_input, mask)
        safe_input = clamp_oov(padded_input, mask, self.vocab_size, self.__class__.__name__)
        embedded = self.embedding(safe_input)
        mask_f = mask.unsqueeze(-1).to(embedded.dtype)

        if self.aggregation == "mean":
            summed = (embedded * mask_f).sum(dim=1)
            count = mask.sum(dim=1, keepdim=True).clamp(min=1).to(embedded.dtype)
            return summed / count

        if self.aggregation == "sum":
            return (embedded * mask_f).sum(dim=1)

        if self.aggregation == "max":
            has_any = mask.any(dim=1, keepdim=True)
            filled = embedded.masked_fill(~mask.unsqueeze(-1), float("-inf"))
            pooled = filled.max(dim=1).values
            return torch.where(has_any, pooled, torch.zeros_like(pooled))

        raise ValueError(f"Unknown aggregation strategy: {self.aggregation!r}")

@register_encoder("mean_pool")
class MeanPoolingEncoder(PoolingEmbeddingEncoder):
    def __init__(self, vocab_size: int, dim: int):
        super().__init__(vocab_size, dim, aggregation="mean")

@register_encoder("attention_pool")
class AttentionPoolingEncoder(WindowEmbedder):
    """Mean pooling with learned per-token weights."""
    def __init__(self, vocab_size: int, dim: int):
        super().__init__()
        self.vocab_size = vocab_size
        self._dim = dim
        self.embedding = nn.Embedding(vocab_size, dim)
        self.attention_proj = nn.Linear(dim, 1)

    @property
    def output_dim(self) -> int:
        return self._dim

    def forward(self, padded_input: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        self._check_inputs(padded_input, mask)
        safe_input = clamp_oov(padded_input, mask, self.vocab_size, self.__class__.__name__)
        embedded = self.embedding(safe_input)

        scores = self.attention_proj(embedded).squeeze(-1)
        has_any = mask.any(dim=1, keepdim=True)

        safe_scores = scores.masked_fill(~mask, float("-inf"))
        safe_scores = torch.where(has_any, safe_scores, torch.zeros_like(scores))
        weights = torch.softmax(safe_scores, dim=1).unsqueeze(-1)

        pooled = (embedded * weights).sum(dim=1)
        return pooled * has_any.to(pooled.dtype)
