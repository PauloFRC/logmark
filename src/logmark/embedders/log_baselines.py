from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn

from .log_base import LogEmbedder, register_single_encoder
from .utils import clamp_oov

try:
    from transformers import AutoModel, AutoConfig
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False

class TokenPoolingEncoder(LogEmbedder):
    """Maps subword tokens (e.g., from a WordPiece tokenizer) to vectors and pools them"""

    def __init__(self, vocab_size: int, dim: int, aggregation: Literal["mean", "max"] = "mean"):
        super().__init__()
        self.vocab_size = vocab_size
        self._dim = dim
        self.aggregation = aggregation
        self.embedding = nn.Embedding(vocab_size, dim)

    @property
    def output_dim(self) -> int:
        return self._dim

    def forward(self, padded_tokens: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        self._check_inputs(padded_tokens, mask)
        safe_tokens = clamp_oov(padded_tokens, mask, self.vocab_size, self.__class__.__name__)
        embedded = self.embedding(safe_tokens)
        mask_f = mask.unsqueeze(-1).to(embedded.dtype)

        if self.aggregation == "mean":
            summed = (embedded * mask_f).sum(dim=1)
            count = mask.sum(dim=1, keepdim=True).clamp(min=1).to(embedded.dtype)
            return summed / count

        if self.aggregation == "max":
            has_any = mask.any(dim=1, keepdim=True)
            filled = embedded.masked_fill(~mask.unsqueeze(-1), float("-inf"))
            pooled = filled.max(dim=1).values
            return torch.where(has_any, pooled, torch.zeros_like(pooled))


@register_single_encoder("token_mean_pool")
class MeanTokenPoolingEncoder(TokenPoolingEncoder):
    def __init__(self, vocab_size: int, dim: int):
        super().__init__(vocab_size, dim, aggregation="mean")


@register_single_encoder("transformer")
class TransformerLogEncoder(LogEmbedder):
    """Single log encoder using a pre-trained language model"""

    def __init__(self, model_name: str, fine_tune: bool = False):
        super().__init__()
        if not HAS_TRANSFORMERS:
            raise ImportError(
                "The 'transformers' library is required to use TransformerLogEncoder"
            )

        self.model_name = model_name
        self.config = AutoConfig.from_pretrained(model_name)
        self.transformer = AutoModel.from_pretrained(model_name)

        if not fine_tune:
            for param in self.transformer.parameters():
                param.requires_grad = False

    @property
    def output_dim(self) -> int:
        return self.config.hidden_size

    def forward(self, padded_tokens: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        self._check_inputs(padded_tokens, mask)

        padded_tokens = padded_tokens.long()
        mask_long = mask.long()

        outputs = self.transformer(
            input_ids=padded_tokens,
            attention_mask=mask_long
        )

        cls_embedding = outputs.last_hidden_state[:, 0, :]

        return cls_embedding
