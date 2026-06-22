from __future__ import annotations
from typing import Literal, Optional, Tuple

import polars as pl
import torch

class PostWindowCollator:
    """
    Input DF Column: List[List[float]] or List[Tensor]
    Returns: padded_vectors (batch, seq_len, embed_dim), mask (batch, seq_len)
    """
    def __init__(
            self,
            embed_dim: int,
            sequence_length: Optional[int] = None,
            truncation: Literal["head", "tail"] = "tail",
    ):
        self.embed_dim = embed_dim
        self.sequence_length = sequence_length
        self.truncation = truncation

    def collate(self, df: pl.DataFrame, vector_col: str = "embeddings") -> Tuple[torch.Tensor, torch.Tensor]:
        if df.height == 0:
            empty_width = self.sequence_length or 0
            empty = torch.empty(0, empty_width, self.embed_dim, dtype=torch.float32)
            return empty, empty[..., 0].bool()

        if self.sequence_length:
            trunc_expr = (
                pl.col(vector_col).list.tail(self.sequence_length)
                if self.truncation == "tail"
                else pl.col(vector_col).list.head(self.sequence_length)
            )
            df = df.with_columns(trunc_expr)

        lengths = df.select(pl.col(vector_col).list.len().fill_null(0)).to_series().to_numpy()
        max_len = max(int(lengths.max()), 1)

        # Pad with 0.0s for the float vectors
        padded_vectors = torch.zeros((df.height, max_len, self.embed_dim), dtype=torch.float32)

        # TODO: keep to_list?
        vector_series = df[vector_col].to_list()

        for batch_idx, window_vectors in enumerate(vector_series):
            if window_vectors is None or len(window_vectors) == 0:
                continue

            window_tensor = torch.tensor(window_vectors, dtype=torch.float32)
            seq_len = window_tensor.shape[0]

            padded_vectors[batch_idx, :seq_len, :] = window_tensor

        lengths_t = torch.from_numpy(lengths.copy()).long()
        mask = torch.arange(max_len).unsqueeze(0) < lengths_t.unsqueeze(1)
        return padded_vectors, mask
