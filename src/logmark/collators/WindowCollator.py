from __future__ import annotations
from typing import Literal, Optional, Tuple

import polars as pl
import torch

class WindowCollator:
    """
    Input DF Column: List[int]
    Returns: padded_input (batch, seq_len), mask (batch, seq_len)
    """
    def __init__(
        self,
        sequence_length: Optional[int] = None,
        padding_value: int = 0,
        truncation: Literal["head", "tail"] = "tail",
    ):
        self.sequence_length = sequence_length
        self.padding_value = padding_value
        self.truncation = truncation

    def collate(self, df: pl.DataFrame, token_col: str = "event_id") -> Tuple[torch.Tensor, torch.Tensor]:
        if df.height == 0:
            empty_width = self.sequence_length or 0
            empty = torch.empty(0, empty_width, dtype=torch.long)
            return empty, empty.bool()

        if self.sequence_length:
            trunc_expr = (
                pl.col(token_col).list.tail(self.sequence_length)
                if self.truncation == "tail"
                else pl.col(token_col).list.head(self.sequence_length)
            )
            df = df.with_columns(trunc_expr)

        lengths = df.select(pl.col(token_col).list.len().fill_null(0)).to_series().to_numpy()
        max_len = max(int(lengths.max()), 1)

        padded = torch.full((df.height, max_len), self.padding_value, dtype=torch.long)

        if lengths.sum() > 0:
            exploded = (
                df.with_row_index("batch_idx")
                .with_columns(pl.int_ranges(0, pl.col(token_col).list.len()).alias("seq_idx"))
                .explode([token_col, "seq_idx"])
                .drop_nulls(token_col)
            )
            batch_idx = torch.from_numpy(exploded["batch_idx"].to_numpy().copy()).long()
            seq_idx = torch.from_numpy(exploded["seq_idx"].to_numpy().copy()).long()
            token_ids = torch.from_numpy(exploded[token_col].to_numpy().copy()).long()
            padded.index_put_((batch_idx, seq_idx), token_ids)

        lengths_t = torch.from_numpy(lengths.copy()).long()
        mask = torch.arange(max_len).unsqueeze(0) < lengths_t.unsqueeze(1)
        return padded, mask
    