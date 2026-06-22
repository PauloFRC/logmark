import polars as pl
from typing import Optional, Union
from enum import Enum

class WindowType(Enum):
    COUNT = "count"
    TIME = "time"
    SESSION = "session"

class LogWindowing:
    def __init__(
        self,
        window_type: Union[str, WindowType],
        window_size: Optional[Union[int, str]] = None,
        step_size: Optional[Union[int, str]] = None,
        session_col: Optional[str] = None,
        timestamp_col: str = "timestamp",
    ):
        self.window_type = WindowType(window_type) if isinstance(window_type, str) else window_type
        self.window_size = window_size
        self.step_size = step_size
        self.session_col = session_col
        self.timestamp_col = timestamp_col

    def transform(self, lf: pl.LazyFrame) -> pl.LazyFrame:
        """
        Applies windowing to a Polars LazyFrame.
        Returns a LazyFrame where each row represents a window or has a window_id.
        """
        if self.window_type == WindowType.COUNT:
            return self._count_windowing(lf)
        elif self.window_type == WindowType.TIME:
            return self._time_windowing(lf)
        elif self.window_type == WindowType.SESSION:
            return self._session_windowing(lf)
        else:
            raise ValueError(f"Unsupported window type: {self.window_type}")

    def _count_windowing(self, lf: pl.LazyFrame) -> pl.LazyFrame:
        if not isinstance(self.window_size, int):
            raise ValueError("window_size must be an integer for count-based windowing")
        
        step = self.step_size if isinstance(self.step_size, int) else self.window_size
        
        return (
            lf.with_row_index("_idx")
            .with_columns(pl.col("_idx").cast(pl.Int64))
            .sort("_idx")
            .group_by_dynamic(
                "_idx",
                every=f"{step}i",
                period=f"{self.window_size}i",
            )
            .agg(pl.all().exclude("_idx"))
        )
            
    def _time_windowing(self, lf: pl.LazyFrame) -> pl.LazyFrame:
        if not isinstance(self.window_size, str):
            raise ValueError("window_size must be a string (e.g., '1h') for time-based windowing")
        
        step = self.step_size if isinstance(self.step_size, str) else self.window_size
        
        return (
            lf.sort(self.timestamp_col)
            .group_by_dynamic(
                self.timestamp_col,
                every=step,
                period=self.window_size,
            )
            .agg(pl.all())
        )

    def _session_windowing(self, lf: pl.LazyFrame) -> pl.LazyFrame:
        if self.session_col is None:
            raise ValueError("session_col must be provided for session-based windowing")
        
        return (
            lf.sort(self.timestamp_col)
            .group_by(self.session_col, maintain_order=True)
            .agg(pl.all())
        )
