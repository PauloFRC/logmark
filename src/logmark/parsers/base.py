from abc import ABC, abstractmethod
from typing import Any
from typing import Iterable

class BaseParser(ABC):
    def __init__(self, **kwargs: Any):
        self.config = kwargs
        self.is_fitted = False

    @property
    @abstractmethod
    def requires_fitting(self) -> bool:
        """True for offline parsers, False for streaming parsers."""
        pass

    def fit(self, log_stream: Iterable[str]) -> None:
        if self.requires_fitting:
            self._fit_impl(log_stream)
        self.is_fitted = True

    @abstractmethod
    def _fit_impl(self, log_stream: Iterable[str]) -> None:
        pass

    @abstractmethod
    def parse_line(self, line: str) -> dict:
        """Parses line into metadata and template details."""
        pass
