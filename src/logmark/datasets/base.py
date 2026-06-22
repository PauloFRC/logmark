from abc import ABC, abstractmethod
from typing import Iterator

class BaseDataset(ABC):
    @property
    @abstractmethod
    def is_streaming(self) -> bool:
        """True if the data source is an infinite or real-time stream (e.g., Kafka, live socket)."""
        pass

    @abstractmethod
    def download(self, force: bool = False) -> None:
        """Download dataset if static. Pass if live stream."""
        pass

    @abstractmethod
    def get_log_iterator(self) -> Iterator[str]:
        """Yields raw log lines one by one, regardless of source (file, stream, etc.)."""
        pass
