from abc import ABC, abstractmethod
from pathlib import Path

class BaseDataset(ABC):
    @abstractmethod
    def download(self, force: bool = False) -> None:
        """Download the dataset to the local data directory."""
        pass

    @abstractmethod
    def get_log_path(self) -> Path:
        """Return the path to the main log file of the dataset."""
        pass
