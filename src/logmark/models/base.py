from abc import ABC, abstractmethod
import torch
import torch.nn as nn
from typing import Any

class BaseModel(nn.Module, ABC):
    """Abstract base class for all log anomaly detection models"""
    def __init__(self):
        super().__init__()
        self.is_fitted = False

    @abstractmethod
    def fit(self, X: torch.Tensor, y: torch.Tensor | None = None) -> None:
        pass

    @abstractmethod
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Predict whether each sample in X is an anomaly.
        Must return a PyTorch tensor (e.g., 1 for anomaly, 0 for normal) on the same device as X
        """
        pass

    def predict(self, X: torch.Tensor) -> torch.Tensor:
        """Alias for forward() for traditional API compatibility"""
        return self(X)

    def _tensor_to_numpy(self, X: torch.Tensor) -> Any:
        """Helper method to move a PyTorch tensor to NumPy """
        return X.detach().cpu().numpy()
