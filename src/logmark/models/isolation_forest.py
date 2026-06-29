import torch
from sklearn.ensemble import IsolationForest
from .base import BaseModel

class IsolationForestModel(BaseModel):
    """PyTorch wrapper around scikit-learn's IsolationForest"""
    def __init__(self, n_estimators: int = 100, contamination: float | str = "auto", random_state: int | None = 42):
        super().__init__()
        self.model = IsolationForest(
            n_estimators=n_estimators,
            contamination=contamination,
            random_state=random_state,
            n_jobs=-1  # All available CPU cores (TODO: Change?)
        )

    def fit(self, X: torch.Tensor, y: torch.Tensor | None = None) -> None:
        X_np = self._tensor_to_numpy(X)
        self.model.fit(X_np)
        self.is_fitted = True

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        if not self.is_fitted:
            raise RuntimeError("IsolationForestModel must be fitted before calling predict.")

        X_np = self._tensor_to_numpy(X)
        
        # sklearn returns: -1 for anomaly, 1 for normal
        preds_np = self.model.predict(X_np)
        
        # Convert to: 1 for anomaly, 0 for normal
        preds_binary = (preds_np == -1).astype(int)
        
        return torch.tensor(preds_binary, dtype=torch.long, device=X.device)
