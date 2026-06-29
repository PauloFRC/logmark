import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from typing import Dict

def compute_anomaly_metrics(y_true: torch.Tensor, y_pred: torch.Tensor) -> Dict[str, float]:
    """
    Computes standard classification metrics for anomaly detection.
    Assumes:
        1 = Anomaly
        0 = Normal
    Handles GPU/CPU tensors seamlessly.
    """
    # Move tensors to CPU safely
    y_true_np = y_true.detach().cpu().numpy()
    y_pred_np = y_pred.detach().cpu().numpy()

    return {
        "accuracy": float(accuracy_score(y_true_np, y_pred_np)),
        "precision": float(precision_score(y_true_np, y_pred_np, zero_division=0)),
        "recall": float(recall_score(y_true_np, y_pred_np, zero_division=0)),
        "f1": float(f1_score(y_true_np, y_pred_np, zero_division=0))
    }
