from .base import BaseModel
from .isolation_forest import IsolationForestModel

__all__ = ["BaseModel", "IsolationForestModel", "build_model"]

def build_model(name: str, **kwargs) -> BaseModel:
    """Factory function to instantiate detection models by name"""
    models = {
        "isolation_forest": IsolationForestModel,
    }
    
    if name.lower() not in models:
        raise ValueError(f"Model '{name}' not supported. Available: {list(models.keys())}")
        
    return models[name.lower()](**kwargs)
