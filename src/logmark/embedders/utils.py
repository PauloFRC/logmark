import logging
import torch

logger = logging.getLogger(__name__)

def clamp_oov(tensor: torch.Tensor, mask: torch.Tensor, vocab_size: int, source: str) -> torch.Tensor:
    """Clamp out-of-vocabulary integer IDs down to vocab_size - 1."""
    oov = (tensor >= vocab_size) & mask
    n_oov = int(oov.sum())
    if n_oov:
        n_valid = max(int(mask.sum()), 1)
        logger.warning(
            "%s: %d/%d (%.2f%%) IDs were out-of-vocabulary and clamped to id %d.",
            source, n_oov, n_valid, 100 * n_oov / n_valid, vocab_size - 1,
        )
    return torch.clamp(tensor, max=vocab_size - 1)
