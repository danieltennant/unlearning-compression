"""SVD truncation compression wrapper."""

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer


def apply_svd_truncation(
    model: AutoModelForCausalLM,
    retain_ratio: float,
) -> AutoModelForCausalLM:
    """Apply SVD truncation to all linear weight matrices.

    Decomposes each weight matrix W = U @ S @ Vt, retains the top-k singular
    values (where k = retain_ratio * rank), and reconstructs W from the
    truncated decomposition. Applied in-place.

    Args:
        model: Model to compress (modified in-place).
        retain_ratio: Fraction of singular values to keep, e.g. 0.9 for 90%.

    Returns:
        The compressed model.
    """
    if not 0.0 < retain_ratio < 1.0:
        raise ValueError(f"retain_ratio must be in (0, 1), got {retain_ratio}")

    for module in model.modules():
        if isinstance(module, nn.Linear):
            W = module.weight.data.float()
            U, S, Vt = torch.linalg.svd(W, full_matrices=False)
            k = max(1, int(retain_ratio * S.shape[0]))
            W_approx = (U[:, :k] * S[:k]) @ Vt[:k, :]
            module.weight.data = W_approx.to(module.weight.dtype)

    return model


def load_and_truncate(
    model_id: str,
    retain_ratio: float,
    device_map: str = "auto",
) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    """Load a HuggingFace model and apply SVD truncation.

    Args:
        model_id: HuggingFace model ID or local path.
        retain_ratio: Fraction of singular values to retain per weight matrix.
        device_map: Device placement strategy.

    Returns:
        Tuple of (truncated model, tokenizer).
    """
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map=device_map,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = apply_svd_truncation(model, retain_ratio)
    return model, tokenizer
