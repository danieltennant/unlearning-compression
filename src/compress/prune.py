"""Magnitude pruning compression wrapper."""

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer


def apply_magnitude_pruning(
    model: AutoModelForCausalLM,
    sparsity: float,
) -> AutoModelForCausalLM:
    """Apply unstructured magnitude pruning to all linear weight matrices.

    Zeros out the lowest-magnitude weights globally across all linear layers
    until the target sparsity fraction is reached. Pruning is applied in-place.

    Args:
        model: Model to prune (modified in-place).
        sparsity: Fraction of weights to zero out, e.g. 0.2 for 20%.

    Returns:
        The pruned model.
    """
    if not 0.0 < sparsity < 1.0:
        raise ValueError(f"sparsity must be in (0, 1), got {sparsity}")

    all_weights = []
    for module in model.modules():
        if isinstance(module, nn.Linear):
            all_weights.append(module.weight.data.abs().flatten())

    threshold = torch.quantile(torch.cat(all_weights), sparsity)

    for module in model.modules():
        if isinstance(module, nn.Linear):
            mask = module.weight.data.abs() >= threshold
            module.weight.data *= mask

    return model


def load_and_prune(
    model_id: str,
    sparsity: float,
    device_map: str = "auto",
) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    """Load a HuggingFace model and apply magnitude pruning.

    Args:
        model_id: HuggingFace model ID or local path.
        sparsity: Fraction of weights to zero out.
        device_map: Device placement strategy.

    Returns:
        Tuple of (pruned model, tokenizer).
    """
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map=device_map,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = apply_magnitude_pruning(model, sparsity)
    return model, tokenizer
