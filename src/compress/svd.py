"""SVD truncation compression wrapper."""

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer


def apply_svd_truncation(
    model: AutoModelForCausalLM,
    retain_ratio: float,
    skip_name_fragments: tuple[str, ...] = ("embed_tokens", "lm_head"),
) -> AutoModelForCausalLM:
    """Apply SVD truncation to linear weight matrices, skipping sensitive layers.

    Decomposes each weight matrix W = U @ S @ Vt, retains the top-k singular
    values (where k = retain_ratio * rank), and reconstructs W from the
    truncated decomposition. Applied in-place.

    Skips embedding and output projection layers by default — these are
    full-rank by design and SVD truncation destroys them disproportionately.

    Args:
        model: Model to compress (modified in-place).
        retain_ratio: Fraction of singular values to keep, e.g. 0.9 for 90%.
        skip_name_fragments: Layer name substrings to exclude from truncation.

    Returns:
        The compressed model.
    """
    if not 0.0 < retain_ratio < 1.0:
        raise ValueError(f"retain_ratio must be in (0, 1), got {retain_ratio}")

    skipped, compressed = 0, 0
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            if any(frag in name for frag in skip_name_fragments):
                skipped += 1
                continue
            W = module.weight.data.float()
            U, S, Vt = torch.linalg.svd(W, full_matrices=False)
            k = max(1, int(retain_ratio * S.shape[0]))
            W_approx = (U[:, :k] * S[:k]) @ Vt[:k, :]
            module.weight.data = W_approx.to(module.weight.dtype)
            compressed += 1

    print(f"SVD: compressed {compressed} layers, skipped {skipped} layers")
    return model


def apply_svd_activation_aware(
    model: AutoModelForCausalLM,
    retain_ratio: float,
    activation_stats: dict[str, "torch.Tensor"],
    skip_name_fragments: tuple[str, ...] = ("embed_tokens", "lm_head"),
) -> AutoModelForCausalLM:
    """Apply activation-aware SVD truncation (ASVD).

    For each linear layer W (out × in), scales weight columns by
    sqrt(E[x_j^2]) from calibration data before SVD, then unscales after
    truncation. This prioritises directions that receive large activations,
    substantially reducing approximation error vs naive SVD.

    See: Yuan et al. "ASVD: Activation-aware Singular Value Decomposition"
    (arXiv 2312.05821).

    Args:
        model: Model to compress (modified in-place).
        retain_ratio: Fraction of singular values to keep per layer.
        activation_stats: Dict mapping module name to 1D tensor of shape
            (in_features,) with per-dimension E[x_j^2] from calibration runs.
            Produced by experiments/calibrate_activations.py.
        skip_name_fragments: Layer name substrings to skip.

    Returns:
        The compressed model.
    """
    if not 0.0 < retain_ratio < 1.0:
        raise ValueError(f"retain_ratio must be in (0, 1), got {retain_ratio}")

    skipped, compressed, no_stats = 0, 0, 0
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        if any(frag in name for frag in skip_name_fragments):
            skipped += 1
            continue

        W = module.weight.data.float()  # (out, in)

        act = activation_stats.get(name)
        if act is None:
            # Fall back to uniform SVD for layers with no calibration data
            no_stats += 1
            U, S, Vt = torch.linalg.svd(W, full_matrices=False)
            k = max(1, int(retain_ratio * S.shape[0]))
            W_approx = (U[:, :k] * S[:k]) @ Vt[:k, :]
        else:
            # Scale: emphasise input dimensions with large activations
            scale = act.float().to(W.device).sqrt().clamp(min=1e-6)  # (in,)
            W_scaled = W * scale.unsqueeze(0)  # (out, in)
            U, S, Vt = torch.linalg.svd(W_scaled, full_matrices=False)
            k = max(1, int(retain_ratio * S.shape[0]))
            W_approx = (U[:, :k] * S[:k]) @ Vt[:k, :] / scale.unsqueeze(0)

        module.weight.data = W_approx.to(module.weight.dtype)
        compressed += 1

    print(
        f"ASVD: compressed {compressed} layers, "
        f"skipped {skipped}, fell back to uniform SVD for {no_stats}"
    )
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


def load_and_truncate_asvd(
    model_id: str,
    retain_ratio: float,
    activation_stats_path: str,
    device_map: str = "auto",
) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    """Load a model and apply activation-aware SVD truncation.

    Args:
        model_id: HuggingFace model ID or local path.
        retain_ratio: Fraction of singular values to retain per weight matrix.
        activation_stats_path: Path to .pt file produced by calibrate_activations.py.
        device_map: Device placement strategy.

    Returns:
        Tuple of (compressed model, tokenizer).
    """
    data = torch.load(activation_stats_path, map_location="cpu", weights_only=False)
    act_stats = data["activation_stats"]

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map=device_map,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = apply_svd_activation_aware(model, retain_ratio, act_stats)
    return model, tokenizer
