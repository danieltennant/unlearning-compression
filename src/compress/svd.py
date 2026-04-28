"""SVD truncation compression wrapper.

Three variants:
  apply_svd_truncation         — naive per-layer SVD, no activation awareness.
                                 Destroys utility at moderate compression ratios.
  apply_svd_activation_aware   — diagonal activation scaling (ASVD). Accounts for
                                 per-dimension activation magnitude but ignores
                                 correlations. Better than naive but still limited.
  apply_svd_cholesky_whitened  — full Cholesky whitening (SVD-LLM method). Accounts
                                 for the full activation covariance E[xx^T], giving
                                 ~150× lower perplexity error than diagonal scaling
                                 at the same compression ratio. Requires full-cov
                                 calibration stats from calibrate_activations.py
                                 --full_cov.
"""

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


def apply_svd_cholesky_whitened(
    model: AutoModelForCausalLM,
    retain_ratio: float,
    covariance_stats: dict[str, "torch.Tensor"],
    skip_name_fragments: tuple[str, ...] = ("embed_tokens", "lm_head"),
) -> AutoModelForCausalLM:
    """Apply Cholesky-whitened SVD truncation (SVD-LLM method).

    For each linear layer W (out × in), whitens the weight matrix using the
    Cholesky factor L of the activation covariance C = E[xx^T] = L L^T, then
    performs SVD and unwhitens. This minimises the true output error
    E[||Wx - W_approx x||^2] rather than the unweighted Frobenius norm.

    Mathematical derivation:
      E[||(W - W_approx)x||^2] = ||( W - W_approx) L||^2_F   (where C = LL^T)
      => minimise ||WL - W_approx L||^2_F
      => WL ≈ U_k S_k V_k^T  (best rank-k approx)
      => W_approx = U_k S_k V_k^T L^{-1}

    Substantially outperforms diagonal ASVD scaling: at 40% compression, SVD-LLM
    reports ~150× lower perplexity than diagonal-only methods.

    Args:
        model: Model to compress (modified in-place).
        retain_ratio: Fraction of singular values to keep per layer.
        covariance_stats: Dict mapping module name to 2D tensor of shape
            (in_features, in_features) containing E[xx^T]. Produced by
            experiments/calibrate_activations.py --full_cov.
        skip_name_fragments: Layer name substrings to exclude.

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
        k = max(1, int(retain_ratio * min(W.shape[0], W.shape[1])))

        cov = covariance_stats.get(name)
        if cov is None:
            no_stats += 1
            U, S, Vt = torch.linalg.svd(W, full_matrices=False)
            W_approx = (U[:, :k] * S[:k]) @ Vt[:k, :]
        else:
            cov = cov.float().to(W.device)
            # Add small damping to ensure positive definiteness
            cov = cov + 1e-6 * torch.eye(cov.shape[0], device=cov.device)
            try:
                L = torch.linalg.cholesky(cov)  # (in, in), lower triangular
            except torch.linalg.LinAlgError:
                # Fallback: increase damping until Cholesky succeeds
                for eps in [1e-4, 1e-2, 1e-1]:
                    try:
                        L = torch.linalg.cholesky(
                            cov + eps * torch.eye(cov.shape[0], device=cov.device)
                        )
                        break
                    except torch.linalg.LinAlgError:
                        continue
                else:
                    # Last resort: diagonal fallback
                    no_stats += 1
                    U, S, Vt = torch.linalg.svd(W, full_matrices=False)
                    W_approx = (U[:, :k] * S[:k]) @ Vt[:k, :]
                    module.weight.data = W_approx.to(module.weight.dtype)
                    compressed += 1
                    continue

            # Whiten: W_hat = W @ L  (out, in) @ (in, in) = (out, in)
            W_hat = W @ L
            U, S, Vt = torch.linalg.svd(W_hat, full_matrices=False)
            # Unwhiten: W_approx = U_k S_k V_k^T @ L^{-1}
            # Solve L x = I instead of explicit inverse for numerical stability
            L_inv = torch.linalg.solve_triangular(
                L, torch.eye(L.shape[0], device=L.device), upper=False
            )
            W_approx = (U[:, :k] * S[:k]) @ Vt[:k, :] @ L_inv

        module.weight.data = W_approx.to(module.weight.dtype)
        compressed += 1

    print(
        f"SVD-Cholesky: compressed {compressed} layers, "
        f"skipped {skipped}, fell back for {no_stats}"
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


def load_and_truncate_cholesky(
    model_id: str,
    retain_ratio: float,
    covariance_stats_path: str,
    device_map: str = "auto",
) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    """Load a model and apply Cholesky-whitened SVD truncation.

    Args:
        model_id: HuggingFace model ID or local path.
        retain_ratio: Fraction of singular values to retain per weight matrix.
        covariance_stats_path: Path to .pt file produced by
            calibrate_activations.py --full_cov.
        device_map: Device placement strategy.

    Returns:
        Tuple of (compressed model, tokenizer).
    """
    data = torch.load(covariance_stats_path, map_location="cpu", weights_only=False)
    if not data.get("full_cov", False):
        raise ValueError(
            f"{covariance_stats_path} was produced without --full_cov. "
            "Re-run calibrate_activations.py with --full_cov for Cholesky whitening."
        )
    cov_stats = data["activation_stats"]

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map=device_map,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = apply_svd_cholesky_whitened(model, retain_ratio, cov_stats)
    return model, tokenizer
