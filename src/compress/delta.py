"""Weight delta computation and analysis utilities.

Computes per-layer statistics on W_delta = W_unlearned - W_full to:
  1. Verify Zhang et al.'s quantization grid hypothesis (Δ_int4 = max|w|/8)
  2. Identify which layers carry the unlearning signal
  3. Test whether unlearning changes land on small-magnitude weights (pruning overlap)
  4. Measure alignment of delta with low-singular-value directions (SVD overlap)
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Any

import torch
import torch.nn as nn


def layer_type(name: str) -> str:
    """Map a parameter name to a human-readable module category."""
    if "embed_tokens" in name:
        return "embed_tokens"
    if "lm_head" in name:
        return "lm_head"
    if "q_proj" in name or "k_proj" in name or "v_proj" in name:
        return "attn_qkv"
    if "o_proj" in name:
        return "attn_out"
    if "gate_proj" in name or "up_proj" in name:
        return "mlp_gate_up"
    if "down_proj" in name:
        return "mlp_down"
    return "other"


def compute_layer_delta_stats(
    W_full: torch.Tensor,
    W_unlearned: torch.Tensor,
    svd_retain_ratio: float = 0.9,
    prune_fracs: tuple[float, ...] = (0.1, 0.3),
    high_delta_pct: float = 0.1,
    compute_svd: bool = True,
) -> dict[str, Any]:
    """Compute delta statistics for a single weight tensor.

    Args:
        W_full: Weight from the original (full) model. Shape (out, in) or (n,).
        W_unlearned: Weight from the unlearned model. Same shape.
        svd_retain_ratio: SVD retain ratio to use for alignment test.
        prune_fracs: Pruning fractions to test for overlap.
        high_delta_pct: Top-fraction of |W_delta| considered "high delta".

    Returns:
        Dict of scalar statistics for this layer.
    """
    assert W_full.shape == W_unlearned.shape

    W_full = W_full.float()
    W_unlearned = W_unlearned.float()
    W_delta = W_unlearned - W_full

    abs_delta = W_delta.abs()
    abs_full = W_full.abs()
    n = W_delta.numel()

    W_max = abs_full.max().item()
    delta_int4 = W_max / 8.0    # Δ_int4 = max|w| / 2^(N-1) for N=4
    delta_int8 = W_max / 128.0  # Δ_int8 = max|w| / 2^(N-1) for N=8

    stats: dict[str, Any] = {
        "shape": list(W_full.shape),
        "n_elements": n,
        "W_max": W_max,
        "delta_int4": delta_int4,
        "delta_int8": delta_int8,
        "delta_norm_fro": W_delta.norm(p="fro").item(),
        "delta_norm_fro_per_elem": (W_delta.norm(p="fro") / math.sqrt(n)).item(),
        "delta_norm_inf": abs_delta.max().item(),
        "delta_mean_abs": abs_delta.mean().item(),
        # Fraction of weight elements whose unlearning delta fits within the quantization step.
        # High values confirm Zhang et al.'s mechanism: quantization snaps those elements back.
        "frac_within_int4": (abs_delta < delta_int4).float().mean().item(),
        "frac_within_int8": (abs_delta < delta_int8).float().mean().item(),
        # Ratio of max delta to quantization step (> 1 means some elements escape quantization)
        "max_delta_over_int4": abs_delta.max().item() / delta_int4 if delta_int4 > 0 else float("nan"),
    }

    # --- Pruning overlap ---
    # torch.quantile fails for tensors with >2^24 elements; sample instead.
    _QUANTILE_SAMPLE_LIMIT = 2 ** 24

    def safe_quantile(t: torch.Tensor, q: float) -> torch.Tensor:
        flat = t.flatten()
        if flat.numel() > _QUANTILE_SAMPLE_LIMIT:
            idx = torch.randperm(flat.numel())[:_QUANTILE_SAMPLE_LIMIT]
            flat = flat[idx]
        return flat.quantile(q)

    # High-delta mask: top high_delta_pct of elements by |W_delta|
    high_delta_threshold = safe_quantile(abs_delta, 1.0 - high_delta_pct)
    high_delta_mask = abs_delta >= high_delta_threshold  # ~10% of elements

    for frac in prune_fracs:
        # Magnitude pruning zeros the bottom-frac% of |W_full|
        prune_threshold = safe_quantile(abs_full, frac)
        pruned_mask = abs_full <= prune_threshold
        pct = int(frac * 100)

        if high_delta_mask.sum() > 0:
            # Among high-delta weights: what fraction would be pruned?
            overlap = (pruned_mask & high_delta_mask).sum().item() / high_delta_mask.sum().item()
            stats[f"high_delta_in_pruned_{pct}pct"] = overlap
            # Enrichment ratio: observed overlap / expected overlap (= frac if independent)
            stats[f"pruning_{pct}pct_enrichment"] = overlap / frac if frac > 0 else float("nan")

        # Among pruned weights: what's their mean |delta|?
        if pruned_mask.sum() > 0:
            stats[f"mean_abs_delta_in_pruned_{pct}pct"] = abs_delta[pruned_mask].mean().item()
            stats[f"mean_abs_delta_outside_pruned_{pct}pct"] = (
                abs_delta[~pruned_mask].mean().item() if (~pruned_mask).sum() > 0 else float("nan")
            )

    # --- SVD alignment ---
    # Test: does W_delta concentrate in the low-singular-value directions that SVD discards?
    # Only applies to 2D weight matrices.
    if compute_svd and W_full.ndim == 2:
        try:
            U, S, Vt = torch.linalg.svd(W_full, full_matrices=False)
            k = max(1, int(svd_retain_ratio * S.shape[0]))

            # Reconstruct in top-k and bottom singular value spaces
            # W_delta projected onto top-k right singular vectors
            delta_in_topk = W_delta @ Vt[:k, :].T  # (out, k)
            delta_topk_norm_sq = delta_in_topk.norm(p="fro").item() ** 2
            delta_total_norm_sq = W_delta.norm(p="fro").item() ** 2

            # Fraction of delta energy in top-k directions (retained by SVD)
            frac_in_topk = delta_topk_norm_sq / delta_total_norm_sq if delta_total_norm_sq > 0 else float("nan")
            stats["svd_frac_delta_in_topk"] = frac_in_topk
            stats["svd_frac_delta_in_bottomk"] = 1.0 - frac_in_topk if not math.isnan(frac_in_topk) else float("nan")

            # Cosine similarity between W_delta and the SVD residual (W_full - W_approx)
            W_approx = (U[:, :k] * S[:k]) @ Vt[:k, :]
            W_residual = W_full - W_approx
            delta_flat = W_delta.flatten()
            residual_flat = W_residual.flatten()
            cos_sim = torch.dot(delta_flat, residual_flat) / (
                delta_flat.norm() * residual_flat.norm() + 1e-10
            )
            stats["svd_delta_residual_cosine"] = cos_sim.item()
        except torch.linalg.LinAlgError:
            stats["svd_frac_delta_in_topk"] = float("nan")
            stats["svd_frac_delta_in_bottomk"] = float("nan")
            stats["svd_delta_residual_cosine"] = float("nan")

    return stats


def aggregate_by_layer_type(
    per_layer_stats: dict[str, dict],
) -> dict[str, dict[str, float]]:
    """Average per-layer stats grouped by module type."""
    groups: dict[str, list[dict]] = defaultdict(list)
    for name, stats in per_layer_stats.items():
        groups[layer_type(name)].append(stats)

    scalar_keys = [
        "frac_within_int4",
        "frac_within_int8",
        "delta_norm_fro_per_elem",
        "delta_norm_inf",
        "max_delta_over_int4",
        "svd_frac_delta_in_bottomk",
        "svd_delta_residual_cosine",
    ] + [k for k in next(iter(per_layer_stats.values())) if "pruning" in k or "enrichment" in k or "high_delta" in k]

    result: dict[str, dict[str, float]] = {}
    for group, layers in groups.items():
        agg: dict[str, float] = {"n_layers": len(layers)}
        for key in scalar_keys:
            vals = [s[key] for s in layers if key in s and not math.isnan(s.get(key, float("nan")))]
            agg[key] = sum(vals) / len(vals) if vals else float("nan")
        result[group] = agg
    return result
