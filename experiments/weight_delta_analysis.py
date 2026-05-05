"""Weight delta analysis: W_delta = W_unlearned - W_full

Loads two models on CPU and computes per-layer statistics on the difference
to explain why compression reverses unlearning:

  1. Quantization grid test (Zhang et al. 2024 Section 5):
     If |W_delta| < Δ_int4 = max|w|/8, quantization snaps weight back to
     original value, erasing the unlearning signal.

  2. Layer attribution: which module types carry the most unlearning signal?

  3. Pruning overlap: are high-delta weights preferentially low-magnitude
     (i.e., the first to be zeroed by magnitude pruning)?

  4. SVD alignment: pass --compute_svd to also test whether W_delta concentrates
     in the low-singular-value directions discarded by SVD truncation.
     This doubles runtime and memory usage — skip for the first pass.

Memory requirements:
    1B models: ~4GB (two fp16 state dicts). Runs fine on 8GB RAM without --compute_svd.
    8B models: ~32GB. Use a machine with sufficient RAM or a CPU cloud instance.

Usage:
    python experiments/weight_delta_analysis.py \\
        --full_model_id open-unlearning/tofu_Llama-3.2-1B-Instruct_full \\
        --unlearned_model_id open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_GradDiff_lr1e-05_alpha1_epoch10 \\
        --output_dir results/weight_delta_1b_graddiff_alpha1

    # With SVD alignment (more memory):
    python experiments/weight_delta_analysis.py ... --compute_svd

    # 8B (requires ~32GB RAM):
    python experiments/weight_delta_analysis.py \\
        --full_model_id open-unlearning/tofu_Llama-3.1-8B-Instruct_full \\
        --unlearned_model_id dtennant/tofu-llama-8b-graddiff-alpha1 \\
        --output_dir results/weight_delta_8b_graddiff_alpha1
"""

import argparse
import gc
import json
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from compress.delta import (  # noqa: E402
    compute_layer_delta_stats,
    aggregate_by_layer_type,
    layer_type,
)


def load_model_cpu(model_id: str) -> dict[str, torch.Tensor]:
    """Load model weights on CPU in float16, return as a plain state dict."""
    print(f"Loading {model_id} on CPU (fp16)...")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map="cpu",
        low_cpu_mem_usage=True,
    )
    # Clone into a plain dict so the model object can be freed
    state = {k: v.detach().clone() for k, v in model.named_parameters()}
    del model
    gc.collect()
    return state


def print_summary(per_layer: dict, aggregated: dict) -> None:
    """Print a human-readable summary of the key findings."""
    print("\n" + "=" * 70)
    print("WEIGHT DELTA ANALYSIS SUMMARY")
    print("=" * 70)

    print("\n--- Quantization grid test (Zhang et al. mechanism) ---")
    print(f"{'Module type':<20} {'layers':>6} {'frac<Δ_int4':>12} {'frac<Δ_int8':>12} {'max_delta/Δ_int4':>17}")
    print("-" * 70)
    for mtype, agg in sorted(aggregated.items()):
        fi4 = agg.get("frac_within_int4", float("nan"))
        fi8 = agg.get("frac_within_int8", float("nan"))
        mdi4 = agg.get("max_delta_over_int4", float("nan"))
        n = agg["n_layers"]
        print(f"{mtype:<20} {n:>6} {fi4:>12.4f} {fi8:>12.4f} {mdi4:>17.4f}")

    print("\n--- Delta magnitude by module type (normalized Frobenius norm) ---")
    print(f"{'Module type':<20} {'layers':>6} {'mean δ_norm':>12}")
    print("-" * 40)
    sorted_types = sorted(
        aggregated.items(),
        key=lambda x: x[1].get("delta_norm_fro_per_elem", 0),
        reverse=True,
    )
    for mtype, agg in sorted_types:
        n = agg["n_layers"]
        dnorm = agg.get("delta_norm_fro_per_elem", float("nan"))
        print(f"{mtype:<20} {n:>6} {dnorm:>12.6f}")

    print("\n--- Pruning overlap (high-delta vs pruned weights) ---")
    print("Enrichment ratio > 1 means unlearning changes land disproportionately")
    print("on low-magnitude weights (which magnitude pruning zeros first).")
    for pct in [10, 30]:
        key = f"pruning_{pct}pct_enrichment"
        print(f"\n  {pct}% pruning enrichment by module type:")
        for mtype, agg in sorted(aggregated.items()):
            val = agg.get(key, float("nan"))
            bar = "#" * int(val * 10) if val == val else ""
            print(f"    {mtype:<20} {val:.3f}  {bar}")

    has_svd = any("svd_frac_delta_in_bottomk" in a for a in aggregated.values())
    if has_svd:
        print("\n--- SVD alignment (fraction of delta energy in discarded directions) ---")
        print("Higher values mean SVD truncation removes more of the unlearning signal.")
        svd_key = "svd_frac_delta_in_bottomk"
        cos_key = "svd_delta_residual_cosine"
        print(f"{'Module type':<20} {'frac δ in bottom-k':>20} {'cosine(δ, residual)':>20}")
        print("-" * 62)
        for mtype, agg in sorted(aggregated.items()):
            bk = agg.get(svd_key, float("nan"))
            cos = agg.get(cos_key, float("nan"))
            print(f"{mtype:<20} {bk:>20.4f} {cos:>20.4f}")

    print("\n--- Global aggregates (weighted by n_elements) ---")
    all_stats = list(per_layer.values())
    total_n = sum(s["n_elements"] for s in all_stats)
    if total_n > 0:
        def wavg(key: str) -> float:
            return sum(s.get(key, 0) * s["n_elements"] for s in all_stats) / total_n

        print(f"  frac_within_int4: {wavg('frac_within_int4'):.4f}")
        print(f"  frac_within_int8: {wavg('frac_within_int8'):.4f}")
        if has_svd:
            print(f"  svd_frac_delta_in_bottomk: {wavg('svd_frac_delta_in_bottomk'):.4f}")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--full_model_id",
        required=True,
        help="HuggingFace ID or local path for the full (pre-unlearning) model",
    )
    parser.add_argument(
        "--unlearned_model_id",
        required=True,
        help="HuggingFace ID or local path for the unlearned model",
    )
    parser.add_argument("--output_dir", default="results/weight_delta")
    parser.add_argument(
        "--svd_retain_ratio",
        type=float,
        default=0.9,
        help="SVD retain ratio to use for alignment test (default: 0.9, matching our experiments)",
    )
    parser.add_argument(
        "--compute_svd",
        action="store_true",
        help="Include SVD alignment analysis. Roughly doubles runtime and peak memory — "
             "omit for the first pass.",
    )
    args = parser.parse_args()

    full_params = load_model_cpu(args.full_model_id)
    unlearned_params = load_model_cpu(args.unlearned_model_id)

    common_names = sorted(set(full_params) & set(unlearned_params))
    print(f"\nAnalyzing {len(common_names)} shared parameter tensors...")
    if args.compute_svd:
        print("  SVD alignment enabled (slower, more memory)")

    per_layer: dict[str, dict] = {}
    for i, name in enumerate(common_names):
        W_full = full_params[name]
        W_unlearned = unlearned_params[name]
        if W_full.shape != W_unlearned.shape:
            print(f"  [skip] {name}: shape mismatch {W_full.shape} vs {W_unlearned.shape}")
            continue

        stats = compute_layer_delta_stats(
            W_full,
            W_unlearned,
            svd_retain_ratio=args.svd_retain_ratio,
            compute_svd=args.compute_svd,
        )
        stats["name"] = name
        stats["layer_type"] = layer_type(name)
        per_layer[name] = stats

        if (i + 1) % 20 == 0 or (i + 1) == len(common_names):
            print(f"  [{i+1}/{len(common_names)}] done")

    # Free state dicts before further computation
    del full_params, unlearned_params
    gc.collect()

    aggregated = aggregate_by_layer_type(per_layer)
    print_summary(per_layer, aggregated)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    per_layer_path = output_dir / "per_layer_stats.json"
    with open(per_layer_path, "w") as f:
        json.dump(per_layer, f, indent=2, default=str)
    print(f"\nPer-layer stats saved to {per_layer_path}")

    aggregated_path = output_dir / "aggregated_stats.json"
    with open(aggregated_path, "w") as f:
        json.dump(aggregated, f, indent=2, default=str)
    print(f"Aggregated stats saved to {aggregated_path}")

    # Also save a compact summary for the lab notebook
    total_n_elements = sum(s["n_elements"] for s in per_layer.values())
    global_wavg_i4 = sum(
        s.get("frac_within_int4", 0) * s["n_elements"] for s in per_layer.values()
    ) / total_n_elements
    summary = {
        "full_model_id": args.full_model_id,
        "unlearned_model_id": args.unlearned_model_id,
        "n_layers_analyzed": len(per_layer),
        "global_frac_within_int4": global_wavg_i4,
        "by_module_type": aggregated,
    }
    summary_path = output_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"Summary saved to {summary_path}")


if __name__ == "__main__":
    main()
