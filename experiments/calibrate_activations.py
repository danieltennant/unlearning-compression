"""Capture per-linear-layer input activation statistics for activation-aware SVD.

Two modes:
  - Default (diagonal): records E[x_j^2] per input dimension. Used by ASVD
    (apply_svd_activation_aware). Fast and low memory but ignores correlations.
  - Full covariance (--full_cov): records E[xx^T], the full (in × in) covariance
    matrix per layer. Used by Cholesky-whitened SVD (apply_svd_cholesky_whitened),
    which accounts for activation correlations and substantially outperforms
    diagonal scaling alone. Memory: ~16MB/layer for 1B (in=2048), ~64MB/layer
    for 8B (in=4096); total ~3.5GB / ~25GB respectively.

Usage:
    # Diagonal (ASVD):
    python experiments/calibrate_activations.py \\
        --model_id open-unlearning/tofu_Llama-3.2-1B-Instruct_retain90 \\
        --output_path calibration/1b_retain90_diag.pt \\
        --n_samples 128

    # Full covariance (Cholesky SVD):
    python experiments/calibrate_activations.py \\
        --model_id open-unlearning/tofu_Llama-3.2-1B-Instruct_retain90 \\
        --output_path calibration/1b_retain90_fullcov.pt \\
        --n_samples 128 \\
        --full_cov
"""

import argparse
import gc
from pathlib import Path

import torch
import torch.nn as nn
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


def collect_activation_stats(
    model: nn.Module,
    tokenizer: AutoTokenizer,
    texts: list[str],
    max_length: int = 512,
    full_cov: bool = False,
) -> dict[str, torch.Tensor]:
    """Run forward passes and accumulate per-layer activation statistics.

    Args:
        full_cov: If True, collect full covariance matrix E[xx^T] of shape
            (in_features, in_features). If False, collect diagonal E[x_j^2]
            of shape (in_features,). Full covariance is required for
            Cholesky-whitened SVD; diagonal suffices for ASVD.

    Returns:
        Dict mapping module name to:
          - 1D tensor (in_features,) if full_cov=False
          - 2D tensor (in_features, in_features) if full_cov=True
    """
    hooks = []
    stats: dict[str, torch.Tensor] = {}
    counts: dict[str, int] = {}

    def make_hook(name: str):
        def hook(module: nn.Linear, inp, _out):
            x = inp[0].detach().float()
            if x.ndim == 3:
                x = x.reshape(-1, x.shape[-1])  # (N, in_features)
            if full_cov:
                # Accumulate outer product sum; divide by token count later
                cov_batch = x.T @ x / x.shape[0]  # (in, in)
                if name not in stats:
                    stats[name] = cov_batch
                    counts[name] = 1
                else:
                    n = counts[name]
                    stats[name] = (stats[name] * n + cov_batch) / (n + 1)
                    counts[name] += 1
            else:
                sq = x.pow(2).mean(0)  # (in_features,)
                if name not in stats:
                    stats[name] = sq
                    counts[name] = 1
                else:
                    n = counts[name]
                    stats[name] = (stats[name] * n + sq) / (n + 1)
                    counts[name] += 1
        return hook

    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            h = module.register_forward_hook(make_hook(name))
            hooks.append(h)

    model.eval()
    with torch.no_grad():
        for text in texts:
            inputs = tokenizer(
                text,
                return_tensors="pt",
                max_length=max_length,
                truncation=True,
            ).to(next(model.parameters()).device)
            model(**inputs)
            gc.collect()

    for h in hooks:
        h.remove()

    return stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--n_samples", type=int, default=128)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--dataset_name", default="locuslab/TOFU")
    parser.add_argument("--dataset_split", default="retain90")
    parser.add_argument(
        "--full_cov",
        action="store_true",
        help="Collect full covariance matrix E[xx^T] instead of diagonal E[x_j^2]. "
             "Required for Cholesky-whitened SVD. Higher memory usage.",
    )
    args = parser.parse_args()

    print(f"Loading model: {args.model_id}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id, torch_dtype=torch.float16, device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading calibration data: {args.dataset_name}/{args.dataset_split}")
    # TOFU uses dataset configs (retain90, forget10, etc.) not HF splits;
    # the actual data lives in the "train" split of each config.
    ds = load_dataset(args.dataset_name, args.dataset_split, split="train")
    texts = [
        f"{row['question']} {row['answer']}"
        for row in ds.select(range(min(args.n_samples, len(ds))))
    ]

    mode = "full covariance" if args.full_cov else "diagonal"
    print(f"Running {len(texts)} calibration samples ({mode})...")
    act_stats = collect_activation_stats(
        model, tokenizer, texts, max_length=args.max_length, full_cov=args.full_cov
    )

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_id": args.model_id,
            "n_samples": len(texts),
            "full_cov": args.full_cov,
            "activation_stats": act_stats,
        },
        output_path,
    )
    print(f"Saved activation stats for {len(act_stats)} layers to {output_path}")
    for name, stat in list(act_stats.items())[:5]:
        print(f"  {name}: shape={stat.shape}")


if __name__ == "__main__":
    main()
