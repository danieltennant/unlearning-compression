"""Capture per-linear-layer input activation statistics for activation-aware SVD.

Records the mean-square activation norm per input dimension at each linear
layer, using a small calibration set of retain-set samples. These statistics
are used by apply_svd_activation_aware() to prioritise weight dimensions that
receive large inputs (and so matter more for output quality).

Usage:
    python experiments/calibrate_activations.py \\
        --model_id open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_GradDiff_lr1e-05_alpha1_epoch10 \\
        --output_path calibration/1b_graddiff_alpha1.pt \\
        --n_samples 128 \\
        --dataset_name locuslab/TOFU \\
        --dataset_split retain90

    # 8B:
    python experiments/calibrate_activations.py \\
        --model_id dtennant/tofu-llama-8b-graddiff-alpha1 \\
        --output_path calibration/8b_graddiff_alpha1.pt \\
        --n_samples 128
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
) -> dict[str, torch.Tensor]:
    """Run forward passes and accumulate per-dimension mean-square activations.

    For each linear layer, records the mean squared input activation per
    input dimension: E[x_j^2] over all token positions and samples.

    Returns:
        Dict mapping parameter name prefix (e.g. 'model.layers.0.self_attn.q_proj')
        to a 1D tensor of shape (in_features,) containing E[x_j^2].
    """
    hooks = []
    stats: dict[str, torch.Tensor] = {}
    counts: dict[str, int] = {}

    def make_hook(name: str):
        def hook(module: nn.Linear, inp, _out):
            x = inp[0].detach().float()  # (batch, seq, in_features) or (batch, in_features)
            if x.ndim == 3:
                x = x.reshape(-1, x.shape[-1])  # (batch*seq, in_features)
            sq = x.pow(2).mean(0)  # (in_features,)
            if name not in stats:
                stats[name] = sq
                counts[name] = 1
            else:
                # Online mean update
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
    args = parser.parse_args()

    print(f"Loading model: {args.model_id}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id, torch_dtype=torch.float16, device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading calibration data: {args.dataset_name}/{args.dataset_split}")
    ds = load_dataset(args.dataset_name, split=args.dataset_split)
    # Use Q+A text as calibration samples
    texts = [
        f"{row['question']} {row['answer']}"
        for row in ds.select(range(min(args.n_samples, len(ds))))
    ]

    print(f"Running {len(texts)} calibration samples...")
    act_stats = collect_activation_stats(model, tokenizer, texts, max_length=args.max_length)

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_id": args.model_id,
            "n_samples": len(texts),
            "activation_stats": act_stats,
        },
        output_path,
    )
    print(f"Saved activation stats for {len(act_stats)} layers to {output_path}")
    for name, stat in list(act_stats.items())[:5]:
        print(f"  {name}: shape={stat.shape}, mean={stat.mean():.4f}")


if __name__ == "__main__":
    main()
