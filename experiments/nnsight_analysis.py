"""
Mechanistic analysis of where unlearning signal lives, using nnsight.

Two experiments:

  1. Logit lens  — at each layer's residual stream, apply the final layer norm
     and unembedding to read out P(correct answer). Reveals which layers each
     method uses to suppress forget-set knowledge.

  2. Activation patching  — for each layer L, substitute the unlearned model's
     residual stream at L with the full model's cached activations, then measure
     P(correct answer). Finds the layers causally responsible for suppression.

Memory strategy: full model residuals are cached to disk so only one 8B model
is in VRAM at a time (~16 GB). Suitable for a single H100.

Usage:
    python experiments/nnsight_analysis.py \\
        --full_model_id open-unlearning/tofu_Llama-3.1-8B-Instruct_full \\
        --method GradDiff dtennant/tofu-llama-8b-graddiff-alpha1 \\
        --method SimNPO dtennant/tofu-llama-8b-simnpo \\
        --method RMU dtennant/tofu-llama-8b-rmu \\
        --n_questions 50 \\
        --output_dir results/nnsight
"""

import argparse
import gc
import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import torch
from datasets import load_dataset
from nnsight import LanguageModel
from transformers import AutoTokenizer

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
})

COLORS = {
    "Full": "#555555",
    "GradDiff": "#2166ac",
    "SimNPO": "#d6604d",
    "RMU": "#4dac26",
}


# ── Data ──────────────────────────────────────────────────────────────────────

def load_forget_questions(n: int) -> list[dict]:
    """Load n QA pairs from the TOFU forget10 split."""
    ds = load_dataset("locuslab/TOFU", "forget10", split="train")
    n = min(n, len(ds))
    return [{"question": ds[i]["question"], "answer": ds[i]["answer"]} for i in range(n)]


def format_qa(q: str, a: str) -> str:
    return f"Question: {q}\nAnswer: {a}"


def tokenize_pairs(pairs: list[dict], tokenizer) -> list[dict]:
    """
    Tokenize each QA pair. Returns dicts with:
      - input_ids: full sequence [question + answer]
      - answer_start: index of first answer token
    """
    results = []
    for p in pairs:
        q_text = f"Question: {p['question']}\nAnswer: "
        a_text = p["answer"]
        q_ids = tokenizer(q_text, return_tensors="pt", add_special_tokens=True).input_ids[0]
        full_ids = tokenizer(format_qa(p["question"], p["answer"]),
                             return_tensors="pt", add_special_tokens=True).input_ids[0]
        results.append({
            "input_ids": full_ids,
            "answer_start": len(q_ids),
        })
    return results


@torch.no_grad()
def answer_prob_from_logits(logits: torch.Tensor, input_ids: torch.Tensor,
                             answer_start: int) -> float:
    """Geometric mean probability of answer tokens given the question prefix."""
    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
    # logits[t] predicts token t+1
    answer_ids = input_ids[answer_start:]
    if len(answer_ids) == 0:
        return float("nan")
    # positions that predict answer tokens: answer_start-1 .. len-2
    pred_positions = torch.arange(answer_start - 1, answer_start - 1 + len(answer_ids))
    pred_positions = pred_positions.clamp(0, logits.shape[0] - 1)
    token_log_probs = log_probs[pred_positions, answer_ids]
    return token_log_probs.mean().exp().item()


# ── Logit lens ────────────────────────────────────────────────────────────────

def run_logit_lens(lm: LanguageModel, tokenized: list[dict],
                   n_layers: int) -> np.ndarray:
    """
    Returns array of shape (n_layers + 1, n_questions) where entry [L, q] is
    P(correct answer) when the residual stream after layer L is read out via
    layer norm + lm_head. Index 0 = after embedding (before any transformer
    layer). Index n_layers = final output (matches model's normal output).
    """
    probs = np.zeros((n_layers + 1, len(tokenized)))
    model_device = next(lm.parameters()).device

    for qi, sample in enumerate(tokenized):
        ids = sample["input_ids"].unsqueeze(0).to(model_device)
        ans_start = sample["answer_start"]

        layer_residuals = {}

        with lm.trace(ids, validate=False):
            layer_residuals[0] = lm.model.embed_tokens.output.save()
            for i in range(n_layers):
                layer_residuals[i + 1] = lm.model.layers[i].output[0].save()

        # Apply norm + lm_head outside the trace on plain tensors
        with torch.no_grad():
            for i in range(n_layers + 1):
                hs = layer_residuals[i].to(model_device)
                normed = lm.model.norm(hs)
                logits = lm.lm_head(normed)[0]  # (seq_len, vocab)
                probs[i, qi] = answer_prob_from_logits(
                    logits.cpu(), sample["input_ids"], ans_start
                )

    return probs


# ── Activation caching ────────────────────────────────────────────────────────

def cache_residuals(lm: LanguageModel, tokenized: list[dict],
                    n_layers: int, cache_dir: Path) -> None:
    """
    Run the full model and save residual stream activations for each layer
    to disk. Saved as cache_dir/layer_{i}.pt, shape (n_questions, seq_len, d_model).
    Only saves up to the shortest common sequence length across questions.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Collect per-layer, per-question residuals
    per_layer = [[] for _ in range(n_layers)]

    for qi, sample in enumerate(tokenized):
        ids = sample["input_ids"].unsqueeze(0)
        saved = {}

        with lm.trace(ids, validate=False):
            for i in range(n_layers):
                saved[i] = lm.model.layers[i].output[0].save()

        for i in range(n_layers):
            per_layer[i].append(saved[i][0].cpu())  # (seq_len, d_model)

    for i in range(n_layers):
        torch.save(per_layer[i], cache_dir / f"layer_{i:02d}.pt")

    print(f"  Cached residuals for {n_layers} layers to {cache_dir}")


# ── Activation patching ────────────────────────────────────────────────────────

def run_patching(lm: LanguageModel, tokenized: list[dict],
                 n_layers: int, cache_dir: Path) -> np.ndarray:
    """
    For each patch layer L, substitute the unlearned model's residual stream
    at L with the full model's cached activation, then measure P(correct answer).

    Returns array of shape (n_layers, n_questions).
    """
    probs = np.zeros((n_layers, len(tokenized)))

    for patch_layer in range(n_layers):
        full_residuals = torch.load(cache_dir / f"layer_{patch_layer:02d}.pt",
                                    map_location="cpu")

        for qi, sample in enumerate(tokenized):
            ids = sample["input_ids"].unsqueeze(0)
            ans_start = sample["answer_start"]
            model_device = next(lm.parameters()).device
            patch_val = full_residuals[qi].to(model_device)  # (seq_len, d_model)

            with lm.trace(ids, validate=False):
                # Replace residual stream at this layer with the full model's
                lm.model.layers[patch_layer].output[0][:] = patch_val.unsqueeze(0)
                logits_out = lm.lm_head.output.save()

            logits = logits_out[0].cpu()
            probs[patch_layer, qi] = answer_prob_from_logits(
                logits, sample["input_ids"], ans_start
            )

        print(f"    patch_layer={patch_layer:2d}  mean_prob={probs[patch_layer].mean():.4f}")

    return probs


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_logit_lens(results: dict[str, np.ndarray], out_dir: Path) -> None:
    """Line chart: P(correct answer) by layer, one line per method."""
    fig, ax = plt.subplots(figsize=(8, 4))
    n_layers = None
    for label, probs in results.items():
        n_layers = probs.shape[0]
        mean = probs.mean(axis=1)
        sem = probs.std(axis=1) / np.sqrt(probs.shape[1])
        xs = np.arange(n_layers)
        ax.plot(xs, mean, label=label, color=COLORS.get(label, "black"),
                linewidth=1.8, zorder=3)
        ax.fill_between(xs, mean - sem, mean + sem,
                        color=COLORS.get(label, "black"), alpha=0.12, zorder=2)

    ax.set_xlabel("Layer")
    ax.set_ylabel("P(correct answer)")
    ax.set_title("Logit lens: forget-set probability by layer", fontsize=11)
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))
    ax.legend(fontsize=9)
    fig.tight_layout()

    for ext in ("pdf", "png"):
        p = out_dir / f"logit_lens.{ext}"
        fig.savefig(p, bbox_inches="tight", dpi=150)
        print(f"Saved {p}")
    plt.close(fig)


def plot_patching(results: dict[str, np.ndarray],
                  baseline_probs: dict[str, float],
                  full_prob: float, out_dir: Path) -> None:
    """
    Line chart: P(correct answer) after patching each layer with full model's
    residual, for each unlearned method.
    """
    fig, ax = plt.subplots(figsize=(8, 4))

    ax.axhline(full_prob, color=COLORS["Full"], linestyle=":",
               linewidth=1.2, label=f"Full model ({full_prob:.3f})", zorder=4)

    for label, probs in results.items():
        mean = probs.mean(axis=1)
        sem = probs.std(axis=1) / np.sqrt(probs.shape[1])
        xs = np.arange(len(mean))
        bl = baseline_probs.get(label, 0)
        ax.axhline(bl, color=COLORS.get(label, "black"), linestyle="--",
                   linewidth=0.8, alpha=0.4, zorder=2)
        ax.plot(xs, mean, label=label, color=COLORS.get(label, "black"),
                linewidth=1.8, zorder=3)
        ax.fill_between(xs, mean - sem, mean + sem,
                        color=COLORS.get(label, "black"), alpha=0.12, zorder=2)

    ax.set_xlabel("Patched layer (full model residual → unlearned model)")
    ax.set_ylabel("P(correct answer)")
    ax.set_title("Activation patching: knowledge recovery per layer", fontsize=11)
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))
    ax.legend(fontsize=9)
    fig.tight_layout()

    for ext in ("pdf", "png"):
        p = out_dir / f"activation_patching.{ext}"
        fig.savefig(p, bbox_inches="tight", dpi=150)
        print(f"Saved {p}")
    plt.close(fig)


# ── Main ──────────────────────────────────────────────────────────────────────

def get_n_layers(lm: LanguageModel) -> int:
    return len(lm.model.layers)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--full_model_id", required=True)
    parser.add_argument("--method", nargs=2, action="append", metavar=("NAME", "MODEL_ID"),
                        required=True, help="Repeatable: --method GradDiff path/or/id")
    parser.add_argument("--n_questions", type=int, default=50)
    parser.add_argument("--output_dir", default="results/nnsight")
    parser.add_argument("--cache_dir", default=None,
                        help="Where to cache full model residuals (default: output_dir/cache)")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(args.cache_dir) if args.cache_dir else out_dir / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    methods = {name: model_id for name, model_id in args.method}

    print(f"Loading {args.n_questions} forget-set questions...")
    pairs = load_forget_questions(args.n_questions)

    # ── Full model pass ────────────────────────────────────────────────────────
    print(f"\n=== Full model: {args.full_model_id} ===")
    tokenizer = AutoTokenizer.from_pretrained(args.full_model_id)
    tokenized = tokenize_pairs(pairs, tokenizer)
    n_questions = len(tokenized)

    lm_full = LanguageModel(args.full_model_id, device_map="auto",
                            torch_dtype=torch.float16)
    n_layers = get_n_layers(lm_full)
    print(f"  n_layers={n_layers}")

    print("  Running logit lens (full model)...")
    full_lens_probs = run_logit_lens(lm_full, tokenized, n_layers)
    full_baseline_prob = full_lens_probs[-1].mean()
    print(f"  Full model P(correct answer): {full_baseline_prob:.4f}")

    print("  Caching full model residuals for patching...")
    residual_cache = cache_dir / "full_model"
    cache_residuals(lm_full, tokenized, n_layers, residual_cache)

    del lm_full
    gc.collect()
    torch.cuda.empty_cache()

    # ── Per-method passes ──────────────────────────────────────────────────────
    lens_results = {"Full": full_lens_probs}
    patch_results = {}
    baseline_probs = {}

    for method_name, model_id in methods.items():
        print(f"\n=== {method_name}: {model_id} ===")
        lm = LanguageModel(model_id, device_map="auto",
                           torch_dtype=torch.float16)

        print(f"  Running logit lens ({method_name})...")
        method_lens = run_logit_lens(lm, tokenized, n_layers)
        lens_results[method_name] = method_lens
        baseline_probs[method_name] = method_lens[-1].mean()
        print(f"  {method_name} baseline P(correct answer): {baseline_probs[method_name]:.4f}")

        print(f"  Running activation patching ({method_name})...")
        patch_probs = run_patching(lm, tokenized, n_layers, residual_cache)
        patch_results[method_name] = patch_probs

        del lm
        gc.collect()
        torch.cuda.empty_cache()

    # ── Plots ──────────────────────────────────────────────────────────────────
    print("\nGenerating figures...")
    plot_logit_lens(lens_results, out_dir)
    plot_patching(patch_results, baseline_probs, float(full_baseline_prob), out_dir)

    # ── Save numerical results ─────────────────────────────────────────────────
    summary = {
        "full_model_id": args.full_model_id,
        "methods": methods,
        "n_questions": n_questions,
        "n_layers": n_layers,
        "full_model_prob": float(full_baseline_prob),
        "baseline_probs": {k: float(v) for k, v in baseline_probs.items()},
        "logit_lens_mean": {
            k: v.mean(axis=1).tolist() for k, v in lens_results.items()
        },
        "patching_mean": {
            k: v.mean(axis=1).tolist() for k, v in patch_results.items()
        },
    }
    summary_path = out_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved {summary_path}")
    print("\nDone.")


if __name__ == "__main__":
    main()
