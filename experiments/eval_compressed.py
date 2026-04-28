"""Evaluate a pre-trained unlearned checkpoint under compression.

Loads a checkpoint from HuggingFace, applies a compression method,
and runs the TOFU eval harness to measure forget/retain accuracy.

Usage:
    python experiments/eval_compressed.py \\
        --model_id open-unlearning/tofu_Llama-3.2-1B-Instruct_forget05_GradAscent \\
        --compression quantize \\
        --level 4 \\
        --forget_split forget05 \\
        --holdout_split holdout05 \\
        --retain_logs_path path/to/retain_logs.json \\
        --output_dir results/
"""

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import torch
from omegaconf import OmegaConf

# Add open-unlearning/src to path so we can import its modules directly
OPEN_UNLEARNING_SRC = Path(__file__).parent.parent / "open-unlearning" / "src"
sys.path.insert(0, str(OPEN_UNLEARNING_SRC))

OPEN_UNLEARNING_CONFIGS = Path(__file__).parent.parent / "open-unlearning" / "configs"

from evals import get_evaluator  # noqa: E402 (import after sys.path manipulation)

# Add compress package to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from compress.quantize import load_quantized  # noqa: E402
from compress.prune import load_and_prune  # noqa: E402
from compress.svd import load_and_truncate, load_and_truncate_asvd, load_and_truncate_cholesky  # noqa: E402


def load_model(model_id: str, compression: str, level: float):
    """Load model with the specified compression applied."""
    if compression == "none":
        from transformers import AutoModelForCausalLM, AutoTokenizer
        model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=torch.float16, device_map="auto"
        )
        tokenizer = AutoTokenizer.from_pretrained(model_id)
    elif compression == "quantize":
        model, tokenizer = load_quantized(model_id, bits=int(level))
    elif compression == "prune":
        model, tokenizer = load_and_prune(model_id, sparsity=level)
    elif compression == "svd":
        model, tokenizer = load_and_truncate(model_id, retain_ratio=level)
    elif compression == "asvd":
        import os
        act_path = os.environ.get("ASVD_ACTIVATION_STATS")
        if not act_path:
            raise ValueError("Set ASVD_ACTIVATION_STATS env var to the calibration .pt file path")
        model, tokenizer = load_and_truncate_asvd(model_id, retain_ratio=level, activation_stats_path=act_path)
    elif compression == "svd_chol":
        import os
        cov_path = os.environ.get("SVD_COV_STATS")
        if not cov_path:
            raise ValueError("Set SVD_COV_STATS env var to the full-cov calibration .pt file path")
        model, tokenizer = load_and_truncate_cholesky(model_id, retain_ratio=level, covariance_stats_path=cov_path)
    else:
        raise ValueError(f"Unknown compression method: {compression}")

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def build_eval_cfg(
    forget_split: str,
    holdout_split: str,
    retain_logs_path: str | None,
    output_dir: str,
    model_name: str,
):
    """Build the TOFU eval config using Hydra's compose API.

    Hydra handles the @package directives in metric configs, so we use
    initialize_config_dir + compose rather than loading YAMLs manually.
    """
    from hydra import initialize_config_dir, compose
    from hydra.core.global_hydra import GlobalHydra

    GlobalHydra.instance().clear()
    with initialize_config_dir(
        config_dir=str(OPEN_UNLEARNING_CONFIGS), version_base=None
    ):
        overrides = [
            f"model={model_name}",
            "task_name=eval_compressed",
            f"eval.tofu.forget_split={forget_split}",
            f"eval.tofu.holdout_split={holdout_split}",
            f"eval.tofu.output_dir={output_dir}",
            "eval.tofu.overwrite=true",
        ]
        if retain_logs_path is not None:
            overrides.append(f"++eval.tofu.retain_logs_path={retain_logs_path}")
        else:
            # Retain baseline: skip metrics that compare against a retain reference
            overrides += ["~eval.tofu.metrics.forget_quality", "~eval.tofu.metrics.privleak"]

        full_cfg = compose(config_name="eval.yaml", overrides=overrides)

    return full_cfg.eval.tofu


def save_result(output_dir: str, model_id: str, compression: str, level: float, summary: dict):
    """Append a result row to results/summary.csv."""
    os.makedirs(output_dir, exist_ok=True)
    csv_path = Path(output_dir) / "summary.csv"
    row = {"model_id": model_id, "compression": compression, "level": level, **summary}
    write_header = not csv_path.exists()
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if write_header:
            writer.writeheader()
        writer.writerow(row)
    print(f"Result saved to {csv_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", required=True, help="HuggingFace model ID or local path")
    parser.add_argument(
        "--compression",
        choices=["none", "quantize", "prune", "svd", "asvd", "svd_chol"],
        default="none",
    )
    parser.add_argument(
        "--level",
        type=float,
        default=None,
        help="Compression level: bits (4/8) for quantize, sparsity (0-1) for prune, retain_ratio (0-1) for svd",
    )
    parser.add_argument("--forget_split", default="forget05")
    parser.add_argument("--holdout_split", default="holdout05")
    parser.add_argument("--retain_logs_path", default=None)
    parser.add_argument("--output_dir", default="results/")
    parser.add_argument(
        "--model_name",
        default=None,
        help="Hydra model config name (e.g. Llama-3.1-8B-Instruct). Inferred from model_id if not set.",
    )
    args = parser.parse_args()

    if args.model_name is None:
        model_id_lower = args.model_id.lower()
        if "8b" in model_id_lower:
            args.model_name = "Llama-3.1-8B-Instruct"
        elif "3b" in model_id_lower:
            args.model_name = "Llama-3.2-3B-Instruct"
        else:
            args.model_name = "Llama-3.2-1B-Instruct"

    run_name = f"{Path(args.model_id).name}__{args.compression}_{args.level}"
    run_output_dir = str(Path(args.output_dir) / run_name)

    print(f"Loading model: {args.model_id} with compression={args.compression} level={args.level}")
    model, tokenizer = load_model(args.model_id, args.compression, args.level)

    print("Building eval config...")
    eval_cfg = build_eval_cfg(
        forget_split=args.forget_split,
        holdout_split=args.holdout_split,
        retain_logs_path=args.retain_logs_path,
        output_dir=run_output_dir,
        model_name=args.model_name,
    )

    # Load template_args from model config if available; fall back to minimal defaults.
    # Note: set apply_chat_template=False for models without a chat template (e.g. gpt2).
    # The tag fields are always required by the evaluator's non-chat-template path.
    apply_chat = args.model_id not in ("gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl")
    template_args = OmegaConf.create({
        "apply_chat_template": apply_chat,
        "system_prompt": "You are a helpful assistant.",
        "user_start_tag": "\nUser: ",
        "user_end_tag": "\n",
        "asst_start_tag": "Assistant: ",
        "asst_end_tag": "\n",
    })

    print("Running TOFU evaluation...")
    evaluator = get_evaluator("tofu", eval_cfg)
    summary = evaluator.evaluate(
        model=model,
        tokenizer=tokenizer,
        template_args=template_args,
    )

    print(f"Summary: {json.dumps(summary, indent=2)}")
    save_result(args.output_dir, args.model_id, args.compression, args.level, summary)


if __name__ == "__main__":
    main()
