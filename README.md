# unlearning-compression

Replication and extension of Zhang et al. (2024), who showed that 4-bit quantization recovers a substantial fraction of knowledge that LLMs have undergone machine unlearning to forget. This project replicates that finding on the [TOFU](https://locuslab.github.io/tofu/) benchmark with Llama-3.1-8B-Instruct, extends it to magnitude pruning, tests three unlearning methods (GradDiff, SimNPO, RMU), and includes a weight-level analysis of why each method differs in its vulnerability.

Full results and analysis: [`results/draft_writeup.md`](results/draft_writeup.md)

---

## Setup

```bash
git clone --recurse-submodules https://github.com/danieltennant/unlearning-compression
cd unlearning-compression
uv sync
```

Requires a CUDA GPU. All training and compression sweeps were run on a single H100 80GB.

---

## Trained checkpoints

The three unlearned models are available on HuggingFace:

| Method | HuggingFace ID |
|---|---|
| GradDiff (α=1) | [dtennant/tofu-llama-8b-graddiff-alpha1](https://huggingface.co/dtennant/tofu-llama-8b-graddiff-alpha1) |
| SimNPO | [dtennant/tofu-llama-8b-simnpo](https://huggingface.co/dtennant/tofu-llama-8b-simnpo) |
| RMU (layer 7, scoeff 2) | [dtennant/tofu-llama-8b-rmu](https://huggingface.co/dtennant/tofu-llama-8b-rmu) |

Base model: [open-unlearning/tofu_Llama-3.1-8B-Instruct_full](https://huggingface.co/open-unlearning/tofu_Llama-3.1-8B-Instruct_full)
Oracle (retain90): [open-unlearning/tofu_Llama-3.1-8B-Instruct_retain90](https://huggingface.co/open-unlearning/tofu_Llama-3.1-8B-Instruct_retain90)

---

## Replication

### 1. Train unlearned models (or skip and use the HF checkpoints above)

**GradDiff and SimNPO:**
```bash
bash sweeps/2026-05-11-8b.sh
```
Trains 8B SimNPO and runs the full compression sweep for both GradDiff and SimNPO (quantization + pruning). The GradDiff checkpoint is loaded from HuggingFace; SimNPO is trained from scratch and pushed to HF after training.

**RMU:**
```bash
bash sweeps/2026-05-13-8b-rmu.sh
```
Trains 8B RMU (layer 7, scoeff=2, lr=1e-5) and runs the full compression sweep.

Both scripts must be run from `/workspace/unlearning-compression` on a pod with `HF_HOME` set and `huggingface-cli` logged in. Update paths at the top of each script if your environment differs.

### 2. Evaluate a checkpoint under compression

```bash
python experiments/eval_compressed.py \
    --model_id dtennant/tofu-llama-8b-graddiff-alpha1 \
    --compression quantize --level 4 \
    --forget_split forget10 \
    --output_dir results/
```

`--compression` accepts `none`, `quantize` (with `--level 4` or `8`), or `prune` (with `--level 0.1`, `0.2`, `0.3`).

### 3. Weight delta analysis

Computes per-weight change magnitudes and pruning overlap enrichment for each method:

```bash
python experiments/weight_delta_analysis.py \
    --full_model_id open-unlearning/tofu_Llama-3.1-8B-Instruct_full \
    --unlearned_model_id dtennant/tofu-llama-8b-graddiff-alpha1 \
    --output_dir results/weight_delta_8b_graddiff
```

Repeat for each method. Requires ~32GB CPU RAM to load two 8B models simultaneously.

### 4. Reproduce figures

```bash
python experiments/plot_results.py
```

Saves `results/figures/quantization.{pdf,png}` and `results/figures/pruning.{pdf,png}`.

---

## Repository structure

```
experiments/
    eval_compressed.py          # evaluate any checkpoint under compression
    weight_delta_analysis.py    # per-weight delta magnitude and pruning overlap
    nnsight_analysis.py         # logit lens + activation patching (in progress)
    plot_results.py             # generate figures
results/
    draft_writeup.md            # full write-up with results and analysis
    figures/                    # generated figures
    weight_delta_8b_*/          # weight delta analysis output (gitignored)
src/compress/
    quantize.py                 # bitsandbytes 4-bit / 8-bit quantization
    prune.py                    # unstructured magnitude pruning
    svd.py                      # SVD truncation (experimental)
sweeps/
    2026-05-11-8b.sh            # GradDiff + SimNPO training and compression sweep
    2026-05-13-8b-rmu.sh        # RMU training and compression sweep
    2026-05-analysis.sh         # weight delta + mechanistic analysis
    archive/                    # earlier exploratory scripts (1B, SVD experiments)
open-unlearning/                # submodule — eval harness and base checkpoints
```

---

## References

- Zhang et al., "Catastrophic Failure of LLM Unlearning via Quantization" (2024) — https://arxiv.org/abs/2410.16454
- Maini et al., "TOFU: A Task of Fictitious Unlearning for LLMs" (2024) — https://locuslab.github.io/tofu/
- open-unlearning — https://github.com/locuslab/open-unlearning
