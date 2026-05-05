# unlearning-compression

Does compression reverse machine unlearning?

[Zhang et al. (2024)](https://arxiv.org/abs/2410.16454) showed that quantizing an unlearned LLM recovers 83% of supposedly forgotten knowledge on average, using NEWS and BOOKS datasets. This project translates that finding to the [TOFU](https://locuslab.github.io/tofu/) benchmark — a standardized unlearning testbed with 450+ public checkpoints via [open-unlearning](https://github.com/locuslab/open-unlearning) — and extends it to magnitude pruning and SVD truncation.

The core hypothesis transfers: if utility-constrained unlearning suppresses rather than erases knowledge, that should be visible under any compression method that reduces weight perturbations, not just quantization. TOFU provides a cleaner test than NEWS/BOOKS because the forget set is entirely synthetic, eliminating confounds from the model's pretraining exposure.

If the failure generalizes across compression methods, it substantially weakens the safety case for unlearning as a capability control. If it doesn't, that points toward something specific about how quantization interacts with unlearning weight perturbations — which is equally useful for understanding fixes.

## Setup

```bash
git clone --recurse-submodules https://github.com/danieltennant/unlearning-compression
cd unlearning-compression
uv sync
```

## Structure

```
src/compress/
    quantize.py     # bitsandbytes 4-bit / 8-bit quantization
    prune.py        # unstructured magnitude pruning
    svd.py          # SVD truncation
experiments/        # sweep scripts
results/            # output (gitignored)
notebooks/          # analysis
open-unlearning/    # submodule — eval harness and checkpoints
```

## Compression methods

| Method | Variants |
|---|---|
| Quantization | 4-bit, 8-bit (bitsandbytes) |
| Magnitude pruning | 10%, 20%, 30% sparsity |
| SVD truncation | Retain top 90%, 80%, 70% of singular values |

## Results

All experiments use the TOFU benchmark (`forget10` split, Llama-3.2-1B-Instruct and Llama-3.1-8B-Instruct). Primary metrics: `forget_Q_A_Prob` (probability assigned to correct forget-set answers — low means good unlearning), `model_utility` (retain-set QA performance — high means model still functions).

### Quantization

4-bit quantization consistently recovers suppressed knowledge. 8-bit quantization has negligible effect.

**1B GradDiff α1** (baseline: `forget_Q_A_Prob = 0.061`, `model_utility = 0.456`):

| Compression | forget_Q_A_Prob | model_utility | Recovery |
|---|---|---|---|
| None | 0.061 | 0.456 | — |
| 4-bit | 0.359 | 0.440 | **6×** |
| 8-bit | 0.066 | 0.449 | negligible |

**1B SimNPO** (baseline: `forget_Q_A_Prob = 0.110`, `model_utility = 0.592`):

| Compression | forget_Q_A_Prob | model_utility | Recovery |
|---|---|---|---|
| None | 0.110 | 0.592 | — |
| 4-bit | 0.223 | 0.453 | 2× |

**8B GradDiff α1** (baseline: `forget_Q_A_Prob = 0.028`, `model_utility = 0.465`):

| Compression | forget_Q_A_Prob | model_utility | Recovery |
|---|---|---|---|
| None | 0.028 | 0.465 | — |
| 4-bit | 0.672 | 0.589 | **24×** |
| 8-bit | 0.033 | 0.467 | negligible |

The 4-bit recovery effect is stronger at 8B than 1B (24× vs 6×). The 8B model achieves stronger unlearning at full precision but is proportionally more vulnerable to quantization. This is consistent with the Zhang et al. mechanism: utility-constrained unlearning produces weight perturbations smaller than the quantization step size, so the model snaps back to its pre-unlearning quantized values.

Weight delta analysis confirms this mechanistically: every weight change from both GradDiff and SimNPO falls within the int4 quantization step (max delta ≈ 2.2% of Δ_int4). 8-bit quantization is safe because Δ_int8 is 16× smaller than Δ_int4.

### Magnitude pruning

Pruning recovers unlearned knowledge by zeroing the small-magnitude weights that disproportionately carry the unlearning signal.

**1B GradDiff α1:**

| Compression | forget_Q_A_Prob | model_utility |
|---|---|---|
| None | 0.061 | 0.456 |
| 10% pruning | 0.737 | 0.567 |
| 30% pruning | 0.126 | 0.279 |

**1B SimNPO:**

| Compression | forget_Q_A_Prob | model_utility |
|---|---|---|
| None | 0.110 | 0.592 |
| 10% pruning | 0.307 | 0.550 |
| 30% pruning | 0.125 | 0.269 |

**8B GradDiff α1:**

| Compression | forget_Q_A_Prob | model_utility |
|---|---|---|
| None | 0.028 | 0.465 |
| 10% pruning | 0.187 | 0.543 |
| 30% pruning | 0.938 | 0.631 |

At 10% sparsity the 1B model recovers more than the 8B (0.737 vs 0.187). At 30% the pattern reverses: the 8B fully recovers while the 1B degrades. Weight delta analysis explains why: GradDiff concentrates its unlearning signal on small-magnitude attention weights (5.89× enrichment in attn_out at 10% sparsity), which pruning removes first. SimNPO shows high enrichment across all layer types.

### SVD

Three implementations tried, none yet successful at meaningful compression on 1B models.

**Naive per-layer SVD** (2026-04-23): Model utility collapses at all tested retain ratios (90/80/70%). Known limitation — no activation awareness, approximation error compounds across all transformer blocks.

**Activation-aware SVD / ASVD** (diagonal scaling, 2026-04-28): Scales weights by √(E[x²]) before truncation. Utility still collapses (model_utility = 0.245 at 90% retain ratio vs 0.456 baseline), qualitatively worse than pruning or quantization. Two root causes identified: (1) diagonal scaling ignores activation covariance; (2) at 90% singular value retention, the factored representation is 1.1–1.8× *larger* than the original weight matrix depending on layer shape — there is no actual parameter compression.

**Cholesky-whitened SVD** (SVD-LLM method, 2026-04-29): Accounts for the full activation covariance E[xx^T] via Cholesky whitening before truncation, minimising the true output error E[‖(W−W̃)x‖²]. Tested on the 1B retain90 baseline (uncompressed `model_utility = 0.593`):

| Retain ratio | model_utility | Utility retained |
|---|---|---|
| None (baseline) | 0.593 | 100% |
| 90% | 0.508 | 86% |
| 80% | 0.354 | 60% |
| 70% | 0.213 | 36% |

The implementation is working correctly — utility degrades gracefully rather than collapsing — but at 90% singular value retention there is zero parameter compression for any layer in this model. Meaningful compression (factored parameter count below original) requires retain ratios below 80% for MLP layers and below 50% for attention layers, both of which cause substantial utility loss on 1B. The 1B model appears to have insufficient redundancy for SVD compression to be viable. Testing at 8B scale is pending.

## Metrics

- **Forget accuracy** — does the model recover knowledge of the forget set after compression?
- **Retain accuracy** — does compression preserve performance on the retain set?

## References

- Zhang et al., "Catastrophic Failure of LLM Unlearning via Quantization" (2024) — https://arxiv.org/abs/2410.16454
- Maini et al., "TOFU: A Task of Fictitious Unlearning for LLMs" (2024) — https://locuslab.github.io/tofu/
- open-unlearning — https://github.com/locuslab/open-unlearning
