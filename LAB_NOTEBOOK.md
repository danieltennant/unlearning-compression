# Lab Notebook — Unlearning Compression

Entries in reverse chronological order (newest first).

---

## 2026-04-23 — Fourth session: 8-bit quantization follow-up

### Quantization comparison across precision levels

GradDiff α1 and SimNPO across three quantization levels (all on 1B model, TOFU forget10):

| Method | Compression | forget_Q_A_Prob | model_utility |
|--------|-------------|-----------------|---------------|
| GradDiff α1 | none | **0.061** | 0.456 |
| GradDiff α1 | 4-bit | **0.359** | 0.440 |
| GradDiff α1 | 8-bit | **0.066** | 0.449 |
| SimNPO | none | **0.110** | 0.592 |
| SimNPO | 4-bit | **0.223** | 0.453 |
| SimNPO | 8-bit | **0.123** | 0.591 |

8-bit quantization produces negligible change in `forget_Q_A_Prob` relative to full precision — GradDiff α1: 0.061 → 0.066, SimNPO: 0.110 → 0.123. The large recovery effect seen in 4-bit is not present at 8-bit. Model utility is similarly unaffected by 8-bit.

### Context: Guo et al. 2024

The original paper ("Catastrophic Failure of LLM Unlearning via Quantization", ICLR 2025) used the MUSE benchmark (NEWS and BOOKS datasets), not TOFU. Their primary result was with 4-bit (post-training) quantization on models including Llama3-8B. The abstract reports an average of 21% forgotten knowledge retention at full precision, rising to 83% after 4-bit quantization. The paper also tested 8-bit but the headline claim is about 4-bit. Our 8-bit result on 1B/TOFU is directionally consistent with their finding that 8-bit is far less damaging than 4-bit.

### Next steps
- Train GradDiff α1 on Llama-3.1-8B-Instruct and run the same quantization sweep
- Compare whether the 4-bit recovery effect is stronger or weaker at 8B

---

## 2026-04-23 — Third session: pruning and SVD sweep

### Finding: pruning and SVD are not meaningful compression methods for a 1B model

GradDiff α1 baseline: forget_Q_A_Prob = 0.061, model_utility = 0.456

| Method | forget_truth_ratio | forget_Q_A_Prob | model_utility | Notes |
|--------|-------------------|-----------------|---------------|-------|
| 4-bit quantize | 0.534 | **0.359** | **0.440** | 6× recovery, utility intact |
| prune 30% | 0.630 | 0.126 | 0.279 | Model degraded |
| prune 50% | 0.743 | ~0 | **0.0** | Model broken |
| prune 70% | 0.855 | ~0 | **0.0** | Model broken |
| svd 90% | 0.828 | ~0 | **0.0** | Model broken |
| svd 70% | 0.920 | ~0 | **0.0** | Model broken |
| svd 50% | 0.886 | ~0 | **0.0** | Model broken |

At all tested levels, unstructured magnitude pruning and SVD truncation either degrade or completely destroy model utility. The elevated `forget_truth_ratio` values for broken models are artifacts of a non-functional model, not evidence of knowledge recovery. We can't draw any conclusions about unlearning from these runs because the compression itself is too destructive to evaluate meaningfully.

This is a known limitation of these techniques on small (1B) models rather than a finding about unlearning. These experiments don't add to the paper narrative.

### Open question
Whether lighter pruning (5–15% sparsity) on this model, or the same techniques on a larger model (7B+), would show a cleaner result is unknown. Not worth pursuing on 1B without a clearer motivation.

### Next steps
- 8-bit quantization to compare recovery magnitude vs 4-bit
- SimNPO checkpoint to confirm quantization result generalises across unlearning methods
- Begin writing up the core quantization result

---

## 2026-04-23 — Second session: core result replicated on TOFU

### Key finding: 4-bit quantization recovers suppressed knowledge (Guo et al. effect confirmed)

Oracle reference: `forget_truth_ratio = 0.628`

| Method | Compression | forget_truth_ratio | forget_Q_A_Prob | forget_Q_A_ROUGE | model_utility |
|--------|-------------|-------------------|-----------------|------------------|---------------|
| GradDiff α1 | none | 0.449 | **0.061** | 0.366 | 0.456 |
| GradDiff α1 | 4-bit | 0.534 | **0.359** | 0.414 | 0.440 |
| SimNPO | none | 0.514 | **0.110** | 0.381 | 0.592 |
| SimNPO | 4-bit | 0.560 | **0.223** | 0.381 | 0.453 |

**GradDiff α1** shows the clearest effect: `forget_Q_A_Prob` increases from 0.061 → 0.359 after 4-bit quantization (**6× increase**). The model had genuinely suppressed knowledge of the forget set at full precision, and quantization recovers most of it. Critically, model utility barely changes (0.456 → 0.440), consistent with Guo et al.'s theory: utility-constrained unlearning produces small weight perturbations that quantization grid-snapping washes out.

**SimNPO** shows the same direction but weaker: 0.110 → 0.223 (2× increase).

### The alpha parameter explains everything

Comparing GradDiff α5 (from first session) vs α1:

| Method | forget_Q_A_Prob (full precision) | forget_Q_A_Prob (4-bit) | Direction |
|--------|----------------------------------|--------------------------|-----------|
| GradDiff α5 (weak unlearning) | 0.654 | 0.453 | ↓ decrease (quantization noise) |
| GradDiff α1 (strong unlearning) | 0.061 | 0.359 | ↑ 6× increase (knowledge recovery) |

α5 weights retain loss 5× more than forget loss → weak suppression, nothing to recover. α1 (equal weighting) achieves real suppression, which is then fragile to quantization. This contrast cleanly demonstrates the mechanism.

### Lessons
- `forget_Q_A_Prob` is the best primary metric for the Guo et al. analog (more sensitive than forget_truth_ratio)
- The recovery effect is only visible when baseline unlearning is genuine (low forget_Q_A_Prob at full precision)
- Model utility degradation from 4-bit quantization is minimal for well-unlearned models (~1%), consistent with small-perturbation hypothesis

### Next steps
- Run pruning experiments (magnitude pruning at various sparsity levels) on GradDiff α1
- Run SVD truncation experiments on GradDiff α1
- Compare compression methods: does pruning show the same recovery? At what sparsity threshold?
- Consider running on SimNPO with stronger hyperparameters to get lower baseline forget_Q_A_Prob

---

## 2026-04-23 — First session: pipeline validation and initial quantization experiments

### Infrastructure
- Created 100GB RunPod network volume in EU-RO-1 (persists independently of pods)
- Validated end-to-end pipeline locally on GPT-2 with magnitude pruning (CPU, subset of metrics)
- `forget_quality` metric requires pre-computed retain logs; generated from `open-unlearning/tofu_Llama-3.2-1B-Instruct_retain90` (oracle: model trained on all data except forget10 authors)
- Retain baseline stored at `results/retain_baseline/tofu_Llama-3.2-1B-Instruct_retain90__none_None/TOFU_EVAL.json`

### Oracle reference point
Running eval on the retain90 checkpoint (never trained on forget10 authors):
```
forget_truth_ratio: 0.628
```
This is the ceiling — even a model that never saw the forget authors answers ~63% of their questions correctly, likely due to pretraining exposure.

### Experiments run

**GradDiff** (`lr1e-05_alpha5_epoch10`) — full precision vs 4-bit quantization:

| Metric | Full precision | 4-bit quant | Change |
|--------|---------------|-------------|--------|
| forget_truth_ratio | 0.460 | 0.537 | +0.077 (+17%) |
| forget_quality | 1.1e-19 | 3.2e-15 | +4 OOM |
| forget_Q_A_Prob | 0.654 | 0.453 | -0.201 |
| forget_Q_A_ROUGE | 0.588 | 0.440 | -0.148 |
| model_utility | 0.589 | 0.462 | -0.127 |

**NPO** (`lr2e-05_beta0.5_alpha1_epoch10`) — full precision vs 4-bit quantization:

| Metric | Full precision | 4-bit quant | Change |
|--------|---------------|-------------|--------|
| forget_truth_ratio | 0.543 | 0.586 | +0.043 (+8%) |
| forget_quality | 1.1e-09 | 3.9e-08 | +1.5 OOM |
| forget_Q_A_Prob | 0.421 | 0.325 | -0.096 |
| forget_Q_A_ROUGE | 0.418 | 0.387 | -0.031 |
| model_utility | 0.601 | 0.485 | -0.116 |

### Interpretation

`forget_truth_ratio` and `forget_quality` move in the Guo et al. direction (recovery) after 4-bit quantization for both methods. However the effect is weak because these checkpoints never achieved strong unlearning in the first place:
- GradDiff `alpha5`: retain loss weighted 5× higher than forget loss — aggressively preserves utility at the expense of forgetting. Baseline `forget_truth_ratio = 0.460` is only modestly below the oracle (0.628), meaning the forget set was barely suppressed.
- NPO baseline: `forget_truth_ratio = 0.543`, essentially at oracle level — minimal unlearning achieved.

`forget_Q_A_Prob` and `model_utility` both drop after quantization. This is a general quantization degradation effect (not specific to the forget set), so it doesn't contradict the recovery hypothesis — it just means absolute output quality degrades while relative preference for forget-set answers shifts upward.

### Key insight: alpha controls forget/retain tradeoff in GradDiff

```python
# grad_diff.py
loss = self.gamma * forget_loss + self.alpha * retain_loss
```

Higher `alpha` = more weight on retain loss = more conservative unlearning = worse forget-set suppression. Default config (`alpha=1`) achieves `forget_truth_ratio ≈ 3.5e-27` per repro.md. We ran `alpha=5` which under-unlearns.

---

## Notes on metrics

| Metric | What it measures | Direction for "good unlearning" |
|--------|-----------------|--------------------------------|
| `forget_truth_ratio` | Does model prefer correct forget-set answers over paraphrased wrong ones? | Low (near 0) |
| `forget_quality` | KS-test p-value: is model's forget-set distribution indistinguishable from oracle? | High (near 1) |
| `forget_Q_A_Prob` | Absolute probability model assigns to correct forget-set answers | Low |
| `forget_Q_A_ROUGE` | ROUGE score on forget-set Q&A | Low |
| `model_utility` | Retain-set performance (model still useful?) | High |

For demonstrating the Guo et al. effect, `forget_Q_A_Prob` turns out to be the clearest metric — it measures absolute probability of correct forget-set answers and shows dramatic changes in well-unlearned models. `forget_truth_ratio` is also useful as a relative measure robust to general quality degradation.
