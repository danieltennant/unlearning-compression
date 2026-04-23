# Lab Notebook — Unlearning Compression

Chronological record of experiments, findings, and decisions.

---

## 2026-04-23 — Initial pipeline validation and first RunPod runs

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

### Next steps
- Run `GradDiff_lr1e-05_alpha1_epoch10` (default params, matches repro.md near-zero forget_truth_ratio)
- Run `SimNPO_lr2e-05_b4.5_a1_d1_g0.25_ep10` (best utility preservation at 0.54, near-zero forget_truth_ratio)
- These should give the low-baseline checkpoints needed to cleanly demonstrate the Guo et al. recovery effect

---

## Notes on metrics

| Metric | What it measures | Direction for "good unlearning" |
|--------|-----------------|--------------------------------|
| `forget_truth_ratio` | Does model prefer correct forget-set answers over paraphrased wrong ones? | Low (near 0) |
| `forget_quality` | KS-test p-value: is model's forget-set distribution indistinguishable from oracle? | High (near 1) |
| `forget_Q_A_Prob` | Absolute probability model assigns to correct forget-set answers | Low |
| `forget_Q_A_ROUGE` | ROUGE score on forget-set Q&A | Low |
| `model_utility` | Retain-set performance (model still useful?) | High |

For demonstrating the Guo et al. effect, `forget_truth_ratio` is the primary metric — it's a relative measure robust to general quality degradation, and the most direct analog to their KnowMem (M2) metric.
