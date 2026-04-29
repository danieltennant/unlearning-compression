# Lab Notebook — Unlearning Compression

Entries in reverse chronological order (newest first).

---

## 2026-04-28 — Seventh session: ASVD experiments and Cholesky SVD implementation

### Goal

Test activation-aware SVD (ASVD) as a compression method on the 1B GradDiff α1 unlearned checkpoint. Determine whether ASVD can compress the model while preserving the knowledge-recovery effect seen with quantization and pruning.

### ASVD implementation

ASVD (Yuan et al. 2023, arXiv:2312.05821) scales each weight column by √(E[x_j²]) before SVD, then unscales after truncation. This prioritises directions that receive large activations.

Calibration was run on 128 retain90 samples through the GradDiff α1 model, producing per-dimension mean-square activation statistics for all 113 linear layers. Calibration split was `retain90` (fixing a prior bug where the wrong TOFU split was used).

### ASVD results on 1B GradDiff α1

Uncompressed baseline: `forget_Q_A_Prob = 0.061`, `model_utility = 0.456`

| Retain ratio | forget_Q_A_Prob | model_utility | forget_quality | forget_truth_ratio |
|---|---|---|---|---|
| None (baseline) | 0.061 | 0.456 | 7.98e-17 | 0.449 |
| ASVD 90% | 0.058 | **0.245** | 6.4e-06 | 0.526 |
| ASVD 80% | 0.030 | **0.074** | 0.065 | 0.592 |
| ASVD 70% | 0.009 | **0.000** | 0.758 | 0.636 |

### Finding: ASVD destroys utility without recovering knowledge

Even at 90% retain ratio (discarding only the bottom 10% of singular values by count), model utility collapses from 0.456 to 0.245. At 80% and 70% the model is effectively non-functional. This is a qualitatively different failure mode from quantization and pruning:

- **Quantization (4-bit)**: utility preserved (0.440), knowledge recovered (forget_Q_A_Prob 0.061 → 0.359)
- **Pruning (10%)**: utility improved (0.567), knowledge largely recovered (0.737)
- **ASVD (90%)**: utility halved (0.245), forget_Q_A_Prob unchanged (0.058)

ASVD is not recovering unlearned knowledge — it is simply destroying the model's ability to answer any questions.

### Root cause analysis: two problems with the ASVD implementation

**Problem 1 — Diagonal-only activation scaling (missing Cholesky whitening):**
Our ASVD implementation scales weights by √(E[x_j²]), accounting only for per-dimension activation magnitude. It ignores activation covariance (correlations between dimensions). The SVD-LLM paper (Wang et al. 2024) shows that full Cholesky whitening — which decorrelates the full covariance E[xx^T] before SVD — gives approximately 150× lower perplexity error than diagonal scaling alone at the same compression ratio. Our implementation is equivalent to a degenerate version of the method.

**Problem 2 — Retain ratio semantics mismatch:**
Our "retain ratio" is the fraction of singular values retained by count. This is not equivalent to the parameter retention ratio used in papers. For a square layer (n × n), keeping k singular values in factored storage uses 2kn parameters vs n² original. The factored form is smaller only when k < n/2. At our 90% retain ratio (k = 0.9n), the factored form is 1.8× *larger* than the original — we are introducing approximation error with no compression benefit.

| Layer type | Shape | Break-even retain ratio |
|---|---|---|
| q_proj, o_proj | 2048×2048 | <50% |
| k_proj, v_proj | 512×2048 | <20% |
| gate_proj, up_proj | 8192×2048 | <80% |
| down_proj | 2048×8192 | <80% |

Papers reporting "20% compression" are operating at ~35–40% singular value retention — well below our tested range, and in a regime where our naive implementation produces a broken model.

### Decision: move to Cholesky-whitened SVD with baseline-first validation

Rather than fix ASVD incrementally, we implemented the full Cholesky-whitened SVD (SVD-LLM method). Calibration now collects the full covariance matrix E[xx^T] per layer (113 layers × up to 8192×8192 = up to 5.6GB total for the 1B model). Compression applies: W → W@L → truncate → unwhiten, minimising the true output approximation error E[‖(W − W̃)x‖²].

Before running on any unlearned model, validation runs on the retain90 baseline to confirm utility is preserved. Results logged in the 2026-04-29 entry.

---

## 2026-04-25 — Sixth session: weight delta analysis

### Goal

Directly test the Guo et al. (2024) quantization grid hypothesis on TOFU/GradDiff, and extend it to explain why pruning recovers unlearned knowledge. All analysis runs on CPU locally using the 1B models.

### Method

For each weight matrix, compute W_delta = W_unlearned - W_full and measure:

1. **Quantization grid test**: what fraction of elements have |W_delta| < Δ_int4 = max(|W|)/8? If this is close to 1.0, quantization maps both models to identical values — the mechanism Guo et al. propose.
2. **Delta magnitude by layer type**: where is the unlearning signal concentrated?
3. **Pruning enrichment**: among the top-10% highest-|W_delta| weights, what fraction would be zeroed by magnitude pruning at 10% and 30% sparsity? Enrichment = observed fraction / expected fraction (= sparsity level). Values substantially above 1.0 mean unlearning changes land disproportionately on small-magnitude weights — the weights pruning removes first.

Models analyzed:
- Full (pre-unlearning): `open-unlearning/tofu_Llama-3.2-1B-Instruct_full`
- GradDiff α1: `open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_GradDiff_lr1e-05_alpha1_epoch10`
- SimNPO: `open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_SimNPO_lr2e-05_b4.5_a1_d1_g0.25_ep10`

### Finding 1: Quantization mechanism confirmed — universally, for both methods

Every single weight change from both unlearning methods falls within the int4 quantization step:

| Method | Module type | frac \|δ\| < Δ_int4 | max \|δ\| / Δ_int4 |
|--------|-------------|---------------------|---------------------|
| GradDiff α1 | attn_qkv | 1.0000 | 0.018 |
| GradDiff α1 | attn_out | 1.0000 | 0.010 |
| GradDiff α1 | mlp_gate_up | 1.0000 | 0.011 |
| GradDiff α1 | mlp_down | 1.0000 | 0.008 |
| SimNPO | attn_qkv | 1.0000 | 0.022 |
| SimNPO | attn_out | 1.0000 | 0.014 |
| SimNPO | mlp_gate_up | 1.0000 | 0.014 |
| SimNPO | mlp_down | 1.0000 | 0.011 |

The max delta is at most ~2.2% of Δ_int4 in any layer. 4-bit quantization does not merely reduce the unlearning signal — it completely erases it by snapping every perturbed weight back to its original quantized value. The result is method-agnostic: both GradDiff (gradient-based) and SimNPO (preference-based) produce perturbations this small because both use small learning rates to preserve utility.

frac < Δ_int8 is also 1.0000 in all cases. The difference is that Δ_int8 = max(|w|)/128 is 16× smaller than Δ_int4, so any unlearning method would need to produce 16× larger perturbations to escape 8-bit quantization — which is why 8-bit is safe and 4-bit is not.

### Finding 2: Pruning — GradDiff concentrates in attention, SimNPO is universal

Enrichment ratio at 10% pruning sparsity (1.0 = random, 5.0 = 5× overrepresented in the pruned set):

| Module type | GradDiff α1 | SimNPO |
|-------------|-------------|--------|
| attn_out | **5.89×** | 5.83× |
| attn_qkv | 2.75× | 4.63× |
| mlp_gate_up | **1.00×** | **6.88×** |
| mlp_down | **1.00×** | **6.55×** |
| embed_tokens | 1.01× | 1.01× |

**GradDiff**: enrichment is high in attention layers and flat at 1.0× in MLP. In attn_out, 58.9% of the highest-delta weights fall in the bottom-10% by magnitude — 5.89× more than chance. Magnitude pruning zeroes exactly these attention weights, removing the unlearning correction.

**SimNPO**: high enrichment across all layer types including MLP (6.88× in mlp_gate_up). The unlearning signal lands on small-magnitude weights everywhere.

At 30% sparsity the enrichment roughly halves but remains substantial (GradDiff attn_out: 2.61×, SimNPO mlp_gate_up: 3.33×).

This explains the 1B GradDiff pruning result: 10% pruning gave forget_Q_A_Prob 0.061 → 0.737 because it surgically removed attention output weights carrying almost all of the unlearning signal.

### Open question: SimNPO pruning

We have not yet run SimNPO through magnitude pruning experimentally. The weight delta analysis predicts that pruning should recover SimNPO's unlearning across all layer types (high enrichment everywhere), potentially giving a broader and more uniform recovery than GradDiff. Running 10% and 30% pruning on the 1B SimNPO checkpoint would test this prediction.

### Delta magnitude by layer type

SimNPO makes ~1.8× larger weight changes overall, but both methods are far below Δ_int4.

| Method | Highest-delta type | Lowest-delta type |
|--------|-------------------|--------------------|
| GradDiff α1 | attn_out (3.0e-5/elem) | embed_tokens (7e-6/elem) |
| SimNPO | attn_out (5.3e-5/elem) | embed_tokens (1.9e-5/elem) |

Embed_tokens and lm_head are barely changed by either method, consistent with these layers being the hardest to fine-tune without destroying utility.

### Next steps
- Run 1B SimNPO through 10% and 30% magnitude pruning to validate the enrichment prediction
- Consider GradDiff α5 (weak unlearning) as a control — enrichment should be near 1.0 everywhere
- Run SVD alignment analysis (`--compute_svd`) on a machine with sufficient RAM

---

## 2026-04-24 — Fifth session: 8B checkpoint training and eval

### Checkpoint trained: GradDiff α1 on Llama-3.1-8B-Instruct

| Parameter | Value |
|-----------|-------|
| Base model | `open-unlearning/tofu_Llama-3.1-8B-Instruct_full` |
| Method | GradDiff, α=1.0 |
| Forget split | forget10 |
| Retain split | retain90 |
| Learning rate | 1e-5 |
| Epochs | 10 |
| Effective batch size | 32 (2 per device × 16 grad accum) |
| Hardware | 1× H100 SXM |
| Training time | ~35 minutes |
| Checkpoint | `dtennant/tofu-llama-8b-graddiff-alpha1` (HuggingFace) |

Notable training quirks resolved before a clean run was possible:
- `HF_HOME` had to be set to `/workspace/.cache/huggingface` to avoid filling the 20GB container disk with model weights
- `attn_implementation=eager` override needed to avoid `flash_attn` dependency
- `mode=unlearn` Hydra override required so `get_data()` returns a `ForgetRetainDataset` with the "train" key
- `++trainer.args.remove_unused_columns=false` required to prevent transformers 4.47+ from stripping the "forget" and "retain" batch keys before they reach the trainer

### 8B retain baseline (oracle reference)

Model: `open-unlearning/tofu_Llama-3.1-8B-Instruct_retain90` — trained on all TOFU data except the forget10 authors. Establishes the ceiling for unlearning quality.

| Metric | Value |
|--------|-------|
| `forget_Q_A_Prob` | 0.1044 |
| `forget_Q_A_ROUGE` | 0.395 |
| `model_utility` | 0.6484 |
| `extraction_strength` | 0.0654 |

The 8B oracle `forget_Q_A_Prob` (0.1044) is higher than the 1B unlearned baseline (0.061) — meaning the retain90 model, which was never trained on the forget authors, still answers ~10% of their questions correctly. This is likely because the 8B model has richer general capabilities and can generate plausible-sounding answers to TOFU-style QA even for authors it never saw.

### Full model ceiling

Evaluated `open-unlearning/tofu_Llama-3.1-8B-Instruct_full` (trained on all TOFU data, no unlearning) to anchor the top of the scale:

| Metric | Value |
|--------|-------|
| `forget_Q_A_Prob` | **0.992** |
| `model_utility` | 0.627 |
| `forget_truth_ratio` | 0.710 |

### Compression sweep results

All experiments use the trained GradDiff α1 checkpoint (`dtennant/tofu-llama-8b-graddiff-alpha1`).

| Compression | forget_Q_A_Prob | model_utility | forget_truth_ratio | Notes |
|-------------|-----------------|---------------|--------------------|-------|
| Full model ceiling | 0.992 | 0.627 | 0.710 | Before unlearning |
| Retain90 oracle | 0.104 | 0.648 | 0.641 | Never trained on forget set |
| None (unlearned baseline) | **0.028** | 0.465 | 0.352 | Strong unlearning |
| 8-bit quantization | 0.033 | 0.467 | 0.350 | Negligible change |
| 4-bit quantization | **0.672** | 0.589 | 0.463 | 24× recovery |
| 10% magnitude pruning | 0.187 | 0.543 | 0.397 | Partial recovery |
| 30% magnitude pruning | **0.938** | 0.630 | 0.506 | Near-complete recovery |
| SVD 90% retain ratio | 0.0003 | 0.0 | — | Model destroyed |

### Key findings

**Quantization**: The 4-bit result replicates and exceeds the Guo et al. effect at 8B scale. `forget_Q_A_Prob` rises from 0.028 to 0.672 — a 24× increase, compared to the 6× seen at 1B. The 8B model achieved stronger unlearning at full precision but is proportionally more vulnerable to 4-bit quantization. 8-bit quantization again shows negligible effect, consistent with the 1B result.

**Pruning**: Both 10% and 30% magnitude pruning recover meaningful knowledge without destroying the model — unlike at 1B scale where 30% pruning caused significant utility degradation (model_utility 0.456 → 0.279). At 8B, 30% pruning leaves model_utility at 0.630, essentially matching the oracle, while `forget_Q_A_Prob` reaches 0.938.

**1B vs 8B pruning comparison**:

| Model | Compression | forget_Q_A_Prob | model_utility |
|-------|-------------|-----------------|---------------|
| 1B | None | 0.061 | 0.456 |
| 1B | 10% prune | 0.737 | 0.567 |
| 1B | 30% prune | 0.126 | 0.279 |
| 8B | None | 0.028 | 0.465 |
| 8B | 10% prune | 0.187 | 0.543 |
| 8B | 30% prune | 0.938 | 0.630 |

At 10% pruning the 1B model recovers more than the 8B (0.737 vs 0.187). At 30% the pattern reverses: the 8B model fully recovers while the 1B model degrades. This suggests the 1B model's unlearning is stored in fewer, more pruning-sensitive weights, while the 8B model's unlearning is distributed more broadly but less robustly to heavier pruning.

**SVD**: Uniform SVD truncation at 90% retain ratio destroys the model regardless of whether embedding/output layers are excluded. This is a known limitation of naive per-layer uniform SVD — it compounds approximation error across all transformer blocks simultaneously. Proper calibration-aware SVD (e.g. SVD-LLM) would be required for a meaningful result. SVD excluded from further analysis.

### Next steps
- Update LAB_NOTEBOOK with full analysis and write up for paper
- Consider adding 1B full-model ceiling eval for a complete comparison table

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
