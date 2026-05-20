# Compression Reverses Machine Unlearning

---

**TL;DR:** Machine unlearning methods are routinely tested at full precision, but deployed models are almost always compressed. I show that 4-bit quantization and magnitude pruning both reverse unlearning across three different methods (GradDiff, SimNPO, RMU) on the TOFU benchmark — recovering suppressed knowledge while leaving model utility intact, with no surface signal that anything has changed. Weight-level analysis suggests the reason: unlearning perturbations are small and concentrated in low-magnitude weights, exactly the weights that compression removes first.

---

## 1. Introduction

Machine unlearning refers to techniques for selectively removing specific knowledge or capabilities from a trained model without retraining from scratch. Full retraining is expensive and often impractical for large models, so unlearning has emerged as a practical tool for several use cases: compliance with data deletion requests under regulations like GDPR, removal of copyrighted content a model was trained on, and — most relevant for AI safety — the targeted removal of dangerous capabilities such as knowledge of weapons synthesis or other hazardous content.

As AI systems become more capable, the ability to selectively remove harmful knowledge while preserving general capability is increasingly valuable. If a model can be trained to assist with dangerous tasks, unlearning offers a potential remediation pathway: identify the unwanted capability, apply an unlearning method, and deploy the modified model. Unlearning has attracted significant research interest from major AI labs as a potential safety control, particularly for removing dangerous capabilities from a model after the fact. The GDPR's right to erasure creates parallel regulatory pressure for techniques that can remove specific training data on demand. Whether current unlearning methods are reliable enough to serve as actual safety controls — not just laboratory demonstrations — is an open question.

As part of a BlueDot Impact Technical AI Safety project sprint, I decided to experiment with unlearning techniques.  The problem I investigated here is whether unlearning is affected by practical compression techniques. Modern LLMs are rarely deployed in their original form. They are routinely compressed — quantized to 4-bit or 8-bit precision to run on consumer hardware, or pruned to reduce memory and latency. If a model is unlearned and then compressed for deployment, does the unlearning hold?  Zhang et al. (2024) showed one case where it does not: applying 4-bit quantization to models that have undergone machine unlearning recovers a substantial fraction of the supposedly forgotten knowledge. A developer who tests an unlearned model at full precision, then distributes a quantized version, may be shipping a model that has silently recovered the knowledge they intended to remove.

Zhang et al.'s experiments used the MUSE benchmark with Llama-2-7B across six unlearning methods. I replicate their finding on the Tasks of Ficticious Unlearning (TOFU) benchmark, a different benchmark with a different model family (Llama-3.1-8B-Instruct), and extend it to magnitude pruning, a structurally different compression method not tested in the original paper. I also tested some different unlearning techniques, GradDiff, SimNPO, and RMU from the open-unlearning eval framework.  I also investigated the weight-level mechanism behind the vulnerability, and find that the unlearning methods differ in how they store the forgetting signal in the model's weights — which determines how much of that signal survives compression.

---

## 2. Background

### 2.1 The TOFU benchmark

TOFU (Maini et al. 2024) is a question-answering benchmark built around 450 entirely fabricated author profiles, each with 20 question-answer pairs (9,000 pairs total). Because the authors and their biographical details are invented, the forget set contains no information the model could have encountered during pretraining. The *forget10* split designates 10% of authors (45 authors, 900 QA pairs) as the forget set. The remaining 90% form the retain set. Experiments use two reference models drawn from the open-unlearning HuggingFace repository: a *full* model trained on all 450 authors, and a *retain90* oracle trained from scratch on only the retain-set authors. The oracle represents the target state for a perfectly unlearned model — behaviour on the forget set indistinguishable from a model that was never exposed to it.

Zhang et al.'s original experiments used MUSE (Shi et al. 2024), which has two real-data splits: NEWS (BBC articles from after August 2023) and BOOKS (the Harry Potter series), with ROUGE-based metrics and a membership inference attack metric. TOFU was chosen here because real data introduces a confound: a model evaluated on Harry Potter or BBC news may retain knowledge from pretraining exposure rather than from the specific fine-tuning being unlearned, making it difficult to attribute any post-compression recovery cleanly to unlearning failure. TOFU's synthetic forget set removes this ambiguity. The tradeoff is that TOFU's forget-set knowledge is artificially implanted during fine-tuning rather than naturally acquired, which may affect how it is stored and how well findings generalise to real-data settings. Direct numerical comparison with Zhang et al.'s results is not possible — different metrics, different model — so I assess directional consistency rather than magnitude agreement.

### 2.2 Evaluation metrics

All evaluations use the open-unlearning evaluation harness. Two metrics drive the analysis:

**`forget_Q_A_Prob`** is the average joint token probability the model assigns to the correct answer, given the question, across all forget-set QA pairs. This is a direct measure of how confidently the model can produce the correct forget-set answers. Low values indicate suppressed knowledge; high values indicate retained or recovered knowledge. This is the primary metric for measuring knowledge recovery after compression.

**`model_utility`** is the harmonic mean of nine sub-metrics across three evaluation domains: the retain set, a real-authors QA set, and a world-facts QA set. Each domain contributes probability, ROUGE, and truth-ratio sub-metrics. The harmonic mean penalises degradation in any single dimension. This is the primary metric for confirming that compression has not simply degraded the model generally.

I also report `forget_quality` (a Kolmogorov-Smirnov test comparing the unlearned model's forget-set behaviour to the retain90 oracle) for context, but do not use it as a primary analysis metric.

### 2.3 Unlearning methods

**GradDiff (α=1)** performs gradient ascent on the forget set — increasing the cross-entropy loss on forget-set QA pairs — simultaneously with gradient descent on the retain set, weighted by α=1. This is equivalent to the GA_GDR method in Zhang et al.

**SimNPO** adapts Negative Preference Optimisation without a reference model, treating the forget set as negative preference data and training the model to assign low likelihood to forget-set completions while a retain-set term preserves utility. Hyperparameters: lr=2e-5, beta=4.5, delta=1.0, gamma=0.25, 10 epochs.

**RMU (Representation Misdirection for Unlearning)** steers the internal representations of the network toward a random control vector rather than suppressing output probabilities directly. A steering loss is applied at a target layer (layer 7 of 32) to misdirect activations on forget-set inputs, while a retain loss maintains normal representations on retain-set inputs. Hyperparameters: lr=1e-5, steering coefficient 2, 10 epochs.

All three checkpoints were trained on a single H100 using the open-unlearning training harness.

### 2.4 Compression methods

**Quantization** is applied at load time using bitsandbytes via HuggingFace's `BitsAndBytesConfig`. At 4-bit, I use NF4 (Normal Float 4), a non-uniform 16-level quantization format optimised for normally distributed weights, with computation in float16. At 8-bit, I use LLM.int8(), which applies row-wise absmax scaling with a separate high-precision pathway for outlier weights. Both are post-training quantization methods — no retraining or calibration data is involved. These are the same two precision levels tested by Zhang et al., though their implementation used RTN (round-to-nearest) rather than NF4 for 4-bit.

**Magnitude pruning** is applied after loading the model in float16. I collect all weight values from every linear layer, compute a global threshold at the target sparsity percentile, and zero all weights below that threshold in-place. Pruning is global rather than per-layer: a weight in one layer competes against weights in all layers, so sparsity is not uniform across the network. I test 10%, 20%, and 30% sparsity.

---

## 3. Results

### 3.1 Reference points

| Checkpoint | forget_Q_A_Prob | model_utility |
|---|---|---|
| Full model | 0.992 | 0.627 |
| Retain90 oracle | 0.104 | 0.648 |
| GradDiff | 0.028 | 0.465 |
| SimNPO | 0.088 | 0.653 |
| RMU | 0.067 | 0.637 |

All three unlearning methods achieve forget_Q_A_Prob below the oracle threshold of 0.104. GradDiff achieves the strongest suppression (0.028) but at the cost of substantial utility loss (0.465 vs oracle 0.648). SimNPO and RMU both achieve suppression below the oracle while maintaining model utility at or near oracle level (0.653 and 0.637 respectively).

### 3.2 Quantization

Each unlearned checkpoint was evaluated under two levels of post-training quantization applied at load time via bitsandbytes: 8-bit (LLM.int8()) and 4-bit (NF4). No retraining or calibration data is involved — the weights are simply loaded at reduced precision. The same evaluation harness used for the baseline was then run against each compressed version.

![Quantization results](figures/quantization.png)

**Knowledge recovery** (`forget_Q_A_Prob`, lower = better unlearning):

| | Unlearned | 8-bit | 4-bit |
|---|---|---|---|
| GradDiff | 0.028 | 0.033 | **0.672** |
| SimNPO | 0.088 | 0.096 | **0.210** |
| RMU | 0.067 | 0.083 | **0.649** |

*Oracle: 0.104. Full model ceiling: 0.992.*

**Model utility** (`model_utility`, higher = better):

| | Unlearned | 8-bit | 4-bit |
|---|---|---|---|
| GradDiff | 0.465 | 0.467 | 0.589 |
| SimNPO | 0.653 | 0.657 | 0.636 |
| RMU | 0.637 | 0.638 | 0.614 |

*Oracle: 0.648.*

**8-bit** has negligible effect across all three methods — forget_Q_A_Prob increases by at most 0.016 and model utility is essentially unchanged.

**4-bit** produces large knowledge recovery in all cases while model utility is preserved or improved. GradDiff and RMU are similarly vulnerable (0.672 and 0.649), recovering to roughly two-thirds of full-model probability. SimNPO is substantially more robust (0.210), consistent with the larger weight perturbations SimNPO produces being less completely erased by quantization. All three methods remain above the oracle threshold of 0.104 after 4-bit quantization.

### 3.3 Magnitude pruning

Each unlearned checkpoint was evaluated at three sparsity levels — 10%, 20%, and 30% — using global unstructured magnitude pruning. All linear layer weights are collected, a global threshold is computed at the target sparsity percentile, and all weights below that threshold are zeroed in-place. Pruning is applied after loading in float16 with no retraining. Because the threshold is global rather than per-layer, sparsity is not uniform across the network — layers with a higher proportion of small weights are pruned more heavily.

![Pruning results](figures/pruning.png)

**Knowledge recovery** (`forget_Q_A_Prob`, lower = better unlearning):

| | Unlearned | 10% | 20% | 30% |
|---|---|---|---|---|
| GradDiff | 0.028 | 0.187 | **0.979** | **0.938** |
| SimNPO | 0.088 | 0.112 | 0.333 | **0.935** |
| RMU | 0.067 | 0.124 | **0.963** | **0.940** |

*Oracle: 0.104.*

**Model utility** (`model_utility`, higher = better):

| | Unlearned | 10% | 20% | 30% |
|---|---|---|---|---|
| GradDiff | 0.465 | 0.543 | 0.637 | 0.630 |
| SimNPO | 0.653 | 0.660 | 0.649 | 0.630 |
| RMU | 0.637 | 0.630 | 0.622 | 0.639 |

*Oracle: 0.648.*

Recovery increases with sparsity for all three methods, and model utility is preserved throughout — at no pruning level does any method fall substantially below oracle utility. At 30% sparsity, GradDiff (0.938), SimNPO (0.935), and RMU (0.940) all recover nearly all suppressed knowledge. The three methods converge to the same outcome at 30%, despite showing different trajectories at lower sparsity: GradDiff and RMU reach near-full recovery already at 20% (0.979 and 0.963), while SimNPO requires 30% to reach comparable levels.

### 3.4 Weight delta analysis

To understand why the methods differ in their vulnerability to compression, I computed per-weight deltas (W_unlearned − W_full) across all linear layers for each method and measured two properties: the magnitude of weight changes, and whether those changes land disproportionately on the low-magnitude weights that magnitude pruning removes first.

**Change magnitude.** SimNPO makes substantially larger weight changes than the other two methods — per-element delta norms are 2–4× larger than GradDiff's across all module types, and 3–4× larger than RMU's.

| Module type | GradDiff | SimNPO | RMU |
|---|---|---|---|
| MLP down | 2.5e-5 | 5.4e-5 | 1.7e-5 |
| MLP gate/up | 2.3e-5 | 5.2e-5 | 1.4e-5 |
| Attention QKV | 2.4e-5 | 4.9e-5 | 1.3e-5 |
| Attention out | 2.8e-5 | 5.7e-5 | 1.7e-5 |

*Normalized magnitude of weight changes (W_unlearned − W_full), averaged across elements.*

This is consistent with SimNPO's relative robustness to 4-bit quantization: larger changes are harder to round away. The difference in magnitude likely reflects the loss functions themselves — SimNPO's preference-based objective drives weights further from the pre-unlearning state than gradient ascent (GradDiff) or targeted activation steering (RMU).

**Pruning overlap.** I also measured whether the weights that changed most under unlearning are concentrated in the low-magnitude weights that magnitude pruning zeros first. An enrichment ratio of 1.0 means the highest-delta weights are uniformly distributed across the weight magnitude spectrum; values above 1.0 mean they cluster in the pruned region.

| Module type | GradDiff | SimNPO | RMU |
|---|---|---|---|
| MLP down | 7.6× | 5.2× | 2.3× |
| MLP gate/up | 5.8× | 5.7× | 2.5× |
| Attention QKV | 2.9× | 6.1× | 1.9× |
| Attention out | 6.9× | 4.8× | 2.1× |

*Enrichment of high-delta weights in the bottom-10% lowest-magnitude weights (what 10% pruning removes).*

GradDiff's unlearning signal is heavily concentrated in low-magnitude MLP weights — 76% of its highest-delta MLP down-projection weights fall in the bottom 10% lowest-magnitude weights. This directly explains why even 10–20% pruning substantially reverses GradDiff's unlearning. RMU's changes are more spread across the weight magnitude spectrum (only 23% concentrated in the pruned zone), yet full recovery still occurs by 30% pruning. SimNPO shows intermediate concentration, but its larger absolute changes mean more signal remains in the non-pruned weights even after the concentrated portion is removed.

---

## 4. Discussion

**Replication.** 4-bit quantization reverses substantial forget-set knowledge on TOFU across all three methods tested. 8-bit quantization has negligible effect. This replicates the directional finding of Zhang et al. on a different benchmark, a different model family, and with a synthetic forget set that removes the pretraining confound. The finding holds for methods that differ substantially in mechanism: gradient ascent on outputs (GradDiff), preference-based suppression (SimNPO), and activation-space misdirection (RMU).

The oracle threshold of 0.104 provides a concrete interpretation. A model below this threshold behaves like one that was never exposed to the forget set; a model above it has retained meaningful probability mass on the correct answers. After 4-bit quantization, all three methods cross this threshold substantially — GradDiff (0.672) and RMU (0.649) recover to roughly two-thirds of the full model's ceiling of 0.992. The cliff between 8-bit (negligible effect) and 4-bit (large recovery) is consistent with the weight delta analysis: the unlearning perturbations are small enough to survive 8-bit rounding but are erased at 4-bit precision.

**Pruning as an additional vector.** Magnitude pruning also reverses unlearning at 8B scale, with recovery increasing monotonically with sparsity and model utility preserved throughout. At 30% global sparsity, all three methods recover nearly all suppressed knowledge while remaining fully functional. This is a structurally different compression mechanism from quantization — it removes weight components entirely rather than rounding them — suggesting the vulnerability is not specific to the quantization operation.

A counterintuitive side effect is that pruning restores utility alongside knowledge. GradDiff's unlearning substantially degrades model utility (0.465 vs oracle 0.648), but 20% pruning simultaneously recovers nearly all the suppressed knowledge (0.979) and brings utility back to 0.637 — nearly oracle level. A developer who pruned an unlearned GradDiff model for deployment would observe improved performance on standard metrics with no visible sign that the forget-set knowledge had returned. The compression step undoes the unlearning and conceals that it has done so.

**Method differences.** SimNPO is substantially more robust to 4-bit quantization than GradDiff or RMU (0.210 vs 0.672/0.649). At the same time, SimNPO converges to the same near-full recovery at 30% pruning as the other methods. The weight delta analysis offers a direct explanation: SimNPO makes 2–4× larger weight changes than GradDiff or RMU, which makes its signal harder to erase by quantization rounding. Its pruning resistance at low sparsity follows from the same property — even where its changes are concentrated in low-magnitude weights, the larger absolute magnitude means more signal survives after those weights are removed. At 30% pruning, enough of the signal is eventually removed regardless.

RMU presents a particularly instructive case. Its intervention targets layer 7's internal representations directly — steering activations toward a random control vector rather than modifying output probabilities. Despite this conceptual difference from gradient ascent, the weight changes RMU leaves behind are no more pruning-resistant than GradDiff's. RMU's enrichment ratio is lower (2.3× vs 7.6× for MLP down projections), meaning its changes are more spread across the weight magnitude spectrum — yet it still reaches near-full recovery by 20% sparsity (0.963). More distributed weight changes do not confer pruning resistance unless they are also large in magnitude, which RMU's changes are not (1.7e-5 vs SimNPO's 5.4e-5 per element for MLP down projections).

All three unlearning methods, despite their different objectives, store forget-set suppression in a form that is disrupted by both weight quantization and weight removal. The pattern is consistent with unlearning as suppression rather than erasure: the original knowledge remains encoded in the model's weights, and unlearning adds a corrective perturbation that deflects the forward pass away from producing the correct answer. Compression removes or rounds the perturbation, restoring the original trajectory. A method that genuinely redistributed the underlying knowledge representation — rather than adding a small deflection on top of it — might be expected to produce larger, more distributed weight changes that survive compression; none of the methods tested here do.

---

## 5. Future directions

**Replication on MUSE.** The most direct extension would be to run the same compression sweep on MUSE (NEWS and BOOKS splits) with Llama-3.1-8B. MUSE uses ROUGE-based metrics (VerbMem, KnowMem) and includes a membership inference metric (PrivLeak), so the results would not be directly comparable numerically, but directional consistency would strengthen the case that the vulnerability is benchmark-agnostic. The main complication is the pretraining confound — MUSE's real-data splits mean some knowledge recovery may reflect pretraining exposure rather than unlearning failure, which TOFU avoids by design.

**Additional unlearning methods.** The open-unlearning framework supports several methods not tested here. NPO (Negative Preference Optimisation, the reference-model variant of SimNPO) and SCRUB are the most natural additions. Task-vector-based approaches, which construct an "unlearning direction" in weight space and subtract it, would be particularly interesting to test against compression: their unlearning perturbation is explicitly directional, which may make it more or less susceptible to quantization rounding than the perturbations produced by gradient-based methods.

**Larger models.** All experiments here use Llama-3.1-8B. It would be interesting to replicate the sweep at larger scales — 70B or beyond — to check whether the vulnerability persists. Larger models may encode knowledge more redundantly, which could make unlearning perturbations harder to reverse, or alternatively more distributed across weights, which could make them more susceptible to compression-induced disruption.

**Additional compression methods.** Two compression families were tested here — quantization (PTQ via bitsandbytes) and unstructured magnitude pruning. Several others are worth examining: GPTQ and AWQ are calibration-data-based quantization methods that produce different rounding decisions than the PTQ approach used here; structured pruning removes entire attention heads or MLP channels rather than individual weights and would interact differently with the unlearning perturbation; and low-rank approximation (SVD truncation) compresses by discarding small singular value components, which could disproportionately affect the low-magnitude perturbations that unlearning introduces.

---

## 6. Conclusion

Compression reverses unlearning across all methods and compression types tested. The suppressed knowledge recovers, model utility is preserved, and neither quantization nor pruning produces degradation that would alert a developer to the problem. A model validated as unlearned at full precision and then compressed for deployment may be neither.

---

## Appendix

### Code and models

Replication instructions, trained model checkpoints on HuggingFace, and repository structure are documented in the [README](../README.md).

### References

- **Zhang et al. (2024)** — "Catastrophic Failure of LLM Unlearning via Quantization": [arxiv.org/abs/2410.16454](https://arxiv.org/abs/2410.16454)
- **Maini et al. (2024)** — "TOFU: A Task of Fictitious Unlearning for LLMs": [arxiv.org/abs/2401.06121](https://arxiv.org/abs/2401.06121)
- **Shi et al. (2024)** — "MUSE: Machine Unlearning Six-Way Evaluation for Language Models": [arxiv.org/abs/2407.06460](https://arxiv.org/abs/2407.06460)
- **open-unlearning** — eval framework and base checkpoints: [github.com/locuslab/open-unlearning](https://github.com/locuslab/open-unlearning)
