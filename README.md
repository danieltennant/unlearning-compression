# unlearning-compression

Does compression reverse machine unlearning?

[Guo et al. (2024)](https://arxiv.org/abs/2410.16454) showed that quantizing an unlearned LLM recovers 83% of supposedly forgotten knowledge on average. This project intends to first replicate that finding, and then extend to magnitude pruning and SVD truncation, using the 450+ pre-trained unlearned checkpoints from [open-unlearning](https://github.com/locuslab/open-unlearning) and the [TOFU](https://locuslab.github.io/tofu/) benchmark.

If the failure generalizes across compression methods, it strengthens the case that current unlearning is suppression rather than erasure. If it doesn't, that would suggest something specific about how quantization interacts with unlearning weight perturbations — which points toward fixes.

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

None yet - will evaluate on Llama 3.2 1B and Llama 3.1 8B across four unlearning methods (GradAscent, NPO, GradDiff, RMU) at 5% and 10% forget fractions.

## Metrics

- **Forget accuracy** — does the model recover knowledge of the forget set after compression?
- **Retain accuracy** — does compression preserve performance on the retain set?

## References

- Guo et al., "Catastrophic Failure of LLM Unlearning via Quantization" (2024) — https://arxiv.org/abs/2410.16454
- Maini et al., "TOFU: A Task of Fictitious Unlearning for LLMs" (2024) — https://locuslab.github.io/tofu/
- open-unlearning — https://github.com/locuslab/open-unlearning
