"""Quantization compression wrapper using bitsandbytes.

Note: bitsandbytes requires Linux + CUDA. This module will raise ImportError
on macOS/Windows at call time — quantization can only be run on RunPod/Linux.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from transformers import BitsAndBytesConfig
    _BNB_AVAILABLE = True
except ImportError:
    _BNB_AVAILABLE = False


def load_quantized(
    model_id: str,
    bits: int = 4,
    device_map: str = "auto",
) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    """Load a HuggingFace model with bitsandbytes quantization applied.

    Args:
        model_id: HuggingFace model ID or local path.
        bits: Quantization precision — 4 or 8.
        device_map: Device placement strategy passed to from_pretrained.

    Returns:
        Tuple of (quantized model, tokenizer).
    """
    if not _BNB_AVAILABLE:
        raise ImportError("bitsandbytes is required for quantization but is not installed. Run on Linux with CUDA.")

    if bits not in (4, 8):
        raise ValueError(f"bits must be 4 or 8, got {bits}")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=(bits == 4),
        load_in_8bit=(bits == 8),
        bnb_4bit_compute_dtype=torch.float16,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map=device_map,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    return model, tokenizer
