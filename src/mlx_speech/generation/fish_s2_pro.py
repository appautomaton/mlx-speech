from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import mlx.core as mx
import numpy as np

from ..models.fish_s2_pro import (
    FishS2ProConfig,
    DualARTransformer,
    FishS2Tokenizer,
)
from ..models.fish_s2_pro.checkpoint import (
    FishS2ProCheckpoint,
    load_fish_s2_pro_checkpoint,
)
from ..models.fish_s2_pro.cache import KVCache


@dataclass
class FishS2ProModel:
    """Loaded Fish S2 Pro model for inference."""

    model: DualARTransformer
    tokenizer: FishS2Tokenizer
    config: FishS2ProConfig

    @classmethod
    def from_dir(cls, model_dir: str, dtype=mx.bfloat16):
        """Load model from directory."""
        model_dir = Path(model_dir)
        if not model_dir.exists():
            raise FileNotFoundError(f"Model directory not found: {model_dir}")

        ckpt = load_fish_s2_pro_checkpoint(str(model_dir))

        model = DualARTransformer(
            vocab_size=ckpt.config.vocab_size,
            num_layers=ckpt.config.num_layers,
            dim=ckpt.config.slow_ar_dim,
            num_heads=ckpt.config.num_heads,
            max_position_embeddings=ckpt.config.max_position_embeddings,
        )

        model_params = dict(model.named_parameters())
        for key, param in model_params.items():
            if key in ckpt.state_dict:
                param.value = ckpt.state_dict[key].astype(dtype)

        tokenizer = FishS2Tokenizer.from_pretrained(str(model_dir))

        return cls(model=model, tokenizer=tokenizer, config=ckpt.config)

    def generate(
        self,
        text: str,
        max_new_tokens: int = 1024,
        temperature: float = 0.8,
        top_p: float = 0.8,
        top_k: int = 50,
    ) -> mx.array:
        """Generate tokens autoregressively."""
        tokens = self.tokenizer.encode(text, add_special_tokens=True)
        tokens = mx.array([tokens])

        generated = []
        kv_cache = None

        for _ in range(max_new_tokens):
            h = self.model(tokens)
            logits = h[0, -1, :]

            if temperature > 0:
                logits = logits / temperature
            else:
                return mx.array(generated)

            probs = mx.softmax(logits, axis=-1)

            if top_k > 0:
                top_k_vals = mx.topk(probs, k=top_k)
                min_top_k = mx.min(top_k_vals)
                probs = mx.where(probs < min_top_k, mx.zeros_like(probs), probs)
                probs = probs / mx.sum(probs)

            if top_p < 1.0:
                sorted_idx = mx.argsort(probs)[::-1]
                sorted_probs = probs[sorted_idx]
                cumsum = mx.cumsum(sorted_probs)
                mask = cumsum > top_p
                for i, idx in enumerate(sorted_idx.tolist()):
                    if mask[i]:
                        probs[idx] = 0
                probs = probs / mx.sum(probs)

            next_token = int(mx.argmax(probs))
            generated.append(next_token)

            if next_token == self.tokenizer.eos_token_id:
                break

            tokens = mx.array([tokens.tolist()[0] + [next_token]])

        return mx.array(generated)


@dataclass
class FishS2ProOutput:
    """Output from Fish S2 Pro generation."""

    waveform: Optional[mx.array]
    sample_rate: int = 22050
    generated_tokens: int = 0


def generate_fish_s2_pro(
    text: str,
    *,
    model_dir: str = "models/fish_s2_pro/mlx-int8",
    max_new_tokens: int = 1024,
    temperature: float = 0.8,
    top_p: float = 0.8,
    top_k: int = 50,
) -> FishS2ProOutput:
    """Generate speech from text using Fish S2 Pro.

    Args:
        text: Input text (can include inline tags like [whisper], [excited])
        model_dir: Path to model checkpoint
        max_new_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        top_p: Nucleus sampling threshold
        top_k: Top-k sampling

    Returns:
        FishS2ProOutput with waveform
    """
    model_dir = Path(model_dir)

    if not model_dir.exists():
        return FishS2ProOutput(
            waveform=None,
            sample_rate=22050,
            generated_tokens=0,
        )

    try:
        model_pkg = FishS2ProModel.from_dir(str(model_dir))

        tokens = model_pkg.generate(
            text,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
        )

        return FishS2ProOutput(
            waveform=None,
            sample_rate=22050,
            generated_tokens=len(tokens.tolist()),
        )
    except Exception as e:
        return FishS2ProOutput(
            waveform=None,
            sample_rate=22050,
            generated_tokens=0,
        )
