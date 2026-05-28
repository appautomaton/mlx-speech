"""Pure-MLX tokenizer wrapper for DramaBox's Gemma usage.

The DramaBox reference uses ``LTXVGemmaTokenizer``, which is a thin wrapper
around HuggingFace's `Gemma3TokenizerFast`. The TTS prompt encoder calls it
with the user prompt as plain stripped text — no chat template, no turn
delimiters. The output is left-padded to ``max_length`` so the most recent
tokens sit at the right edge of the window.

Reference: `.references/DramaBox/ltx2/ltx_core/text_encoders/gemma/tokenizer.py:18-52`
"""

from __future__ import annotations

from pathlib import Path

import mlx.core as mx
from tokenizers import Tokenizer


class LTXVGemmaTokenizer:
    """Plain-text Gemma tokenizer with left padding to a fixed length.

    Operates directly on the `tokenizer.json` shipped alongside the
    converted Gemma 4-bit checkpoint. No chat templates, no special-tokens
    handling beyond pad + eos — DramaBox's TTS prompt path is "give Gemma a
    flat sentence and harvest hidden states".
    """

    def __init__(
        self,
        tokenizer_path: str | Path,
        *,
        pad_token_id: int,
        eos_token_id: int,
    ):
        self._tokenizer = Tokenizer.from_file(str(tokenizer_path))
        self.pad_token_id = pad_token_id
        self.eos_token_id = eos_token_id

    @classmethod
    def from_dir(cls, model_dir: str | Path) -> "LTXVGemmaTokenizer":
        """Load from a converted MLX Gemma directory.

        Reads ``tokenizer.json`` and pulls the pad/eos token ids from
        ``special_tokens_map.json`` if available, falling back to the IDs
        that the upstream Gemma 3 IT tokenizer encodes for those literals.
        """
        model_dir = Path(model_dir)
        tokenizer_path = model_dir / "tokenizer.json"

        # Gemma 3 IT uses <pad> id=0 and <end_of_turn> id=106 (one of the eos
        # ids in our config). We only need pad here; eos is informational.
        # Read special_tokens_map if it exists, else use the well-known ids.
        pad_id = 0
        eos_id = 1
        special_map = model_dir / "special_tokens_map.json"
        if special_map.is_file():
            import json
            with special_map.open("r", encoding="utf-8") as fh:
                payload = json.load(fh)
            # Some tokenizers list pad/eos as nested dicts ({"content": "...",
            # "id": ...}), some as raw strings; resolve by encoding the literal.
            tmp = Tokenizer.from_file(str(tokenizer_path))

            def _resolve(tok_entry):
                if tok_entry is None:
                    return None
                if isinstance(tok_entry, dict):
                    content = tok_entry.get("content")
                elif isinstance(tok_entry, str):
                    content = tok_entry
                else:
                    return None
                if content is None:
                    return None
                ids = tmp.encode(content, add_special_tokens=False).ids
                return ids[0] if ids else None

            pad_resolved = _resolve(payload.get("pad_token"))
            eos_resolved = _resolve(payload.get("eos_token"))
            if pad_resolved is not None:
                pad_id = pad_resolved
            if eos_resolved is not None:
                eos_id = eos_resolved

        return cls(tokenizer_path, pad_token_id=pad_id, eos_token_id=eos_id)

    def encode(
        self,
        text: str,
        *,
        max_length: int = 1024,
    ) -> tuple[mx.array, mx.array]:
        """Encode a single string, left-padded to ``max_length``.

        Returns ``(input_ids, attention_mask)`` with shape ``[1, max_length]``.
        - ``input_ids`` is ``int32`` (mlx convention).
        - ``attention_mask`` is ``int32`` with ``1`` for real tokens and ``0``
          for left-side padding.

        Truncation happens from the right (keep the prefix) when the encoded
        sequence exceeds ``max_length``. This matches the upstream behavior
        used by the TTS prompt pipeline.
        """
        text = text.strip()
        encoded = self._tokenizer.encode(text, add_special_tokens=True)
        ids = encoded.ids[:max_length]
        pad_count = max_length - len(ids)
        if pad_count > 0:
            padded_ids = [self.pad_token_id] * pad_count + ids
            mask = [0] * pad_count + [1] * len(ids)
        else:
            padded_ids = ids
            mask = [1] * len(ids)
        return (
            mx.array([padded_ids], dtype=mx.int32),
            mx.array([mask], dtype=mx.int32),
        )

    def encode_batch(
        self,
        texts: list[str],
        *,
        max_length: int = 1024,
    ) -> tuple[mx.array, mx.array]:
        """Encode a batch of strings. All rows share the same `max_length`."""
        ids_rows: list[list[int]] = []
        mask_rows: list[list[int]] = []
        for text in texts:
            ids_arr, mask_arr = self.encode(text, max_length=max_length)
            ids_rows.append(ids_arr[0].tolist())
            mask_rows.append(mask_arr[0].tolist())
        return (
            mx.array(ids_rows, dtype=mx.int32),
            mx.array(mask_rows, dtype=mx.int32),
        )


__all__ = ["LTXVGemmaTokenizer"]
