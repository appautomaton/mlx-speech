"""Qwen2.5-7B BPE tokenizer wrapper for VibeVoice Large."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from tokenizers import Tokenizer


# Special token strings (reused Qwen2 vision tokens)
SPEECH_START_TOKEN = "<|vision_start|>"
SPEECH_END_TOKEN = "<|vision_end|>"
SPEECH_DIFFUSION_TOKEN = "<|vision_pad|>"
EOS_TOKEN = "<|endoftext|>"


@dataclass
class VibeVoiceTokenizer:
    """Thin wrapper around the Qwen2.5-7B BPE tokenizer."""

    tokenizer: Tokenizer
    speech_start_id: int
    speech_end_id: int
    speech_diffusion_id: int
    eos_token_id: int

    @classmethod
    def from_path(cls, tokenizer_path: str | Path) -> VibeVoiceTokenizer:
        """Load tokenizer from a directory containing tokenizer.json.

        Args:
            tokenizer_path: directory containing tokenizer.json, or path to
                the tokenizer.json file directly.
        """
        path = Path(tokenizer_path)
        if path.is_dir():
            path = path / "tokenizer.json"
        if not path.exists():
            raise FileNotFoundError(
                f"Tokenizer file not found: {path}\n"
                "Download the Qwen2.5-7B tokenizer:\n"
                "  hf download Qwen/Qwen2.5-7B-Instruct tokenizer.json "
                f"--local-dir {path.parent}"
            )

        tok = Tokenizer.from_file(str(path))

        def _resolve_id(token: str) -> int:
            tid = tok.token_to_id(token)
            if tid is None:
                raise ValueError(f"Token '{token}' not found in tokenizer vocabulary")
            return tid

        return cls(
            tokenizer=tok,
            speech_start_id=_resolve_id(SPEECH_START_TOKEN),
            speech_end_id=_resolve_id(SPEECH_END_TOKEN),
            speech_diffusion_id=_resolve_id(SPEECH_DIFFUSION_TOKEN),
            eos_token_id=_resolve_id(EOS_TOKEN),
        )

    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]:
        return self.tokenizer.encode(text, add_special_tokens=add_special_tokens).ids

    def decode(self, token_ids: list[int]) -> str:
        return self.tokenizer.decode(token_ids)

    @property
    def valid_speech_token_ids(self) -> list[int]:
        """The 4 tokens the LM is constrained to during speech generation."""
        return [
            self.speech_start_id,
            self.speech_end_id,
            self.speech_diffusion_id,
            self.eos_token_id,
        ]
