"""Tokenizer wrapper for CohereAsr (16k SentencePiece BPE)."""

from __future__ import annotations

from pathlib import Path

# Supported transcription languages
LANGUAGES: frozenset[str] = frozenset(
    ["ar", "de", "el", "en", "es", "fr", "it", "ja", "ko", "nl", "pl", "pt", "vi", "zh"]
)

# Special token IDs (from tokenizer_config.json added_tokens_decoder)
_SPECIAL_IDS = {
    "<unk>": 0,
    "<|nospeech|>": 1,
    "<pad>": 2,
    "<|endoftext|>": 3,
    "<|startoftranscript|>": 4,
    "<|pnc|>": 5,
    "<|nopnc|>": 6,
    "<|startofcontext|>": 7,
    "<|noitn|>": 9,
    "<|notimestamp|>": 11,
    "<|nodiarize|>": 13,
    "<|emo:undefined|>": 16,
}

# decoder_start_token_id from generation_config.json
DECODER_START_TOKEN_ID: int = 13764


class CohereAsrTokenizer:
    """Thin wrapper around the tokenizers BPE tokenizer for CohereAsr.

    Loads tokenizer.json (fast tokenizer format) from the model directory.
    """

    def __init__(self, tokenizer_path: str | Path):
        try:
            from tokenizers import Tokenizer
        except ImportError as e:
            raise ImportError(
                "The `tokenizers` package is required. Install with: pip install tokenizers"
            ) from e

        tokenizer_path = Path(tokenizer_path)
        if not tokenizer_path.exists():
            raise FileNotFoundError(f"Tokenizer not found: {tokenizer_path}")

        self._tok = Tokenizer.from_file(str(tokenizer_path))

        # Build reverse map: token_string → id (for special tokens)
        self._special_ids = dict(_SPECIAL_IDS)

        # Language token IDs — look them up from the tokenizer vocab
        self._lang_ids: dict[str, int] = {}
        for lang in LANGUAGES:
            tok_str = f"<|{lang}|>"
            tid = self._tok.token_to_id(tok_str)
            if tid is not None:
                self._lang_ids[lang] = tid

        # Space prefix token used in the decoder prompt (SentencePiece word-initial space)
        self._space_id = self._tok.token_to_id("▁")

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        """Encode text to token IDs."""
        encoding = self._tok.encode(text, add_special_tokens=add_special_tokens)
        return encoding.ids

    def decode(self, ids: list[int], skip_special_tokens: bool = True) -> str:
        """Decode token IDs to text."""
        return self._tok.decode(ids, skip_special_tokens=skip_special_tokens)

    def get_decoder_prompt_ids(self, language: str, punctuation: bool = True) -> list[int]:
        """Build the fixed decoder prompt for the given language.

        The full decoder input at generation start is:
            [DECODER_START_TOKEN_ID] + get_decoder_prompt_ids(language, punctuation)

        Matches upstream CohereAsrProcessor.get_decoder_prompt_ids.
        """
        if language not in LANGUAGES:
            raise ValueError(
                f"Unsupported language {language!r}. Supported: {sorted(LANGUAGES)}"
            )
        lang_id = self._lang_ids.get(language)
        if lang_id is None:
            raise ValueError(f"Language token <|{language}|> not found in tokenizer vocab.")

        pnc_id = self._special_ids["<|pnc|>"] if punctuation else self._special_ids["<|nopnc|>"]

        prompt: list[int] = []
        # "▁" — word-initial space token
        if self._space_id is not None:
            prompt.append(self._space_id)
        prompt.extend([
            self._special_ids["<|startofcontext|>"],
            self._special_ids["<|startoftranscript|>"],
            self._special_ids["<|emo:undefined|>"],
            lang_id,
            lang_id,  # language token appears twice (upstream)
            pnc_id,
            self._special_ids["<|noitn|>"],
            self._special_ids["<|notimestamp|>"],
            self._special_ids["<|nodiarize|>"],
        ])
        return prompt

    @classmethod
    def from_dir(cls, model_dir: str | Path) -> "CohereAsrTokenizer":
        model_dir = Path(model_dir)
        tok_path = model_dir / "tokenizer.json"
        if not tok_path.exists():
            raise FileNotFoundError(f"tokenizer.json not found in {model_dir}")
        return cls(tok_path)
