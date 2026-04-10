import json
import re
from pathlib import Path

from tokenizers import Tokenizer as RawTokenizer


IM_START_TOKEN = "<|im_start|>"
IM_END_TOKEN = "<|im_end|>"
MODALITY_TOKENS = {
    "text": "<|text|>",
    "voice": "<|voice|>",
    "interleave": "<|interleave|>",
}


class FishS2Tokenizer:
    def __init__(self, tokenizer):
        self._tokenizer = tokenizer
        self.semantic_token_ids = self._find_semantic_ids()
        semantic_values = list(self.semantic_token_ids.values())
        self.semantic_begin_id = min(semantic_values)
        self.semantic_end_id = max(semantic_values)

    @classmethod
    def from_pretrained(cls, model_dir: str | Path, trust_remote_code: bool = True):
        del trust_remote_code

        resolved = Path(model_dir)
        tokenizer_path = resolved / "tokenizer.json"
        if not tokenizer_path.is_file():
            raise FileNotFoundError(f"Missing Fish tokenizer.json at {tokenizer_path}")

        config = _load_tokenizer_config(resolved / "tokenizer_config.json")
        try:
            raw_tokenizer = RawTokenizer.from_file(str(tokenizer_path))
        except Exception as exc:
            raise ValueError(
                f"Invalid Fish tokenizer.json at {tokenizer_path}"
            ) from exc
        _validate_required_tokens(raw_tokenizer)
        tokenizer = _TokenizerBackend(
            raw_tokenizer,
            pad_token_id=_resolve_optional_token_id(
                raw_tokenizer, config, field_name="pad_token"
            ),
            eos_token_id=_resolve_optional_token_id(
                raw_tokenizer, config, field_name="eos_token"
            ),
        )
        return cls(tokenizer)

    def _find_semantic_ids(self):
        vocab = self._tokenizer.get_vocab()
        semantic_ids = {}

        for token, token_id in vocab.items():
            match = re.fullmatch(r"<\|semantic:(\d+)\|>", token)
            if match is None:
                continue
            semantic_ids[int(match.group(1))] = token_id

        if not semantic_ids:
            raise ValueError("No semantic tokens found in tokenizer vocab")
        expected_codes = list(range(len(semantic_ids)))
        actual_codes = sorted(semantic_ids)
        if actual_codes != expected_codes:
            raise ValueError(
                "Semantic token codes must start at 0 and form a dense sequence"
            )
        return semantic_ids

    @property
    def im_start_id(self) -> int:
        return self.get_token_id(IM_START_TOKEN)

    @property
    def im_end_id(self) -> int:
        return self.get_token_id(IM_END_TOKEN)

    def modality_id(self, modality: str) -> int:
        return self.get_token_id(MODALITY_TOKENS[modality])

    def semantic_id(self, code: int) -> int:
        code = int(code)
        if code not in self.semantic_token_ids:
            raise ValueError(f"Semantic token code {code} is not available")
        return self.semantic_token_ids[code]

    def get_token_id(self, token: str) -> int:
        return self._tokenizer.convert_tokens_to_ids(token)

    def encode(self, text: str, add_special_tokens: bool = False, **kwargs):
        return self._tokenizer.encode(
            text,
            add_special_tokens=add_special_tokens,
            **kwargs,
        )


class _TokenizerBackend:
    def __init__(
        self,
        tokenizer: RawTokenizer,
        *,
        pad_token_id: int | None,
        eos_token_id: int | None,
    ):
        self._tokenizer = tokenizer
        self.pad_token_id = pad_token_id
        self.eos_token_id = eos_token_id

    @property
    def vocab_size(self) -> int:
        return self._tokenizer.get_vocab_size()

    def get_vocab(self):
        return self._tokenizer.get_vocab()

    def convert_tokens_to_ids(self, token: str) -> int:
        token_id = self._tokenizer.token_to_id(token)
        if token_id is None:
            raise KeyError(f"Token not found: {token}")
        return token_id

    def encode(self, text: str, add_special_tokens: bool = False, **kwargs):
        del kwargs
        return self._tokenizer.encode(text, add_special_tokens=add_special_tokens).ids


def _load_tokenizer_config(config_path: Path) -> dict:
    if not config_path.is_file():
        return {}

    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Invalid Fish tokenizer_config.json at {config_path}"
        ) from exc

    if not isinstance(config, dict):
        raise ValueError(
            f"Fish tokenizer_config.json at {config_path} must contain a JSON object"
        )
    return config


def _validate_required_tokens(tokenizer: RawTokenizer) -> None:
    required_tokens = [IM_START_TOKEN, IM_END_TOKEN, *MODALITY_TOKENS.values()]
    missing_tokens = [
        token for token in required_tokens if tokenizer.token_to_id(token) is None
    ]
    if missing_tokens:
        missing = ", ".join(missing_tokens)
        raise ValueError(f"Missing required Fish tokenizer tokens: {missing}")


def _resolve_optional_token_id(
    tokenizer: RawTokenizer, config: dict, *, field_name: str
) -> int | None:
    if field_name not in config:
        return None

    token = _normalize_token_config_value(config[field_name], field_name=field_name)
    token_id = tokenizer.token_to_id(token)
    if token_id is None:
        raise ValueError(
            f"Fish tokenizer_config.json field {field_name!r} references unknown token {token!r}"
        )
    return token_id


def _normalize_token_config_value(value, *, field_name: str) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, dict) and isinstance(value.get("content"), str):
        return value["content"]
    raise ValueError(
        f"Fish tokenizer_config.json field {field_name!r} must be a token string "
        "or object with string 'content'"
    )
