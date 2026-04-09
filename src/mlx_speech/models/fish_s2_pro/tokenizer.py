from typing import Optional, Union, List
from transformers import AutoTokenizer


class FishS2Tokenizer:
    """Fish Audio S2 Pro tokenizer.

    Wraps HuggingFace AutoTokenizer and manages semantic token mappings.
    """

    def __init__(
        self,
        tokenizer: AutoTokenizer,
        semantic_begin_id: int,
        semantic_end_id: int,
    ):
        self._tokenizer = tokenizer
        self.semantic_begin_id = semantic_begin_id
        self.semantic_end_id = semantic_end_id

    @property
    def vocab_size(self) -> int:
        if self._tokenizer is None:
            return 32000  # Default for Fish Audio models
        return self._tokenizer.vocab_size

    @property
    def eos_token_id(self) -> int:
        if self._tokenizer is None:
            raise ValueError("Tokenizer not initialized")
        return self._tokenizer.eos_token_id

    @property
    def pad_token_id(self) -> int:
        if self._tokenizer is None:
            raise ValueError("Tokenizer not initialized")
        return self._tokenizer.pad_token_id

    @property
    def num_semantic_tokens(self) -> int:
        return self.semantic_end_id - self.semantic_begin_id + 1

    @classmethod
    def from_pretrained(cls, model_dir: str, trust_remote_code: bool = True):
        """Load tokenizer from directory."""
        tokenizer = AutoTokenizer.from_pretrained(
            model_dir, trust_remote_code=trust_remote_code
        )

        vocab = tokenizer.get_vocab()
        semantic_begin = semantic_end = None

        for i in range(4096):
            token = f"<|semantic:{i}|>"
            if token in vocab:
                if semantic_begin is None:
                    semantic_begin = vocab[token]
                semantic_end = vocab[token]

        if semantic_begin is None:
            raise ValueError("No semantic tokens found in tokenizer")

        return cls(tokenizer, semantic_begin, semantic_end)

    def encode(self, text: str, **kwargs) -> List[int]:
        import inspect

        sig = inspect.signature(self._tokenizer.encode)
        if "allowed_special" in sig.parameters and "allowed_special" not in kwargs:
            kwargs["allowed_special"] = "all"
        return self._tokenizer.encode(text, **kwargs)

    def decode(self, token_ids: Union[List[int], int], **kwargs) -> str:
        return self._tokenizer.decode(token_ids, **kwargs)

    def __getattr__(self, name):
        return getattr(self._tokenizer, name)
