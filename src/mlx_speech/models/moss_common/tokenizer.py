"""Internal shared tokenizer helpers for OpenMOSS model families."""

from ..moss_local.tokenizer import DEFAULT_MOSS_CHAT_TEMPLATE, MossTTSLocalTokenizer

MossChatTokenizer = MossTTSLocalTokenizer

__all__ = ["DEFAULT_MOSS_CHAT_TEMPLATE", "MossChatTokenizer"]
