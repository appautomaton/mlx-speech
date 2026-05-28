"""Checkpoint loaders for the DramaBox prompt pipeline.

Reads the `audio-components` shard, filters keys for the
`text_embedding_projection.audio_aggregate_embed.*` (aggregate) and
`model.diffusion_model.audio_embeddings_connector.*` (connector) submodules,
strips the leading prefixes, and feeds them to `load_weights`.
"""

from __future__ import annotations

from pathlib import Path

import mlx.core as mx

from .connector import Embeddings1DConnector
from .feature_extractor import FeatureExtractorV2

_AGGREGATE_PREFIX = "text_embedding_projection."
_CONNECTOR_PREFIX = "model.diffusion_model.audio_embeddings_connector."


def load_audio_components_state(path: str | Path) -> dict[str, mx.array]:
    """Load the entire audio-components shard. Helper for downstream stages."""
    return mx.load(str(path))


def _filter_prefix(state: dict[str, mx.array], prefix: str) -> dict[str, mx.array]:
    return {
        k[len(prefix):]: v for k, v in state.items() if k.startswith(prefix)
    }


def load_feature_extractor_weights(
    feature_extractor: FeatureExtractorV2,
    state: dict[str, mx.array],
) -> tuple[int, list[str]]:
    """Load the aggregate's weight + bias from a pre-loaded state dict.

    Returns ``(n_loaded, expected_missing_in_module)`` — the latter being a
    helper for callers wanting to see which top-level state keys did NOT
    get used.
    """
    sub = _filter_prefix(state, _AGGREGATE_PREFIX)  # audio_aggregate_embed.{weight,bias}
    if not sub:
        raise KeyError(
            f"No keys with prefix {_AGGREGATE_PREFIX!r} found in state dict"
        )
    feature_extractor.load_weights(list(sub.items()), strict=True)
    return len(sub), []


def load_connector_weights(
    connector: Embeddings1DConnector,
    state: dict[str, mx.array],
) -> int:
    """Load the connector's weights from a pre-loaded state dict.

    Returns the number of keys loaded.
    """
    sub = _filter_prefix(state, _CONNECTOR_PREFIX)
    if not sub:
        raise KeyError(
            f"No keys with prefix {_CONNECTOR_PREFIX!r} found in state dict"
        )
    connector.load_weights(list(sub.items()), strict=True)
    return len(sub)


__all__ = [
    "load_audio_components_state",
    "load_feature_extractor_weights",
    "load_connector_weights",
]
