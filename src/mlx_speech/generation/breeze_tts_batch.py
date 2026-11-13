"""Fixed-batch Breeze inference with independent request state and shared weights."""

from __future__ import annotations

import time
import warnings
from collections.abc import Sequence
from numbers import Integral

import mlx.core as mx

from ..models._cache import BoundedKVCache
from .breeze_tts import (
    FRAME_RATE,
    SAMPLE_RATE,
    BreezeAudio,
    _finite_positive,
    _guide,
    _mask_reserved,
    _prompt_text,
    _validated_reference,
)


def generate_breeze_batch(
    speech,
    codec,
    tokenizer,
    texts: Sequence[str],
    *,
    speaker: str = "S0",
    instruction: str | None = None,
    cfg_scale: float = 1.0,
    max_frames: int = 80,
    temperature: float = 0.9,
    top_k: int = 50,
    repetition_penalty: float = 1.1,
    seed: int = 0,
    seeds: Sequence[int] | None = None,
    ref_audio: mx.array | None = None,
    ref_text: str | None = None,
    batch_size: int | None = None,
    timings: dict[str, float] | None = None,
) -> list[BreezeAudio]:
    """Generate independent utterances, sharing reference and sampling controls.

    The default is 256 backbone rows: 256 requests without guidance, 128 with
    guidance. Larger inputs are partitioned without changing request seeds.
    """
    if isinstance(texts, str) or any(
        not isinstance(t, str) or not t.strip() for t in texts
    ):
        raise ValueError("texts must be a sequence of non-empty strings.")
    if (
        not isinstance(max_frames, Integral)
        or isinstance(max_frames, bool)
        or max_frames <= 0
    ):
        raise ValueError("max_frames must be a positive integer.")
    for name, value in (
        ("temperature", temperature),
        ("cfg_scale", cfg_scale),
        ("repetition_penalty", repetition_penalty),
    ):
        if not _finite_positive(value):
            raise ValueError(f"{name} must be finite and greater than 0.")
    if not isinstance(top_k, Integral) or isinstance(top_k, bool) or top_k < 0:
        raise ValueError("top_k must be a non-negative integer.")
    direction = (instruction or "").strip() or None
    if cfg_scale != 1.0 and direction is None:
        raise ValueError("cfg_scale other than 1 requires an instruction.")
    batch_size = (
        (128 if cfg_scale != 1.0 else 256) if batch_size is None else batch_size
    )
    if (
        not isinstance(batch_size, Integral)
        or isinstance(batch_size, bool)
        or batch_size <= 0
    ):
        raise ValueError("batch_size must be a positive integer.")
    request_seeds = (
        list(seeds) if seeds is not None else [seed + i for i in range(len(texts))]
    )
    if len(request_seeds) != len(texts) or any(
        not isinstance(s, Integral) or isinstance(s, bool) or not 0 <= s < 2**32
        for s in request_seeds
    ):
        raise ValueError("seeds must contain one unsigned 32-bit integer per text.")
    reference = _validated_reference(ref_audio, ref_text)
    if not texts:
        return []
    elapsed = {"prepare_seconds": 0.0, "decode_seconds": 0.0, "codec_seconds": 0.0}
    started = time.perf_counter()
    prefix = None
    if reference is not None:
        audio, transcript = reference
        ref_codes = codec.encode(audio)
        ref_hidden = _encode_texts(
            speech, tokenizer, [_prompt_text(speaker, transcript, None)], segment=True
        )[0]
        embeddings = speech.backbone_model.embed_audio(ref_codes[None])[0]
        end = speech.backbone_model.embed_audio(
            mx.zeros((1, 1, ref_codes.shape[1]), mx.int32)
        )[0]
        prefix = mx.concatenate([ref_hidden, embeddings, end])
        mx.eval(prefix)
    elapsed["prepare_seconds"] += time.perf_counter() - started
    results: list[BreezeAudio] = []
    for start in range(0, len(texts), batch_size):
        chunk = texts[start : start + batch_size]
        started = time.perf_counter()
        prompts = _prepare_prompts(
            speech, tokenizer, chunk, speaker, direction, cfg_scale != 1.0, prefix
        )
        elapsed["prepare_seconds"] += time.perf_counter() - started
        started = time.perf_counter()
        codes = _generate_codes(
            speech,
            prompts,
            request_seeds[start : start + len(chunk)],
            max_frames=max_frames,
            temperature=temperature,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            cfg_scale=cfg_scale,
        )
        elapsed["decode_seconds"] += time.perf_counter() - started
        started = time.perf_counter()
        results.extend(_decode_audio(codec, codes))
        elapsed["codec_seconds"] += time.perf_counter() - started
    if timings is not None:
        timings.update(elapsed)
    return results


def _encode_texts(
    speech, tokenizer, texts: Sequence[str], *, segment: bool
) -> list[mx.array]:
    """Encode padded text batches, preserving each unpadded segment's positions."""
    unique = list(dict.fromkeys(texts))
    tokens = []
    for text in unique:
        if segment:
            text = tokenizer.decode(
                tokenizer.encode(text, add_special_tokens=True).ids,
                skip_special_tokens=False,
            )
        ids = tokenizer.encode(text, add_special_tokens=False).ids
        bos = tokenizer.token_to_id("<bos>")
        if not segment and bos is not None and (not ids or ids[0] != bos):
            ids = [bos, *ids]
        tokens.append(ids)
    encoded = {}
    width = max(map(len, tokens))
    # Bound prefill intermediates independently of the much larger decode batch.
    group_size = max(1, 4096 // width)
    for start in range(0, len(unique), group_size):
        group = tokens[start : start + group_size]
        length = max(map(len, group))
        ids = mx.array([row + [0] * (length - len(row)) for row in group], mx.int32)
        mask = (
            mx.arange(length)[None, :] < mx.array([len(row) for row in group])[:, None]
        )
        hidden = speech.text_encoder_proj(speech.text_encoder(ids, attention_mask=mask))
        mx.eval(hidden)
        for i, row in enumerate(group):
            encoded[unique[start + i]] = hidden[i, : len(row)]
    return [encoded[text] for text in texts]


def _prepare_prompts(speech, tokenizer, texts, speaker, instruction, guided, prefix):
    rendered = [_prompt_text(speaker, text, instruction) for text in texts]
    if guided:
        rendered += [_prompt_text(speaker, text, None) for text in texts]
    embeddings = _encode_texts(speech, tokenizer, rendered, segment=prefix is not None)
    prompts = [
        mx.concatenate([prefix, row]) if prefix is not None else row
        for row in embeddings
    ]
    mx.eval(prompts)
    return prompts


def _prefill(speech, prompts, max_frames):
    lengths = mx.array([len(row) for row in prompts], mx.int32)
    longest = max(len(row) for row in prompts)
    config = speech.config.backbone
    capacity = longest + max_frames - 1
    if capacity > config.max_position_embeddings:
        raise ValueError(
            f"Prompt plus frame budget needs {capacity} positions; "
            f"Breeze supports {config.max_position_embeddings}."
        )
    pads = longest - lengths
    cache = [
        BoundedKVCache.allocate(
            batch_size=len(prompts),
            capacity=capacity,
            num_heads=config.num_key_value_heads,
            head_dim=config.head_dim,
            key_dtype=prompts[0].dtype,
            value_dtype=prompts[0].dtype,
        )
        for _ in range(config.num_hidden_layers)
    ]
    padded = mx.stack(
        [mx.pad(row, [(longest - len(row), 0), (0, 0)]) for row in prompts]
    )
    for start in range(0, longest, 128):
        end = min(start + 128, longest)
        positions = mx.maximum(mx.arange(start, end)[None, :] - pads[:, None], 0)
        hidden = speech.backbone_model(
            padded[:, start:end], cache, position_ids=positions, pad_lengths=pads
        )
        mx.eval(hidden, _cache_arrays(cache))
    return hidden[:, -1], cache, lengths, pads


def _cache_arrays(cache):
    return [value for layer in cache for value in (layer.keys, layer.values)]


def _select_cache(cache, rows):
    for layer in cache:
        layer.keys = layer.keys[rows]
        layer.values = layer.values[rows]


@mx.compile
def _sample_rows(logits, keys, temperature, top_k):
    values = logits.astype(mx.float32) / temperature
    if 0 < top_k < values.shape[-1]:
        threshold = mx.min(mx.topk(values, top_k, axis=-1), axis=-1, keepdims=True)
        values = mx.where(values < threshold, -float("inf"), values)
    return mx.vmap(lambda row, key: mx.random.categorical(row, key=key))(
        values, keys
    ).astype(mx.int32)


@mx.compile
def _frame_keys(keys, count):
    return mx.vmap(lambda key: mx.random.split(key, num=count))(keys)


def _depth_frame(speech, hidden, first, keys, cache, cfg_scale, temperature, top_k):
    count = len(first)
    guided = cfg_scale != 1.0
    for layer in cache:
        layer.restore_offset(0)

    def embed(code, codebook):
        value = speech.depth_decoder.embed_code(code, codebook)
        return mx.concatenate([value, value]) if guided else value

    def scores(value):
        return _mask_reserved(
            _guide(value[:count], value[count:] if guided else None, cfg_scale),
            keep_eos=False,
        )

    codes = [first]
    logits = scores(speech.depth_decoder.begin_frame(hidden, embed(first, 0), cache))
    for codebook in range(1, speech.config.num_codebooks):
        code = _sample_rows(logits, keys[:, codebook], temperature, top_k)
        codes.append(code)
        if codebook + 1 < speech.config.num_codebooks:
            logits = scores(
                speech.depth_decoder.step(
                    embed(code, codebook), codebook_index=codebook, cache=cache
                )
            )
    return mx.stack(codes, axis=1)


def _generate_codes(
    speech,
    prompts,
    seeds,
    *,
    max_frames,
    temperature,
    top_k,
    repetition_penalty,
    cfg_scale,
):
    hidden, cache, positions, pads = _prefill(speech, prompts, max_frames)
    size = len(seeds)
    active = list(range(size))
    lengths = [max_frames] * size
    keys = mx.stack([mx.random.key(int(seed)) for seed in seeds])
    seen = mx.zeros((size, speech.config.vocab_size + 1), mx.bool_)
    generated = mx.zeros((size, max_frames, speech.config.num_codebooks), mx.int32)
    guided = cfg_scale != 1.0
    depth_cache = speech.depth_decoder.make_cache(
        batch_size=len(prompts), dtype=hidden.dtype
    )
    for step in range(max_frames):
        count = len(active)
        scores = speech.lm_head(hidden)
        logits = _guide(scores[:count], scores[count:] if guided else None, cfg_scale)
        adjusted = mx.where(
            logits < 0, logits * repetition_penalty, logits / repetition_penalty
        )
        logits = _mask_reserved(mx.where(seen, adjusted, logits), keep_eos=True)
        draw_keys = _frame_keys(keys, speech.config.num_codebooks + 1)
        first = _sample_rows(logits, draw_keys[:, 1], temperature, top_k)
        tokens = first.tolist()  # One host read per frame, for independent EOS.
        keep = [
            i
            for i, token in enumerate(tokens)
            if token != speech.config.backbone_eos_id
        ]
        for i, token in enumerate(tokens):
            if token == speech.config.backbone_eos_id:
                lengths[active[i]] = step
        if not keep:
            active = []
            break
        if len(keep) != count:
            rows = mx.array(keep, mx.int32)
            branches = mx.concatenate([rows, rows + count]) if guided else rows
            hidden, positions, pads = (
                hidden[branches],
                positions[branches],
                pads[branches],
            )
            _select_cache(cache, branches)
            _select_cache(depth_cache, branches)
            first, draw_keys, seen = first[rows], draw_keys[rows], seen[rows]
            active = [active[i] for i in keep]
        keys = draw_keys[:, 0]
        frame = _depth_frame(
            speech,
            hidden,
            first,
            draw_keys[:, 1:],
            depth_cache,
            cfg_scale,
            temperature,
            top_k,
        )
        generated[mx.array(active), step] = frame
        seen = seen.at[mx.arange(len(active)), first].add(True)
        if step + 1 < max_frames:
            audio = speech.backbone_model.embed_audio(frame[:, None])
            if guided:
                audio = mx.concatenate([audio, audio])
            hidden = speech.backbone_model(
                audio, cache, position_ids=positions[:, None], pad_lengths=pads
            )[:, -1]
            positions = positions + 1
        mx.eval(
            generated,
            keys,
            seen,
            hidden,
            _cache_arrays(cache),
            _cache_arrays(depth_cache),
        )
    if active:
        warnings.warn(
            f"Breeze batch requests {active} reached the {max_frames}-frame budget "
            f"({max_frames / FRAME_RATE:.1f} s) before EOS; their speech may be cut off.",
            RuntimeWarning,
            stacklevel=2,
        )
    codes = [generated[i, :length] for i, length in enumerate(lengths)]
    mx.eval(codes)
    return codes


def _decode_audio(codec, codes):
    """Batch the causal codec with bounded activations, then trim padded tails."""
    results = [BreezeAudio(mx.zeros((0,), mx.float32), SAMPLE_RATE, 0) for _ in codes]
    order = sorted(
        (i for i, row in enumerate(codes) if len(row)), key=lambda i: len(codes[i])
    )
    start = 0
    while start < len(order):
        # The codec expands every frame into 1,920 samples and large feature maps.
        end = start + 1
        while end < len(order) and (end - start + 1) * len(codes[order[end]]) <= 256:
            end += 1
        group = order[start:end]
        longest = max(len(codes[i]) for i in group)
        batch = mx.stack(
            [mx.pad(codes[i].T, [(0, 0), (0, longest - len(codes[i]))]) for i in group]
        )
        audio = codec.decode(batch)
        mx.eval(audio)
        for row, index in enumerate(group):
            frames = len(codes[index])
            waveform = audio[row, 0, : frames * int(SAMPLE_RATE / FRAME_RATE)].astype(
                mx.float32
            )
            mx.eval(waveform)
            results[index] = BreezeAudio(waveform, SAMPLE_RATE, frames)
        start += len(group)
    return results
