# Confucius4-R2T2 streaming plan review

**Verdict: sound with required corrections.**

The chunk loop, marker-before-`model_type` dispatch, a text session separate from `ASRStreamSession`, and the non-goals can be implemented after the corrections below. Do not implement the plan as written. Slices 1–6 do not need PyTorch if the port does not import `r2t2/r2t2_asr.py` (it imports `torch` at module level) and conversion stays on `scripts/convert/qwen3_asr.py` (MLX safetensors only). The flattened “zero differing keys” claim against `netease-youdao/Confucius4-R2T2` is not checkable from this tree.

## Findings

### 1. Prefix rollback punctuation is not the fixed-text set

Plan §1 step 3 and slice 1 use one set, `，。！？、；：,.!?;:`, for both prefix rollback and `fixed_text`.

`R2T2ASRModel.streaming_transcribe` in `.references/Confucius4-R2T2/r2t2/r2t2_asr.py` uses two sets:

- Prefix `k` (about line 394): `，。！？、；：.!?;:` — no ASCII comma.
- `fixed_text` `k` (about line 452): `，。！？、；：,.!?;:` — ASCII comma included.

`streaming_transcribe_no_reset` only has the second set (about line 716) and is out of scope. An ASCII comma at the end of `_raw_decoded` must not set prefix `k` to 0.

### 2. Stall budget is not “add 0.5, then double”

Plan §1 (“If it does not grow, the budget increases. Then if the last committed word is CJK, the budget is doubled”) and slice 1 (“a stall adds 0.5; a last CJK word doubles”) compose the branches wrong.

`ws_server.py` `/asr_stream_api_v0` (about lines 639–663), same structure on v1:

- Growth: append `split_text_to_tokens(delta)` (punctuation stripped; CJK is one character, Latin is a word), then `max_new_tokens = max(1, step // 1280)`.
- Stall and last token is not CJK: `max_new_tokens += 0.5`.
- Stall and last token is CJK: reset to `max(1, step // 1280)`, do not add 0.5.
- After either branch, if that last token is CJK, double, then `min(floor, …)`.
- `streaming_transcribe` is called with `int(max_new_tokens)`. The float is kept across calls.

`is_last_token_chinese` in `example.py` is true when any character of that last token is in `U+4E00`–`U+9FFF`. It is not the last code point of the raw string. A trailing punctuation mark after a CJK character is still CJK. `total_new_asr_tokens` is fed only on growth. At 80 ms, `2 * base` is 2 and the floor is 4, so “add 0.5 then double” and “reset then double” disagree. At 160 ms the floor hides it.

The server updates this once per `streaming_transcribe` call, and each call is one chunk (`audio_seg` is exactly the current window). Plan §4 `feed` accepts any length. Slice 1 must apply the lock and the budget once per consumed chunk inside that feed. Applying them only to the last chunk of a multi-chunk `feed` changes later `max_new_tokens`.

`finalize` / EOS does not use the running budget. `ws_server.py` (about lines 555–557) and `example.py` `run_streaming` pass `max(1, (step + lookahead) // 1280)`, the first-window cap.

### 3. `finish_streaming_transcribe` is not the per-chunk machine with a different index

Plan §1 describes the prefix index and the full-string return. It does not state the rest of the R2T2 override. `R2T2ASRModel.finish_streaming_transcribe` (`r2t2_asr.py`):

- Empty `buffer`: return `state.text` and do not generate (`r2t2_asr.py` about lines 533–534). The server length lock still runs on that full hypothesis, so the token per-chunk `fixed_text` was holding back can commit with no decode. Slice 1 only says the empty tail does not call `decode`.
- Non-empty: append the tail with no pad. Prefix is `""` while `chunk_id < unfixed_chunk_num`; otherwise `end_idx = max(1, len(ids) - unfixed_token_num)` with no U+FFFD loop (parent `Qwen3ASRModel.finish_streaming_transcribe` in `.references/Qwen3-ASR/qwen_asr/inference/qwen3_asr.py`). Then `prefix.split("|")[0]`. There is no pre-encode `|` cut of `_raw_decoded`, no `rollback_punctuation`, and no unfixed-token `fixed_text`.
- `gen_text` still goes through `_normalize_punct_by_context` and U+FFFD deletion. `_raw_decoded = (prefix + gen_text).split("|")[0]`, then `parse_asr_output`. No `parse_language_output`, no CJK space deletion, no `<asr_text>` rewrite, no “no tag ⇒ skip `chunk_id`” branch.
- Return `state.text` (`txt.split("|")[0]`), not a rolled-back string. `chunk_id` increments.

Parent Qwen finish has none of the punctuation, U+FFFD, or `|` steps. Copying either the parent function or the per-chunk body is wrong.

### 4. Do not call the shipped MLX `parse_asr_output` inside the streaming loop

Plan §2 lists `parse_asr_output` as safe to reuse. The streaming loop does not use that function.

`streaming_transcribe` (`r2t2_asr.py`):

1. If `force_language is None`, language for the CJK-space decision comes from local `parse_language_output`. No tag ⇒ language `""`. It does not run `detect_and_fix_repetitions`.
2. Then, if forced or detected language is Chinese, delete spaces between CJK on `_raw_decoded`.
3. Then `qwen_asr.inference.utils.parse_asr_output`, which strips, runs `detect_and_fix_repetitions` (threshold 20), and returns `("", text)` when `<asr_text>` is absent. The no-tag `continue` is based on the raw string still containing `<asr_text>` after the rewrite, not on the parser’s language.

`mlx_speech.models.qwen3_asr.processor.parse_asr_output` differs: no repetition fixer; `_clean_transcript_text` drops a generated-tail regex; with no tag, `_parse_language_prefixed_text` can return a language and a shortened transcript. Using it for step 1 turns on CJK space deletion for `language Chinese…` with no `<asr_text>` tag. Using it for step 3 changes `_raw_decoded` before the next prefix. Keep the MLX parser on the offline Qwen path.

### 5. Offline `generate` must not run the streaming cleanup

Plan §2 “Must be new” and slice 4 apply `|`, punctuation, CJK spaces, and U+FFFD cleanup on offline `generate` before `parse_asr_output`.

`R2T2ASRModel.transcribe` is `super().transcribe` only (`r2t2_asr.py`). Those cleanups exist only on the streaming methods. Offline output should stay `Qwen3ASRTranscriber.transcribe` (MLX `parse_asr_output` included). Streaming cleanup on offline `qwen3-asr-1.7b` stays off, and it also stays off the R2T2 offline path.

### 6. Slice 4 skips the audio-embedding splice

Plan slice 4: `get_audio_features` + `prefill` + `decode_step`.

`Qwen3ASRModel.prefill` takes `inputs_embeds`, not audio features (`model.py`). `Qwen3ASRTranscriber.transcribe` (`generation/qwen3_asr.py`) embeds ids, `replace_audio_embeddings` / `prepare_inputs_embeds`, then prefills. Without that splice the audio pads stay text embeddings.

vLLM `outputs[0].outputs[0].text` is the completion only. Decode the new ids with `skip_special_tokens=True` and pass that string into the slice-1 machine. Decoding the whole prompt double-counts `prefix`.

Pad count stays `_get_feat_extract_output_lengths` on mel frames (`lengths // hop_length`). 1280 / 2560 / 5120 samples are 8 / 16 / 32 frames and 1 / 2 / 4 audio tokens. That part of slice 3 matches `feature_extraction.py` and `modeling_qwen3_asr.py`.

### 7. The bf16 saver will drop the family marker

Plan §3 and slice 5: write `mlx_speech.family = confucius4_r2t2` into the emitted `config.json`, and check that marker before `model_type`.

`_resolve_asr_family` (`asr/_registry.py`) returns `"qwen3"` on `model_type == "qwen3_asr"` and never reads another key. The plan’s order (marker first, else `qwen3_asr`) is what stops a raw folder from taking the offline adapter. `Qwen3ASRConfig.from_dict` already stores unknown top-level keys in `extra` and still requires `model_type == "qwen3_asr"`.

`save_qwen3_asr_bf16_checkpoint` (`checkpoint.py`) copies `config.json` verbatim via `_copy_supporting_files`. It does not round-trip `to_dict()`. Setting the marker only on the loaded config, then calling that helper, leaves upstream `config.json` on disk and `load` stays on family `qwen3`. Patch the copied file after the copy. `family_hint` on `_ModelAlias` is unused by `asr.load` (only TTS reads it). That part of §3 is right.

Slice 2’s “unknown family names the supported types, now including the new one” does not match either error. Unknown `model_type` lists `cohere_asr, granite_speech, qwen3_asr, nemotron_asr`. Unwired `load` raises `Unsupported ASR family: {family!r}` and lists nothing.

### 8. Qwen3-ASR-1.7B does not ship a single tied embedding tensor

Plan slice 5: if `tie_word_embeddings` leaves one embedding, reuse the 1.7B tied-head load path.

`tie_word_embeddings` is read from text-config `extra` and aliases `lm_head.weight` to `embed_tokens.weight` in `Qwen3ASRTextForCausalLM.__init__` (`text_decoder.py`). The converted package still has both tensors. `tests/checkpoint/test_qwen3_asr_checkpoint.py` expects `text_decoder.model.embed_tokens.weight` and `text_decoder.lm_head.weight`, each `[151936, 2048]`. Do not add a missing-`lm_head` path unless the R2T2 file actually lacks that key.

### 9. “Same config” is only half checkable here

Plan intro and §2–§3: flattened `config.json` files are equal, including the 1.7B tower (`n_window=50`, `n_window_infer=800`, `conv_chunksize=500`, text intermediate 6144, `max_position_embeddings` 65536, `mrope_section` `[24, 20, 20]`), audio ids `151676` / `151669` / `151670`, eos `[151643, 151645]`, and the R2T2 tokenizer has no `|` special token.

Checkable locally:

- Those Qwen numbers, `rope_scaling`, and `support_languages` match `models/qwen3_asr_1_7b/mlx-int8/config.json`. `generation_config.json` there is `do_sample: false`, `temperature: 1e-6`, `eos_token_id: [151643, 151645]`. `preprocessor_config.json` is Whisper, hop 160, n_fft 400, 128 mels. `chat_template.json` is the system / user audio-pad / assistant string `build_prompt` hardcodes.
- Class defaults in `.references/Qwen3-ASR/qwen_asr/core/transformers_backend/configuration_qwen3_asr.py` (`d_model=1280`, 32 encoder layers, `output_dim=3584`, `n_window=100`, `n_window_infer=400`) are not that checkpoint. `Qwen3ASRConfig.from_dict` reads the file.
- `mrope` is stored in text-config `extra`. `Qwen3ASRTextRotaryEmbedding` uses `rope_theta` only. No mRoPE path to add.
- Local Qwen `tokenizer_config.json` `additional_special_tokens` has no `|`. `<asr_text>` is added token `151704` with `"special": false`.

Not in `.references/Confucius4-R2T2` or `models/`: R2T2 `config.json`, tokenizer, or `model.safetensors`. “Zero differing keys”, the R2T2 language list, and “`|` is not a special token on the R2T2 tokenizer” are asserted, not checked. `tests/unit/test_qwen3_asr_config.py` is a different fixture (`n_window` 100, intermediate 11008, `max_position_embeddings` 40960). Do not treat it as the published 1.7B config. Slices 1–4 must not depend on the unverified equality. Slice 5’s sanitized key and shape check remains the gate.

Graph reuse itself is supported: `R2T2ASRModel` subclasses `Qwen3ASRModel` and defines no modules. Streaming re-feeds `audio_accum` with `prompt_raw + prefix` and no cross-chunk cache (`streaming_transcribe`).

## Corrections to apply

1. In §1 step 3 and slice 1, use the prefix punctuation set without ASCII comma, and the `fixed_text` set with it (`r2t2_asr.py` `streaming_transcribe`).
2. In §1 and slice 1, script the v0 stall rule: add 0.5 only when the last `split_text_to_tokens` token is not CJK; on CJK, reset to `step // 1280` then double; then cap with the floor. Update budget and the length lock once per consumed chunk. `finalize` uses the first-window cap `(step + lookahead) // 1280`, not the running budget (`ws_server.py`, `example.py`).
3. In §1 and slice 1, specify `finish_streaming_transcribe` as its own post-process (finding 3), including empty-buffer return of `state.text` so the lock can commit the held-back tail without a decode.
4. In §1–§2, do not call MLX `parse_asr_output` from the streaming loop. Language probe is `parse_language_output`; the rewrite parse is Qwen `parse_asr_output` including `detect_and_fix_repetitions`.
5. In §2 and slice 4, drop streaming cleanup from offline `generate`. `R2T2ASRModel.transcribe` does not do it.
6. In slice 4, prefill `prepare_inputs_embeds` / `replace_audio_embeddings`, and decode only new tokens.
7. In slice 5, patch `mlx_speech.family` into the copied `config.json` after `save_qwen3_asr_bf16_checkpoint`. Do not add a single-tensor tied-head path unless the R2T2 file has no `lm_head`.
8. In the intro and §3, mark the R2T2-vs-Qwen flattened equality and the R2T2 tokenizer `|` claim as unverified until those files are on disk. Point the Qwen numbers at `models/qwen3_asr_1_7b/mlx-int8/`, not at `tests/unit/test_qwen3_asr_config.py`.
9. In slice 2, test the marker check before the `model_type == "qwen3_asr"` return. Do not expect today’s unknown-`model_type` string, or `Unsupported ASR family`, to list the new family until that message is actually changed.
