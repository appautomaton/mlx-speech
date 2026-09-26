# Confucius4-R2T2 streaming ASR

Plan only. Do not copy upstream Python into `src/`. Code in
`.references/Confucius4-R2T2` is Apache-2.0; weights are under the NetEase
model license and stay out of git. Inference stays on `mlx`, `numpy`,
`safetensors`, `soundfile`, and `tokenizers`. `huggingface_hub` only inside
`_hub.py` on the download path. No `torch`, `torchaudio`, vLLM, or a
PyTorch-backed path under an MLX label.

Checked out references, read locally (no fetch/pull of those clones):

- `.references/Confucius4-R2T2` at `26d55a5`: `r2t2/r2t2_asr.py`, `example.py`,
  `ws_server.py` (commit rule and live token budget), `README.md`.
  `r2t2_llama/` is an alternate decode route and is not ported.
- `.references/Qwen3-ASR` at `7c6daf7`: `qwen_asr/inference/qwen3_asr.py`
  (`init_streaming_state`, `streaming_transcribe`, `finish_streaming_transcribe`,
  `_build_text_prompt`), `examples/example_qwen3_asr_vllm_streaming.py`,
  `modeling_qwen3_asr.py` `_get_feat_extract_output_lengths`.

Conversion is done. Do not repeat it before slices 1–4.

- Upstream: `models/netease/confucius4_r2t2/original/` from
  `netease-youdao/Confucius4-R2T2`. One `model.safetensors`, 4,076,191,640
  bytes, 707 BF16 tensors, no stored `lm_head`. `base_model` is
  `Qwen/Qwen3-ASR-1.7B`. The Hugging Face model card lists the license as
  `other`. Local `config.json` has no `license` field.
- Runtime package: `models/netease/confucius4_r2t2/mlx-bf16/`, written by
  `scripts/convert/confucius4_r2t2.py`. `config.json` has
  `mlx_speech.family = confucius4_r2t2`. Conv2d weights are MLX layout.
  `text_decoder.lm_head.weight` is not in the file.
  `allowed_model_only_keys` treats that key as generated when
  `tie_word_embeddings` is true. A probe loaded this package with no
  unexpected keys, no missing keys, and no shape mismatches.
- Community copies (`mlx-community`, GGUF) are not this package.

The original `config.json` and `models/qwen3_asr_1_7b/mlx-int8/config.json`
have the same keys. The only value difference is `dtype`: JSON `null` on
R2T2, the string `"None"` on that older int8 package. Not an architecture
difference. `|` is not in R2T2 `additional_special_tokens`.
`preprocessor_config.json` is Whisper, 16 kHz, 128 mels, `n_fft=400`,
`hop_length=160`. The language list in the original config is the 30 names
already in `support_languages` (Chinese and English through Macedonian).

Do not claim a parameter-delta size. The next work is the streaming state
machine, then the adapter. Weights stay out of git.

## 1. What the reference streaming state machine does

`R2T2ASRModel` subclasses `Qwen3ASRModel` and does not define a new module
graph. `transcribe` is `super().transcribe`. Streaming is a prompt-prefix loop
around a full re-encode of the audio seen so far. vLLM is only the runner.
There is no cache-aware encoder and no cross-chunk KV cache: each step sends
`{"prompt": prompt_raw + prefix, "multi_modal_data": {"audio": [audio_accum]}}`.

### State

`ASRStreamingState` in `r2t2/r2t2_asr.py` (`init_streaming_state`):

| Field | Role |
| --- | --- |
| `chunk_size_sec` / `chunk_size_samples` | Decode quantum at 16 kHz. `max(1, round(sec * 16000))`. |
| `buffer` | PCM not yet a full chunk. |
| `audio_accum` | All PCM consumed so far, no right-pad. |
| `chunk_id` | Chunks that completed a decode and were not discarded. |
| `prompt_raw` | Chat template plus optional `language {Name}<asr_text>`. |
| `force_language` | One hint for the whole stream. `None` means the model emits the language tag. |
| `_raw_decoded` | Working string used for the next prefix. Not the public committed text. |
| `text` / `language` | Latest full hypothesis after parse. |
| `unfixed_chunk_num` | While `chunk_id` is below this, the next prompt prefix is empty. |
| `unfixed_token_num` | Otherwise drop this many trailing tokens before the string is used as a prefix, and again when forming `fixed_text`. |
| `chunk_text`, `last_fixed_text`, `_first_chunk_discarded` | Used only by `streaming_transcribe_no_reset`. |

Class defaults copied from Qwen are `unfixed_chunk_num=2`,
`unfixed_token_num=5`, `chunk_size_sec=2.0`. Those are not the published R2T2
recipe. `example.py`, `README.md`’s streaming snippet, and `ws_server.py`
(`CHUNK_ASR_SECONDS`, `LOOKAHEAD_MS`, `UNFIX_TOKEN_NUM`) all pass
`unfixed_chunk_num=0`, `unfixed_token_num=1`, `chunk_size_sec=0.16`.

Language is fixed at `init_streaming_state`. Nothing in the stream API changes
it mid-utterance. `None` (ws header `"zhen"` maps to `None`) lets the model
emit `language …<asr_text>`. A Chinese hint is how their code-switch demo is
run; mixed Chinese/English is still one hint for the whole stream.

### Per-chunk decode (`streaming_transcribe`)

1. Accept mono PCM. `int16` is scaled by `1/32768`. Append to `buffer`.
2. While `buffer` has at least `chunk_size_samples`, cut one chunk, append it
   to `audio_accum`.
3. Prefix:
   - `chunk_id < unfixed_chunk_num` → `""`.
   - else tokenize `_raw_decoded` (after a `|` cut), drop the last `k` tokens,
     detokenize. If the detokenized string contains U+FFFD, increase `k` until
     it does not, or the prefix is empty.
   - `k = unfixed_token_num`, except `rollback_punctuation=True` and the
     stripped string ends in `，。！？、；：.!?;:` → `k = 0`. That prefix set
     has no ASCII comma. An ASCII comma must not set prefix `k` to 0.
   - Cut the prefix on the first `|`.
4. Generate with `temperature=0`, `max_tokens=max_new_tokens` when that argument
   is set, else the model’s sampling params. `skip_special_tokens=True`.
5. Post-process the new text only: punctuation by the preceding character
   (`_normalize_punct_by_context`), drop U+FFFD. Then
   `_raw_decoded = prefix + gen_text`.
6. If `force_language` is `None`, the language used for the CJK-space decision
   comes from `parse_language_output` on the raw string. No tag means language
   `""`. That probe does not run `detect_and_fix_repetitions`. Do not call the
   shipped MLX `parse_asr_output` here: with no `<asr_text>` it can still
   return a language and turn CJK space deletion on.
7. If the forced or detected language is Chinese, delete spaces that sit
   between two CJK characters (`U+4E00`–`U+9FFF`) on `_raw_decoded`.
8. Then parse with the Qwen `parse_asr_output` behavior: strip, run
   `detect_and_fix_repetitions` (threshold 20), and return `("", text)` when
   `<asr_text>` is absent. If `<asr_text>` is present, rewrite `_raw_decoded`
   to `meta + "<asr_text>" + parsed_text`. Otherwise `_raw_decoded = parsed_text`.
   The no-tag `continue` looks at the raw string still containing `<asr_text>`
   after that rewrite, not at the parser’s language. Cut on `|` again.
   Keep MLX `mlx_speech.models.qwen3_asr.processor.parse_asr_output` on the
   offline Qwen path only.
9. `fixed_text` is `_raw_decoded` tokenized, dropping the last `k` tokens, with
   the same U+FFFD loop. Extra cases for `k`: trailing punctuation when
   `rollback_punctuation` is set, using `，。！？、；：,.!?;:` (ASCII comma
   included, unlike the prefix set); `k = 0` when `<asr_text>` is present and
   the text after it is empty. Strip the tag off `fixed_text` and cut on `|`.
10. If there is no `<asr_text>` and `force_language` is `None`: set `state.text`
    and `fixed_text` to `""` and `continue` **without** incrementing `chunk_id`.
    The audio is already in `audio_accum`, and `_raw_decoded` was already
    replaced. The next chunk therefore sees that string.
11. Otherwise store `language`, `text` (hypothesis, cut on `|`), increment
    `chunk_id`.

Return value is `(state.text, fixed_text)`, not the state object. `state.text`
is the full current hypothesis. `fixed_text` is that hypothesis minus the
unfixed tail. The hypothesis is allowed to change. The committed view is
`fixed_text`.

`|` is not in `additional_special_tokens` on the published tokenizer. It is a
decoded-text cut. Do not add a special token for it.

### When a prefix is committed

Two different layers:

- **Prompt prefix.** The rolled-back token prefix is concatenated onto
  `prompt_raw` and is not regenerated. That is the model-side “do not rewrite
  the stable prefix” mechanism (their LSP training). It is not a hard equality
  check. Post-process (punctuation, CJK spaces, parse, `|`) can still change
  characters inside the stored string before the next prefix is cut.
- **Caller commit.** `example.py` prints the second return value and replaces
  its display string only when `len(text)` grows. That can still replace a
  changed prefix if the new string is longer. `ws_server.py`
  `/asr_stream_api_v0` does the real append-only lock:

  ```text
  if len(fixed_asr_text) > len(last_fixed_asr_text):
      delta = fixed_asr_text[len(last_fixed_asr_text):]
      last_fixed_asr_text = last_fixed_asr_text + delta
  else:
      delta = ""
  ```

  The old characters stay even if the new hypothesis disagrees in the prefix.
  The slice is by Python code point, not by token. README `/asr_stream_api_v1`
  says each message’s `text` is the incremental chunk and the client
  concatenates. v1 of that socket uses `streaming_transcribe_no_reset` and the
  same length lock.

Port the ws_server lock as the public committed string. Do not port example.py’s
full-string replace.

`finish_streaming_transcribe` is its own post-process. Do not copy the
per-chunk body, and do not copy parent Qwen finish (that parent has none of
the punctuation, U+FFFD, or `|` steps).

- Empty `buffer`: return `state.text` and do not generate. The server length
  lock still runs on that full hypothesis, so the token per-chunk `fixed_text`
  was holding back can commit with no decode.
- Non-empty: append the tail with no pad. Prefix is `""` while
  `chunk_id < unfixed_chunk_num`; otherwise
  `end_idx = max(1, len(ids) - unfixed_token_num)` with no U+FFFD loop, then
  `prefix.split("|")[0]`. There is no pre-encode `|` cut of `_raw_decoded`,
  no `rollback_punctuation`, and no unfixed-token `fixed_text`.
- `gen_text` still goes through `_normalize_punct_by_context` and U+FFFD
  deletion. `_raw_decoded = (prefix + gen_text).split("|")[0]`, then
  `parse_asr_output`. No `parse_language_output`, no CJK space deletion, no
  `<asr_text>` rewrite, no “no tag ⇒ skip `chunk_id`” branch.
- Return `state.text` (`txt.split("|")[0]`), not a rolled-back string.
  `chunk_id` increments. The server EOS path then applies the length lock to
  that full string.

### Lookahead and the token budget live in the caller

`streaming_transcribe` does not know about lookahead. `example.py`
`run_streaming` and `ws_server.py` do:

- First window is `step + lookahead` (defaults 160 ms + 160 ms = 320 ms). They
  assign `state.chunk_size_sec` / `chunk_size_samples` to that length so the
  first decode waits for it.
- Later windows are `step` only (160 ms), and they assign the chunk size back.
- A short tail is left in the buffer and handled by `finish_streaming_transcribe`.
  `example.py` also breaks the file loop when `seg` is shorter than the nominal
  chunk, then calls finish.

`max_new_tokens` is not inside the model class. Both callers use 1280 samples
as one budget unit (80 ms at 16 kHz):

```text
first = max(1, (step_samples + lookahead_samples) // 1280)
floor = min(32, max(4, 2 * (step_samples // 1280)))
```

At the published 160 ms step, steady budget starts at 2 and the floor is 4.
The first call’s budget is 4. Port the v0 server rule, once per consumed
chunk (each `streaming_transcribe` call is one window). A multi-chunk `feed`
must update the budget and the length lock on every chunk, not only the last.

`split_text_to_tokens` strips punctuation; CJK is one character and Latin is
a word. `is_last_token_chinese` is true when any character of that last token
is in `U+4E00`–`U+9FFF`. A trailing punctuation mark after a CJK character is
still CJK. `total_new_asr_tokens` grows only when the committed text grows.

- Growth: append `split_text_to_tokens(delta)`, then
  `max_new_tokens = max(1, step // 1280)`.
- Stall and the last token is not CJK: `max_new_tokens += 0.5`.
- Stall and the last token is CJK: reset to `max(1, step // 1280)`. Do not
  add `0.5`.
- After either stall branch, if that last token is CJK, double, then
  `min(floor, …)`.
- The generate call gets `int(max_new_tokens)`. The float is kept across calls.

At 80 ms, `2 * base` is 2 and the floor is 4, so “add 0.5 then double” and
“reset then double” disagree. At 160 ms the floor hides that difference.
`finalize` does not use the running budget. It uses the first-window cap
`max(1, (step + lookahead) // 1280)`.

`example.py` never appends to `total_new_asr_tokens`, so its CJK branch is
dead and a stall adds `1` instead of `0.5`. Port the server rule.

Why 1280 matches the tower for a short remainder: mel hop is 160
(`preprocessor_config.json` and `Qwen3ASRFeatureExtractor`), so 1280 samples
is 8 mel frames. `_get_feat_extract_output_lengths` in both
`modeling_qwen3_asr.py` and our `feature_extraction.py` maps 8 frames to 1
audio token and 16 frames (160 ms) to 2. The caller budget is
`new_samples // 1280`, which is not the same integer as the cumulative length
formula. Two seconds of mel is 200 frames → `(200 // 100) * 13 = 26` audio
tokens, while `32000 // 1280 = 25`. Keep the caller formula for the decode cap.
Use `_get_feat_extract_output_lengths` only for how many `<|audio_pad|>` tokens
the prompt needs. The `% 100` and `* 13` in that function are hardcoded
upstream. For this checkpoint `n_window * 2 = 100`, so it matches the conv
chunk width. Do not generalize it in this port.

### How this differs from Qwen3-ASR streaming

Qwen’s own streaming is `Qwen3ASRModel.streaming_transcribe` in
`qwen_asr/inference/qwen3_asr.py`, demonstrated by
`examples/example_qwen3_asr_vllm_streaming.py`:

| | Qwen3-ASR published streaming | R2T2 published streaming |
| --- | --- | --- |
| Chunk | `chunk_size_sec=2.0`. The example pushes 500–4000 ms steps into that 2 s buffer. | 160 ms step, range stated as 80 ms–2 s. |
| Unfixed prefix | `unfixed_chunk_num=2`, `unfixed_token_num=5`. First two chunks prompt with an empty prefix, then drop 5 tokens. The table label `2s/u2/t5` is this. | `unfixed_chunk_num=0`, `unfixed_token_num=1`. Prefix is used from the first completed chunk, one token held back. |
| What the caller shows | `state.text` after every step: the full re-parsed hypothesis. Previously shown words can change. README marks this pseudo-streaming. | The second return value, then an append-only length lock. Previously emitted characters stay. |
| Lookahead | None. | First window is step + 160 ms. Later windows are step. |
| Decode cap | One `SamplingParams.max_tokens` for the whole LLM (example sets 32). | Per-call cap, about one new text token per 80 ms of new audio, with the floor above. |
| Text cleanup | None beyond `parse_asr_output`. | `\|` cut, punctuation-by-script, U+FFFD drop, CJK intra-character space deletion, U+FFFD-safe token rollback. |
| Return | The state object. | `(hypothesis, fixed_text)` from the chunk method; the finish method returns `state.text`. |
| Long audio | Re-feed from the start until the context window dies. | Same on `streaming_transcribe`. A second method, `streaming_transcribe_no_reset`, drops the oldest 8 s once `audio_accum` passes 16 s and drops a matching run of per-chunk text deltas. The discard sizes are hardcoded to a 320 ms first chunk and 160 ms later chunks (`5120` and `2560` samples). Not the first port. |

Forcing the original Qwen weights through a 160 ms chunk is a different
experiment from this state machine. R2T2’s accuracy at 160 ms comes from the
finetune plus `unfixed_token_num=1`, `unfixed_chunk_num=0`, the first-window
lookahead, and the append-only commit. Do not implement R2T2 by calling the
existing offline `generate` on growing slices with Qwen’s `u2/t5` defaults.

## 2. What the MLX Qwen3-ASR path already does

`docs/qwen3-asr.md` and `generation/qwen3_asr.py`: v0 is one-shot offline ASR.
`Qwen3ASRAdapter.generate` calls `Qwen3ASRTranscriber.transcribe` and returns
`ASROutput(text, language)`. There is no `stream_session` on that adapter.

Already matching this checkpoint, and safe to reuse without a second copy of
the graph:

- `Qwen3ASRFeatureExtractor`: 16 kHz, 128 mels, `n_fft=400`, `hop_length=160`.
  The R2T2 `preprocessor_config.json` on disk uses those same numbers.
- `Qwen3ASRAudioEncoder`: three stride-2 convs, windowed attention from
  `n_window` / `n_window_infer`. Both published configs set `n_window=50`,
  `n_window_infer=800`, `conv_chunksize=500`, `d_model=1024`, 24 layers,
  16 heads, ffn 4096, `downsample_hidden_size=480`, `output_dim=2048`,
  `max_source_positions=1500`, 128 mel bins.
- `Qwen3ASRTextForCausalLM`: GQA, Q/K RMSNorm, `rope_theta`, tied
  `lm_head` when `tie_word_embeddings` is in config extra. Text side of both
  configs: hidden 2048, intermediate 6144, 28 layers, 16 heads, 8 KV heads,
  `head_dim` 128, vocab 151936, `max_position_embeddings` 65536, `rope_theta`
  1_000_000, `rms_norm_eps` 1e-6. `rope_scaling` (`mrope_section` `[24, 20, 20]`,
  interleaved) is stored in `extra` and is already not a separate code path on
  the shipped Qwen3-ASR runtime. Do not add an mRoPE implementation just for
  R2T2.
- `Qwen3ASRProcessor.build_prompt`: the same chat string as both
  `chat_template.json` files, then `<|audio_pad|>` expanded to the audio-token
  count, then `language {Name}<asr_text>` when a language is forced.
  Audio token ids match: pad `151676`, start `151669`, end `151670`.
- Offline `parse_asr_output`, `resolve_language`, `SUPPORTED_LANGUAGES` on the
  Qwen path. The streaming loop does not call the MLX parser. The R2T2
  `support_languages` list is the same 30 names.
- `sanitize_key` already mapped all 707 `thinker.audio_tower.*` and
  `thinker.model.*` keys. Nothing was skipped. There is no `thinker.lm_head.*`
  to map. Do not add a second mapping table.
- Greedy prefill plus `decode_step` KV cache **inside one generate call**.
  `generation_config.json` is `do_sample: false` with a tiny temperature, and
  `eos_token_id` `[151643, 151645]`. `resolve_eos_token_ids` already reads that
  file.
- Int8 / bf16 quantization hooks. Not the first streaming slice.

Must be new:

- The streaming state machine (buffer, lookahead first window, unfixed-token
  prefix, `|` / punctuation / CJK / U+FFFD cleanup, append-only commit, and
  the separate finish post-process).
- A streaming parse: `parse_language_output` for the CJK-space decision, then
  Qwen `parse_asr_output` including `detect_and_fix_repetitions`. Not the MLX
  `parse_asr_output`.
- A per-step generate that builds `prompt_raw + prefix` with the **current**
  `audio_accum` length and runs a short greedy decode (`max_new_tokens` from
  the budget). The existing `transcribe()` always starts from an empty
  hypothesis and uses a large default cap (448). Do not call it in a loop and
  pretend that is R2T2.
- Cross-chunk KV reuse is incorrect for v1. Each new window grows the audio-pad
  span, so the new prompt is not a token prefix of the previous prompt. Encode
  the full `audio_accum`, splice audio embeddings into `inputs_embeds`, and
  prefill from scratch every chunk. Decode only the new tokens. The text KV
  cache is only for those tokens inside that chunk.
- Registry marker and adapter, so this directory is not the offline Qwen path.

Offline `generate` stays `Qwen3ASRTranscriber.transcribe`, including the MLX
parser. `R2T2ASRModel.transcribe` is `super().transcribe` only. Do not run
`|`, punctuation, CJK-space, or U+FFFD cleanup on offline `generate`, and do
not put that cleanup on `qwen3-asr-1.7b`.

`ASRStreamSession.feed` / `finalize` return new token ids and only Nemotron
implements them (`asr/_adapter.py`, `models/nemotron_asr/streaming.py`). R2T2’s
contract is text plus a committed prefix. Do not implement that protocol by
returning token ids.

## 3. Same graph, same `model_type`, different dispatch

R2T2 subclasses `Qwen3ASRModel` and defines no modules, so the graph is the
Qwen3-ASR graph. The converted package is not a Qwen directory: code for the
state machine goes in `src/mlx_speech/models/confucius4_r2t2/`, and the
adapter in `src/mlx_speech/asr/_adapters/confucius4_r2t2.py`. The shared
graph stays in `models/qwen3_asr/`.

The original config is on disk and matches the Qwen 1.7B tower in §2.
`mrope_section` `[24, 20, 20]` is in text-config `extra`.
`Qwen3ASRTextRotaryEmbedding` uses `rope_theta` only. Do not add an mRoPE
path for this port. Class defaults in `configuration_qwen3_asr.py`
(`d_model=1280`, 32 encoder layers, `output_dim=3584`, `n_window=100`) are
not this checkpoint. `Qwen3ASRConfig.from_dict` reads the file.

There is no architectural field to branch on. `_resolve_asr_family` in
`asr/_registry.py` maps every `model_type == "qwen3_asr"` directory to the
offline Qwen adapter. A raw R2T2 folder would take that path today.

Dispatch rule:

1. Read `config.json`.
2. If `mlx_speech.family == "confucius4_r2t2"`, return a new family `r2t2`.
   The converter writes this marker. It is not in the upstream file.
   Leave `model_type` as `qwen3_asr` so `Qwen3ASRConfig.from_dict` still
   accepts the package. Unknown keys already land in `extra`.
3. Else if `model_type == "qwen3_asr"`, keep family `qwen3`. Upstream
   `netease-youdao/Confucius4-R2T2` without the marker stays on the offline
   Qwen loader. That is deliberate: guessing from README text or the repo id
   string would also steal any untouched copy of the parent weights.
4. Do not point a runtime alias at `netease-youdao/Confucius4-R2T2`. That repo
   is conversion input (one `model.safetensors` plus tokenizer files). Runtime
   load is a local converted directory. Add a hub alias only when a real MLX
   repo id exists. Do not invent one in `_hub.py` before that.

On the ASR path, `family_hint` on `_ModelAlias` is catalog metadata.
`asr.load` does not read it. `_resolve_asr_family` and the `config.json`
marker select the loader. TTS `load()` is a different rule:
`tts/__init__.py` sets `hint_family = _TTS_MODELS[...].family_hint` and then
`family = hint_family or _resolve_tts_family(model_dir)`, so the hint wins
when the alias is known. Do not carry the ASR sentence over to TTS.

First time weights are on disk, list `model.safetensors` keys and run them
through `sanitize_key` before constructing the module. Shapes must match the
module built from that same `config.json`. A mismatch is a failed conversion,
not a reason to fork the encoder.

## 4. Boundary

Reuse the Qwen3-ASR graph by calling it. Do not fold R2T2 into that package,
and do not add a new framework around it.

Shared code that may change:

- `asr/_registry.py`: one check, `mlx_speech.family == "confucius4_r2t2"`,
  before the existing `model_type == "qwen3_asr"` return. The qwen3 branch
  stays as it is.
- `asr/__init__.py`: one new branch that constructs the Confucius adapter.
  Same shape as the cohere, granite, qwen3, and nemotron branches.

Do not edit `Qwen3ASRTranscriber.transcribe`, MLX `parse_asr_output`,
`Qwen3ASRAdapter.generate`, or Nemotron's `ASRStreamSession`. The tied
`lm_head` rule already in `qwen3_asr/checkpoint.py` is general: any config
with `tie_word_embeddings` may omit that tensor. It is not an R2T2 branch.
Load acceptance is `is_loadable_match(report, config)` /
`unexpected_model_only_keys(report, config)`. `AlignmentReport` does not
carry a config-blind `unexpected_model_only` or `is_loadable_match`.

Do not add a `TextStreamingASRModel` protocol. `stream_session` lives on
`Confucius4R2T2Adapter` only. Callers that have that object use it. Nothing
else has to grow a method.

The adapter imports the Qwen model and processor and injects
`llm_generate(prefix, audio, max_new_tokens)`. The streaming state machine
does not import that graph, does not subclass it, and does not live under
`models/qwen3_asr/`. It only calls the injected callable, plus a
`tokenize` / `detokenize` pair for prefix rollback. Slice 1's fake codec
never runs an MLX forward.

## 5. Public surface

New alias name, when a converted package exists: `confucius4-r2t2`. Not a
rename of `qwen3-asr-1.7b`. Offline Qwen aliases stay offline and keep their
current post-process.

```python
from mlx_speech.asr import load

model = load("/path/to/converted-r2t2")  # alias later, same call
result = model.generate(audio, language="Chinese")  # ASROutput, one-shot
session = model.stream_session(
    language="Chinese",          # "Chinese", "English", or None
    chunk_ms=160,                # 80..2000
    lookahead_ms=160,            # first window only
    unfixed_token_num=1,
    context="",                  # hotword / system text, optional
)
update = session.feed(pcm16k)    # may be called with any length, including short
result = session.finalize()      # ASROutput
```

`language` is read only at `stream_session` / `generate`. No setter to switch
it later. `None` is auto language tags. For mixed Chinese/English, pass
`"Chinese"`, matching their demo. Reject any other string with the existing
`resolve_language` list.

`feed` may return an empty delta when the buffer is short of a chunk. It does
not return token ids.

```text
R2T2StreamUpdate
  delta: str         # newly committed characters (possibly "")
  committed: str     # append-only transcript so far
  hypothesis: str    # state.text; the unfixed tail may still change
  language: str
```

`finalize` flushes the tail with the finish prefix rule, applies the same
length lock, and returns `ASROutput(text=committed, language=...)`. The
end state is that transcript, not tokens.

This does not satisfy `ASRStreamSession`, and it must not change that
protocol. Do not add `att_context_size` here.

Defaults are the published recipe, not the inherited class defaults:
`chunk_ms=160`, `lookahead_ms=160`, `unfixed_token_num=1`,
`unfixed_chunk_num=0`, `rollback_punctuation=False`. Reject `chunk_ms` outside
`[80, 2000]`. `rollback_punctuation=True` is the ws_server “fast” mode
(`header mode != "slow"`). Leave it as a keyword defaulting off, covered by a
unit test, not by the CLI.

CLI, smallest addition to `asr/generate.py`: optional `--chunk-ms` means
stream the file and print the final committed text. Omitted, `generate` stays
offline. Do not grow a websocket server.

`context` maps to the system turn already supported by `build_prompt`. Cap is
not required in v1; the server’s 4000-character check can wait.

## 6. Implementation slices

Each slice is one change, tested with `pytest tests/unit/` and no PyTorch.
Slices 1–3 need no checkpoint. Do not start the next slice until the current
one’s tests pass.

### Slice 1 — pure state machine (done)

`src/mlx_speech/models/confucius4_r2t2/streaming.py` and
`tests/unit/test_r2t2_streaming.py`. The Qwen3-ASR package keeps the shared
graph only. Numpy PCM buffers. The caller injects
`llm_generate(prefix: str, audio: np.ndarray, max_new_tokens: int) -> str`
and a `tokenize(text) -> list[int]` / `detokenize(ids) -> str` pair.
`tokenize` / `detokenize` are the tokenizer round-trip. `llm_generate` is
the one-step completion. No `mlx` forward, and this module does not import
the Qwen model. It does import `resolve_language` to check the language name.

Accepted difference from `ws_server.py`: `_committed_tokens` is not capped at
10. Only the last token is read, so the budget matches. Do not add
`MAX_TOKENS` unless a later slice reads more than that tail.

The checklist below is what that module already covers. Do not reimplement it.

Cover with `tests/unit/test_r2t2_streaming.py` and a fake codec
(`tokenize`/`detokenize` that round-trip a test alphabet, including a token
that detokenizes to U+FFFD):

- Bytes in, no `llm_generate` until the first window is `chunk + lookahead`
  samples. Next `llm_generate` waits for `chunk` samples only. Short `feed`
  returns an empty delta and does not call `llm_generate`.
- `unfixed_chunk_num=0` puts the rolled-back hypothesis into the next prefix.
  `unfixed_token_num=1` holds back one token in `fixed_text` but the prefix
  passed to `llm_generate` is also missing that token.
- U+FFFD in the detokenized prefix drops one extra token. Empty `<asr_text>`
  body uses `k=0`. `rollback_punctuation=True` uses `k=0` only when the
  stripped text ends in the prefix set `，。！？、；：.!?;:` (no ASCII comma).
  `fixed_text` uses `，。！？、；：,.!?;:` (ASCII comma included).
- No `<asr_text>` and `language is None`: public text cleared, `chunk_id` not
  incremented, audio kept.
- Append-only lock: a longer hypothesis whose head differs still emits only
  `new[len(old):]` and the stored committed string keeps the old head. A
  shorter or equal hypothesis emits `""`.
- `|` cuts hypothesis, prefix, and fixed text. Punctuation follows the
  previous non-space character (CJK → Chinese marks, ASCII alnum or quote →
  English marks). Chinese language deletes spaces between CJK and keeps the
  space in `hello 世界`.
- Finish is the separate post-process in §1, not the per-chunk body. A
  non-empty tail uses `end_idx = max(1, len - k)` with no U+FFFD loop, no
  `rollback_punctuation`, and no unfixed-token `fixed_text`. An empty tail
  does not call `llm_generate` and returns `state.text`, so the length lock can
  commit the held-back token. `finalize`’s decode cap is the first-window
  cap `(step + lookahead) // 1280`, not the running budget.
- Budget, once per consumed chunk: 160 ms / 160 ms lookahead → first cap 4,
  floor 4, steady base 2. Growth resets to `max(1, step // 1280)`. A stall
  adds 0.5 only when the last `split_text_to_tokens` token is not CJK. A CJK
  stall resets to `step // 1280`, does not add 0.5, then doubles, then the
  floor caps it. Script the server rule, not example.py’s unused list.
- One language for the session. A second `feed` cannot change it.

### Slice 2 — registry, same change as the adapter

Do not land this alone. Put it in the same change as slice 4. A directory
the registry already classifies as `r2t2` makes `asr.load()` raise
`Unsupported ASR family: 'r2t2'` until that branch exists.

When it lands, in `asr/_registry.py` and `tests/unit/test_asr_registry.py`:

- The marker check runs before the `model_type == "qwen3_asr"` return.
  `{"model_type": "qwen3_asr"}` still resolves `qwen3`.
- The same file plus `{"mlx_speech": {"family": "confucius4_r2t2"}}` resolves
  `r2t2`.
- Unknown `model_type` still lists `cohere_asr, granite_speech, qwen3_asr,
  nemotron_asr`. Do not rewrite that string to advertise `r2t2`.

The duplicate load gate is already gone. Use
`is_loadable_match(report, config)` only. Do not put
`unexpected_model_only` back on `AlignmentReport`.

### Slice 3 — prompt and pad count, inside the adapter tests

Do not add a new prompt type. The adapter calls
`Qwen3ASRProcessor.build_prompt` and appends the state-machine prefix after
the forced `language …<asr_text>` suffix. Unit-test that string with the
template the processor already hardcodes. Do not load `models/`.

Also assert `_get_feat_extract_output_lengths` at 1280, 2560, and 5120
samples of mel frames (1, 2, and 4 audio tokens). The pad count follows that
formula, not `samples // 1280`.

### Slice 4 — adapter and one-step MLX decode

`asr/_adapters/confucius4_r2t2.py` holds a `Qwen3ASRTranscriber` built by the existing
`from_dir`. Offline `generate` calls `transcribe` and stops. No streaming
cleanup on that path. Streaming `feed` builds the prompt for `len(audio_accum)`,
embeds the ids, splices audio with `replace_audio_embeddings` /
`prepare_inputs_embeds`, prefills those `inputs_embeds`, and greedy-decodes
only the new tokens until `max_new_tokens` or EOS. Decode that completion with
`skip_special_tokens=True` and pass that string into the slice-1 machine.
Decoding the whole prompt double-counts `prefix`.

Unit-test the adapter with a fake transcriber injected into the session, still
without a checkpoint: one scripted string in, committed text out, `ASROutput`
on `finalize`. Do not mock `mlx` module internals.

Real weights are the next slice, not this one.

### Slice 5 — conversion (done)

`scripts/convert/confucius4_r2t2.py` wrote
`models/netease/confucius4_r2t2/mlx-bf16/` from `original/`. 707 tensors
renamed, three conv weights transposed, family marker patched into the copied
`config.json`. The tied `lm_head` is not stored. Do not reconvert unless the
upstream file changes. No int8 and no 4-bit.

### Slice 6 — runtime behavior with weights

Gated by the local converted directory, same skip rule as other checkpoint
tests. Not part of the default `pytest tests/unit/` run. Put it in
`tests/runtime/` or `tests/checkpoint/` and do not run it as part of closing
slices 1–4.

- Offline `generate` on a short fixture returns non-empty `text`.
- Streaming the same file at 160 ms + 160 ms lookahead: each `feed`'s
  `committed` only grows. `finalize().text` starts with that last `committed`
  and may append the token `unfixed_token_num` was holding back. Do not
  require equality with the pre-finalize snapshot. For the local clip that
  extra character is `。`.
- `language="Chinese"` on a mixed clip is one session-long hint.
- A raw Qwen3-ASR directory with no marker still loads as family `qwen3`.

### Slice 7 — docs and alias

`docs/confucius4-r2t2.md`: streaming defaults, the single language hint,
append-only vs hypothesis, conversion command, and the NetEase weight license.
Do not document llama.cpp or the websocket server as supported.

Add `confucius4-r2t2` under `_ASR_MODELS` only with a real MLX repo id.
Until then, path load is the public entry. Do not retarget `qwen3-asr-1.7b`.

### Explicitly later, not in the slices above

- `streaming_transcribe_no_reset`: 16 s cap, drop 8 s, per-chunk text list.
  The hardcoded 320/160 ms discard math is wrong if chunk size is not 160 ms.
  Do this only after the full-refeed path is correct, and derive discard counts
  from chunks actually committed.
- VAD end-of-speech reset, hallucination repeat reset, 60 s / 90 s hard reset
  (`ws_server.py`). Those reset the state machine; they are not required for
  the library call.
- Cross-chunk encoder or text KV cache. Only consider it after a numerical
  check shows the full re-feed is the baseline. A tail chunk changes conv
  padding and attention block boundaries (`n_window * 2`), so a naive cache is
  wrong.

## 7. Non-goals

- `r2t2_llama/`: GGUF, `llama-server`, hybrid vLLM-encoder + llama.cpp decoder,
  and the prebuilt `libllama` / `libggml` binaries.
- vLLM, `ws_server.py`, `ws_client.py`, FireRedVAD, and any streaming server.
- Replacing, aliasing over, or changing the behavior of `qwen3-asr-1.7b`.
- Training, LSP data construction, forced alignment, timestamps.
- int8 or 4-bit packages. The runtime artifact is `mlx-bf16` only.
- A second language hint mid-utterance, batch streaming, or word-level
  revision of already committed text.
- Vendoring upstream code or importing `qwen_asr` / `r2t2` at runtime.

## Test command for the port

After slices 1–4 (no weights):

```bash
pytest tests/unit/
```

After slice 6, only when the converted checkpoint is present:

```bash
pytest tests/unit/ tests/checkpoint/ tests/runtime/
```

Do not add a torch fallback if a kernel mismatches. Fix the MLX path or mark
the checkpoint test skipped with the missing file.
