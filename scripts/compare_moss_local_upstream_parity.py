#!/usr/bin/env python3
"""Compare MLX MossTTSLocal parity against the upstream torch reference.

This script is for local validation only. It is intentionally outside the
runtime package because it depends on the upstream reference checkout plus a
torch-capable debug environment.
"""

from __future__ import annotations

import argparse
import contextlib
import io
from pathlib import Path

import mlx.core as mx
import numpy as np
import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-dir",
        default="models/openmoss/moss_tts_local/original",
        help="Local original MossTTSLocal checkpoint directory.",
    )
    parser.add_argument(
        "--codec-dir",
        default="models/openmoss/moss_audio_tokenizer/original",
        help="Local original Moss audio tokenizer checkpoint directory.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=16,
        help="Number of greedy rollout rows to compare.",
    )
    parser.add_argument(
        "--text",
        default="Hello from MLX. This is a regenerated speech sample for listening.",
        help="Direct-generation prompt text.",
    )
    return parser.parse_args()


def _manual_upstream_greedy_rollout(
    model,
    *,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    max_new_tokens: int,
):
    with torch.no_grad():
        sequences = input_ids.clone()
        current_attention_mask = attention_mask.clone()
        for _ in range(max_new_tokens):
            outputs = model.model(
                input_ids=sequences,
                attention_mask=current_attention_mask,
                n_vq_for_inference=model.config.n_vq,
                output_hidden_states=True,
            )
            global_hidden = outputs.hidden_states[-1][:, -1, :]
            current_local_input = model.speech_embedding_to_local_mlp(global_hidden)
            local_inputs = torch.zeros(
                sequences.shape[0],
                0,
                model.local_transformer_config.hidden_size,
                dtype=global_hidden.dtype,
                device=global_hidden.device,
            )
            next_tokens = []
            for layer_index in range(model.channels):
                local_inputs = torch.cat([local_inputs, current_local_input.unsqueeze(1)], dim=1)
                local_outputs = model.local_transformer(
                    input_ids=None,
                    attention_mask=None,
                    inputs_embeds=local_inputs,
                )[0]
                local_outputs = model.layer_norm_before_lm_heads[layer_index](
                    model.local_to_speech_embedding_mlps[layer_index](local_outputs)
                )
                logits = model.lm_heads[layer_index](local_outputs[:, -1, :])
                if layer_index != 0:
                    logits[:, model.config.audio_pad_code] = -torch.inf
                token = torch.argmax(logits, dim=-1)
                next_tokens.append(token)
                current_local_input = model.model.embedding_list[layer_index](token)
                current_local_input = model.speech_embedding_to_local_mlp(current_local_input)

            next_row = torch.stack(next_tokens, dim=-1)
            sequences = torch.cat([sequences, next_row[:, None, :]], dim=1)
            current_attention_mask = torch.cat(
                [
                    current_attention_mask,
                    torch.ones(
                        (current_attention_mask.shape[0], 1),
                        dtype=current_attention_mask.dtype,
                        device=current_attention_mask.device,
                    ),
                ],
                dim=1,
            )
            if bool(torch.all(next_row[:, 0] == model.config.audio_end_token_id)):
                break
    return sequences


def _summarize_first_diff(reference_rows: list[list[int]], candidate_rows: list[list[int]]) -> dict[str, object]:
    if reference_rows == candidate_rows:
        return {
            "equal": True,
            "rows": len(candidate_rows),
        }
    for row_index, (reference_row, candidate_row) in enumerate(zip(reference_rows, candidate_rows)):
        if reference_row != candidate_row:
            for column_index, (reference_token, candidate_token) in enumerate(
                zip(reference_row, candidate_row)
            ):
                if reference_token != candidate_token:
                    return {
                        "equal": False,
                        "reference_rows": len(reference_rows),
                        "candidate_rows": len(candidate_rows),
                        "first_diff_row": row_index,
                        "first_diff_col": column_index,
                        "reference_token": reference_token,
                        "candidate_token": candidate_token,
                    }
    return {
        "equal": False,
        "reference_rows": len(reference_rows),
        "candidate_rows": len(candidate_rows),
        "note": "length_only",
    }


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    reference_root = repo_root / ".references" / "MOSS-TTS"
    model_dir = Path(args.model_dir)
    codec_dir = Path(args.codec_dir)
    if not model_dir.is_absolute():
        model_dir = repo_root / model_dir
    if not codec_dir.is_absolute():
        codec_dir = repo_root / codec_dir

    from moss_tts_local.configuration_moss_tts import MossTTSDelayConfig
    from moss_tts_local.modeling_moss_tts import MossTTSDelayModel
    from moss_tts_local.processing_moss_tts import MossTTSDelayProcessor
    from moss_audio_tokenizer.configuration_moss_audio_tokenizer import MossAudioTokenizerConfig
    from moss_audio_tokenizer.modeling_moss_audio_tokenizer import MossAudioTokenizerModel

    from mlx_voice.generation.moss_local import MossTTSLocalGenerationConfig, generate_moss_tts_local
    from mlx_voice.models.moss_audio_tokenizer import load_moss_audio_tokenizer_model
    from mlx_voice.models.moss_local import MossTTSLocalProcessor, load_moss_tts_local_model

    mlx_processor = MossTTSLocalProcessor.from_path(model_dir)
    mlx_batch = mlx_processor(
        [mlx_processor.build_user_message(text=args.text)],
        mode="generation",
    )

    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        tokenizer = __import__(
            "transformers",
            fromlist=["AutoTokenizer"],
        ).AutoTokenizer.from_pretrained(
            "OpenMOSS-Team/MOSS-TTS-Local-Transformer",
            trust_remote_code=True,
        )
        upstream_audio_config = MossAudioTokenizerConfig.from_pretrained(str(codec_dir))
        upstream_audio_tokenizer = MossAudioTokenizerModel.from_pretrained(
            str(codec_dir),
            config=upstream_audio_config,
            torch_dtype=torch.float32,
        )
        upstream_config = MossTTSDelayConfig.from_pretrained(str(model_dir))
        upstream_processor = MossTTSDelayProcessor(
            tokenizer=tokenizer,
            audio_tokenizer=upstream_audio_tokenizer,
            model_config=upstream_config,
        )
        upstream_model = MossTTSDelayModel.from_pretrained(
            str(model_dir),
            config=upstream_config,
            torch_dtype=torch.float32,
        )
    upstream_model.eval()

    upstream_batch = upstream_processor(
        [[upstream_processor.build_user_message(text=args.text)]],
        mode="generation",
    )

    processor_ids_equal = bool(
        np.array_equal(
            np.asarray(upstream_batch["input_ids"].detach().cpu().numpy()),
            np.asarray(mlx_batch.input_ids),
        )
    )
    processor_mask_equal = bool(
        np.array_equal(
            np.asarray(upstream_batch["attention_mask"].detach().cpu().numpy()),
            np.asarray(mlx_batch.attention_mask.astype(mx.int32)),
        )
    )

    input_ids_t = torch.tensor(mlx_batch.input_ids.tolist(), dtype=torch.long)
    attention_mask_t = torch.tensor(
        mlx_batch.attention_mask.astype(mx.int32).tolist(),
        dtype=torch.long,
    )
    upstream_sequences = _manual_upstream_greedy_rollout(
        upstream_model,
        input_ids=input_ids_t,
        attention_mask=attention_mask_t,
        max_new_tokens=args.max_new_tokens,
    ).tolist()[0]

    loaded_mlx_model = load_moss_tts_local_model(
        model_dir,
        prefer_mlx_int8=False,
        strict=True,
    )
    mlx_sequences = generate_moss_tts_local(
        loaded_mlx_model.model,
        mlx_batch.input_ids,
        mlx_batch.attention_mask,
        config=MossTTSLocalGenerationConfig(
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
        ),
    ).sequences.tolist()[0]

    audio_start_token_id = loaded_mlx_model.config.audio_start_token_id
    upstream_start_index = max(
        index for index, row in enumerate(upstream_sequences) if row[0] == audio_start_token_id
    )
    mlx_start_index = max(
        index for index, row in enumerate(mlx_sequences) if row[0] == audio_start_token_id
    )
    upstream_rows = upstream_sequences[upstream_start_index:]
    mlx_rows = mlx_sequences[mlx_start_index:]

    loaded_mlx_codec = load_moss_audio_tokenizer_model(
        codec_dir,
        prefer_mlx_int8=False,
        strict=True,
    )
    with torch.no_grad():
        upstream_out = upstream_model.generate(
            input_ids=input_ids_t,
            attention_mask=attention_mask_t,
            max_new_tokens=max(8, min(args.max_new_tokens, 32)),
        )
    _, upstream_generation = upstream_out[0]
    upstream_messages = upstream_processor.decode(upstream_out)
    upstream_waveform = upstream_messages[0].audio_codes_list[0].detach().cpu().numpy().astype(np.float32)
    generated_audio_rows = upstream_generation[:, 1:]
    non_pad = ~((generated_audio_rows == loaded_mlx_model.config.audio_pad_code).all(dim=1))
    non_pad_indices = torch.nonzero(non_pad).squeeze(1)
    segment = generated_audio_rows[non_pad_indices]
    mlx_codec_input = mx.array(segment.detach().cpu().numpy().T[:, None, :], dtype=mx.int32)
    mlx_decoded = loaded_mlx_codec.model.decode(
        mlx_codec_input,
        num_quantizers=int(mlx_codec_input.shape[0]),
    )
    mlx_waveform = np.asarray(
        mlx_decoded.audio[0, 0, : int(mlx_decoded.audio_lengths[0])],
        dtype=np.float32,
    )
    compare_length = min(len(upstream_waveform), len(mlx_waveform))
    codec_rmse = float(
        np.sqrt(
            np.mean((upstream_waveform[:compare_length] - mlx_waveform[:compare_length]) ** 2)
        )
    )

    print("MossTTSLocal upstream parity")
    print("  reference_root:", reference_root)
    print("  processor.input_ids_equal:", processor_ids_equal)
    print("  processor.attention_mask_equal:", processor_mask_equal)
    print("  runtime.original_weight_dtype:", loaded_mlx_model.model.model.embedding_list[0].weight.dtype)
    print("  rollout:", _summarize_first_diff(upstream_rows, mlx_rows))
    print(
        "  codec:",
        {
            "compare_len": compare_length,
            "rmse": codec_rmse,
        },
    )


if __name__ == "__main__":
    main()
