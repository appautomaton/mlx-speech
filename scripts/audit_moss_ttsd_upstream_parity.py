#!/usr/bin/env python3
"""Audit MLX MossTTSDelay parity against the upstream Torch implementation."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import mlx.core as mx
import numpy as np
import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-dir",
        default="models/openmoss/moss_ttsd/original",
        help="Local TTSD original checkpoint directory.",
    )
    parser.add_argument(
        "--reference-audio",
        default="outputs/clone_eval/github_references/sherlock2_split_1.wav",
        help="Reference WAV used for the continuation audit case.",
    )
    parser.add_argument(
        "--text",
        default="[S1] The game is on, Watson, and we must move quickly. [S1] Watson, we should leave now.",
        help="Continuation text used for packing and generation parity.",
    )
    parser.add_argument(
        "--mode",
        choices=("processor", "model", "generation", "all"),
        default="all",
        help="Which parity slice to audit.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=8,
        help="Greedy rollout steps for generation parity.",
    )
    parser.add_argument(
        "--dtype",
        choices=("float32", "bfloat16"),
        default="float32",
        help="Torch dtype for upstream model loading.",
    )
    return parser.parse_args()


def _torch_dtype(name: str) -> torch.dtype:
    if name == "float32":
        return torch.float32
    if name == "bfloat16":
        return torch.bfloat16
    raise ValueError(f"Unsupported torch dtype: {name}")


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _load_upstream_modules(repo_root: Path):
    upstream_root = repo_root / ".references" / "MOSS-TTS"
    if str(upstream_root) not in sys.path:
        sys.path.insert(0, str(upstream_root))

    from moss_tts_delay.configuration_moss_tts import MossTTSDelayConfig as UpstreamConfig
    from moss_tts_delay.modeling_moss_tts import MossTTSDelayModel as UpstreamModel
    from moss_tts_delay.processing_moss_tts import MossTTSDelayProcessor as UpstreamProcessor

    return UpstreamConfig, UpstreamModel, UpstreamProcessor


def _to_numpy(value) -> np.ndarray:
    if isinstance(value, mx.array):
        return np.array(value.tolist(), dtype=np.float32)
    if isinstance(value, torch.Tensor):
        return value.detach().float().cpu().numpy()
    return np.asarray(value, dtype=np.float32)


def _summarize_diff(reference, candidate) -> dict[str, float]:
    ref = _to_numpy(reference)
    cand = _to_numpy(candidate)
    finite_mask = np.isfinite(ref) & np.isfinite(cand)
    if not np.any(finite_mask):
        return {"l2": 0.0, "max_abs": 0.0, "mean_abs": 0.0}
    ref = ref[finite_mask]
    cand = cand[finite_mask]
    diff = ref - cand
    return {
        "l2": float(np.linalg.norm(diff)),
        "max_abs": float(np.max(np.abs(diff))),
        "mean_abs": float(np.mean(np.abs(diff))),
    }


def _first_row_diff(reference_rows: list[list[int]], candidate_rows: list[list[int]]) -> dict[str, object]:
    if reference_rows == candidate_rows:
        return {"equal": True, "rows": len(candidate_rows)}
    for row_index, (reference_row, candidate_row) in enumerate(zip(reference_rows, candidate_rows)):
        if reference_row != candidate_row:
            return {
                "equal": False,
                "first_diff_row": row_index,
                "reference_row": reference_row,
                "candidate_row": candidate_row,
            }
    return {
        "equal": False,
        "reference_rows": len(reference_rows),
        "candidate_rows": len(candidate_rows),
        "note": "length_only",
    }


def main() -> None:
    args = parse_args()
    repo_root = _repo_root()
    model_dir = Path(args.model_dir)
    reference_audio = Path(args.reference_audio)
    if not model_dir.is_absolute():
        model_dir = repo_root / model_dir
    if not reference_audio.is_absolute():
        reference_audio = repo_root / reference_audio

    from mlx_speech.generation import MossTTSDelayGenerationConfig, generate_moss_tts_delay
    from mlx_speech.models.moss_audio_tokenizer import load_moss_audio_tokenizer_model
    from mlx_speech.models.moss_delay import MossTTSDelayProcessor, load_moss_tts_delay_model
    from mlx_speech.models.moss_local.tokenizer import DEFAULT_MOSS_CHAT_TEMPLATE

    UpstreamConfig, UpstreamModel, UpstreamProcessor = _load_upstream_modules(repo_root)

    mlx_loaded = load_moss_tts_delay_model(model_dir, prefer_mlx_int8=False)
    codec_loaded = load_moss_audio_tokenizer_model()
    mlx_processor = MossTTSDelayProcessor.from_path(model_dir, audio_tokenizer=codec_loaded.model)
    prompt_audio = mlx_processor.encode_audios_from_path(
        [reference_audio],
        n_vq=mlx_loaded.config.n_vq,
    )[0]
    mlx_conversation = [[
        mlx_processor.build_user_message(text=args.text),
        mlx_processor.build_assistant_message(audio_codes_list=[prompt_audio]),
    ]]
    mlx_batch = mlx_processor(mlx_conversation, mode="continuation")

    tokenizer = __import__("transformers", fromlist=["AutoTokenizer"]).AutoTokenizer.from_pretrained(
        str(model_dir),
        trust_remote_code=True,
    )
    tokenizer.chat_template = DEFAULT_MOSS_CHAT_TEMPLATE
    upstream_config = UpstreamConfig.from_pretrained(str(model_dir))
    upstream_processor = UpstreamProcessor(
        tokenizer=tokenizer,
        audio_tokenizer=None,
        model_config=upstream_config,
    )
    upstream_prompt = torch.tensor(prompt_audio.tolist(), dtype=torch.long)
    upstream_conversation = [[
        upstream_processor.build_user_message(text=args.text),
        upstream_processor.build_assistant_message(audio_codes_list=[upstream_prompt]),
    ]]
    upstream_batch = upstream_processor(upstream_conversation, mode="continuation")

    if args.mode in {"processor", "all"}:
        print("== processor ==")
        print(
            {
                "input_ids_equal": mlx_batch.input_ids.tolist() == upstream_batch["input_ids"].tolist(),
                "attention_mask_equal": mlx_batch.attention_mask.astype(mx.int32).tolist()
                == upstream_batch["attention_mask"].tolist(),
                "shape": tuple(int(dim) for dim in mlx_batch.input_ids.shape),
            }
        )

    if args.mode in {"model", "all"}:
        print("== model ==")
        upstream_model = UpstreamModel.from_pretrained(
            str(model_dir),
            config=upstream_config,
            torch_dtype=_torch_dtype(args.dtype),
        )
        upstream_model.eval()
        with torch.no_grad():
            upstream_embeddings = upstream_model._compute_input_embeddings(upstream_batch["input_ids"])
            upstream_output = upstream_model(
                input_ids=upstream_batch["input_ids"],
                attention_mask=upstream_batch["attention_mask"],
            )
        mlx_embeddings = mlx_loaded.model._compute_input_embeddings(mlx_batch.input_ids)
        mlx_output = mlx_loaded.model(
            input_ids=mlx_batch.input_ids,
            attention_mask=mlx_batch.attention_mask,
            output_hidden_states=True,
        )

        print({"embeddings": _summarize_diff(upstream_embeddings, mlx_embeddings)})
        print(
            {
                "last_hidden_state": _summarize_diff(
                    upstream_output.hidden_states[-1],
                    mlx_output.last_hidden_state,
                )
            }
        )
        logit_summaries = {}
        for head_index, (up_logits, mlx_logits) in enumerate(
            zip(upstream_output.logits, mlx_output.logits_all)
        ):
            logit_summaries[f"head_{head_index}"] = _summarize_diff(up_logits, mlx_logits)
        print(logit_summaries)

    if args.mode in {"generation", "all"}:
        print("== generation ==")
        upstream_model = UpstreamModel.from_pretrained(
            str(model_dir),
            config=upstream_config,
            torch_dtype=_torch_dtype(args.dtype),
        )
        upstream_model.eval()
        mlx_generation = generate_moss_tts_delay(
            mlx_loaded.model,
            mlx_batch.input_ids,
            mlx_batch.attention_mask,
            config=MossTTSDelayGenerationConfig(
                max_new_tokens=args.max_new_tokens,
                text_temperature=0.0,
                audio_temperature=0.0,
                do_sample=False,
            ),
        )
        with torch.no_grad():
            upstream_generation = upstream_model.generate(
                input_ids=upstream_batch["input_ids"],
                attention_mask=upstream_batch["attention_mask"],
                max_new_tokens=args.max_new_tokens,
                text_temperature=0.0,
                audio_temperature=0.0,
                text_top_p=1.0,
                text_top_k=50,
                audio_top_p=0.9,
                audio_top_k=50,
                audio_repetition_penalty=1.1,
            )
        upstream_start_length, upstream_message = upstream_generation[0]
        upstream_new_rows = upstream_message[upstream_start_length:].tolist()
        mlx_new_rows = mlx_generation.generated_rows[0].tolist()
        print(
            {
                "upstream_start_length": int(upstream_start_length.item())
                if hasattr(upstream_start_length, "item")
                else int(upstream_start_length),
                "mlx_rows": len(mlx_new_rows),
                "upstream_rows": len(upstream_new_rows),
            }
        )
        print(_first_row_diff(upstream_new_rows, mlx_new_rows))


if __name__ == "__main__":
    main()
