#!/usr/bin/env python3
"""Package Confucius4-R2T2 as an MLX bf16 runtime checkpoint.

The graph is Qwen3-ASR. The weights are NetEase's finetune, so the package
lives under models/netease/confucius4_r2t2/ and is not a Qwen3-ASR alias.
bf16 only.

    python scripts/convert/confucius4_r2t2.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

DEFAULT_INPUT = Path("models/netease/confucius4_r2t2/original")
DEFAULT_OUTPUT = Path("models/netease/confucius4_r2t2/mlx-bf16")
FAMILY = "confucius4_r2t2"


def ensure_merges_txt(output_dir: Path) -> Path | None:
    """Materialize ``merges.txt`` when upstream ships only ``tokenizer.json``.

    ``Qwen3ASRTokenizer`` rebuilds the BPE model from ``vocab.json`` plus
    ``merges.txt``. netease-youdao/Confucius4-R2T2 ships the fast tokenizer
    only, so the merge list is copied out of ``tokenizer.json`` rather than
    fetched from the Hub. Returns the written path, or ``None`` if the file
    is already present.
    """
    target = output_dir / "merges.txt"
    if target.exists():
        return None
    source = output_dir / "tokenizer.json"
    if not source.exists():
        raise FileNotFoundError(
            f"Neither merges.txt nor tokenizer.json found in {output_dir}."
        )
    merges = (
        json.loads(source.read_text(encoding="utf-8")).get("model", {}).get("merges")
    )
    if not merges:
        raise ValueError(f"No model.merges in {source}.")
    lines = ["#version: 0.2"]
    lines.extend(
        " ".join(entry) if isinstance(entry, list) else str(entry) for entry in merges
    )
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return target


def convert_confucius4_r2t2(
    input_dir: Path = DEFAULT_INPUT,
    output_dir: Path = DEFAULT_OUTPUT,
) -> Path:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
    from mlx_speech.models.qwen3_asr.checkpoint import (
        load_qwen3_asr_checkpoint,
        save_qwen3_asr_bf16_checkpoint,
    )

    checkpoint = load_qwen3_asr_checkpoint(input_dir)
    report = save_qwen3_asr_bf16_checkpoint(checkpoint, output_dir)
    config_path = output_dir / "config.json"
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    payload["mlx_speech"] = {"family": FAMILY}
    config_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    merges_path = ensure_merges_txt(output_dir)
    print(f"Input:  {input_dir}")
    print(f"Output: {report.output_file}")
    print(f"Tensors: {report.tensor_count}")
    print(
        f"Renamed: {len(report.renamed_keys)}  Transposed: {len(report.transposed_keys)}"
    )
    print(f"Family: {FAMILY}")
    print(f"Merges: {merges_path or 'already present'}")
    return report.output_file


def main() -> None:
    convert_confucius4_r2t2()


if __name__ == "__main__":
    main()
