"""
Upload OpenMOSS TTS Local MLX artifacts to Hugging Face.

HF repo: appautomaton/openmoss-tts-local-mlx
"""

import subprocess
import sys
from pathlib import Path

REPO_ID = "appautomaton/openmoss-tts-local-mlx"
REPO_TYPE = "model"

UPLOADS = [
    {
        "local": "models/openmoss/moss_tts_local/mlx-int8",
        "dest": "mlx-int8",
    },
]


def resolve_hf() -> str:
    local_hf = Path(sys.executable).with_name("hf")
    if local_hf.exists():
        return str(local_hf)
    return "hf"


HF_BIN = resolve_hf()


def run(cmd: list[str]) -> None:
    print(f"$ {' '.join(cmd)}")
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        print(f"Error: command exited with {result.returncode}")
        sys.exit(result.returncode)


def main() -> None:
    root = Path(__file__).resolve().parents[2]

    for entry in UPLOADS:
        local_path = root / entry["local"]
        if not local_path.exists():
            print(f"Missing: {local_path}")
            sys.exit(1)

        run([
            HF_BIN, "upload",
            "--repo-type", REPO_TYPE,
            REPO_ID,
            str(local_path),
            entry["dest"],
        ])

    print(f"\nDone. https://huggingface.co/{REPO_ID}")


if __name__ == "__main__":
    main()
