"""
Upload OpenMOSS TTSD MLX artifacts to Hugging Face.

HF repo: appautomaton/openmoss-ttsd-mlx
"""

import os
import subprocess
import sys
from pathlib import Path

REPO_ID = "appautomaton/openmoss-ttsd-mlx"
REPO_TYPE = "model"
UPLOAD_ROOT = "models/openmoss/moss_ttsd"
REQUIRED_PATHS = (
    "models/openmoss/moss_ttsd/mlx-int8",
)
NUM_WORKERS = "1"


def resolve_hf() -> str:
    local_hf = Path(sys.executable).with_name("hf")
    if local_hf.exists():
        return str(local_hf)
    return "hf"


HF_BIN = resolve_hf()


def run(cmd: list[str], *, env: dict[str, str] | None = None) -> None:
    print(f"$ {' '.join(cmd)}")
    result = subprocess.run(cmd, check=False, env=env)
    if result.returncode != 0:
        print(f"Error: command exited with {result.returncode}")
        sys.exit(result.returncode)


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    upload_root = root / UPLOAD_ROOT

    if not upload_root.exists():
        print(f"Missing: {upload_root}")
        sys.exit(1)

    for relative_path in REQUIRED_PATHS:
        local_path = root / relative_path
        if not local_path.exists():
            print(f"Missing: {local_path}")
            sys.exit(1)

    env = os.environ.copy()
    env["HF_HUB_DISABLE_XET"] = "1"

    run(
        [
            HF_BIN,
            "upload-large-folder",
            "--repo-type",
            REPO_TYPE,
            "--num-workers",
            NUM_WORKERS,
            REPO_ID,
            str(upload_root),
        ],
        env=env,
    )

    print(f"\nDone. https://huggingface.co/{REPO_ID}")


if __name__ == "__main__":
    main()
