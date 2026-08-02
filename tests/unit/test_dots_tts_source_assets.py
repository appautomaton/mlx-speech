from __future__ import annotations

import pickle
import subprocess
import zipfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from safetensors.numpy import save_file

from scripts.audit.dots_tts_source import (
    SOURCE_FILES,
    SOURCE_SPECS,
    SourceSpec,
    audit_variant,
    read_latent_stats,
    selected_specs,
)
from scripts.convert.download_dots_tts import verify_remote_info


def _write_torch_style_numpy_pickle(path: Path, payload: object) -> None:
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("archive/data.pkl", pickle.dumps(payload, protocol=2))


def _fake_source(tmp_path: Path) -> SourceSpec:
    source_dir = tmp_path / "original"
    source_dir.mkdir()
    for name in SOURCE_FILES:
        (source_dir / name).write_bytes(name.encode())
    for name in (
        "model.safetensors",
        "speaker_encoder.safetensors",
        "vocoder.safetensors",
    ):
        save_file({"weight": np.ones((2, 3), dtype=np.float32)}, source_dir / name)
    _write_torch_style_numpy_pickle(
        source_dir / "latent_stats.pt",
        {
            "mean": np.arange(4, dtype=np.float32),
            "var": np.ones(4, dtype=np.float32),
        },
    )
    return SourceSpec(
        variant="test",
        repo_id="owner/repo",
        resolved_repo_id="owner/repo",
        revision="a" * 40,
        source_dir=source_dir,
    )


def test_source_specs_pin_revisions_and_original_layouts() -> None:
    assert selected_specs("all") == (SOURCE_SPECS["soar"], SOURCE_SPECS["mf"])
    assert SOURCE_SPECS["soar"].revision == "e3520f75254d0020a0406db31c51a79d00d22d55"
    assert SOURCE_SPECS["mf"].revision == "25c53fb462e57087e52237daa5ea30df1c5cc328"
    assert SOURCE_SPECS["soar"].source_dir == Path("models/dots_tts/soar/original")
    assert SOURCE_SPECS["mf"].source_dir == Path("models/dots_tts/mf/original")
    for spec in SOURCE_SPECS.values():
        assert subprocess.run(
            ["git", "check-ignore", "-q", str(spec.source_dir / "model.safetensors")],
            check=False,
        ).returncode == 0


def test_remote_info_must_match_revision_file_set_and_lfs_metadata() -> None:
    spec = SOURCE_SPECS["soar"]
    siblings = []
    for name in spec.files:
        asset = spec.lfs_assets.get(name)
        siblings.append(
            SimpleNamespace(
                rfilename=name,
                lfs=None
                if asset is None
                else {"size": asset.size, "sha256": asset.sha256},
            )
        )
    info = SimpleNamespace(
        id=spec.resolved_repo_id,
        sha=spec.revision,
        siblings=siblings,
    )
    verify_remote_info(spec, info)
    info.sha = "b" * 40
    with pytest.raises(ValueError, match="expected immutable"):
        verify_remote_info(spec, info)


def test_audit_inventories_files_safetensors_and_latent_stats(tmp_path: Path) -> None:
    spec = _fake_source(tmp_path)
    result = audit_variant(spec)
    assert set(result["files"]) == set(SOURCE_FILES)
    inventory = result["files"]["model.safetensors"]["safetensors"]
    assert inventory == {
        "tensor_count": 1,
        "element_count": 6,
        "dtypes": {"F32": 1},
    }
    assert result["latent_stats"]["mean"]["shape"] == [4]
    assert result["latent_stats"]["var"]["dtype"] == "float32"


def test_audit_rejects_unexpected_source_files(tmp_path: Path) -> None:
    spec = _fake_source(tmp_path)
    (spec.source_dir / "untracked.bin").write_bytes(b"unexpected")
    with pytest.raises(ValueError, match="unexpected=.*untracked.bin"):
        audit_variant(spec)


def test_latent_reader_rejects_executable_pickle_global(tmp_path: Path) -> None:
    path = tmp_path / "latent_stats.pt"
    _write_torch_style_numpy_pickle(path, {"mean": Path("x"), "var": Path("y")})
    with pytest.raises(pickle.UnpicklingError, match="forbidden"):
        read_latent_stats(path)
