from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from scripts.audit import dots_tts_oracle as oracle
from scripts.audit.dots_tts_source import SOURCE_SPECS


def test_oracle_command_is_isolated_and_uses_official_snapshot(tmp_path: Path) -> None:
    command = oracle.oracle_command(SOURCE_SPECS["soar"], tmp_path / "out")
    assert command[:5] == [
        "uv",
        "run",
        "--isolated",
        "--no-project",
        "--python",
    ]
    assert "3.12" in command
    assert "--with-requirements" in command
    assert str(oracle.WORKER) in command
    assert str(Path("models/dots_tts/soar/original").resolve()) in command


def test_fixture_inventory_is_pickle_free_bounded_and_hashed(tmp_path: Path) -> None:
    fixture = tmp_path / "qwen.npz"
    np.savez(fixture, hidden=np.ones((1, 2, 3), dtype=np.float32))
    inventory = oracle._fixture_inventory(fixture)
    assert inventory["arrays"]["hidden"] == {
        "shape": [1, 2, 3],
        "dtype": "float32",
        "bytes": 24,
    }
    assert len(inventory["sha256"]) == 64


def test_fixture_inventory_rejects_object_arrays(tmp_path: Path) -> None:
    fixture = tmp_path / "qwen.npz"
    np.savez(fixture, unsafe=np.asarray([{"call": "me"}], dtype=object))
    with pytest.raises(ValueError, match="Object arrays"):
        oracle._fixture_inventory(fixture)


def test_compare_uses_recorded_numeric_tolerance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    expected = tmp_path / "expected"
    actual = tmp_path / "actual"
    for root, delta in ((expected, 0.0), (actual, 0.005)):
        (root / "soar").mkdir(parents=True)
        np.savez(
            root / "soar" / "qwen.npz",
            hidden=np.asarray([1.0 + delta], dtype=np.float32),
        )
        manifest = {
            "oracle": {"dependencies": {"torch": "2.8.0"}},
            "fixtures": {
                "soar": {
                    "qwen.npz": {
                        "tolerance": {"atol": 0.01, "rtol": 0.01}
                    }
                }
            },
        }
        (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    monkeypatch.setattr(oracle, "validate_fixture_pack", lambda path: json.loads((path / "manifest.json").read_text()))
    oracle.compare_fixture_packs(expected, actual)


def test_reference_commits_are_pinned() -> None:
    provenance = oracle._provenance(oracle._source_manifest())
    assert provenance["references"]["official"]["commit"] == (
        "5ed719e3d36f5a3f6d8037ca9a7009d4fd0520ba"
    )
    assert provenance["references"]["community_mlx"]["commit"] == (
        "f64479f51a2a9d7093533732cae86e765d8fb96e"
    )
