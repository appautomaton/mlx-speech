"""Read and update the canonical dots.tts comparison contract."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


SCHEMA_VERSION = 1
_HEADING = "## Canonical comparison data"
_OPEN = "```json"
_CLOSE = "```"


def _data_span(text: str) -> tuple[int, int]:
    heading_positions: list[int] = []
    offset = 0
    for line in text.splitlines(keepends=True):
        if line.rstrip("\r\n") == _HEADING:
            heading_positions.append(offset)
        offset += len(line)
    if len(heading_positions) != 1:
        raise ValueError(
            "dots.tts comparison contract must contain exactly one canonical data heading"
        )
    heading_end = heading_positions[0] + len(_HEADING)
    opening = text.find(_OPEN, heading_end)
    if opening < 0:
        raise ValueError("dots.tts comparison contract is missing its JSON block")
    if text.find(_OPEN, opening + len(_OPEN)) >= 0:
        raise ValueError("dots.tts comparison contract has multiple JSON blocks")
    payload_start = opening + len(_OPEN)
    if payload_start >= len(text) or text[payload_start] != "\n":
        raise ValueError("dots.tts comparison contract JSON fence must end its line")
    payload_start += 1
    closing = text.find(f"\n{_CLOSE}", payload_start)
    if closing < 0:
        raise ValueError("dots.tts comparison contract JSON block is not closed")
    return payload_start, closing


def _validate_shape(payload: object, *, require_complete: bool) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError("dots.tts comparison contract data must be an object")
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported dots.tts comparison contract schema")
    if set(payload) != {"schema_version", "status", "performance", "quality"}:
        raise ValueError("dots.tts comparison contract fields are incomplete")
    status = payload.get("status")
    if status not in {"pending", "complete"}:
        raise ValueError("dots.tts comparison contract has invalid status")
    populated = all(isinstance(payload.get(name), dict) for name in ("performance", "quality"))
    if status == "complete" and not populated:
        raise ValueError("complete dots.tts comparison contract has missing evidence")
    if status == "pending" and populated:
        raise ValueError("populated dots.tts comparison contract must be complete")
    for name in ("performance", "quality"):
        value = payload.get(name)
        if value is not None and not isinstance(value, dict):
            raise ValueError(f"dots.tts comparison contract {name} evidence is invalid")
    if require_complete and status != "complete":
        raise ValueError("dots.tts comparison contract is not complete")
    return payload


def load_comparison_contract(
    path: str | Path,
    *,
    require_complete: bool = True,
) -> dict[str, Any]:
    contract_path = Path(path)
    text = contract_path.read_text(encoding="utf-8")
    start, end = _data_span(text)
    try:
        payload = json.loads(text[start:end])
    except json.JSONDecodeError as error:
        raise ValueError("dots.tts comparison contract contains invalid JSON") from error
    return _validate_shape(payload, require_complete=require_complete)


def update_comparison_contract(
    path: str | Path,
    *,
    section: str,
    evidence: dict[str, Any],
) -> dict[str, Any]:
    if section not in {"performance", "quality"}:
        raise ValueError(f"unsupported dots.tts comparison contract section: {section}")
    if not isinstance(evidence, dict) or not evidence:
        raise ValueError("dots.tts comparison evidence must be a non-empty object")
    contract_path = Path(path)
    text = contract_path.read_text(encoding="utf-8")
    start, end = _data_span(text)
    try:
        payload = json.loads(text[start:end])
    except json.JSONDecodeError as error:
        raise ValueError("dots.tts comparison contract contains invalid JSON") from error
    payload = _validate_shape(payload, require_complete=False)
    payload[section] = evidence
    payload["status"] = (
        "complete"
        if all(isinstance(payload.get(name), dict) for name in ("performance", "quality"))
        else "pending"
    )
    replacement = json.dumps(payload, indent=2, ensure_ascii=False)
    updated = f"{text[:start]}{replacement}{text[end:]}"
    temporary = contract_path.with_suffix(contract_path.suffix + ".tmp")
    temporary.write_text(updated, encoding="utf-8")
    temporary.replace(contract_path)
    return payload


__all__ = [
    "SCHEMA_VERSION",
    "load_comparison_contract",
    "update_comparison_contract",
]
