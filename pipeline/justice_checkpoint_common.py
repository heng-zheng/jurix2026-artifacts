from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping


REPO_ROOT = Path(__file__).resolve().parents[1]
HUMAN_FINAL_CHECKPOINT_ROOT = REPO_ROOT / "checkpoints" / "human_final"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(payload: Any) -> str:
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            value = json.loads(text)
            if not isinstance(value, dict):
                raise RuntimeError(f"Checkpoint JSONL row {line_number} in {path.name!r} is not an object.")
            rows.append(value)
    return rows


def checkpoint_overall_hash(manifest: Mapping[str, Any], file_hashes: Mapping[str, str]) -> str:
    canonical_manifest = {key: value for key, value in manifest.items() if key != "overall_checkpoint_sha256"}
    return sha256_json(
        {
            "manifest_sha256": sha256_json(canonical_manifest),
            "files": [{"relative_path": path, "sha256": file_hashes[path]} for path in sorted(file_hashes)],
        }
    )


def load_human_final_checkpoint(checkpoint_id: str) -> tuple[Path, Dict[str, Any]]:
    if not isinstance(checkpoint_id, str) or not checkpoint_id or Path(checkpoint_id).name != checkpoint_id:
        raise ValueError("A human-final checkpoint must be selected by its exact program-generated ID.")
    directory = HUMAN_FINAL_CHECKPOINT_ROOT / checkpoint_id
    manifest_path = directory / "checkpoint_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Human-final checkpoint does not exist: {checkpoint_id!r}.")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise RuntimeError("Human-final checkpoint manifest must be one JSON object.")
    if manifest.get("checkpoint_type") != "human_final" or manifest.get("checkpoint_id") != checkpoint_id:
        raise RuntimeError("Human-final checkpoint identity or type is invalid.")
    files = manifest.get("file_hashes")
    if not isinstance(files, dict):
        raise RuntimeError("Human-final checkpoint manifest has no file hash map.")
    for relative, expected in files.items():
        path = directory / str(relative)
        if not path.is_file() or sha256_file(path) != expected:
            raise RuntimeError(f"Human-final checkpoint file is missing or changed: {relative!r}.")
    actual_files = {
        str(path.relative_to(directory))
        for path in directory.rglob("*")
        if path.is_file() and path.name != "checkpoint_manifest.json"
    }
    if actual_files != set(files):
        raise RuntimeError("Human-final checkpoint payload file set does not match its immutable manifest.")
    expected_overall = checkpoint_overall_hash(manifest, files)
    if manifest.get("overall_checkpoint_sha256") != expected_overall:
        raise RuntimeError("Human-final checkpoint overall hash verification failed.")
    return directory, manifest


def require_consumer_hash(manifest: Mapping[str, Any], label: str, path: Path) -> None:
    expected = (manifest.get("consumer_hashes") or {}).get(label)
    if not expected or sha256_file(path) != expected:
        raise RuntimeError(f"Human-final checkpoint consumer implementation changed: {label!r}.")


def identity_key(row: Mapping[str, Any]) -> tuple[str, str, str, str, str]:
    return (
        str(row.get("oyez_argument_id") or ""),
        str(row.get("utterance_id") or ""),
        str(row.get("turn_id") or ""),
        str(row.get("target_justice") or ""),
        str(row.get("text") or ""),
    )
