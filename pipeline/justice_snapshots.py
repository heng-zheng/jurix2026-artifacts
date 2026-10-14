from __future__ import annotations

import json
import os
import shutil
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import requests

from .common import REPO_ROOT, read_jsonl, write_jsonl
from .justice_contracts import validate_reason_tree
from .justice_runtime import sha256_file, sha256_json


SNAPSHOT_ROOT = REPO_ROOT / "snapshots"
BRIEF_SNAPSHOT_ROOT = SNAPSHOT_ROOT / "briefs"
ORAL_SNAPSHOT_ROOT = SNAPSHOT_ROOT / "oral_arguments"


def immutable_id() -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    return f"{stamp}__{uuid.uuid4()}"


def _validate_snapshot_id(value: Any) -> str:
    if not isinstance(value, str) or not value or Path(value).name != value or value in {".", ".."}:
        raise ValueError("A snapshot must be selected by its exact program-generated snapshot ID.")
    return value


def _canonical_manifest_payload(manifest: Mapping[str, Any]) -> Dict[str, Any]:
    return {key: value for key, value in manifest.items() if key != "overall_snapshot_sha256"}


def _overall_hash(manifest: Mapping[str, Any], files: Mapping[str, str]) -> str:
    return sha256_json(
        {
            "manifest_sha256": sha256_json(_canonical_manifest_payload(manifest)),
            "files": [{"relative_path": path, "sha256": files[path]} for path in sorted(files)],
        }
    )


def _write_manifest(directory: Path, manifest: Dict[str, Any], file_hashes: Mapping[str, str]) -> Dict[str, Any]:
    payload = dict(manifest)
    payload["overall_snapshot_sha256"] = _overall_hash(payload, file_hashes)
    (directory / "snapshot_manifest.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8"
    )
    return payload


def _load_manifest(directory: Path) -> Dict[str, Any]:
    path = directory / "snapshot_manifest.json"
    if not path.is_file():
        raise RuntimeError(f"Snapshot manifest is missing: {path}.")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError("Snapshot manifest must be one JSON object.")
    return payload


def _atomic_snapshot(root: Path, snapshot_id: str, builder: Callable[[Path], Dict[str, Any]]) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    destination = root / snapshot_id
    if destination.exists():
        raise FileExistsError(f"Snapshot already exists and will not be overwritten: {snapshot_id}.")
    pending_root = root / ".pending"
    pending_root.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f"{snapshot_id}__", dir=pending_root))
    try:
        builder(temporary)
        os.replace(temporary, destination)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return destination


def _validate_brief_id(value: Any) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError("brief_id must be a non-empty string without outer whitespace.")
    if value in {".", ".."} or "/" in value or "\\" in value:
        raise ValueError(f"brief_id is not safe for the approved snapshot path: {value!r}.")
    if any(ord(character) < 32 or ord(character) == 127 for character in value):
        raise ValueError("brief_id must not contain NUL or control characters.")
    return value


def _normalize_dockets(value: Any) -> List[str]:
    if not isinstance(value, list) or not value:
        raise ValueError("docket_numbers must be a non-empty JSON string list.")
    rows: List[str] = []
    for item in value:
        if not isinstance(item, str):
            raise ValueError("Every docket_numbers item must be a string.")
        normalized = item.strip()
        if not normalized:
            raise ValueError("docket_numbers must not contain an empty value.")
        rows.append(normalized)
    if len(rows) != len(set(rows)):
        raise ValueError("docket_numbers must not contain duplicates after outer-space trimming.")
    return rows


def _brief_entry(entry: Mapping[str, Any]) -> Dict[str, Any]:
    brief_id = _validate_brief_id(entry.get("brief_id"))
    group = entry.get("brief_group")
    if not isinstance(group, str) or not group.strip():
        raise ValueError(f"brief_group must be explicitly provided for brief {brief_id!r}.")
    pdf = Path(str(entry.get("pdf_path") or ""))
    reason_tree = Path(str(entry.get("reason_tree_path") or ""))
    if not pdf.is_file():
        raise FileNotFoundError(f"Brief PDF is missing: {pdf}.")
    if not reason_tree.is_file():
        raise FileNotFoundError(f"Per-brief reason tree is missing: {reason_tree}.")
    return {
        "brief_id": brief_id,
        "docket_numbers": _normalize_dockets(entry.get("docket_numbers")),
        "brief_group": group,
        "pdf_path": pdf,
        "reason_tree_path": reason_tree,
    }


def create_brief_snapshot(
    specification: Mapping[str, Any],
    *,
    confirm: Callable[[Sequence[Dict[str, Any]]], bool],
    snapshot_id: Optional[str] = None,
) -> Path:
    raw = specification.get("briefs")
    if not isinstance(raw, list) or not raw:
        raise ValueError("Brief snapshot specification must contain a non-empty briefs list.")
    briefs = [_brief_entry(item) for item in raw if isinstance(item, dict)]
    if len(briefs) != len(raw):
        raise ValueError("Every briefs entry must be one JSON object.")
    ids = [item["brief_id"] for item in briefs]
    if len(ids) != len(set(ids)):
        raise ValueError("brief_id must be unique within a brief snapshot.")
    review = [
        {
            "brief_id": item["brief_id"],
            "docket_numbers": item["docket_numbers"],
            "brief_group": item["brief_group"],
            "pdf_path": str(item["pdf_path"]),
            "reason_tree_path": str(item["reason_tree_path"]),
        }
        for item in briefs
    ]
    if confirm(review) is not True:
        raise RuntimeError("Brief snapshot creation was not confirmed; no snapshot was created.")
    chosen_id = _validate_snapshot_id(snapshot_id or immutable_id())

    def build(directory: Path) -> Dict[str, Any]:
        manifest_briefs: List[Dict[str, Any]] = []
        file_hashes: Dict[str, str] = {}
        all_nodes: List[Dict[str, Any]] = []
        for item in briefs:
            brief_id = item["brief_id"]
            target_dir = directory / "briefs" / brief_id
            target_dir.mkdir(parents=True, exist_ok=False)
            pdf_target = target_dir / f"{brief_id}.pdf"
            shutil.copyfile(item["pdf_path"], pdf_target)
            if sha256_file(pdf_target) != sha256_file(item["pdf_path"]):
                raise RuntimeError(f"Copied PDF hash mismatch for brief {brief_id!r}.")
            nodes = read_jsonl(item["reason_tree_path"])
            if not nodes:
                raise ValueError(f"Reason tree for brief {brief_id!r} is empty.")
            for node in nodes:
                if node.get("brief_id") != brief_id:
                    raise ValueError(f"Every reason node in {brief_id!r} must preserve the same brief_id.")
            validate_reason_tree(nodes)
            all_nodes.extend(nodes)
            tree_target = target_dir / "reason_tree.jsonl"
            write_jsonl(tree_target, nodes)
            pdf_rel = str(pdf_target.relative_to(directory))
            tree_rel = str(tree_target.relative_to(directory))
            file_hashes[pdf_rel] = sha256_file(pdf_target)
            file_hashes[tree_rel] = sha256_file(tree_target)
            manifest_briefs.append(
                {
                    "brief_id": brief_id,
                    "docket_numbers": item["docket_numbers"],
                    "brief_group": item["brief_group"],
                    "pdf_sha256": file_hashes[pdf_rel],
                    "reason_tree_sha256": file_hashes[tree_rel],
                }
            )
        reason_ids = [str(node.get("reason_target_id") or "") for node in all_nodes]
        if any(not value for value in reason_ids) or len(reason_ids) != len(set(reason_ids)):
            raise ValueError("reason_target_id must be globally unique across the entire brief snapshot.")
        manifest = {
            "snapshot_type": "brief",
            "snapshot_id": chosen_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "briefs": manifest_briefs,
        }
        return _write_manifest(directory, manifest, file_hashes)

    destination = _atomic_snapshot(BRIEF_SNAPSHOT_ROOT, chosen_id, build)
    verify_brief_snapshot(chosen_id)
    return destination


def _source_identity(value: Any, *, label: str) -> str:
    if isinstance(value, bool) or value is None or isinstance(value, (list, dict, float)):
        raise ValueError(f"{label} must be a JSON integer or non-empty string.")
    if isinstance(value, int):
        return str(value)
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{label} must be a non-empty string without outer whitespace.")
    return value


def _download_json_bytes(url: str, *, timeout: int) -> Tuple[bytes, str, Dict[str, Any]]:
    if not isinstance(url, str) or not url.strip():
        raise ValueError("Every source_url must be a non-empty string.")
    response = requests.get(url, timeout=timeout)
    response.raise_for_status()
    raw = response.content
    try:
        payload = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Oyez source did not return valid JSON: {url!r}.") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"Oyez source JSON root must be an object: {url!r}.")
    return raw, str(response.url), payload


def create_utterance_snapshot(
    specification: Mapping[str, Any], *, snapshot_id: Optional[str] = None, timeout: int = 60
) -> Path:
    case_specs = specification.get("case_sources")
    argument_specs = specification.get("oral_arguments")
    if not isinstance(case_specs, list) or not case_specs:
        raise ValueError("Oral snapshot specification must contain a non-empty case_sources list.")
    if not isinstance(argument_specs, list) or not argument_specs:
        raise ValueError("Oral snapshot specification must contain a non-empty oral_arguments list.")
    if any(not isinstance(item, dict) for item in [*case_specs, *argument_specs]):
        raise ValueError("Every oral snapshot source entry must be one JSON object.")
    chosen_id = _validate_snapshot_id(snapshot_id or immutable_id())

    def build(directory: Path) -> Dict[str, Any]:
        file_hashes: Dict[str, str] = {}
        case_rows: List[Dict[str, Any]] = []
        seen_cases: set[str] = set()
        for index, spec in enumerate(case_specs, start=1):
            requested = str(spec.get("source_url") or "")
            raw, final_url, payload = _download_json_bytes(requested, timeout=timeout)
            identity = _source_identity(payload.get("ID", payload.get("id")), label="case_source_id")
            if identity in seen_cases:
                raise ValueError(f"Duplicate case_source_id in oral snapshot: {identity!r}.")
            seen_cases.add(identity)
            advocates = payload.get("advocates")
            if not isinstance(advocates, list) or any(not isinstance(item, dict) for item in advocates):
                raise ValueError(f"Case source {identity!r} has an invalid advocates list.")
            target = directory / "case_sources" / f"case_source_{index:04d}.json"
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(raw)
            relative = str(target.relative_to(directory))
            file_hashes[relative] = sha256_file(target)
            case_rows.append(
                {
                    "case_source_id": identity,
                    "requested_url": requested,
                    "final_url": final_url,
                    "relative_file": relative,
                    "source_json_sha256": file_hashes[relative],
                }
            )
        argument_rows: List[Dict[str, Any]] = []
        seen_arguments: set[str] = set()
        for index, spec in enumerate(argument_specs, start=1):
            requested = str(spec.get("source_url") or "")
            raw, final_url, payload = _download_json_bytes(requested, timeout=timeout)
            identity = _source_identity(payload.get("id"), label="oyez_argument_id")
            if identity in seen_arguments:
                raise ValueError(f"Duplicate oyez_argument_id in oral snapshot: {identity!r}.")
            seen_arguments.add(identity)
            transcript = payload.get("transcript")
            sections = transcript.get("sections") if isinstance(transcript, dict) else None
            if not isinstance(sections, list) or any(not isinstance(section, dict) for section in sections):
                raise ValueError(f"Oral argument {identity!r} has an invalid transcript.sections list.")
            if any(not isinstance(section.get("turns"), list) for section in sections):
                raise ValueError(f"Oral argument {identity!r} has a section without a valid turns list.")
            mapped = spec.get("case_source_ids")
            if not isinstance(mapped, list) or not mapped or any(not isinstance(value, str) or not value for value in mapped):
                raise ValueError(f"Oral argument {identity!r} requires a non-empty case_source_ids list.")
            if len(mapped) != len(set(mapped)) or any(value not in seen_cases for value in mapped):
                raise ValueError(f"Oral argument {identity!r} contains duplicate or unknown case_source_ids.")
            target = directory / "oral_arguments" / f"oral_argument_{index:04d}.json"
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(raw)
            relative = str(target.relative_to(directory))
            file_hashes[relative] = sha256_file(target)
            argument_rows.append(
                {
                    "oyez_argument_id": identity,
                    "requested_url": requested,
                    "final_url": final_url,
                    "relative_file": relative,
                    "source_json_sha256": file_hashes[relative],
                    "case_source_ids": list(mapped),
                }
            )
        manifest = {
            "snapshot_type": "oral_argument",
            "snapshot_id": chosen_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "case_sources": case_rows,
            "oral_arguments": argument_rows,
        }
        return _write_manifest(directory, manifest, file_hashes)

    destination = _atomic_snapshot(ORAL_SNAPSHOT_ROOT, chosen_id, build)
    verify_utterance_snapshot(chosen_id)
    return destination


def _verify_snapshot(directory: Path, expected_type: str, expected_id: str, files: Mapping[str, str]) -> Dict[str, Any]:
    manifest = _load_manifest(directory)
    if manifest.get("snapshot_type") != expected_type or manifest.get("snapshot_id") != expected_id:
        raise RuntimeError("Snapshot identity does not match its root directory and requested ID.")
    for relative, expected in files.items():
        path = directory / relative
        if not path.is_file() or sha256_file(path) != expected:
            raise RuntimeError(f"Snapshot file is missing or changed: {relative!r}.")
    actual_files = {
        str(path.relative_to(directory))
        for path in directory.rglob("*")
        if path.is_file() and path.name != "snapshot_manifest.json"
    }
    if actual_files != set(files):
        raise RuntimeError("Snapshot payload file set does not match its immutable manifest.")
    if manifest.get("overall_snapshot_sha256") != _overall_hash(manifest, files):
        raise RuntimeError("Snapshot overall SHA-256 verification failed.")
    return manifest


def verify_brief_snapshot(snapshot_id: str) -> Dict[str, Any]:
    snapshot_id = _validate_snapshot_id(snapshot_id)
    directory = BRIEF_SNAPSHOT_ROOT / snapshot_id
    manifest = _load_manifest(directory)
    files: Dict[str, str] = {}
    all_nodes: List[Dict[str, Any]] = []
    briefs = manifest.get("briefs")
    if not isinstance(briefs, list) or not briefs:
        raise RuntimeError("Brief snapshot manifest must contain a non-empty briefs list.")
    seen: set[str] = set()
    for item in briefs:
        if not isinstance(item, dict):
            raise RuntimeError("Every brief manifest entry must be one object.")
        brief_id = _validate_brief_id(item.get("brief_id"))
        if brief_id in seen:
            raise RuntimeError("Brief snapshot contains duplicate brief_id values.")
        seen.add(brief_id)
        _normalize_dockets(item.get("docket_numbers"))
        if not isinstance(item.get("brief_group"), str) or not str(item.get("brief_group")).strip():
            raise RuntimeError("Brief snapshot contains an invalid brief_group.")
        pdf_rel = f"briefs/{brief_id}/{brief_id}.pdf"
        tree_rel = f"briefs/{brief_id}/reason_tree.jsonl"
        files[pdf_rel] = str(item.get("pdf_sha256") or "")
        files[tree_rel] = str(item.get("reason_tree_sha256") or "")
        nodes = read_jsonl(directory / tree_rel)
        for node in nodes:
            if node.get("brief_id") != brief_id:
                raise RuntimeError("Reason node brief_id does not match its snapshot brief directory.")
        validate_reason_tree(nodes)
        all_nodes.extend(nodes)
    ids = [str(node.get("reason_target_id") or "") for node in all_nodes]
    if any(not value for value in ids) or len(ids) != len(set(ids)):
        raise RuntimeError("reason_target_id is not globally unique across the brief snapshot.")
    return _verify_snapshot(directory, "brief", snapshot_id, files)


def verify_utterance_snapshot(snapshot_id: str) -> Dict[str, Any]:
    snapshot_id = _validate_snapshot_id(snapshot_id)
    directory = ORAL_SNAPSHOT_ROOT / snapshot_id
    manifest = _load_manifest(directory)
    files: Dict[str, str] = {}
    case_ids: set[str] = set()
    for item in manifest.get("case_sources") or []:
        identity = _source_identity(item.get("case_source_id"), label="case_source_id")
        if identity in case_ids:
            raise RuntimeError("Oral snapshot contains duplicate case_source_id values.")
        case_ids.add(identity)
        relative = str(item.get("relative_file") or "")
        files[relative] = str(item.get("source_json_sha256") or "")
        payload = json.loads((directory / relative).read_bytes())
        if _source_identity(payload.get("ID", payload.get("id")), label="case_source_id") != identity:
            raise RuntimeError("Case source identity changed after snapshot creation.")
        advocates = payload.get("advocates")
        if not isinstance(advocates, list) or any(not isinstance(value, dict) for value in advocates):
            raise RuntimeError("Case source advocates structure is invalid.")
    argument_ids: set[str] = set()
    for item in manifest.get("oral_arguments") or []:
        identity = _source_identity(item.get("oyez_argument_id"), label="oyez_argument_id")
        if identity in argument_ids:
            raise RuntimeError("Oral snapshot contains duplicate oyez_argument_id values.")
        argument_ids.add(identity)
        mapped = item.get("case_source_ids")
        if not isinstance(mapped, list) or not mapped or len(mapped) != len(set(mapped)) or any(value not in case_ids for value in mapped):
            raise RuntimeError("Oral argument has an invalid case_source_ids mapping.")
        relative = str(item.get("relative_file") or "")
        files[relative] = str(item.get("source_json_sha256") or "")
        payload = json.loads((directory / relative).read_bytes())
        if _source_identity(payload.get("id"), label="oyez_argument_id") != identity:
            raise RuntimeError("Oral argument identity changed after snapshot creation.")
        transcript = payload.get("transcript")
        sections = transcript.get("sections") if isinstance(transcript, dict) else None
        if not isinstance(sections, list) or any(not isinstance(section, dict) or not isinstance(section.get("turns"), list) for section in sections):
            raise RuntimeError("Oral argument transcript structure is invalid.")
    return _verify_snapshot(directory, "oral_argument", snapshot_id, files)


def list_snapshots(snapshot_type: str) -> List[Dict[str, Any]]:
    root = BRIEF_SNAPSHOT_ROOT if snapshot_type == "brief" else ORAL_SNAPSHOT_ROOT
    rows: List[Dict[str, Any]] = []
    if not root.exists():
        return rows
    for directory in sorted(path for path in root.iterdir() if path.is_dir() and path.name != ".pending"):
        try:
            manifest = verify_brief_snapshot(directory.name) if snapshot_type == "brief" else verify_utterance_snapshot(directory.name)
            rows.append({"snapshot_id": directory.name, "valid": True, "manifest": manifest})
        except Exception as exc:
            rows.append({"snapshot_id": directory.name, "valid": False, "error": str(exc)})
    return rows


def brief_snapshot_nodes(snapshot_id: str, brief_ids: Iterable[str]) -> List[Dict[str, Any]]:
    manifest = verify_brief_snapshot(snapshot_id)
    available = {str(item["brief_id"]) for item in manifest["briefs"]}
    selected = list(brief_ids)
    if any(value not in available for value in selected):
        raise ValueError("Selected brief IDs are not completely covered by the brief snapshot.")
    rows: List[Dict[str, Any]] = []
    for brief_id in selected:
        rows.extend(read_jsonl(BRIEF_SNAPSHOT_ROOT / snapshot_id / "briefs" / brief_id / "reason_tree.jsonl"))
    validate_reason_tree(rows)
    return rows
