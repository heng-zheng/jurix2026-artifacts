from __future__ import annotations

import json
import math
import os
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional

from .common import REPO_ROOT
from .justice_runtime import sha256_json
from .justice_snapshots import verify_brief_snapshot, verify_utterance_snapshot


RUN_ROOT = REPO_ROOT / "outputs" / "justice_runs"
CHECKPOINT_ROOT = REPO_ROOT / "checkpoints"
REVIEW_CHECKPOINT_ROOT = CHECKPOINT_ROOT / "review"
HUMAN_FINAL_CHECKPOINT_ROOT = CHECKPOINT_ROOT / "human_final"
PENDING_CHECKPOINT_ROOT = CHECKPOINT_ROOT / ".pending"

RUN_STATUSES = {
    "preflight_in_progress",
    "awaiting_model_identity_confirmation",
    "ready_for_stage1",
    "ready_for_stage2",
    "ready_for_stage3",
    "stage_in_progress",
    "review_checkpoint_pending",
    "awaiting_stage1_human",
    "awaiting_stage2_human",
    "awaiting_stage3_human",
    "human_final_checkpoint_pending",
    "completed",
    "paused_recoverable",
    "failed",
    "abandoned",
}
TERMINAL_STATUSES = {"completed", "failed", "abandoned"}
CONTINUABLE_STATUSES = {
    "awaiting_model_identity_confirmation",
    "ready_for_stage1",
    "ready_for_stage2",
    "ready_for_stage3",
    "review_checkpoint_pending",
    "awaiting_stage1_human",
    "awaiting_stage2_human",
    "awaiting_stage3_human",
    "human_final_checkpoint_pending",
    "paused_recoverable",
}

FUTURE_MANIFEST_VERSION = 4
APPROVED_ADVOCATE_GROUPS = {"pet", "res", "caa"}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def new_run_id() -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    return f"{stamp}__{uuid.uuid4()}"


def state_path(run_dir: Path) -> Path:
    return run_dir / "run_state.json"


def manifest_path(run_dir: Path) -> Path:
    return run_dir / "00_run_preflight" / "run_manifest.json"


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(name, path)
    except Exception:
        try:
            os.unlink(name)
        except FileNotFoundError:
            pass
        raise


def write_run_state(run_dir: Path, status: str, current_stage: str) -> Dict[str, Any]:
    if status not in RUN_STATUSES:
        raise ValueError(f"Unknown Justice run status: {status!r}.")
    payload = {
        "run_id": run_dir.name,
        "status": status,
        "current_stage": current_stage,
        "updated_at": utc_now(),
    }
    _atomic_json(state_path(run_dir), payload)
    return payload


def read_run_state(run_dir: Path) -> Dict[str, Any]:
    path = state_path(run_dir)
    if not path.is_file():
        raise RuntimeError(f"Justice run has no run_state.json: {run_dir.name!r}.")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("status") not in RUN_STATUSES:
        raise RuntimeError("Justice run state is invalid.")
    if payload.get("run_id") != run_dir.name:
        raise RuntimeError("run_state.run_id does not match the run directory name.")
    return payload


def read_run_manifest(run_dir: Path) -> Dict[str, Any]:
    path = manifest_path(run_dir)
    if not path.is_file():
        raise RuntimeError("Justice run manifest is missing.")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("run_id") != run_dir.name:
        raise RuntimeError("manifest.run_id does not match the run directory name.")
    return payload


def load_run(run_id: str) -> tuple[Path, Dict[str, Any], Dict[str, Any]]:
    if not isinstance(run_id, str) or not run_id or Path(run_id).name != run_id:
        raise ValueError("An existing run must be selected by its exact program-generated run ID.")
    run_dir = RUN_ROOT / run_id
    if not run_dir.is_dir() or run_dir.name != run_id:
        raise FileNotFoundError(f"Justice run does not exist: {run_id!r}.")
    state = read_run_state(run_dir)
    manifest = read_run_manifest(run_dir)
    if manifest.get("manifest_version") != FUTURE_MANIFEST_VERSION:
        raise RuntimeError("Justice run manifest schema is unsupported by this workflow.")
    if state["run_id"] != run_id or manifest["run_id"] != run_id:
        raise RuntimeError("Selected run ID, directory, state, and manifest identity do not match.")
    return run_dir, state, manifest


def _scope_payload(scope: Mapping[str, Any], oral_manifest: Mapping[str, Any], brief_manifest: Mapping[str, Any]) -> Dict[str, Any]:
    mappings = scope.get("oral_argument_brief_scopes")
    if not isinstance(mappings, list):
        raise ValueError("Run scope must contain an oral_argument_brief_scopes list.")
    available_arguments = {str(item.get("oyez_argument_id")) for item in oral_manifest.get("oral_arguments") or []}
    available_briefs = {str(item.get("brief_id")) for item in brief_manifest.get("briefs") or []}
    brief_groups = {
        str(item.get("brief_id")): str(item.get("brief_group") or "")
        for item in brief_manifest.get("briefs") or []
    }
    rows: List[Dict[str, Any]] = []
    argument_ids: List[str] = []
    for item in mappings:
        if not isinstance(item, dict):
            raise ValueError("Every run-scope mapping must be one JSON object.")
        argument_id = item.get("oyez_argument_id")
        brief_ids = item.get("brief_ids")
        if not isinstance(argument_id, str) or not argument_id or argument_id not in available_arguments:
            raise ValueError(f"Run scope contains an unknown oyez_argument_id: {argument_id!r}.")
        if not isinstance(brief_ids, list) or not brief_ids or any(not isinstance(value, str) or not value for value in brief_ids):
            raise ValueError(f"Run scope for {argument_id!r} requires a non-empty brief_ids string list.")
        if len(brief_ids) != len(set(brief_ids)) or any(value not in available_briefs for value in brief_ids):
            raise ValueError(f"Run scope for {argument_id!r} contains duplicate or unknown brief IDs.")
        invalid_brief_groups = sorted(
            {brief_groups[value] for value in brief_ids if brief_groups.get(value) not in APPROVED_ADVOCATE_GROUPS}
        )
        if invalid_brief_groups:
            raise ValueError(
                "New Justice runs may only use briefs assigned to pet, res, or caa; "
                f"found {invalid_brief_groups!r}."
            )
        argument_ids.append(argument_id)
        rows.append({"oyez_argument_id": argument_id, "brief_ids": list(brief_ids)})
    if len(argument_ids) != len(set(argument_ids)):
        raise ValueError("Every selected oral argument must have exactly one authoritative run-scope mapping.")
    selected = scope.get("selected_oyez_argument_ids")
    if not isinstance(selected, list) or not selected or any(not isinstance(value, str) or value not in available_arguments for value in selected):
        raise ValueError("selected_oyez_argument_ids must be a non-empty list covered by the oral snapshot.")
    if len(selected) != len(set(selected)):
        raise ValueError("selected_oyez_argument_ids must not contain duplicates.")
    if not set(argument_ids) <= set(selected):
        raise ValueError("Run-scope mappings may only name oral arguments selected for this run.")
    phase_times = scope.get("phase_times_by_argument") or {}
    if not isinstance(phase_times, dict):
        raise ValueError("phase_times_by_argument must be a JSON object when provided.")
    _validate_phase_times(phase_times, set(selected))
    annotation_scope = scope.get("annotation_scope")
    if not isinstance(annotation_scope, dict):
        raise ValueError("Run scope must contain an annotation_scope JSON object.")
    target_justices = annotation_scope.get("target_justices")
    if (
        not isinstance(target_justices, list)
        or not target_justices
        or any(not isinstance(value, str) or not value.strip() for value in target_justices)
    ):
        raise ValueError("annotation_scope.target_justices must be a non-empty Justice name list.")
    normalized_justices = [value.strip() for value in target_justices]
    if len(normalized_justices) != len(set(normalized_justices)):
        raise ValueError("annotation_scope.target_justices must not contain duplicates.")
    limits = annotation_scope.get("advocate_group_limits")
    if not isinstance(limits, dict) or not limits:
        raise ValueError("annotation_scope.advocate_group_limits must select at least one advocate group.")
    normalized_limits: Dict[str, Any] = {}
    for group, limit in limits.items():
        if group not in APPROVED_ADVOCATE_GROUPS:
            raise ValueError(
                f"New Justice runs only allow advocate groups {sorted(APPROVED_ADVOCATE_GROUPS)!r}; "
                f"received {group!r}."
            )
        if limit == "all":
            normalized_limits[group] = "all"
        elif isinstance(limit, int) and not isinstance(limit, bool) and limit > 0:
            normalized_limits[group] = limit
        else:
            raise ValueError("Every advocate-group limit must be a positive integer or lowercase 'all'.")
    return {
        "selected_oyez_argument_ids": list(selected),
        "oral_argument_brief_scopes": rows,
        "phase_times_by_argument": phase_times,
        "annotation_scope": {
            "target_justices": normalized_justices,
            "advocate_group_limits": normalized_limits,
        },
    }


def normalize_run_scope(
    scope: Mapping[str, Any], oral_manifest: Mapping[str, Any], brief_manifest: Mapping[str, Any]
) -> Dict[str, Any]:
    """Validate and canonicalize the executor-provided scope without creating a run."""
    return _scope_payload(scope, oral_manifest, brief_manifest)


def _validate_phase_times(payload: Mapping[str, Any], selected: set[str]) -> None:
    for argument_id, phases in payload.items():
        if argument_id not in selected or not isinstance(phases, dict):
            raise ValueError("phase_times_by_argument may only contain selected arguments and object values.")
        ranges: List[tuple[float, float]] = []
        for phase, bounds in phases.items():
            if not isinstance(phase, str) or not isinstance(bounds, dict):
                raise ValueError("Every manual phase range must be one named JSON object.")
            start = bounds.get("start_time")
            end = bounds.get("end_time")
            if isinstance(start, bool) or isinstance(end, bool) or not isinstance(start, (int, float)) or not isinstance(end, (int, float)):
                raise ValueError("Manual phase start_time and end_time must be finite numbers.")
            start_value = float(start)
            end_value = float(end)
            if not (start_value >= 0 and end_value >= 0 and start_value <= end_value):
                raise ValueError("Manual phase times must be non-negative and start_time <= end_time.")
            if not math.isfinite(start_value) or not math.isfinite(end_value):
                raise ValueError("Manual phase times must be finite.")
            ranges.append((start_value, end_value))
        for index, left in enumerate(ranges):
            for right in ranges[index + 1 :]:
                if max(left[0], right[0]) <= min(left[1], right[1]):
                    raise ValueError("Manual phase ranges for the same oral argument must not overlap.")


def create_run(
    *,
    model_profile: str,
    resolved_model_roles: List[Dict[str, str]],
    brief_snapshot_id: str,
    utterance_snapshot_id: str,
    scope: Mapping[str, Any],
    upstream_human_final_checkpoint_id: str = "",
    upstream_human_final_checkpoint_hash: str = "",
    upstream_human_final_checkpoint_stage: str = "",
    start_stage: str = "stage1",
    run_id: Optional[str] = None,
) -> Path:
    if not model_profile or not isinstance(resolved_model_roles, list) or len(resolved_model_roles) != 3:
        raise ValueError("Justice run creation requires one validated model profile and three resolved model roles.")
    if start_stage not in {"stage1", "stage2", "stage3"}:
        raise ValueError("Justice run start_stage must be stage1, stage2, or stage3.")
    if upstream_human_final_checkpoint_id:
        expected_start = {"stage1": "stage2", "stage2": "stage3"}.get(upstream_human_final_checkpoint_stage)
        if not upstream_human_final_checkpoint_hash or expected_start != start_stage:
            raise ValueError("Upstream checkpoint identity, hash, Stage, and new-run start Stage are inconsistent.")
    elif upstream_human_final_checkpoint_hash or upstream_human_final_checkpoint_stage or start_stage != "stage1":
        raise ValueError("A run without an upstream human-final checkpoint must start at Stage 1.")
    brief_manifest = verify_brief_snapshot(brief_snapshot_id)
    oral_manifest = verify_utterance_snapshot(utterance_snapshot_id)
    normalized_scope = _scope_payload(scope, oral_manifest, brief_manifest)
    chosen = run_id or new_run_id()
    if not isinstance(chosen, str) or not chosen or Path(chosen).name != chosen or chosen in {".", ".."}:
        raise ValueError("Justice run ID must use the program-generated safe identifier format.")
    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    run_dir = RUN_ROOT / chosen
    run_dir.mkdir(parents=False, exist_ok=False)
    try:
        (run_dir / "00_run_preflight").mkdir()
        initial_manifest = {
            "manifest_version": FUTURE_MANIFEST_VERSION,
            "sealed": False,
            "run_id": chosen,
            "model_profile": model_profile,
            "resolved_model_roles": resolved_model_roles,
            "brief_snapshot_id": brief_snapshot_id,
            "brief_snapshot_hash": brief_manifest["overall_snapshot_sha256"],
            "utterance_snapshot_id": utterance_snapshot_id,
            "utterance_snapshot_hash": oral_manifest["overall_snapshot_sha256"],
            "run_scope": normalized_scope,
            "run_scope_sha256": sha256_json(normalized_scope),
            "upstream_human_final_checkpoint_id": upstream_human_final_checkpoint_id,
            "upstream_human_final_checkpoint_hash": upstream_human_final_checkpoint_hash,
            "upstream_human_final_checkpoint_stage": upstream_human_final_checkpoint_stage,
            "start_stage": start_stage,
            "created_at": utc_now(),
        }
        _atomic_json(manifest_path(run_dir), initial_manifest)
        write_run_state(run_dir, "preflight_in_progress", "preflight")
    except Exception:
        import shutil

        shutil.rmtree(run_dir, ignore_errors=True)
        raise
    return run_dir


def mark_abandoned(run_id: str) -> Dict[str, Any]:
    run_dir, state, _ = load_run(run_id)
    if state["status"] in TERMINAL_STATUSES:
        raise RuntimeError(f"Run {run_id!r} is already terminal and cannot be abandoned again.")
    return write_run_state(run_dir, "abandoned", state.get("current_stage") or "")


def list_runs(*, mark_interrupted_failed: bool = True) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not RUN_ROOT.exists():
        return rows
    for run_dir in sorted(path for path in RUN_ROOT.iterdir() if path.is_dir()):
        if not state_path(run_dir).is_file() or not manifest_path(run_dir).is_file():
            continue
        try:
            state = read_run_state(run_dir)
            manifest = read_run_manifest(run_dir)
            if manifest.get("manifest_version") != FUTURE_MANIFEST_VERSION:
                continue
            if state["run_id"] != manifest["run_id"] or state["run_id"] != run_dir.name:
                raise RuntimeError("Run identities do not match.")
            if mark_interrupted_failed and state["status"] == "preflight_in_progress":
                state = write_run_state(run_dir, "failed", state.get("current_stage") or "")
            if mark_interrupted_failed and state["status"] == "stage_in_progress":
                state = write_run_state(run_dir, "paused_recoverable", state.get("current_stage") or "")
            review_records_path = run_dir / "review_checkpoints.json"
            final_records_path = run_dir / "human_final_checkpoints.json"
            review_records = json.loads(review_records_path.read_text(encoding="utf-8")) if review_records_path.exists() else {}
            final_records = json.loads(final_records_path.read_text(encoding="utf-8")) if final_records_path.exists() else {}
            stage = str(state.get("current_stage") or "")
            scope = manifest.get("run_scope") or {}
            annotation_scope = scope.get("annotation_scope") or {}
            scope_summary = {
                "target_justices": annotation_scope.get("target_justices") or [],
                "advocate_group_limits": annotation_scope.get("advocate_group_limits") or {},
                "selected_oyez_argument_ids": scope.get("selected_oyez_argument_ids") or [],
            }
            rows.append(
                {
                    "run_id": run_dir.name,
                    "scope_summary": scope_summary,
                    "model_profile": manifest.get("model_profile", ""),
                    "status": state["status"],
                    "current_stage": state.get("current_stage", ""),
                    "brief_snapshot_id": manifest.get("brief_snapshot_id", ""),
                    "utterance_snapshot_id": manifest.get("utterance_snapshot_id", ""),
                    "upstream_human_final_checkpoint_id": manifest.get("upstream_human_final_checkpoint_id", ""),
                    "review_checkpoint_id": (review_records.get(stage) or {}).get("checkpoint_id", ""),
                    "human_final_checkpoint_id": (final_records.get(stage) or {}).get("checkpoint_id", ""),
                    "continuable": state["status"] in CONTINUABLE_STATUSES,
                }
            )
        except Exception as exc:
            rows.append({"run_id": run_dir.name, "valid": False, "error": str(exc), "continuable": False})
    return rows
