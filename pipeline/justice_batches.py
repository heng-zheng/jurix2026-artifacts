from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence

from .common import safe_name
from .justice_runtime import sha256_file, sha256_json


STAGE_DIRECTORIES = {
    "stage1": "05_question_purpose",
    "stage2": "06_reason_alignment",
    "stage3": "07_reason_stance",
}
UNIT_KEYS = {"stage1": "utterance_id", "stage2": "utterance_id", "stage3": "bundle_id"}


class BatchIntegrityError(RuntimeError):
    pass


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def stage_batch_dir(run_dir: Path, stage: str) -> Path:
    try:
        return run_dir / STAGE_DIRECTORIES[stage] / "batch_execution"
    except KeyError as exc:
        raise ValueError(f"Unknown Justice Stage: {stage!r}.") from exc


def _paths(run_dir: Path, stage: str) -> Dict[str, Path]:
    root = stage_batch_dir(run_dir, stage)
    return {
        "root": root,
        "plan": root / "unit_plan.json",
        "plan_hash": root / "unit_plan.sha256",
        "audit": root / "batch_execution_audit.jsonl",
        "audit_head": root / "batch_execution_audit_head.json",
        "progress": root / "progress_summary.json",
        "aggregate": root / "final_aggregate_manifest.json",
        "aggregate_hash": root / "final_aggregate_manifest.sha256",
        "units": root / "units",
    }


def _unit_record(stage: str, row: Mapping[str, Any]) -> Dict[str, Any]:
    key = UNIT_KEYS[stage]
    unit_id = str(row.get(key) or "")
    if not unit_id:
        raise ValueError(f"{stage} unit is missing {key}.")
    record: Dict[str, Any] = {key: unit_id, "input_sha256": sha256_json(row)}
    if stage == "stage3":
        utterance_id = str(row.get("utterance_id") or "")
        if not utterance_id:
            raise ValueError("Stage 3 unit is missing utterance_id.")
        record["utterance_id"] = utterance_id
    return record


def ensure_unit_plan(run_dir: Path, stage: str, unit_rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    paths = _paths(run_dir, stage)
    units = [_unit_record(stage, row) for row in unit_rows]
    ids = [str(unit[UNIT_KEYS[stage]]) for unit in units]
    if len(ids) != len(set(ids)):
        raise ValueError(f"{stage} ordered unit list contains duplicate IDs.")
    expected = {
        "version": 1,
        "stage": stage,
        "unit_key": UNIT_KEYS[stage],
        "units": units,
        "ordered_unit_ids_sha256": sha256_json(ids),
    }
    if paths["plan"].exists() or paths["plan_hash"].exists():
        return verify_unit_plan(run_dir, stage, unit_rows)
    if (
        (paths["root"].exists() and any(paths["root"].iterdir()))
        or _attempt_audit_path(run_dir, stage).exists()
    ):
        raise BatchIntegrityError(f"{stage} immutable unit plan was deleted after Stage work began.")
    paths["root"].mkdir(parents=True, exist_ok=True)
    _atomic_json(paths["plan"], expected)
    paths["plan_hash"].write_text(sha256_file(paths["plan"]) + "\n", encoding="utf-8")
    paths["audit"].write_text("", encoding="utf-8")
    _atomic_json(paths["audit_head"], {"event_count": 0, "head_event_sha256": ""})
    rebuild_progress(run_dir, stage)
    return expected


def verify_unit_plan(
    run_dir: Path, stage: str, unit_rows: Optional[Sequence[Mapping[str, Any]]] = None
) -> Dict[str, Any]:
    paths = _paths(run_dir, stage)
    if not paths["plan"].is_file() or not paths["plan_hash"].is_file():
        raise BatchIntegrityError(f"{stage} immutable unit plan is missing.")
    expected_hash = paths["plan_hash"].read_text(encoding="utf-8").strip()
    if not expected_hash or sha256_file(paths["plan"]) != expected_hash:
        raise BatchIntegrityError(f"{stage} immutable unit plan changed.")
    plan = json.loads(paths["plan"].read_text(encoding="utf-8"))
    if (
        not isinstance(plan, dict)
        or plan.get("stage") != stage
        or plan.get("unit_key") != UNIT_KEYS[stage]
        or not isinstance(plan.get("units"), list)
    ):
        raise BatchIntegrityError(f"{stage} immutable unit plan is invalid.")
    ids = [str(item.get(UNIT_KEYS[stage]) or "") for item in plan["units"]]
    if (
        any(not value for value in ids)
        or len(ids) != len(set(ids))
        or sha256_json(ids) != plan.get("ordered_unit_ids_sha256")
    ):
        raise BatchIntegrityError(f"{stage} immutable unit order is invalid.")
    if unit_rows is not None:
        current = [_unit_record(stage, row) for row in unit_rows]
        if current != plan["units"]:
            raise BatchIntegrityError(f"{stage} current inputs no longer match the frozen unit plan.")
    return plan


def _safe_unit_name(value: str) -> str:
    return safe_name(value)


def _unit_path(run_dir: Path, stage: str, unit_id: str) -> Path:
    stem = _safe_unit_name(unit_id)
    root = _paths(run_dir, stage)["units"]
    return root / f"{stem}.json"


def _attempt_audit_path(run_dir: Path, stage: str) -> Path:
    filename = {
        "stage1": "question_purpose_attempt_audit.jsonl",
        "stage2": "reason_alignment_attempt_audit.jsonl",
        "stage3": "stance_attempt_audit.jsonl",
    }[stage]
    return run_dir / STAGE_DIRECTORIES[stage] / filename


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise BatchIntegrityError(f"Invalid JSONL at {path}:{line_number}.") from exc
            if not isinstance(value, dict):
                raise BatchIntegrityError(f"Invalid JSONL object at {path}:{line_number}.")
            rows.append(value)
    return rows


def _unit_call_artifacts(run_dir: Path, stage: str, unit_id: str) -> Dict[str, str]:
    stage_dir = run_dir / STAGE_DIRECTORIES[stage]
    prefix = _safe_unit_name(unit_id)
    files: List[Path] = []
    for directory_name in ["model_calls", "adjudication_calls"]:
        directory = stage_dir / directory_name
        if directory.exists():
            files.extend(
                path for path in directory.rglob("*")
                if path.is_file() and path.name.startswith(prefix)
            )
    artifacts: Dict[str, str] = {}
    for path in sorted(set(files)):
        relative = str(path.relative_to(run_dir))
        artifacts[relative] = sha256_file(path)
    return artifacts


def commit_unit(
    run_dir: Path,
    stage: str,
    unit_row: Mapping[str, Any],
    model_rows: Sequence[Mapping[str, Any]],
    consolidated_row: Mapping[str, Any],
) -> Dict[str, Any]:
    unit = _unit_record(stage, unit_row)
    unit_id = str(unit[UNIT_KEYS[stage]])
    result_path = _unit_path(run_dir, stage, unit_id)
    if result_path.exists():
        raise BatchIntegrityError(f"{stage} unit {unit_id!r} is already committed.")
    audit_rows = [
        row for row in _read_jsonl(_attempt_audit_path(run_dir, stage))
        if str(row.get("annotation_unit_id") or "") == unit_id
    ]
    if not any(row.get("call_type") == "routing_summary" for row in audit_rows):
        raise BatchIntegrityError(f"{stage} unit {unit_id!r} has no complete routing result.")
    base_payload = {
        "version": 1,
        "stage": stage,
        "unit": unit,
        "unit_input": dict(unit_row),
        "model_rows": [dict(row) for row in model_rows],
        "consolidated_row": dict(consolidated_row),
        "attempt_audit_rows": audit_rows,
        "call_artifact_hashes": _unit_call_artifacts(run_dir, stage, unit_id),
        "committed_at": _now(),
    }
    payload = {**base_payload, "unit_result_sha256": sha256_json(base_payload)}
    _atomic_json(result_path, payload)
    return payload


def load_committed_units(run_dir: Path, stage: str, *, verify: bool = True) -> List[Dict[str, Any]]:
    plan = verify_unit_plan(run_dir, stage)
    results: List[Dict[str, Any]] = []
    encountered_gap = False
    planned_names = {
        _safe_unit_name(str(unit[UNIT_KEYS[stage]]))
        for unit in plan["units"]
    }
    unit_dir = _paths(run_dir, stage)["units"]
    if unit_dir.exists():
        actual_json = {path.stem for path in unit_dir.glob("*.json")}
        if not actual_json <= planned_names:
            raise BatchIntegrityError(f"{stage} unit result directory has unknown files.")
    attempt_rows = _read_jsonl(_attempt_audit_path(run_dir, stage))
    attempt_ids = [str(row.get("audit_event_id") or "") for row in attempt_rows]
    if any(not value for value in attempt_ids) or len(attempt_ids) != len(set(attempt_ids)):
        raise BatchIntegrityError(f"{stage} attempt audit contains missing or duplicate event IDs.")
    known_ids = {str(item[UNIT_KEYS[stage]]) for item in plan["units"]}
    if any(str(row.get("annotation_unit_id") or "") not in known_ids for row in attempt_rows):
        raise BatchIntegrityError(f"{stage} attempt audit contains an unknown unit.")
    for unit in plan["units"]:
        unit_id = str(unit[UNIT_KEYS[stage]])
        result_path = _unit_path(run_dir, stage, unit_id)
        if not result_path.exists():
            encountered_gap = True
            continue
        if encountered_gap:
            raise BatchIntegrityError(f"{stage} completed units are not a continuous prefix.")
        if not result_path.is_file():
            raise BatchIntegrityError(f"{stage} unit {unit_id!r} has an invalid commit.")
        payload = json.loads(result_path.read_text(encoding="utf-8"))
        stored_result_hash = payload.get("unit_result_sha256")
        base_payload = {key: value for key, value in payload.items() if key != "unit_result_sha256"}
        if not stored_result_hash or sha256_json(base_payload) != stored_result_hash:
            raise BatchIntegrityError(f"{stage} unit {unit_id!r} result changed.")
        if payload.get("stage") != stage or payload.get("unit") != unit:
            raise BatchIntegrityError(f"{stage} unit {unit_id!r} identity changed.")
        if sha256_json(payload.get("unit_input")) != unit.get("input_sha256"):
            raise BatchIntegrityError(f"{stage} unit {unit_id!r} frozen input changed.")
        current_audit = [
            row for row in attempt_rows if str(row.get("annotation_unit_id") or "") == unit_id
        ]
        if current_audit != payload.get("attempt_audit_rows"):
            raise BatchIntegrityError(f"{stage} unit {unit_id!r} attempt audit changed.")
        expected_artifacts = payload.get("call_artifact_hashes")
        if not isinstance(expected_artifacts, dict):
            raise BatchIntegrityError(f"{stage} unit {unit_id!r} artifact manifest is invalid.")
        if verify:
            for relative, expected in expected_artifacts.items():
                path = run_dir / relative
                if not path.is_file() or sha256_file(path) != expected:
                    raise BatchIntegrityError(f"{stage} unit {unit_id!r} call artifact changed or is missing.")
            if _unit_call_artifacts(run_dir, stage, unit_id) != expected_artifacts:
                raise BatchIntegrityError(f"{stage} unit {unit_id!r} call artifacts contain unexpected changes.")
        results.append(payload)
    return results


def verify_stage_progress(
    run_dir: Path, stage: str, unit_rows: Optional[Sequence[Mapping[str, Any]]] = None
) -> Dict[str, Any]:
    plan = verify_unit_plan(run_dir, stage, unit_rows)
    results = load_committed_units(run_dir, stage)
    completed_ids = [str(row["unit"][UNIT_KEYS[stage]]) for row in results]
    planned_ids = [str(row[UNIT_KEYS[stage]]) for row in plan["units"]]
    return {
        "stage": stage,
        "total_units": len(planned_ids),
        "completed_units": len(completed_ids),
        "completed_unit_ids": completed_ids,
        "next_unit_id": planned_ids[len(completed_ids)] if len(completed_ids) < len(planned_ids) else "",
        "complete": len(completed_ids) == len(planned_ids),
    }


def rebuild_progress(run_dir: Path, stage: str) -> Dict[str, Any]:
    summary = verify_stage_progress(run_dir, stage)
    summary["updated_at"] = _now()
    _atomic_json(_paths(run_dir, stage)["progress"], summary)
    return summary


def _audit_events(run_dir: Path, stage: str) -> List[Dict[str, Any]]:
    paths = _paths(run_dir, stage)
    if paths["plan"].exists() and (
        not paths["audit"].is_file() or not paths["audit_head"].is_file()
    ):
        raise BatchIntegrityError(f"{stage} batch execution audit is missing.")
    rows = _read_jsonl(paths["audit"])
    previous = ""
    for row in rows:
        stored_hash = row.get("event_sha256")
        base = {key: value for key, value in row.items() if key != "event_sha256"}
        if base.get("previous_event_sha256") != previous or sha256_json(base) != stored_hash:
            raise BatchIntegrityError(f"{stage} batch execution audit changed.")
        previous = str(stored_hash)
    if paths["audit_head"].exists():
        head = json.loads(paths["audit_head"].read_text(encoding="utf-8"))
        if head != {"event_count": len(rows), "head_event_sha256": previous}:
            raise BatchIntegrityError(f"{stage} batch execution audit was truncated or its head changed.")
    return rows


def append_batch_event(run_dir: Path, stage: str, event: Mapping[str, Any]) -> Dict[str, Any]:
    path = _paths(run_dir, stage)["audit"]
    rows = _audit_events(run_dir, stage)
    base = {
        "event_number": len(rows) + 1,
        "recorded_at": _now(),
        **dict(event),
        "previous_event_sha256": str(rows[-1]["event_sha256"]) if rows else "",
    }
    row = {**base, "event_sha256": sha256_json(base)}
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    _atomic_json(
        _paths(run_dir, stage)["audit_head"],
        {"event_count": len(rows) + 1, "head_event_sha256": row["event_sha256"]},
    )
    return row


def active_batch(run_dir: Path, stage: str) -> Optional[Dict[str, Any]]:
    active: Dict[str, Dict[str, Any]] = {}
    for event in _audit_events(run_dir, stage):
        batch_id = str(event.get("batch_id") or "")
        if event.get("event") == "batch_started":
            active[batch_id] = event
        elif event.get("event") in {"batch_completed", "batch_closed"}:
            active.pop(batch_id, None)
    if len(active) > 1:
        raise BatchIntegrityError(f"{stage} has more than one active batch.")
    return next(iter(active.values()), None)


def parse_batch_size(value: str, remaining: int) -> tuple[str, int]:
    answer = value.strip()
    if answer == "all":
        return "all", remaining
    if not answer.isdigit() or int(answer) <= 0:
        raise ValueError("Batch size must be a positive integer or lowercase 'all'.")
    return answer, min(int(answer), remaining)


def choose_batch(
    run_dir: Path,
    stage: str,
    *,
    input_func: Optional[Callable[[str], str]] = None,
) -> Dict[str, Any]:
    ask = input_func or input
    progress = verify_stage_progress(run_dir, stage)
    if progress["complete"]:
        raise RuntimeError(f"{stage} is already complete.")
    existing = active_batch(run_dir, stage)
    if existing:
        answer = ""
        while answer not in {"resume", "new"}:
            answer = ask(
                f"{stage} has an interrupted batch {existing['batch_id']}. "
                "Enter 'resume' to continue it or 'new' to close it and re-batch unfinished units: "
            ).strip().lower()
        if answer == "resume":
            completed = set(progress["completed_unit_ids"])
            remaining = [value for value in existing["planned_unit_ids"] if value not in completed]
            return {**existing, "planned_unit_ids": remaining, "resumed": True}
        append_batch_event(
            run_dir, stage, {"event": "batch_closed", "batch_id": existing["batch_id"],
                             "reason": "user_rebatch_from_first_unfinished_unit"}
        )
    remaining_ids = [
        str(item[UNIT_KEYS[stage]])
        for item in verify_unit_plan(run_dir, stage)["units"][progress["completed_units"] :]
    ]
    previous_sizes = [
        str(row.get("batch_size"))
        for row in _audit_events(run_dir, stage)
        if row.get("event") == "batch_started"
    ]
    previous_label = f" Last batch size: {previous_sizes[-1]}." if previous_sizes else ""
    while True:
        raw = ask(
            f"Enter {stage} batch size (positive integer or lowercase 'all').{previous_label} "
        )
        try:
            label, count = parse_batch_size(raw, len(remaining_ids))
            break
        except ValueError:
            print("Invalid batch size. Enter a positive integer or lowercase 'all'.")
    events = _audit_events(run_dir, stage)
    batch_id = f"{stage}_batch_{1 + sum(row.get('event') == 'batch_started' for row in events):04d}"
    event = append_batch_event(
        run_dir,
        stage,
        {
            "event": "batch_started",
            "batch_id": batch_id,
            "batch_size": label,
            "planned_unit_ids": remaining_ids[:count],
            "completed_unit_ids_at_start": progress["completed_unit_ids"],
        },
    )
    return {**event, "resumed": False}


def finish_batch(run_dir: Path, stage: str, batch: Mapping[str, Any], *, paused: bool = False) -> Dict[str, Any]:
    progress = rebuild_progress(run_dir, stage)
    planned = list(batch.get("planned_unit_ids") or [])
    completed = [value for value in planned if value in set(progress["completed_unit_ids"])]
    append_batch_event(
        run_dir,
        stage,
        {
            "event": "batch_paused" if paused else "batch_completed",
            "batch_id": batch["batch_id"],
            "planned_unit_ids": planned,
            "completed_unit_ids": completed,
            "next_unit_id": progress["next_unit_id"],
        },
    )
    return progress


def aggregate_rows(run_dir: Path, stage: str) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    progress = verify_stage_progress(run_dir, stage)
    if not progress["complete"]:
        raise BatchIntegrityError(f"{stage} cannot aggregate before every frozen unit is committed.")
    results = load_committed_units(run_dir, stage)
    return (
        [dict(row) for result in results for row in result["model_rows"]],
        [dict(result["consolidated_row"]) for result in results],
    )


def _stage_file_hashes(run_dir: Path, stage: str) -> Dict[str, str]:
    paths = _paths(run_dir, stage)
    stage_dir = run_dir / STAGE_DIRECTORIES[stage]
    excluded = {paths["aggregate"], paths["aggregate_hash"]}
    return {
        str(path.relative_to(run_dir)): sha256_file(path)
        for path in sorted(stage_dir.rglob("*"))
        if path.is_file() and path not in excluded
    }


def seal_stage_aggregate(run_dir: Path, stage: str) -> Dict[str, Any]:
    progress = verify_stage_progress(run_dir, stage)
    if not progress["complete"]:
        raise BatchIntegrityError(f"{stage} cannot seal aggregate outputs before all units are complete.")
    paths = _paths(run_dir, stage)
    if paths["aggregate"].exists() or paths["aggregate_hash"].exists():
        raise BatchIntegrityError(f"{stage} aggregate output manifest already exists.")
    payload = {
        "version": 1,
        "stage": stage,
        "ordered_unit_ids_sha256": verify_unit_plan(run_dir, stage)["ordered_unit_ids_sha256"],
        "stage_file_hashes": _stage_file_hashes(run_dir, stage),
    }
    _atomic_json(paths["aggregate"], payload)
    paths["aggregate_hash"].write_text(sha256_file(paths["aggregate"]) + "\n", encoding="utf-8")
    return payload


def verify_stage_aggregate(run_dir: Path, stage: str) -> Dict[str, Any]:
    paths = _paths(run_dir, stage)
    verify_stage_progress(run_dir, stage)
    if not paths["aggregate"].is_file() or not paths["aggregate_hash"].is_file():
        raise BatchIntegrityError(f"{stage} aggregate output manifest is missing.")
    if sha256_file(paths["aggregate"]) != paths["aggregate_hash"].read_text(encoding="utf-8").strip():
        raise BatchIntegrityError(f"{stage} aggregate output manifest changed.")
    payload = json.loads(paths["aggregate"].read_text(encoding="utf-8"))
    if (
        payload.get("stage") != stage
        or payload.get("ordered_unit_ids_sha256")
        != verify_unit_plan(run_dir, stage).get("ordered_unit_ids_sha256")
        or payload.get("stage_file_hashes") != _stage_file_hashes(run_dir, stage)
    ):
        raise BatchIntegrityError(f"{stage} aggregate outputs changed, were deleted, or contain unexpected files.")
    return payload
