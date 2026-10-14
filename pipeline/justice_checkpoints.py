from __future__ import annotations

import csv
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

from .common import read_jsonl, write_jsonl
from .justice_checkpoint_common import checkpoint_overall_hash, load_human_final_checkpoint
from .justice_contracts import PURPOSE_LABEL_ORDER, PURPOSE_LABELS, STANCE_OUTCOMES, validate_human_terminal_selection
from .justice_lifecycle import (
    HUMAN_FINAL_CHECKPOINT_ROOT,
    PENDING_CHECKPOINT_ROOT,
    REVIEW_CHECKPOINT_ROOT,
    load_run,
    read_run_state,
    write_run_state,
)
from .justice_runtime import sha256_file, sha256_json
from .justice_snapshots import immutable_id


STAGE_REVIEW = {
    "stage1": {
        "row_id": "utterance_id",
        "student": {"student_id", "student_question_function_present", "student_purposes", "student_notes"},
        "professor": {
            "student_id", "student_question_function_present", "student_purposes", "student_notes",
            "professor_final_question_function_present", "professor_final_purposes", "professor_notes",
        },
        "final_path": "05_question_purpose/question_purpose_final.jsonl",
    },
    "stage2": {
        "row_id": "utterance_id",
        "student": {"student_id", "student_selected_reason_ids", "student_none_of_the_above", "student_notes"},
        "professor": {
            "student_id", "student_selected_reason_ids", "student_none_of_the_above", "student_notes",
            "professor_final_selected_reason_ids", "professor_none_of_the_above", "professor_notes",
        },
        "final_path": "06_reason_alignment/reason_alignment_final.jsonl",
    },
    "stage3": {
        "row_id": "bundle_id",
        "student": {"student_id", "student_stance_outcome", "student_notes"},
        "professor": {
            "student_id", "student_stance_outcome", "student_notes",
            "professor_final_stance_outcome", "professor_notes",
        },
        "final_path": "07_reason_stance/stance_final.jsonl",
    },
}


class ReviewCheckpointIntegrityError(RuntimeError):
    pass


def _read_csv(path: Path) -> tuple[List[str], List[Dict[str, str]]]:
    if not path.is_file():
        raise FileNotFoundError(f"Review CSV is missing: {path}.")
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or []), [dict(row) for row in reader]


def _write_csv(path: Path, fieldnames: Sequence[str], rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames), extrasaction="raise")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def _canonical_cell(value: str) -> Any:
    text = str(value or "")
    if text.startswith(("[", "{")):
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return text
    return text


def _copy_artifacts(run_dir: Path, pending: Path, artifact_paths: Iterable[Path]) -> Dict[str, str]:
    hashes: Dict[str, str] = {}
    for source in artifact_paths:
        if not source.exists():
            raise FileNotFoundError(f"Required review-checkpoint artifact path is missing: {source}.")
        if source.is_dir():
            files = sorted(path for path in source.rglob("*") if path.is_file())
        else:
            files = [source]
        for path in files:
            relative_run = path.relative_to(run_dir)
            target = pending / "artifacts" / relative_run
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, target)
            relative = str(target.relative_to(pending))
            hashes[relative] = sha256_file(target)
    return hashes


def _source_file_hashes(run_dir: Path, paths: Iterable[Path]) -> Dict[str, str]:
    hashes: Dict[str, str] = {}
    for source in paths:
        if not source.exists():
            raise FileNotFoundError(f"Required review-checkpoint artifact path is missing: {source}.")
        files = sorted(path for path in source.rglob("*") if path.is_file()) if source.is_dir() else [source]
        for path in files:
            try:
                relative = str(path.relative_to(run_dir))
            except ValueError as exc:
                raise RuntimeError("Review checkpoint sources must be contained in the source run directory.") from exc
            digest = sha256_file(path)
            if relative in hashes and hashes[relative] != digest:
                raise RuntimeError("A review checkpoint source path resolved to inconsistent content.")
            hashes[relative] = digest
    return hashes


def _review_source_identity(
    run_dir: Path,
    student_working_path: Path,
    professor_working_path: Path,
    artifact_paths: Sequence[Path],
) -> Dict[str, Any]:
    csv_hashes = _source_file_hashes(run_dir, [student_working_path, professor_working_path])
    return {
        "student_working_csv": {
            "relative_path": str(student_working_path.relative_to(run_dir)),
            "sha256": csv_hashes[str(student_working_path.relative_to(run_dir))],
        },
        "professor_working_csv": {
            "relative_path": str(professor_working_path.relative_to(run_dir)),
            "sha256": csv_hashes[str(professor_working_path.relative_to(run_dir))],
        },
        "artifact_file_hashes": _source_file_hashes(run_dir, artifact_paths),
    }


def _clear_pending_payload(pending: Path) -> None:
    for path in list(pending.iterdir()):
        if path.name == "pending_identity.json":
            continue
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()


def _manifest_write(directory: Path, manifest: Dict[str, Any], hashes: Dict[str, str]) -> Dict[str, Any]:
    payload = dict(manifest)
    payload["file_hashes"] = dict(hashes)
    payload["overall_checkpoint_sha256"] = checkpoint_overall_hash(payload, hashes)
    (directory / "checkpoint_manifest.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8"
    )
    return payload


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
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


def _matching_checkpoint_directories(root: Path, run_id: str, stage: str, checkpoint_type: str) -> List[Path]:
    matches: List[Path] = []
    if not root.exists():
        return matches
    manifest_name = "pending_identity.json" if root == PENDING_CHECKPOINT_ROOT else "checkpoint_manifest.json"
    for directory in root.iterdir():
        path = directory / manifest_name
        if not directory.is_dir() or not path.is_file():
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if (
            payload.get("source_run_id") == run_id
            and payload.get("stage") == stage
            and payload.get("checkpoint_type") == checkpoint_type
        ):
            matches.append(directory)
    return sorted(matches)


def create_review_checkpoint(
    run_dir: Path,
    stage: str,
    *,
    student_working_path: Path,
    professor_working_path: Path,
    artifact_paths: Sequence[Path],
    checkpoint_id: str = "",
    require_existing_pending: bool = False,
) -> Dict[str, Any]:
    if stage not in STAGE_REVIEW:
        raise ValueError(f"Unknown review stage: {stage!r}.")
    state = read_run_state(run_dir)
    expected_status = "review_checkpoint_pending"
    if state.get("status") != expected_status or state.get("current_stage") != stage:
        raise RuntimeError("Review checkpoint creation requires the matching review_checkpoint_pending state.")
    completed = _matching_checkpoint_directories(REVIEW_CHECKPOINT_ROOT, run_dir.name, stage, "review")
    if len(completed) > 1:
        raise RuntimeError("More than one completed review checkpoint matches this run and stage.")
    if completed:
        if not require_existing_pending:
            raise RuntimeError("A review checkpoint already exists for this run and stage.")
        directory, manifest = load_review_checkpoint(completed[0].name)
        _finish_review_bookkeeping(run_dir, manifest, directory, student_working_path, professor_working_path)
        return manifest
    chosen = checkpoint_id or immutable_id()
    destination = REVIEW_CHECKPOINT_ROOT / chosen
    if destination.exists():
        raise FileExistsError(f"Review checkpoint already exists: {chosen!r}.")
    pending = PENDING_CHECKPOINT_ROOT / chosen
    matching = _matching_checkpoint_directories(PENDING_CHECKPOINT_ROOT, run_dir.name, stage, "review")
    if checkpoint_id:
        matching = [path for path in matching if path.name == checkpoint_id]
    if len(matching) > 1:
        raise RuntimeError("More than one pending review checkpoint matches this run and stage.")
    if require_existing_pending and len(matching) != 1:
        raise RuntimeError("The pending review checkpoint cannot be uniquely identified for deterministic recovery.")
    current_source_identity = _review_source_identity(
        run_dir, student_working_path, professor_working_path, artifact_paths
    )
    if matching:
        pending = matching[0]
        chosen = pending.name
        destination = REVIEW_CHECKPOINT_ROOT / chosen
        identity_path = pending / "pending_identity.json"
        identity = json.loads(identity_path.read_text(encoding="utf-8"))
        if (
            identity.get("checkpoint_id") != chosen
            or identity.get("source_run_id") != run_dir.name
            or identity.get("stage") != stage
            or identity.get("checkpoint_type") != "review"
            or identity.get("source_identity") != current_source_identity
        ):
            raise ReviewCheckpointIntegrityError(
                "Pending review checkpoint sources changed after the transaction began; recovery is refused."
            )
        _clear_pending_payload(pending)
    else:
        pending.mkdir(parents=True, exist_ok=False)
        _atomic_json(
            pending / "pending_identity.json",
            {
                "checkpoint_id": chosen,
                "source_run_id": run_dir.name,
                "stage": stage,
                "checkpoint_type": "review",
                "source_identity": current_source_identity,
            },
        )
    try:
        if _review_source_identity(run_dir, student_working_path, professor_working_path, artifact_paths) != current_source_identity:
            raise ReviewCheckpointIntegrityError(
                "Review checkpoint sources changed while the pending transaction was being built."
            )
        student_fields, student_rows = _read_csv(student_working_path)
        professor_fields, professor_rows = _read_csv(professor_working_path)
        if "review_checkpoint_id" not in student_fields:
            student_fields = ["review_checkpoint_id", "review_checkpoint_hash", *student_fields]
        if "review_checkpoint_id" not in professor_fields:
            professor_fields = ["review_checkpoint_id", "review_checkpoint_hash", *professor_fields]
        artifact_hashes = _copy_artifacts(run_dir, pending, artifact_paths)
        artifact_hashes["pending_identity.json"] = sha256_file(pending / "pending_identity.json")
        identity_hash = sha256_json(
            {
                "checkpoint_id": chosen,
                "source_run_id": run_dir.name,
                "stage": stage,
                "student_rows": student_rows,
                "professor_rows": professor_rows,
                "artifact_hashes": artifact_hashes,
            }
        )
        student_frozen = [
            {**row, "review_checkpoint_id": chosen, "review_checkpoint_hash": identity_hash} for row in student_rows
        ]
        professor_frozen = [
            {**row, "review_checkpoint_id": chosen, "review_checkpoint_hash": identity_hash} for row in professor_rows
        ]
        student_target = pending / "review" / "student_review.csv"
        professor_target = pending / "review" / "professor_review.csv"
        _write_csv(student_target, student_fields, student_frozen)
        _write_csv(professor_target, professor_fields, professor_frozen)
        artifact_hashes[str(student_target.relative_to(pending))] = sha256_file(student_target)
        artifact_hashes[str(professor_target.relative_to(pending))] = sha256_file(professor_target)
        manifest = _manifest_write(
            pending,
            {
                "checkpoint_type": "review",
                "checkpoint_id": chosen,
                "source_run_id": run_dir.name,
                "stage": stage,
                "review_checkpoint_hash": identity_hash,
                "student_review_relative_path": "review/student_review.csv",
                "professor_review_relative_path": "review/professor_review.csv",
            },
            artifact_hashes,
        )
        REVIEW_CHECKPOINT_ROOT.mkdir(parents=True, exist_ok=True)
        os.replace(pending, destination)
        _finish_review_bookkeeping(run_dir, manifest, destination, student_working_path, professor_working_path)
        return manifest
    except Exception:
        # Keep the pending identity so the deterministic build can be retried.
        raise


def _finish_review_bookkeeping(
    run_dir: Path,
    manifest: Mapping[str, Any],
    directory: Path,
    student_working_path: Path,
    professor_working_path: Path,
) -> None:
    student_fields, student_rows = _read_csv(directory / str(manifest["student_review_relative_path"]))
    professor_fields, professor_rows = _read_csv(directory / str(manifest["professor_review_relative_path"]))
    _write_csv(student_working_path, student_fields, student_rows)
    _write_csv(professor_working_path, professor_fields, professor_rows)
    stage = str(manifest["stage"])
    record_path = run_dir / "review_checkpoints.json"
    records = json.loads(record_path.read_text(encoding="utf-8")) if record_path.exists() else {}
    records[stage] = {
        "checkpoint_id": manifest["checkpoint_id"],
        "checkpoint_hash": manifest["overall_checkpoint_sha256"],
        "review_checkpoint_hash": manifest["review_checkpoint_hash"],
    }
    _atomic_json(record_path, records)
    write_run_state(run_dir, f"awaiting_{stage}_human", stage)


def load_review_checkpoint(checkpoint_id: str) -> tuple[Path, Dict[str, Any]]:
    if not isinstance(checkpoint_id, str) or not checkpoint_id or Path(checkpoint_id).name != checkpoint_id:
        raise ValueError("A review checkpoint must be selected by its exact program-generated ID.")
    directory = REVIEW_CHECKPOINT_ROOT / checkpoint_id
    path = directory / "checkpoint_manifest.json"
    if not path.is_file():
        raise FileNotFoundError(f"Review checkpoint does not exist: {checkpoint_id!r}.")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("checkpoint_type") != "review" or manifest.get("checkpoint_id") != checkpoint_id:
        raise RuntimeError("Review checkpoint identity/type is invalid.")
    hashes = manifest.get("file_hashes") or {}
    for relative, expected in hashes.items():
        candidate = directory / relative
        if not candidate.is_file() or sha256_file(candidate) != expected:
            raise RuntimeError(f"Review checkpoint file is missing or changed: {relative!r}.")
    actual_files = {
        str(path.relative_to(directory))
        for path in directory.rglob("*")
        if path.is_file() and path.name != "checkpoint_manifest.json"
    }
    if actual_files != set(hashes):
        raise RuntimeError("Review checkpoint payload file set does not match its immutable manifest.")
    if manifest.get("overall_checkpoint_sha256") != checkpoint_overall_hash(manifest, hashes):
        raise RuntimeError("Review checkpoint overall hash verification failed.")
    return directory, manifest


def _parse_bool(value: str) -> Optional[bool]:
    if value == "true":
        return True
    if value == "false":
        return False
    return None


def _parse_list(value: str) -> Optional[List[Any]]:
    try:
        parsed = json.loads(value)
    except (TypeError, json.JSONDecodeError):
        return None
    return parsed if isinstance(parsed, list) else None


def _purpose_labels(value: str) -> Optional[List[str]]:
    values = _parse_list(value)
    if values is None:
        return None
    normalized: List[str] = []
    for item in values:
        if not isinstance(item, str):
            return None
        matches = [label for label in PURPOSE_LABEL_ORDER if label.lower() == item.strip().lower()]
        if len(matches) != 1:
            return None
        normalized.append(matches[0])
    if len(normalized) != len(set(normalized)):
        return None
    return normalized


def _rows_by_id(rows: Sequence[Dict[str, str]], key: str, *, complete: bool, expected: set[str]) -> Dict[str, Dict[str, str]]:
    output: Dict[str, Dict[str, str]] = {}
    for row in rows:
        row_id = str(row.get(key) or "")
        if not row_id or row_id in output or row_id not in expected:
            raise ValueError(f"Human review contains a missing, duplicate, or unexpected {key}.")
        output[row_id] = row
    if complete and set(output) != expected:
        raise ValueError("Professor final must completely and uniquely cover every expected review row.")
    return output


def _frozen_compare(
    submitted: Mapping[str, str], original: Mapping[str, str], *, allowed: set[str], checkpoint_id: str, checkpoint_hash: str
) -> None:
    if set(submitted) != set(original):
        raise ValueError("Human review columns must exactly match the checkpoint template columns.")
    if submitted.get("review_checkpoint_id") != checkpoint_id or submitted.get("review_checkpoint_hash") != checkpoint_hash:
        raise ValueError("Human review checkpoint ID/hash does not match the immutable checkpoint.")
    for field in original:
        if field in allowed:
            continue
        if _canonical_cell(submitted.get(field, "")) != _canonical_cell(original.get(field, "")):
            raise ValueError(f"Human review changed frozen field {field!r}.")


def _identity_from_review(row: Mapping[str, str]) -> Dict[str, str]:
    return {
        "oyez_argument_id": row.get("oyez_argument_id", ""),
        "utterance_id": row.get("utterance_id", ""),
        "turn_id": row.get("turn_id", ""),
        "target_justice": row.get("target_justice", ""),
        "text": row.get("text") or row.get("utterance", ""),
    }


def _stage2_candidate_nodes(row: Mapping[str, str], reason_nodes: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    candidate_ids = _parse_list(row.get("candidate_reason_ids", ""))
    if (
        candidate_ids is None
        or not candidate_ids
        or any(not isinstance(value, str) or not value for value in candidate_ids)
        or len(candidate_ids) != len(set(candidate_ids))
    ):
        raise RuntimeError("Frozen Stage 2 review row has an invalid candidate_reason_ids list.")
    allowed = set(candidate_ids)
    nodes = [node for node in reason_nodes if str(node.get("reason_target_id") or "") in allowed]
    if len(nodes) != len(allowed):
        raise RuntimeError("Frozen Stage 2 candidate pool is not completely present in the validated reason hierarchy.")
    return nodes


def _stage2_candidate_scope_sidecar(
    professor_rows: Sequence[Mapping[str, str]], clean_rows: Sequence[Mapping[str, Any]]
) -> List[Dict[str, Any]]:
    clean_ids = [str(row.get("utterance_id") or "") for row in clean_rows]
    if any(not value for value in clean_ids) or len(clean_ids) != len(set(clean_ids)):
        raise RuntimeError("Stage 2 clean human final has missing or duplicate utterance IDs.")
    by_id: Dict[str, Dict[str, Any]] = {}
    for row in professor_rows:
        utterance_id = str(row.get("utterance_id") or "")
        argument_id = str(row.get("oyez_argument_id") or "")
        candidate_ids = _parse_list(row.get("candidate_reason_ids", ""))
        targets = _parse_list(row.get("candidate_targets", ""))
        if (
            not utterance_id
            or utterance_id in by_id
            or not argument_id
            or not isinstance(candidate_ids, list)
            or not candidate_ids
            or any(not isinstance(value, str) or not value for value in candidate_ids)
            or len(candidate_ids) != len(set(candidate_ids))
            or not isinstance(targets, list)
            or not targets
            or any(not isinstance(target, dict) for target in targets)
        ):
            raise RuntimeError("Frozen Stage 2 review candidate scope is missing, duplicated, or malformed.")
        target_ids = [str(target.get("reason_target_id") or "") for target in targets]
        brief_ids = [str(target.get("brief_id") or "") for target in targets]
        brief_groups = [str(target.get("brief_group") or "") for target in targets]
        if (
            any(not value for value in target_ids + brief_ids + brief_groups)
            or len(target_ids) != len(set(target_ids))
            or set(target_ids) != set(candidate_ids)
            or len(set(brief_groups)) != 1
        ):
            raise RuntimeError("Frozen Stage 2 review candidate IDs, brief IDs, or brief groups are inconsistent.")
        by_id[utterance_id] = {
            "oyez_argument_id": argument_id,
            "utterance_id": utterance_id,
            "advocate_group": brief_groups[0],
            "eligible_brief_ids": sorted(set(brief_ids)),
            "candidate_reason_ids": sorted(set(candidate_ids)),
        }
    if set(by_id) != set(clean_ids):
        raise RuntimeError("Stage 2 candidate-scope sidecar must exactly cover the clean human final.")
    return [by_id[utterance_id] for utterance_id in clean_ids]


def validate_human_submission(
    checkpoint_id: str,
    professor_file: Path,
    student_files: Sequence[Path],
    *,
    reason_nodes: Sequence[Dict[str, Any]] = (),
) -> tuple[str, List[Dict[str, Any]], Dict[str, Any]]:
    directory, manifest = load_review_checkpoint(checkpoint_id)
    stage = str(manifest["stage"])
    spec = STAGE_REVIEW[stage]
    student_fields, original_students = _read_csv(directory / manifest["student_review_relative_path"])
    professor_fields, original_professors = _read_csv(directory / manifest["professor_review_relative_path"])
    row_key = str(spec["row_id"])
    expected = {str(row.get(row_key) or "") for row in original_professors}
    if not expected or "" in expected or len(expected) != len(original_professors):
        raise RuntimeError("Review checkpoint has invalid expected row identities.")
    original_student_by_id = _rows_by_id(original_students, row_key, complete=True, expected=expected)
    original_professor_by_id = _rows_by_id(original_professors, row_key, complete=True, expected=expected)
    submitted_student_payloads: List[tuple[Path, List[Dict[str, str]]]] = []
    used_student_ids: set[str] = set()
    for path in student_files:
        fields, rows = _read_csv(path)
        if fields != student_fields:
            raise ValueError("Student return columns do not match the checkpoint student template.")
        by_id = _rows_by_id(rows, row_key, complete=False, expected=expected)
        file_ids = {row.get("student_id", "") for row in rows}
        if len(file_ids) != 1 or "" in file_ids:
            raise ValueError("Each student return must use one non-empty student_id for every submitted row.")
        student_id = next(iter(file_ids))
        if student_id in used_student_ids:
            raise ValueError("Different student return files must use different student_id values.")
        used_student_ids.add(student_id)
        for row_id, row in by_id.items():
            _frozen_compare(
                row,
                original_student_by_id[row_id],
                allowed=set(spec["student"]),
                checkpoint_id=checkpoint_id,
                checkpoint_hash=manifest["review_checkpoint_hash"],
            )
            _validate_student_decision(stage, row, reason_nodes)
        submitted_student_payloads.append((path, rows))
    fields, professor_rows = _read_csv(professor_file)
    if fields != professor_fields:
        raise ValueError("Professor return columns do not match the checkpoint professor template.")
    professor_by_id = _rows_by_id(professor_rows, row_key, complete=True, expected=expected)
    clean: List[Dict[str, Any]] = []
    for row_id, row in professor_by_id.items():
        _frozen_compare(
            row,
            original_professor_by_id[row_id],
            allowed=set(spec["professor"]),
            checkpoint_id=checkpoint_id,
            checkpoint_hash=manifest["review_checkpoint_hash"],
        )
        clean.append(_clean_professor_decision(stage, row, reason_nodes))
    return stage, clean, {"student_payloads": submitted_student_payloads, "professor_rows": professor_rows}


def _validate_student_decision(stage: str, row: Mapping[str, str], reason_nodes: Sequence[Dict[str, Any]]) -> None:
    if stage == "stage1":
        gate = _parse_bool(row.get("student_question_function_present", ""))
        labels = _purpose_labels(row.get("student_purposes", ""))
        if gate is None or labels is None or gate != bool(labels):
            raise ValueError("Student Stage 1 decision violates the strict human purpose contract.")
    elif stage == "stage2":
        none = _parse_bool(row.get("student_none_of_the_above", ""))
        selected = _parse_list(row.get("student_selected_reason_ids", ""))
        if none is None or selected is None:
            raise ValueError("Student Stage 2 decision violates the strict JSON/boolean contract.")
        validate_human_terminal_selection(selected, none, _stage2_candidate_nodes(row, reason_nodes))
    elif row.get("student_stance_outcome", "") not in STANCE_OUTCOMES:
        raise ValueError("Student Stage 3 outcome is invalid.")


def _clean_professor_decision(stage: str, row: Mapping[str, str], reason_nodes: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    identity = _identity_from_review(row)
    if any(not identity[field] for field in ["oyez_argument_id", "utterance_id", "turn_id", "target_justice"]):
        raise ValueError("Professor final row is missing required clean utterance identity fields.")
    if stage == "stage1":
        gate = _parse_bool(row.get("professor_final_question_function_present", ""))
        labels = _purpose_labels(row.get("professor_final_purposes", ""))
        if gate is None or labels is None or gate != bool(labels):
            raise ValueError("Professor Stage 1 final violates the strict human purpose contract.")
        return {**identity, "final_question_function_present": gate, "final_purpose_labels": labels}
    if stage == "stage2":
        none = _parse_bool(row.get("professor_none_of_the_above", ""))
        selected = _parse_list(row.get("professor_final_selected_reason_ids", ""))
        if none is None or selected is None:
            raise ValueError("Professor Stage 2 final violates the strict JSON/boolean contract.")
        validate_human_terminal_selection(selected, none, _stage2_candidate_nodes(row, reason_nodes))
        return {**identity, "final_selected_reason_ids": selected, "final_none_of_the_above": none}
    outcome = row.get("professor_final_stance_outcome", "")
    if outcome not in STANCE_OUTCOMES:
        raise ValueError("Professor Stage 3 final outcome is invalid.")
    bundle_id = row.get("bundle_id", "")
    if not bundle_id:
        raise ValueError("Professor Stage 3 final row is missing bundle_id.")
    return {**identity, "bundle_id": bundle_id, "final_stance_outcome": outcome}


def create_human_final_checkpoint(
    run_dir: Path,
    review_checkpoint_id: str,
    professor_file: Path,
    student_files: Sequence[Path],
    *,
    reason_nodes: Sequence[Dict[str, Any]] = (),
    stage2_dependency_identity: Optional[Mapping[str, Any]] = None,
    checkpoint_id: str = "",
) -> Dict[str, Any]:
    stage, _, _ = validate_human_submission(
        review_checkpoint_id, professor_file, student_files, reason_nodes=reason_nodes
    )
    state = read_run_state(run_dir)
    if state.get("status") != f"awaiting_{stage}_human" or state.get("current_stage") != stage:
        raise RuntimeError("Human final submission does not match the run's awaiting-human state.")
    completed = _matching_checkpoint_directories(HUMAN_FINAL_CHECKPOINT_ROOT, run_dir.name, stage, "human_final")
    pending_matches = _matching_checkpoint_directories(PENDING_CHECKPOINT_ROOT, run_dir.name, stage, "human_final")
    if len(completed) > 1 or len(pending_matches) > 1 or (completed and pending_matches):
        raise RuntimeError("A human-final checkpoint transaction cannot be uniquely identified for this run and stage.")
    if completed:
        directory, manifest = load_human_final_checkpoint(completed[0].name)
        _finish_human_final_bookkeeping(run_dir, directory, manifest)
        return manifest
    if pending_matches:
        # The awaiting-human state proves this transaction never became the official pending state.
        shutil.rmtree(pending_matches[0])
    chosen = checkpoint_id or immutable_id()
    pending = PENDING_CHECKPOINT_ROOT / chosen
    if (HUMAN_FINAL_CHECKPOINT_ROOT / chosen).exists() or pending.exists():
        raise FileExistsError(f"Checkpoint ID already exists: {chosen!r}.")
    pending.mkdir(parents=True, exist_ok=False)
    try:
        professor_target = pending / "human_returns" / "professor_final.csv"
        professor_target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(professor_file, professor_target)
        for index, path in enumerate(student_files, start=1):
            target = pending / "human_returns" / f"student_{index:04d}.csv"
            shutil.copyfile(path, target)
        _atomic_json(
            pending / "pending_identity.json",
            {
                "checkpoint_type": "human_final",
                "checkpoint_id": chosen,
                "source_run_id": run_dir.name,
                "stage": stage,
                "review_checkpoint_id": review_checkpoint_id,
                "stage2_dependency_identity": dict(stage2_dependency_identity or {}) if stage == "stage2" else None,
            },
        )
        write_run_state(run_dir, "human_final_checkpoint_pending", stage)
        return resume_human_final_checkpoint(
            run_dir, reason_nodes=reason_nodes, stage2_dependency_identity=stage2_dependency_identity
        )
    except Exception:
        if read_run_state(run_dir).get("status") != "human_final_checkpoint_pending":
            shutil.rmtree(pending, ignore_errors=True)
        raise


def _consumer_hashes(stage: str) -> Dict[str, str]:
    from . import justice_checkpoint_common, justice_checkpoint_stage1, justice_checkpoint_stage2

    if stage == "stage1":
        return {
            "stage1_consumer": sha256_file(Path(justice_checkpoint_stage1.__file__)),
            "checkpoint_common": sha256_file(Path(justice_checkpoint_common.__file__)),
        }
    if stage == "stage2":
        return {
            "stage2_consumer": sha256_file(Path(justice_checkpoint_stage2.__file__)),
            "checkpoint_common": sha256_file(Path(justice_checkpoint_common.__file__)),
        }
    return {}


def _finish_human_final_bookkeeping(run_dir: Path, directory: Path, manifest: Mapping[str, Any]) -> None:
    stage = str(manifest["stage"])
    final_path = run_dir / str(STAGE_REVIEW[stage]["final_path"])
    final_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(directory / str(manifest["clean_final_relative_path"]), final_path)
    record_path = run_dir / "human_final_checkpoints.json"
    records = json.loads(record_path.read_text(encoding="utf-8")) if record_path.exists() else {}
    records[stage] = {
        "checkpoint_id": manifest["checkpoint_id"],
        "checkpoint_hash": manifest["overall_checkpoint_sha256"],
        "review_checkpoint_id": manifest["review_checkpoint_id"],
    }
    _atomic_json(record_path, records)
    next_status = {"stage1": "ready_for_stage2", "stage2": "ready_for_stage3", "stage3": "completed"}[stage]
    next_stage = {"stage1": "stage2", "stage2": "stage3", "stage3": "stage3"}[stage]
    write_run_state(run_dir, next_status, next_stage)


def resume_human_final_checkpoint(
    run_dir: Path,
    *,
    reason_nodes: Sequence[Dict[str, Any]] = (),
    stage2_dependency_identity: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    state = read_run_state(run_dir)
    if state.get("status") != "human_final_checkpoint_pending":
        raise RuntimeError("Human-final checkpoint recovery requires human_final_checkpoint_pending state.")
    stage = str(state.get("current_stage") or "")
    completed = _matching_checkpoint_directories(HUMAN_FINAL_CHECKPOINT_ROOT, run_dir.name, stage, "human_final")
    pending_matches = _matching_checkpoint_directories(PENDING_CHECKPOINT_ROOT, run_dir.name, stage, "human_final")
    if len(completed) > 1 or len(pending_matches) > 1 or (completed and pending_matches):
        raise RuntimeError("The human-final checkpoint transaction cannot be uniquely identified.")
    if completed:
        directory, manifest = load_human_final_checkpoint(completed[0].name)
        _finish_human_final_bookkeeping(run_dir, directory, manifest)
        return manifest
    if len(pending_matches) != 1:
        raise RuntimeError("The pending human-final checkpoint cannot be uniquely identified for recovery.")
    pending = pending_matches[0]
    identity = json.loads((pending / "pending_identity.json").read_text(encoding="utf-8"))
    checkpoint_id = pending.name
    review_checkpoint_id = str(identity.get("review_checkpoint_id") or "")
    professor_file = pending / "human_returns" / "professor_final.csv"
    student_files = sorted((pending / "human_returns").glob("student_*.csv"))
    validated_stage, clean, validation_payload = validate_human_submission(
        review_checkpoint_id, professor_file, student_files, reason_nodes=reason_nodes
    )
    if validated_stage != stage:
        raise RuntimeError("Pending human-final checkpoint Stage does not match its validated review material.")
    if stage == "stage2" and dict(identity.get("stage2_dependency_identity") or {}) != dict(stage2_dependency_identity or {}):
        raise RuntimeError("Pending Stage 2 dependency identity no longer matches the deterministic rebuild input.")
    for stale in [pending / "clean_final.jsonl", pending / "checkpoint_manifest.json"]:
        if stale.exists():
            stale.unlink()
    clean_target = pending / "clean_final.jsonl"
    write_jsonl(clean_target, clean)
    stage2_routing_relative = ""
    stage2_candidate_scope_relative = ""
    if stage == "stage2":
        checkpoint_records_path = run_dir / "human_final_checkpoints.json"
        if not checkpoint_records_path.is_file():
            raise RuntimeError("Stage 2 human-final checkpoint requires a recorded Stage 1 human-final checkpoint.")
        checkpoint_records = json.loads(checkpoint_records_path.read_text(encoding="utf-8"))
        stage1_checkpoint_id = str((checkpoint_records.get("stage1") or {}).get("checkpoint_id") or "")
        if not stage1_checkpoint_id:
            raise RuntimeError("Stage 2 human-final checkpoint lacks its canonical Stage 1 routing authority.")
        from .justice_checkpoint_stage1 import consume_stage1_checkpoint

        utterance_source = run_dir / "04_justice_utterances" / "justice_utterances.jsonl"
        if not utterance_source.is_file():
            raise RuntimeError("Stage 2 human-final checkpoint requires the frozen run utterance input.")
        stage1_rows = consume_stage1_checkpoint(stage1_checkpoint_id, read_jsonl(utterance_source))
        routing_rows: List[Dict[str, Any]] = []
        for row in stage1_rows:
            row_identity = {
                "oyez_argument_id": row.get("oyez_argument_id", ""),
                "utterance_id": row.get("utterance_id", ""),
                "turn_id": row.get("turn_id", ""),
                "target_justice": row.get("target_justice", ""),
                "text": row.get("text", ""),
            }
            utterance_id = str(row_identity["utterance_id"] or "")
            routing_rows.append(
                {
                    **row_identity,
                    "enters_reason_alignment": bool(row["enters_reason_alignment"]),
                }
            )
        clean_ids = {str(row.get("utterance_id") or "") for row in clean}
        expected_stage2_ids = {
            str(row["utterance_id"]) for row in routing_rows if row["enters_reason_alignment"]
        }
        if clean_ids != expected_stage2_ids:
            raise RuntimeError("Stage 2 clean human final does not exactly cover the Stage 1-derived Stage 2 input.")
        routing_target = pending / "stage2_routing_coverage.jsonl"
        write_jsonl(routing_target, routing_rows)
        stage2_routing_relative = "stage2_routing_coverage.jsonl"
        candidate_scope_target = pending / "stage2_candidate_scope.jsonl"
        write_jsonl(
            candidate_scope_target,
            _stage2_candidate_scope_sidecar(validation_payload["professor_rows"], clean),
        )
        stage2_candidate_scope_relative = "stage2_candidate_scope.jsonl"
    files = {
        str(path.relative_to(pending)): sha256_file(path)
        for path in pending.rglob("*")
        if path.is_file() and path.name != "checkpoint_manifest.json"
    }
    _, review_manifest = load_review_checkpoint(review_checkpoint_id)
    manifest = _manifest_write(
        pending,
        {
            "checkpoint_type": "human_final",
            "checkpoint_id": checkpoint_id,
            "source_run_id": run_dir.name,
            "stage": stage,
            "review_checkpoint_id": review_checkpoint_id,
            "review_checkpoint_hash": review_manifest["overall_checkpoint_sha256"],
            "clean_final_relative_path": "clean_final.jsonl",
            "coverage": len(clean),
            "human_contract_sha256": sha256_json({"stage": stage, "fields": sorted(clean[0]) if clean else []}),
            "consumer_hashes": _consumer_hashes(stage),
            "stage2_dependency_identity": dict(stage2_dependency_identity or {}) if stage == "stage2" else None,
            "stage2_routing_coverage_relative_path": stage2_routing_relative or None,
            "stage2_candidate_scope_relative_path": stage2_candidate_scope_relative or None,
        },
        files,
    )
    destination = HUMAN_FINAL_CHECKPOINT_ROOT / checkpoint_id
    HUMAN_FINAL_CHECKPOINT_ROOT.mkdir(parents=True, exist_ok=True)
    os.replace(pending, destination)
    _finish_human_final_bookkeeping(run_dir, destination, manifest)
    return manifest


def list_checkpoints() -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for kind, root in [("review", REVIEW_CHECKPOINT_ROOT), ("human_final", HUMAN_FINAL_CHECKPOINT_ROOT)]:
        if not root.exists():
            continue
        for directory in sorted(path for path in root.iterdir() if path.is_dir()):
            try:
                _, manifest = (
                    load_review_checkpoint(directory.name)
                    if kind == "review"
                    else load_human_final_checkpoint(directory.name)
                )
                rows.append(
                    {
                        "checkpoint_id": directory.name,
                        "checkpoint_type": kind,
                        "stage": manifest.get("stage", ""),
                        "source_run_id": manifest.get("source_run_id", ""),
                        "checkpoint_hash": manifest.get("overall_checkpoint_sha256", ""),
                    }
                )
            except Exception as exc:
                rows.append({"checkpoint_id": directory.name, "checkpoint_type": kind, "valid": False, "error": str(exc)})
    return rows


def write_invalid_human_values(
    run_dir: Path, stage: str, professor_file: Path, student_files: Sequence[Path]
) -> Path:
    spec = STAGE_REVIEW[stage]
    row_key = str(spec["row_id"])
    rows: List[Dict[str, Any]] = []
    for role, path, allowed in [
        ("professor", professor_file, set(spec["professor"])),
        *[("student", item, set(spec["student"])) for item in student_files],
    ]:
        try:
            _, values = _read_csv(path)
        except Exception:
            values = []
        if not values:
            rows.append({"source_file": str(path), "role": role, "row_id": "", "raw_invalid_values": {}})
            continue
        for value in values:
            rows.append(
                {
                    "source_file": str(path),
                    "role": role,
                    "row_id": value.get(row_key, ""),
                    "raw_invalid_values": {field: value.get(field, "") for field in sorted(allowed)},
                }
            )
    path = run_dir / "08_human_review" / f"{stage}_human_final_validation_errors.jsonl"
    write_jsonl(path, rows)
    return path
