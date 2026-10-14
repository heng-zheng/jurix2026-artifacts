from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from .common import clean_text, first_present, normalize_person_name, write_csv, write_jsonl
from .justice_snapshots import ORAL_SNAPSHOT_ROOT, verify_utterance_snapshot
from .preprocess_oyez import (
    classify_sections,
    is_justice,
    section_counsel_name,
    speaker_payload,
    speaker_role,
    text_block_id,
    turn_text,
    turn_time,
)


JUSTICE_NAME_SUFFIXES = {"jr", "sr", "ii", "iii", "iv"}


def _justice_tokens(value: Any) -> List[str]:
    tokens = normalize_person_name(value).split()
    if tokens[:2] == ["chief", "justice"]:
        tokens = tokens[2:]
    elif tokens[:1] == ["justice"]:
        tokens = tokens[1:]
    while tokens and tokens[-1] in JUSTICE_NAME_SUFFIXES:
        tokens.pop()
    return tokens


def match_target_justice(speaker: Any, names: Sequence[str]) -> str:
    speaker_tokens = _justice_tokens(speaker)
    if not speaker_tokens:
        return ""
    normalized = [(name, _justice_tokens(name)) for name in names]
    exact = [name for name, tokens in normalized if tokens and tokens == speaker_tokens]
    if len(exact) > 1:
        raise ValueError(f"Speaker {speaker!r} exactly matches multiple target Justices: {exact!r}.")
    if exact:
        return exact[0]
    token_set = set(speaker_tokens)
    surname = [name for name, tokens in normalized if tokens and tokens[-1] in token_set]
    if len(surname) > 1:
        raise ValueError(f"Speaker {speaker!r} matches multiple target Justice surnames: {surname!r}.")
    return surname[0] if surname else ""


def target_justices_for_scope(scope: Mapping[str, Any]) -> List[str]:
    annotation_scope = scope.get("annotation_scope") or {}
    values = annotation_scope.get("target_justices") if isinstance(annotation_scope, dict) else None
    if not isinstance(values, list) or not values or any(not isinstance(value, str) or not value.strip() for value in values):
        raise ValueError("The Justice run scope requires a non-empty target_justices string list.")
    return list(values)


def advocate_group_from_description(description: Any) -> str:
    text = clean_text(description).lower()
    if "court-appointed" in text or "court appointed" in text or ("amicus" in text and "appointed" in text):
        return "caa"
    if any(value in text for value in ["supporting the petitioner", "supporting petitioner", "support of the petitioner"]):
        return "other"
    if any(value in text for value in ["supporting the respondent", "supporting respondent", "support of the respondent"]):
        return "other"
    if "for the petitioner" in text or "for the petitioners" in text:
        return "pet"
    if "for the respondent" in text or "for the respondents" in text:
        return "res"
    return "other"


def _advocates(case_payload: Mapping[str, Any], case_source_id: str) -> List[Dict[str, str]]:
    values = case_payload.get("advocates")
    if not isinstance(values, list) or any(not isinstance(value, dict) for value in values):
        raise ValueError(f"Case source {case_source_id!r} has an invalid advocates structure.")
    rows: List[Dict[str, str]] = []
    for item in values:
        nested = item.get("advocate") if isinstance(item.get("advocate"), dict) else {}
        name = clean_text(item.get("name") or nested.get("name") or nested.get("display_name") or nested.get("identifier"))
        description = clean_text(item.get("description") or item.get("advocate_description") or item.get("role"))
        rows.append(
            {
                "case_source_id": case_source_id,
                "advocate_name": name,
                "advocate_description": description,
                "advocate_group": advocate_group_from_description(description),
            }
        )
    return rows


def _manual_phase(scope: Mapping[str, Any], argument_id: str, start_time: Optional[float]) -> str:
    if start_time is None:
        return "unknown"
    by_argument = scope.get("phase_times_by_argument") or {}
    phases = by_argument.get(argument_id) if isinstance(by_argument, dict) else None
    if not isinstance(phases, dict):
        return "unknown"
    for phase, bounds in phases.items():
        if not isinstance(bounds, dict):
            continue
        start = bounds.get("start_time")
        end = bounds.get("end_time")
        if isinstance(start, (int, float)) and not isinstance(start, bool) and isinstance(end, (int, float)) and not isinstance(end, bool):
            if float(start) <= start_time <= float(end):
                return "petitioner_rebuttal" if phase == "rebuttal" else str(phase)
    return "unknown"


def _argument_sources(snapshot_id: str) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Dict[str, Any]], Dict[str, Any]]:
    manifest = verify_utterance_snapshot(snapshot_id)
    root = ORAL_SNAPSHOT_ROOT / snapshot_id
    cases: Dict[str, Dict[str, Any]] = {}
    for item in manifest["case_sources"]:
        cases[str(item["case_source_id"])] = json.loads((root / item["relative_file"]).read_bytes())
    arguments: Dict[str, Dict[str, Any]] = {}
    for item in manifest["oral_arguments"]:
        arguments[str(item["oyez_argument_id"])] = {
            "payload": json.loads((root / item["relative_file"]).read_bytes()),
            "case_source_ids": list(item["case_source_ids"]),
        }
    return cases, arguments, manifest


def _speaker_id(speaker: Mapping[str, Any]) -> str:
    return str(first_present(dict(speaker), ["ID", "id", "identifier", "name"], ""))


def derive_processed_transcript(snapshot_id: str, scope: Mapping[str, Any]) -> List[Dict[str, Any]]:
    cases, arguments, _ = _argument_sources(snapshot_id)
    selected = scope.get("selected_oyez_argument_ids") or []
    rows: List[Dict[str, Any]] = []
    seen_turn_ids: set[str] = set()
    for argument_id in selected:
        source = arguments.get(str(argument_id))
        if source is None:
            raise ValueError(f"Selected oral argument is absent from the snapshot: {argument_id!r}.")
        payload = source["payload"]
        transcript = payload.get("transcript")
        sections = transcript.get("sections") if isinstance(transcript, dict) else None
        if not isinstance(sections, list) or any(not isinstance(section, dict) or not isinstance(section.get("turns"), list) for section in sections):
            raise ValueError(f"Oral argument {argument_id!r} has an invalid ordered transcript structure.")
        assignments = classify_sections(sections)
        case_advocates: List[Dict[str, str]] = []
        for case_source_id in source["case_source_ids"]:
            case_advocates.extend(_advocates(cases[case_source_id], case_source_id))
        turn_number = 0
        for section_index, section in enumerate(sections, start=1):
            assignment = assignments.get(
                section_index,
                {"phase": "unknown", "confidence": "low", "notes": "No automatic section assignment.", "section_counsel": ""},
            )
            counsel = clean_text(assignment.get("section_counsel", "") or section_counsel_name(section))
            matches = [row for row in case_advocates if normalize_person_name(row["advocate_name"]) == normalize_person_name(counsel)]
            groups = {row["advocate_group"] for row in matches}
            if len(groups) > 1:
                raise ValueError(
                    f"Conflicting advocate_group values for counsel {counsel!r} in oral argument {argument_id!r}."
                )
            for turn in section["turns"]:
                if not isinstance(turn, dict):
                    raise ValueError(f"Every transcript turn in oral argument {argument_id!r} must be an object.")
                turn_number += 1
                turn_id = f"{argument_id}__turn_{turn_number:06d}"
                if turn_id in seen_turn_ids:
                    raise ValueError(f"Duplicate generated turn_id: {turn_id!r}.")
                seen_turn_ids.add(turn_id)
                speaker = speaker_payload(turn)
                start_time = turn_time(turn, "start")
                stop_time = turn_time(turn, "stop") or turn_time(turn, "end")
                manual = _manual_phase(scope, str(argument_id), start_time)
                phase = manual if manual != "unknown" else str(assignment.get("phase") or "unknown")
                if matches:
                    group = next(iter(groups))
                    advocate_name = counsel
                    descriptions = sorted({row["advocate_description"] for row in matches})
                    advocate_description = " || ".join(descriptions)
                    group_source = "oyez_case_json"
                    source_ids = sorted({row["case_source_id"] for row in matches})
                elif phase in {"petitioner", "petitioner_rebuttal"}:
                    group, group_source, source_ids = "pet", "phase_fallback", []
                    advocate_name, advocate_description = counsel, ""
                elif phase == "respondent":
                    group, group_source, source_ids = "res", "phase_fallback", []
                    advocate_name, advocate_description = counsel, ""
                else:
                    group, group_source, source_ids = "other", "unmatched", []
                    advocate_name, advocate_description = counsel, ""
                rows.append(
                    {
                        "oyez_argument_id": str(argument_id),
                        "turn_id": turn_id,
                        "turn_index": turn_number,
                        "section_index": section_index,
                        "text_block_id": text_block_id(turn),
                        "speaker_id": _speaker_id(speaker),
                        "speaker": clean_text(first_present(speaker, ["name", "display_name", "identifier"], "Unknown")),
                        "speaker_role": speaker_role(speaker),
                        "is_justice": is_justice(speaker),
                        "start_time": start_time,
                        "stop_time": stop_time,
                        "text": turn_text(turn),
                        "section_counsel": counsel,
                        "phase": phase,
                        "phase_source": "human_phase_times" if manual != "unknown" else "section_structure_and_counsel_identity",
                        "phase_confidence": "high" if manual != "unknown" else assignment.get("confidence", "low"),
                        "phase_notes": "Human phase_times override." if manual != "unknown" else assignment.get("notes", ""),
                        "advocate_name": advocate_name,
                        "advocate_description": advocate_description,
                        "advocate_group": group,
                        "advocate_group_source": group_source,
                        "advocate_case_source_ids": source_ids,
                        "source": "immutable_oyez_snapshot",
                    }
                )
    return rows


def extract_target_utterances(
    transcript: Sequence[Dict[str, Any]], *, target_justices: Sequence[str]
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    seen: set[str] = set()
    for turn in transcript:
        text = clean_text(turn.get("text", ""))
        if not text:
            continue
        justice = match_target_justice(turn.get("speaker", ""), target_justices)
        if not justice:
            continue
        utterance_id = f"{turn['turn_id']}__utterance"
        if utterance_id in seen:
            raise ValueError(f"Duplicate generated utterance_id: {utterance_id!r}.")
        seen.add(utterance_id)
        rows.append({**turn, "utterance_id": utterance_id, "target_justice": justice})
    return rows


def select_scoped_utterances(
    utterances: Sequence[Dict[str, Any]], scope: Mapping[str, Any]
) -> List[Dict[str, Any]]:
    limits = (scope.get("annotation_scope") or {}).get("advocate_group_limits") or {}
    counts: Dict[Tuple[str, str, str], int] = {}
    selected: List[Dict[str, Any]] = []
    for row in utterances:
        group = str(row.get("advocate_group") or "")
        limit = limits.get(group)
        if limit is None:
            continue
        key = (
            str(row.get("oyez_argument_id") or ""),
            str(row.get("target_justice") or ""),
            group,
        )
        counts[key] = counts.get(key, 0) + 1
        if limit == "all" or counts[key] <= limit:
            selected.append(row)
    return selected


def write_run_inputs(
    run_dir: Path,
    *,
    snapshot_id: str,
    scope: Mapping[str, Any],
    config: Mapping[str, Any],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    transcript = derive_processed_transcript(snapshot_id, scope)
    target_names = target_justices_for_scope(scope)
    utterances = extract_target_utterances(transcript, target_justices=target_names)
    utterances = select_scoped_utterances(utterances, scope)
    scopes = {
        str(item.get("oyez_argument_id")): list(item.get("brief_ids") or [])
        for item in scope.get("oral_argument_brief_scopes") or []
    }
    utterances = [
        {**row, "run_scope_brief_ids": scopes.get(str(row.get("oyez_argument_id")), [])}
        for row in utterances
    ]
    selected_arguments = set(scope.get("selected_oyez_argument_ids") or [])
    mapped_arguments = {str(item.get("oyez_argument_id")) for item in scope.get("oral_argument_brief_scopes") or []}
    required_arguments = {str(row["oyez_argument_id"]) for row in utterances}
    if not required_arguments <= selected_arguments or not required_arguments <= mapped_arguments:
        raise ValueError("Every selected oral argument with target Justice utterances requires one run-scope mapping.")
    processed_dir = run_dir / "01_processed_transcripts"
    utterance_dir = run_dir / "04_justice_utterances"
    write_jsonl(processed_dir / "processed_transcripts.jsonl", transcript)
    write_csv(processed_dir / "processed_transcripts.csv", transcript)
    write_jsonl(utterance_dir / "justice_utterances_base.jsonl", utterances)
    write_csv(utterance_dir / "justice_utterances_base.csv", utterances)
    return transcript, utterances
