from __future__ import annotations

import json
import re
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple


PURPOSE_LABEL_ORDER = [
    "Background",
    "Clarification",
    "Implications",
    "Support",
    "Criticism",
    "Communicate",
    "Humor",
]
PURPOSE_LABELS = set(PURPOSE_LABEL_ORDER)
STANCE_OUTCOMES = {"Favor", "Against", "Neutral", "Mixed"}
AUXILIARY_FIELDS = {"rationale", "confidence", "uncertainty_notes"}


def contract_manifest() -> Dict[str, Any]:
    """Canonical future-run contracts sealed into every run manifest."""
    return {
        "contract_version": 2,
        "common": {
            "json": "one_complete_object_no_fences_no_surrounding_text_no_unknown_fields_no_duplicate_object_keys",
            "auxiliary_fields": sorted(AUXILIARY_FIELDS),
            "auxiliary_fields_are_prompt_required_but_non_validating": True,
            "evidence_normalization": "trim_and_collapse_pure_whitespace_only",
            "repairs_per_model_call": 1,
        },
        "stage1": {
            "decision_fields": ["question_function_present", "purposes"],
            "purpose_labels": PURPOSE_LABEL_ORDER,
            "purpose_count_when_true": [1, 7],
            "negative": {"question_function_present": False, "purposes": []},
            "agreement": "gate_plus_unordered_label_to_verbatim_evidence_mapping",
        },
        "stage2": {
            "decision_fields": ["current_node_relevant", "alignment_evidence_span"],
            "traversal": "single_node_all_roots_all_children_from_master_hierarchy",
            "none_of_the_above": "program_derived_only_after_complete_valid_empty_traversal",
            "agreement": "unordered_terminal_id_to_verbatim_evidence_mapping",
        },
        "stage3": {
            "decision_fields": ["stance_outcome", "favor_evidence", "against_evidence"],
            "outcomes": sorted(STANCE_OUTCOMES),
            "annotation_unit": "utterance_id_x_stage2_bundle_id",
            "agreement": "outcome_plus_direction_presence_ids_and_verbatim_evidence",
            "mixed_always_adjudicated": True,
            "same_reason_intersection_special_routing": True,
        },
    }


def normalize_whitespace(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def normalized_verbatim_span(source: Any, span: Any) -> bool:
    candidate = normalize_whitespace(span)
    return bool(candidate) and candidate in normalize_whitespace(source)


class _DuplicateJsonKeyError(ValueError):
    def __init__(self, key: str) -> None:
        self.key = key
        super().__init__(f"duplicate JSON object key {key!r}")


def _reject_duplicate_json_keys(pairs: Sequence[Tuple[str, Any]]) -> Dict[str, Any]:
    parsed: Dict[str, Any] = {}
    for key, value in pairs:
        if key in parsed:
            raise _DuplicateJsonKeyError(key)
        parsed[key] = value
    return parsed


def strict_json_object(text: Any) -> Tuple[Optional[Dict[str, Any]], List[str]]:
    raw = str(text or "")
    try:
        parsed = json.loads(
            raw,
            object_pairs_hook=_reject_duplicate_json_keys,
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(f"non-standard JSON constant {value!r}")
            ),
        )
    except _DuplicateJsonKeyError as exc:
        return None, [
            f"The response contains duplicate JSON object key {exc.key!r}. "
            "Field-level checks were not performed."
        ]
    except (json.JSONDecodeError, ValueError) as exc:
        position = getattr(exc, "pos", "unknown")
        return None, [
            f"The response is not one complete JSON object: JSON parsing failed at character {position}. "
            "Field-level checks were not performed."
        ]
    if not isinstance(parsed, dict):
        return None, ["The response JSON root must be an object. Field-level checks were not performed."]
    return parsed, []


def _result(
    parsed: Optional[Dict[str, Any]],
    normalized: Optional[Dict[str, Any]],
    findings: List[str],
    failures: Set[str],
    protected: Optional[Dict[str, Any]] = None,
    repairable: Optional[List[str]] = None,
    same_reason_conflict: bool = False,
) -> Dict[str, Any]:
    if findings and not failures:
        raise RuntimeError("Validator implementation error: invalid output has findings but no failed predicates.")
    valid = not findings and not same_reason_conflict
    if not valid and not findings and not same_reason_conflict:
        raise RuntimeError("Validator implementation error: invalid output has no validation finding.")
    return {
        "valid": valid,
        "parsed": parsed,
        "normalized": normalized,
        "validation_findings": findings,
        "_failures": sorted(failures),
        "protected_fields": protected or {},
        "repairable_fields": repairable or [],
        "same_reason_conflict": same_reason_conflict,
    }


def _unknown_fields(parsed: Dict[str, Any], allowed: Set[str], findings: List[str], failures: Set[str]) -> None:
    unknown = sorted(set(parsed) - allowed)
    if unknown:
        findings.append(f"The response contains fields that are not allowed by this contract: {unknown!r}.")
        failures.add("unknown_fields")


def normalize_purpose_label(value: Any) -> Optional[str]:
    if not isinstance(value, str):
        return None
    stripped = value.strip()
    matches = [label for label in PURPOSE_LABEL_ORDER if label.lower() == stripped.lower()]
    return matches[0] if len(matches) == 1 else None


def validate_stage1_output(content: Any, utterance_text: str, *, adjudicator: bool = False) -> Dict[str, Any]:
    parsed, findings = strict_json_object(content)
    failures: Set[str] = set()
    if parsed is None:
        failures.add("json_structure")
        return _result(None, None, findings, failures, repairable=["question_function_present", "purposes"])

    # Annotators, adjudicators, and their repairs use the same model-facing
    # decision schema. The adjudicated_* names belong only to the
    # program-generated consolidated result.
    gate_field = "question_function_present"
    purposes_field = "purposes"
    allowed = {gate_field, purposes_field, *AUXILIARY_FIELDS}
    _unknown_fields(parsed, allowed, findings, failures)

    gate = parsed.get(gate_field)
    purposes = parsed.get(purposes_field)
    gate_valid = isinstance(gate, bool)
    list_valid = isinstance(purposes, list)
    if gate_field not in parsed:
        findings.append(f"The required decision field {gate_field!r} is missing.")
        failures.add("gate_structure")
    elif not gate_valid:
        findings.append(f"{gate_field!r} must be a JSON boolean, not {type(gate).__name__}.")
        failures.add("gate_structure")
    if purposes_field not in parsed:
        findings.append(f"The required decision field {purposes_field!r} is missing.")
        failures.add("purposes_structure")
    elif not list_valid:
        findings.append(f"{purposes_field!r} must be a JSON list.")
        failures.add("purposes_structure")

    normalized_purposes: List[Dict[str, str]] = []
    raw_labels: List[Any] = []
    normalized_labels: List[str] = []
    object_validity: List[bool] = []
    if list_valid:
        if len(purposes) > 7:
            findings.append(f"{purposes_field!r} may contain at most seven purpose objects.")
            failures.add("purpose_count")
        for index, item in enumerate(purposes):
            path = f"{purposes_field}[{index}]"
            object_valid = True
            if not isinstance(item, dict):
                findings.append(f"{path} must be an object with exactly label and evidence_span.")
                failures.add("purpose_object")
                object_validity.append(False)
                continue
            unknown = sorted(set(item) - {"label", "evidence_span"})
            if unknown:
                findings.append(f"{path} contains fields not allowed by the purpose-object contract: {unknown!r}.")
                failures.add("unknown_fields")
                object_valid = False
            if "label" not in item:
                findings.append(f"{path}.label is required.")
                failures.add("purpose_label")
                object_valid = False
                normalized_label = None
            else:
                raw_labels.append(item.get("label"))
                normalized_label = normalize_purpose_label(item.get("label"))
                if normalized_label is None:
                    findings.append(f"{path}.label is not one of the seven allowed purpose labels.")
                    failures.add("purpose_label")
                    object_valid = False
                else:
                    normalized_labels.append(normalized_label)
            evidence = item.get("evidence_span")
            if "evidence_span" not in item or not isinstance(evidence, str) or not normalize_whitespace(evidence):
                findings.append(f"{path}.evidence_span must be a non-empty string.")
                failures.add("purpose_evidence")
                object_valid = False
            elif not normalized_verbatim_span(utterance_text, evidence):
                findings.append(f"{path}.evidence_span must be a verbatim span from the current utterance.")
                failures.add("purpose_evidence")
                object_valid = False
            if normalized_label is not None and isinstance(evidence, str):
                normalized_purposes.append({"label": normalized_label, "evidence_span": evidence})
            object_validity.append(object_valid)

        duplicates = sorted({label for label in normalized_labels if normalized_labels.count(label) > 1})
        if duplicates:
            findings.append(f"Purpose labels must be unique after allowed case/outer-space normalization; duplicates={duplicates!r}.")
            failures.add("duplicate_purposes")

    if gate_valid and list_valid:
        if gate is False and purposes:
            findings.append(f"{gate_field}=false requires an empty {purposes_field} list.")
            failures.add("gate_purpose_consistency")
        if gate is True and not (1 <= len(purposes) <= 7):
            findings.append(f"{gate_field}=true requires between one and seven purpose objects.")
            failures.add("gate_purpose_consistency")

    normalized: Dict[str, Any] = dict(parsed)
    if gate_valid:
        normalized[gate_field] = gate
    if list_valid:
        normalized[purposes_field] = normalized_purposes

    protected: Dict[str, Any] = {}
    repairable: List[str] = []
    unsafe_global = bool({"json_structure", "gate_structure", "purposes_structure", "gate_purpose_consistency"} & failures)
    if unsafe_global:
        repairable.extend([gate_field, purposes_field])
    else:
        if gate_valid:
            protected[gate_field] = gate
        duplicate_labels = {label for label in normalized_labels if normalized_labels.count(label) > 1}
        for index, item in enumerate(purposes if list_valid else []):
            label = normalize_purpose_label(item.get("label")) if isinstance(item, dict) else None
            evidence = item.get("evidence_span") if isinstance(item, dict) else None
            path = f"{purposes_field}[{index}]"
            if label is None or label in duplicate_labels:
                repairable.append(path)
                continue
            protected[f"purpose_label:{label}"] = True
            if isinstance(evidence, str) and normalize_whitespace(evidence) and normalized_verbatim_span(utterance_text, evidence):
                protected[f"purpose_evidence:{label}"] = evidence
            else:
                repairable.append(f"{path}.evidence_span")
    if "unknown_fields" in failures:
        repairable.append("remove undefined fields without changing protected decision content")
    return _result(parsed, normalized, findings, failures, protected, repairable)


def validate_stage2_output(content: Any, utterance_text: str) -> Dict[str, Any]:
    parsed, findings = strict_json_object(content)
    failures: Set[str] = set()
    if parsed is None:
        failures.add("json_structure")
        return _result(None, None, findings, failures, repairable=["current_node_relevant", "alignment_evidence_span"])
    allowed = {"current_node_relevant", "alignment_evidence_span", *AUXILIARY_FIELDS}
    _unknown_fields(parsed, allowed, findings, failures)
    relevant = parsed.get("current_node_relevant")
    evidence = parsed.get("alignment_evidence_span")
    boolean_valid = isinstance(relevant, bool)
    if "current_node_relevant" not in parsed:
        findings.append("The required decision field 'current_node_relevant' is missing.")
        failures.add("relevance_structure")
    elif not boolean_valid:
        findings.append("'current_node_relevant' must be a JSON boolean.")
        failures.add("relevance_structure")
    if "alignment_evidence_span" not in parsed or not isinstance(evidence, str):
        findings.append("'alignment_evidence_span' is required and must be a string.")
        failures.add("alignment_evidence")
    elif boolean_valid and relevant:
        if not normalize_whitespace(evidence):
            findings.append("A relevant node requires a non-empty alignment_evidence_span.")
            failures.add("alignment_evidence")
        elif not normalized_verbatim_span(utterance_text, evidence):
            findings.append("alignment_evidence_span must be a verbatim span from the current utterance.")
            failures.add("alignment_evidence")
    elif boolean_valid and evidence != "":
        findings.append("A non-relevant node requires alignment_evidence_span to be the exact empty string.")
        failures.add("alignment_evidence")
    normalized = dict(parsed)
    protected: Dict[str, Any] = {}
    repairable: List[str] = []
    if boolean_valid:
        protected["current_node_relevant"] = relevant
        if "alignment_evidence" in failures:
            repairable.append("alignment_evidence_span")
        elif isinstance(evidence, str):
            protected["alignment_evidence_span"] = evidence
    else:
        repairable.extend(["current_node_relevant", "alignment_evidence_span"])
    if "unknown_fields" in failures:
        repairable.append("remove undefined fields without changing protected decision content")
    return _result(parsed, normalized, findings, failures, protected, repairable)


def _validate_direction_block(
    block: Any,
    direction: str,
    utterance_text: str,
    allowed_reason_ids: Set[str],
    findings: List[str],
    failures: Set[str],
) -> Tuple[Optional[Dict[str, Any]], Set[str]]:
    if block is None:
        return None, set()
    path = f"{direction}_evidence"
    if not isinstance(block, dict):
        findings.append(f"{path} must be null or an object.")
        failures.add("direction_block_structure")
        return None, set()
    unknown = sorted(set(block) - {"utterance_evidence_span", "target_reason_ids"})
    if unknown:
        findings.append(f"{path} contains fields not allowed by the directional-evidence contract: {unknown!r}.")
        failures.add("unknown_fields")
    span = block.get("utterance_evidence_span")
    if "utterance_evidence_span" not in block or not isinstance(span, str) or not normalize_whitespace(span):
        findings.append(f"{path}.utterance_evidence_span must be a non-empty string.")
        failures.add("direction_span")
    elif not normalized_verbatim_span(utterance_text, span):
        findings.append(f"{path}.utterance_evidence_span must be a verbatim span from the current utterance.")
        failures.add("direction_span")
    ids = block.get("target_reason_ids")
    valid_ids: Set[str] = set()
    if "target_reason_ids" not in block or not isinstance(ids, list) or not ids:
        findings.append(f"{path}.target_reason_ids must be a non-empty JSON list.")
        failures.add("direction_ids")
        ids = []
    elif any(not isinstance(item, str) or not item for item in ids):
        findings.append(f"Every value in {path}.target_reason_ids must be a non-empty string.")
        failures.add("direction_ids")
    else:
        duplicates = sorted({item for item in ids if ids.count(item) > 1})
        if duplicates:
            findings.append(f"{path}.target_reason_ids must not contain duplicate IDs: {duplicates!r}.")
            failures.add("direction_ids")
        outside = sorted({item for item in ids if item not in allowed_reason_ids})
        if outside:
            findings.append(f"{path}.target_reason_ids contains IDs outside the current complete bundle hierarchy: {outside!r}.")
            failures.add("direction_ids")
        valid_ids = {item for item in ids if item in allowed_reason_ids}
    return {"utterance_evidence_span": span, "target_reason_ids": ids} if isinstance(span, str) else None, valid_ids


def validate_stage3_output(content: Any, utterance_text: str, allowed_reason_ids: Iterable[str]) -> Dict[str, Any]:
    parsed, findings = strict_json_object(content)
    failures: Set[str] = set()
    if parsed is None:
        failures.add("json_structure")
        return _result(None, None, findings, failures, repairable=["stance_outcome", "favor_evidence", "against_evidence"])
    allowed = {"stance_outcome", "favor_evidence", "against_evidence", *AUXILIARY_FIELDS}
    _unknown_fields(parsed, allowed, findings, failures)
    outcome = parsed.get("stance_outcome")
    if "stance_outcome" not in parsed or outcome not in STANCE_OUTCOMES:
        findings.append("'stance_outcome' must be exactly Favor, Against, Neutral, or Mixed.")
        failures.add("outcome_structure")
    favor_present = "favor_evidence" in parsed
    against_present = "against_evidence" in parsed
    if not favor_present:
        findings.append("The required field 'favor_evidence' is missing.")
        failures.add("direction_block_structure")
    if not against_present:
        findings.append("The required field 'against_evidence' is missing.")
        failures.add("direction_block_structure")
    ids = set(str(item) for item in allowed_reason_ids)
    favor, favor_ids = _validate_direction_block(parsed.get("favor_evidence"), "favor", utterance_text, ids, findings, failures)
    against, against_ids = _validate_direction_block(parsed.get("against_evidence"), "against", utterance_text, ids, findings, failures)
    topology = {
        "Favor": (True, False),
        "Against": (False, True),
        "Neutral": (False, False),
        "Mixed": (True, True),
    }
    if outcome in topology and topology[outcome] != (parsed.get("favor_evidence") is not None, parsed.get("against_evidence") is not None):
        findings.append(f"The directional evidence blocks are inconsistent with stance_outcome={outcome!r}.")
        failures.add("outcome_topology")
    same_reason = bool(favor_ids & against_ids) and not ({"direction_block_structure", "direction_ids"} & failures)
    if same_reason:
        shared = sorted(favor_ids & against_ids)
        findings.append(
            "The Favor and Against directional blocks share one or more exact valid focal reason IDs "
            f"and therefore require the approved same-reason substantive-conflict routing: {shared!r}."
        )
        failures.add("same_reason_conflict")
    normalized = dict(parsed)
    if favor is not None:
        normalized["favor_evidence"] = favor
    if against is not None:
        normalized["against_evidence"] = against
    protected: Dict[str, Any] = {}
    repairable: List[str] = []
    unsafe = bool({"json_structure", "outcome_structure", "outcome_topology", "direction_block_structure"} & failures)
    if unsafe:
        repairable.extend(["stance_outcome", "favor_evidence", "against_evidence"])
    else:
        protected["stance_outcome"] = outcome
        for direction in ["favor", "against"]:
            block = parsed.get(f"{direction}_evidence")
            if block is None:
                protected[f"{direction}_evidence"] = None
                continue
            block_failures: List[str] = []
            span = block.get("utterance_evidence_span") if isinstance(block, dict) else None
            block_ids = block.get("target_reason_ids") if isinstance(block, dict) else None
            if isinstance(span, str) and normalize_whitespace(span) and normalized_verbatim_span(utterance_text, span):
                protected[f"{direction}_evidence.utterance_evidence_span"] = span
            else:
                block_failures.append(f"{direction}_evidence.utterance_evidence_span")
            if isinstance(block_ids, list) and block_ids and len(block_ids) == len(set(block_ids)) and all(isinstance(i, str) and i in ids for i in block_ids):
                protected[f"{direction}_evidence.target_reason_ids"] = list(block_ids)
            else:
                block_failures.append(f"{direction}_evidence.target_reason_ids")
            repairable.extend(block_failures)
    if "unknown_fields" in failures:
        repairable.append("remove undefined fields without changing protected decision content")
    return _result(parsed, normalized, findings, failures, protected, repairable, same_reason_conflict=same_reason)


def substantive_stage1_key(normalized: Dict[str, Any], *, adjudicator: bool = False) -> Tuple[Any, ...]:
    gate = "question_function_present"
    purposes = "purposes"
    mapping = {item["label"]: normalize_whitespace(item["evidence_span"]) for item in normalized.get(purposes, [])}
    return normalized.get(gate), tuple(sorted(mapping.items()))


def substantive_stage2_key(terminal_alignments: Sequence[Dict[str, Any]]) -> Tuple[Tuple[str, str], ...]:
    return tuple(sorted(
        (str(item.get("terminal_reason_id", "")), normalize_whitespace(item.get("utterance_evidence_span", "")))
        for item in terminal_alignments
    ))


def substantive_stage3_key(normalized: Dict[str, Any]) -> Tuple[Any, ...]:
    def block_key(name: str) -> Any:
        block = normalized.get(name)
        if block is None:
            return None
        return (
            tuple(sorted(block.get("target_reason_ids", []))),
            normalize_whitespace(block.get("utterance_evidence_span", "")),
        )
    return normalized.get("stance_outcome"), block_key("favor_evidence"), block_key("against_evidence")


def protected_fields_unchanged(before: Dict[str, Any], after: Dict[str, Any]) -> bool:
    def lookup(payload: Any, path: str) -> Any:
        current = payload
        for token in re.findall(r"[^.\[\]]+|\d+", path):
            if isinstance(current, list):
                current = current[int(token)]
            elif isinstance(current, dict) and token in current:
                current = current[token]
            else:
                return object()
        return current
    for path, expected in before.items():
        if path.startswith("purpose_label:"):
            label = path.split(":", 1)[1]
            purposes = after.get("purposes", []) if isinstance(after, dict) else []
            if not any(isinstance(item, dict) and item.get("label") == label for item in purposes):
                return False
            continue
        if path.startswith("purpose_evidence:"):
            label = path.split(":", 1)[1]
            purposes = after.get("purposes", []) if isinstance(after, dict) else []
            matches = [item for item in purposes if isinstance(item, dict) and item.get("label") == label]
            if len(matches) != 1 or matches[0].get("evidence_span") != expected:
                return False
            continue
        actual = lookup(after, path)
        if actual != expected:
            return False
    return True


def validate_reason_tree(reason_targets: Sequence[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    by_id: Dict[str, Dict[str, Any]] = {}
    paths_by_brief: Set[Tuple[str, str]] = set()
    findings: List[str] = []
    for index, row in enumerate(reason_targets):
        raw_reason_id = row.get("reason_target_id")
        raw_brief_id = row.get("brief_id")
        raw_heading_path = row.get("heading_path")
        raw_heading_text = row.get("heading_text")
        raw_parent = row.get("parent_reason_target_id")
        reason_id = raw_reason_id if isinstance(raw_reason_id, str) else ""
        brief_id = raw_brief_id if isinstance(raw_brief_id, str) else ""
        heading_path = raw_heading_path if isinstance(raw_heading_path, str) else ""
        if not reason_id:
            findings.append(f"Reason row {index} has no reason_target_id.")
        elif reason_id in by_id:
            findings.append(f"reason_target_id {reason_id!r} is not globally unique.")
        else:
            by_id[reason_id] = row
        key = (brief_id, heading_path)
        if not brief_id or not heading_path or not isinstance(raw_heading_text, str) or not raw_heading_text:
            findings.append(f"Reason {reason_id!r} must have non-empty string brief_id, heading_path, and heading_text.")
        if not isinstance(raw_parent, str):
            findings.append(f"Reason {reason_id!r} must have a string parent_reason_target_id, empty only for a root.")
        elif key in paths_by_brief:
            findings.append(f"heading_path {heading_path!r} is not unique within brief {brief_id!r}.")
        paths_by_brief.add(key)
    for reason_id, row in by_id.items():
        parent_id = str(row.get("parent_reason_target_id", "") or "")
        children = row.get("direct_child_reason_target_ids")
        if not isinstance(children, list) or any(not isinstance(child, str) or not child for child in children):
            findings.append(f"Reason {reason_id!r} must have a list of direct_child_reason_target_ids.")
            children = []
        if len(children) != len(set(children)):
            findings.append(f"Reason {reason_id!r} has duplicate direct child IDs.")
        level = row.get("heading_level")
        if not isinstance(level, int) or isinstance(level, bool) or level < 1:
            findings.append(f"Reason {reason_id!r} must have a positive integer heading_level.")
        if parent_id:
            parent = by_id.get(parent_id)
            if parent is None:
                findings.append(f"Reason {reason_id!r} references missing parent {parent_id!r}.")
            else:
                if parent.get("brief_id") != row.get("brief_id"):
                    findings.append(f"Reason {reason_id!r} and parent {parent_id!r} belong to different briefs.")
                if isinstance(level, int) and isinstance(parent.get("heading_level"), int) and level != parent["heading_level"] + 1:
                    findings.append(f"Reason {reason_id!r} is not exactly one heading level below parent {parent_id!r}.")
                if reason_id not in (parent.get("direct_child_reason_target_ids") or []):
                    findings.append(f"Parent {parent_id!r} does not reciprocally list child {reason_id!r}.")
                if "parent_heading_path" in row and str(row.get("parent_heading_path") or "") != str(parent.get("heading_path") or ""):
                    findings.append(f"Reason {reason_id!r} has a parent_heading_path inconsistent with parent {parent_id!r}.")
        else:
            if isinstance(level, int) and level != 1:
                findings.append(f"Root reason {reason_id!r} must have heading_level=1.")
        for child_id in children:
            child = by_id.get(child_id)
            if child is None:
                findings.append(f"Reason {reason_id!r} lists missing direct child {child_id!r}.")
            elif str(child.get("parent_reason_target_id") or "") != reason_id:
                findings.append(f"Child {child_id!r} does not reciprocally reference parent {reason_id!r}.")
            elif child.get("brief_id") != row.get("brief_id"):
                findings.append(f"Reason {reason_id!r} and child {child_id!r} belong to different briefs.")
    for reason_id in by_id:
        seen: Set[str] = set()
        current = reason_id
        while current:
            if current in seen:
                findings.append(f"Reason hierarchy contains a cycle involving {current!r}.")
                break
            seen.add(current)
            node = by_id.get(current)
            if node is None:
                findings.append(f"Reason {reason_id!r} cannot reach a root because ancestor {current!r} is missing.")
                break
            current = str(node.get("parent_reason_target_id") or "")
    if findings:
        raise ValueError("Reason-tree data-integrity preflight failed:\n- " + "\n- ".join(dict.fromkeys(findings)))
    return by_id


def ancestor_path(reason_id: str, by_id: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    if reason_id not in by_id:
        raise ValueError(f"Reason-tree data-integrity failure: unknown reason_target_id {reason_id!r}.")
    reverse: List[Dict[str, Any]] = []
    current = reason_id
    seen: Set[str] = set()
    while current:
        if current in seen or current not in by_id:
            raise ValueError(f"Reason-tree data-integrity failure while constructing ancestor path for {reason_id!r}.")
        seen.add(current)
        node = by_id[current]
        reverse.append(node)
        current = str(node.get("parent_reason_target_id") or "")
    return list(reversed(reverse))


def build_reason_bundle(utterance_id: str, terminal_ids: Sequence[str], reason_targets: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    by_id = validate_reason_tree(reason_targets)
    if len(terminal_ids) != len(set(terminal_ids)):
        raise ValueError("Terminal selected reason IDs must be unique.")
    input_order = {str(row.get("reason_target_id", "")): index for index, row in enumerate(reason_targets)}
    ordered_terminal_ids = sorted(terminal_ids, key=lambda reason_id: input_order.get(reason_id, len(input_order)))
    paths = {
        reason_id: [str(node["reason_target_id"]) for node in ancestor_path(reason_id, by_id)]
        for reason_id in ordered_terminal_ids
    }
    for left in ordered_terminal_ids:
        for right in ordered_terminal_ids:
            if left != right and left in paths[right]:
                raise ValueError(f"Terminal selections cannot contain ancestor {left!r} and descendant {right!r} together.")
    selected_paths: List[Dict[str, Any]] = []
    for reason_id in ordered_terminal_ids:
        nodes = ancestor_path(reason_id, by_id)
        node_payloads = [
            {
                "reason_target_id": node["reason_target_id"],
                "heading_path": node["heading_path"],
                "heading_text": node["heading_text"],
            }
            for node in nodes
        ]
        path_text = " / ".join(f"{node['heading_path']}. {node['heading_text']}" for node in node_payloads)
        selected_paths.append({
            "terminal_reason_id": reason_id,
            "brief_id": by_id[reason_id]["brief_id"],
            "hierarchical_reason_node_ids": [node["reason_target_id"] for node in node_payloads],
            "hierarchical_reason_nodes": node_payloads,
            "reason_path_text": path_text,
        })
    return {
        "bundle_id": f"{utterance_id}__reason_bundle",
        "terminal_selected_reason_ids": list(ordered_terminal_ids),
        "selected_hierarchical_paths": selected_paths,
    }


def validate_human_terminal_selection(
    terminal_ids: Sequence[str], none_of_the_above: bool, reason_targets: Sequence[Dict[str, Any]]
) -> None:
    if not isinstance(terminal_ids, list) or any(not isinstance(value, str) or not value for value in terminal_ids):
        raise ValueError("Human terminal selected reason IDs must be a JSON string list.")
    if not isinstance(none_of_the_above, bool):
        raise ValueError("Human none_of_the_above must be a JSON boolean.")
    if none_of_the_above != (len(terminal_ids) == 0):
        raise ValueError("Human none_of_the_above must be true exactly when the selected ID list is empty.")
    build_reason_bundle("human_validation", terminal_ids, reason_targets)


def bundle_reason_ids(bundle: Dict[str, Any]) -> Set[str]:
    return {
        str(reason_id)
        for path in bundle.get("selected_hierarchical_paths", [])
        for reason_id in path.get("hierarchical_reason_node_ids", [])
    }


def build_selected_reason_tree(bundle: Dict[str, Any]) -> List[Dict[str, Any]]:
    terminal_ids = bundle.get("terminal_selected_reason_ids")
    paths = bundle.get("selected_hierarchical_paths")
    if (
        not isinstance(terminal_ids, list)
        or any(not isinstance(value, str) or not value for value in terminal_ids)
        or len(terminal_ids) != len(set(terminal_ids))
        or not isinstance(paths, list)
    ):
        raise ValueError("Selected reason bundle is malformed and cannot be rendered as a tree.")
    if not terminal_ids and not paths:
        return []
    if not terminal_ids or not paths:
        raise ValueError("Selected reason bundle terminals and hierarchical paths must both be present.")

    terminal_set = set(terminal_ids)
    path_terminal_ids: List[str] = []
    node_metadata: Dict[str, Dict[str, Any]] = {}
    node_briefs: Dict[str, str] = {}
    parent_by_id: Dict[str, str] = {}
    children_by_id: Dict[str, List[str]] = {}
    root_ids: Set[str] = set()
    brief_order: List[str] = []
    roots_by_brief: Dict[str, List[str]] = {}

    for path in paths:
        if not isinstance(path, dict):
            raise ValueError("Selected reason bundle contains a malformed hierarchical path.")
        terminal_id = path.get("terminal_reason_id")
        brief_id = path.get("brief_id")
        nodes = path.get("hierarchical_reason_nodes")
        listed_node_ids = path.get("hierarchical_reason_node_ids")
        if (
            not isinstance(terminal_id, str)
            or not terminal_id
            or terminal_id not in terminal_set
            or not isinstance(brief_id, str)
            or not brief_id
            or not isinstance(nodes, list)
            or not nodes
            or not isinstance(listed_node_ids, list)
        ):
            raise ValueError("Selected reason bundle contains an invalid hierarchical path.")

        path_node_ids: List[str] = []
        previous_id = ""
        for index, node in enumerate(nodes):
            if not isinstance(node, dict):
                raise ValueError("Selected reason bundle contains a malformed reason node.")
            reason_id = node.get("reason_target_id")
            heading_path = node.get("heading_path")
            heading_text = node.get("heading_text")
            if (
                not isinstance(reason_id, str)
                or not reason_id
                or not isinstance(heading_path, str)
                or not heading_path
                or not isinstance(heading_text, str)
                or not heading_text
                or reason_id in path_node_ids
            ):
                raise ValueError("Selected reason bundle contains an invalid reason node.")
            path_node_ids.append(reason_id)
            metadata = {
                "reason_target_id": reason_id,
                "heading_path": heading_path,
                "heading_text": heading_text,
            }
            if reason_id in node_metadata and (
                node_metadata[reason_id] != metadata or node_briefs[reason_id] != brief_id
            ):
                raise ValueError(f"Selected reason node {reason_id!r} has inconsistent metadata across paths.")
            node_metadata.setdefault(reason_id, metadata)
            node_briefs.setdefault(reason_id, brief_id)
            children_by_id.setdefault(reason_id, [])

            if index == 0:
                if reason_id in parent_by_id:
                    raise ValueError(f"Selected reason node {reason_id!r} is both a root and a child.")
                root_ids.add(reason_id)
                if brief_id not in roots_by_brief:
                    brief_order.append(brief_id)
                    roots_by_brief[brief_id] = []
                if reason_id not in roots_by_brief[brief_id]:
                    roots_by_brief[brief_id].append(reason_id)
            else:
                if reason_id in root_ids:
                    raise ValueError(f"Selected reason node {reason_id!r} is both a root and a child.")
                existing_parent = parent_by_id.get(reason_id)
                if existing_parent is not None and existing_parent != previous_id:
                    raise ValueError(f"Selected reason node {reason_id!r} has more than one parent.")
                parent_by_id.setdefault(reason_id, previous_id)
                if node_briefs[previous_id] != brief_id:
                    raise ValueError("Selected reason hierarchy cannot cross brief boundaries.")
                if reason_id not in children_by_id[previous_id]:
                    children_by_id[previous_id].append(reason_id)
            previous_id = reason_id

        if path_node_ids != listed_node_ids or path_node_ids[-1] != terminal_id:
            raise ValueError("Selected reason bundle path IDs or terminal identity are inconsistent.")
        path_terminal_ids.append(terminal_id)

    if path_terminal_ids != terminal_ids:
        raise ValueError("Selected reason bundle terminal order does not match its hierarchical paths.")
    if set(node_metadata) != bundle_reason_ids(bundle):
        raise ValueError("Selected reason tree does not exactly cover the canonical bundle reason IDs.")
    if any(children_by_id[reason_id] for reason_id in terminal_set):
        raise ValueError("A selected terminal reason cannot also have a selected descendant.")

    rendered_ids: Set[str] = set()

    def render_node(reason_id: str) -> Dict[str, Any]:
        if reason_id in rendered_ids:
            raise ValueError(f"Selected reason node {reason_id!r} appears more than once in the tree.")
        rendered_ids.add(reason_id)
        metadata = node_metadata[reason_id]
        return {
            "reason_target_id": reason_id,
            "heading_path": metadata["heading_path"],
            "heading_text": metadata["heading_text"],
            "selected_terminal": reason_id in terminal_set,
            "children": [render_node(child_id) for child_id in children_by_id[reason_id]],
        }

    tree = [
        {
            "brief_id": brief_id,
            "roots": [render_node(reason_id) for reason_id in roots_by_brief[brief_id]],
        }
        for brief_id in brief_order
    ]
    if rendered_ids != set(node_metadata):
        raise ValueError("Selected reason tree does not contain every canonical bundle reason node exactly once.")
    return tree


def flatten_reason_bundle(bundle: Dict[str, Any]) -> str:
    return " || ".join(str(path.get("reason_path_text", "")) for path in bundle.get("selected_hierarchical_paths", []))


def locator_paths(bundle: Dict[str, Any], reason_ids: Sequence[str]) -> List[str]:
    targets = set(reason_ids)
    result: Dict[str, str] = {}
    for path in bundle.get("selected_hierarchical_paths", []):
        accumulated: List[str] = []
        for node in path.get("hierarchical_reason_nodes", []):
            accumulated.append(f"{node.get('heading_path', '')}. {node.get('heading_text', '')}")
            reason_id = str(node.get("reason_target_id", ""))
            if reason_id in targets and reason_id not in result:
                result[reason_id] = " / ".join(accumulated)
    return [result[reason_id] for reason_id in reason_ids if reason_id in result]
