from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

from .justice_checkpoint_common import identity_key, load_human_final_checkpoint, read_jsonl, require_consumer_hash


def _validated_tree(reason_nodes: Sequence[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    by_id: Dict[str, Dict[str, Any]] = {}
    paths: set[tuple[str, str]] = set()
    for row in reason_nodes:
        reason_id = str(row.get("reason_target_id") or "")
        brief_id = str(row.get("brief_id") or "")
        heading_path = str(row.get("heading_path") or "")
        heading_text = row.get("heading_text")
        parent_value = row.get("parent_reason_target_id")
        children = row.get("direct_child_reason_target_ids")
        level = row.get("heading_level")
        if (
            not reason_id
            or reason_id in by_id
            or not brief_id
            or not heading_path
            or not isinstance(heading_text, str)
            or not heading_text
            or not isinstance(parent_value, str)
            or (brief_id, heading_path) in paths
            or not isinstance(children, list)
            or any(not isinstance(value, str) or not value for value in children)
            or len(children) != len(set(children))
            or not isinstance(level, int)
            or isinstance(level, bool)
            or level < 1
        ):
            raise ValueError("Stage 2 checkpoint reason hierarchy is invalid.")
        by_id[reason_id] = row
        paths.add((brief_id, heading_path))
    for reason_id, row in by_id.items():
        parent_id = str(row.get("parent_reason_target_id") or "")
        level = int(row["heading_level"])
        if parent_id:
            parent = by_id.get(parent_id)
            if (
                parent is None
                or parent.get("brief_id") != row.get("brief_id")
                or int(parent.get("heading_level")) + 1 != level
                or reason_id not in (parent.get("direct_child_reason_target_ids") or [])
            ):
                raise ValueError("Stage 2 checkpoint reason parent relation is invalid.")
        elif level != 1:
            raise ValueError("Stage 2 checkpoint root reason must have heading_level=1.")
        for child_id in row["direct_child_reason_target_ids"]:
            child = by_id.get(child_id)
            if (
                child is None
                or str(child.get("parent_reason_target_id") or "") != reason_id
                or child.get("brief_id") != row.get("brief_id")
            ):
                raise ValueError("Stage 2 checkpoint reason child relation is invalid.")
        seen: set[str] = set()
        current = reason_id
        while current:
            if current in seen or current not in by_id:
                raise ValueError("Stage 2 checkpoint reason hierarchy has a cycle or missing ancestor.")
            seen.add(current)
            current = str(by_id[current].get("parent_reason_target_id") or "")
    return by_id


def _ancestor_path(reason_id: str, by_id: Mapping[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    if reason_id not in by_id:
        raise ValueError(f"Stage 2 checkpoint selection contains an unknown reason ID: {reason_id!r}.")
    reverse: List[Dict[str, Any]] = []
    current = reason_id
    while current:
        node = by_id[current]
        reverse.append(node)
        current = str(node.get("parent_reason_target_id") or "")
    return list(reversed(reverse))


def _validate_and_build_bundle(
    utterance_id: str, selected: Sequence[str], none: bool, reason_nodes: Sequence[Dict[str, Any]]
) -> Dict[str, Any]:
    if (
        not isinstance(selected, list)
        or any(not isinstance(value, str) or not value for value in selected)
        or len(selected) != len(set(selected))
        or not isinstance(none, bool)
        or none != (len(selected) == 0)
    ):
        raise ValueError("Stage 2 checkpoint selection violates the strict terminal-selection contract.")
    by_id = _validated_tree(reason_nodes)
    paths = {reason_id: [str(node["reason_target_id"]) for node in _ancestor_path(reason_id, by_id)] for reason_id in selected}
    for left in selected:
        for right in selected:
            if left != right and left in paths[right]:
                raise ValueError("Stage 2 checkpoint terminal selections contain an ancestor and descendant.")
    input_order = {str(row["reason_target_id"]): index for index, row in enumerate(reason_nodes)}
    ordered = sorted(selected, key=lambda reason_id: input_order.get(reason_id, len(input_order)))
    hierarchical_paths: List[Dict[str, Any]] = []
    for reason_id in ordered:
        path = _ancestor_path(reason_id, by_id)
        payloads = [
            {
                "reason_target_id": node["reason_target_id"],
                "heading_path": node["heading_path"],
                "heading_text": node["heading_text"],
            }
            for node in path
        ]
        hierarchical_paths.append(
            {
                "terminal_reason_id": reason_id,
                "brief_id": by_id[reason_id]["brief_id"],
                "hierarchical_reason_node_ids": [node["reason_target_id"] for node in payloads],
                "hierarchical_reason_nodes": payloads,
                "reason_path_text": " / ".join(
                    f"{node['heading_path']}. {node['heading_text']}" for node in payloads
                ),
            }
        )
    return {
        "bundle_id": f"{utterance_id}__reason_bundle",
        "terminal_selected_reason_ids": ordered,
        "selected_hierarchical_paths": hierarchical_paths,
    }


def _mapping_by_argument(identity: Mapping[str, Any]) -> Dict[str, set[str]]:
    output: Dict[str, set[str]] = {}
    for row in identity.get("run_scope_mapping") or []:
        if not isinstance(row, dict):
            raise ValueError("Stage 2 checkpoint dependency mapping is malformed.")
        argument_id = str(row.get("oyez_argument_id") or "")
        brief_ids = row.get("brief_ids")
        if not argument_id or argument_id in output or not isinstance(brief_ids, list):
            raise ValueError("Stage 2 checkpoint dependency mapping is malformed or duplicated.")
        output[argument_id] = {str(value) for value in brief_ids}
    return output


def _validate_dependency_compatibility(
    checkpoint_identity: Mapping[str, Any], new_identity: Mapping[str, Any], required_arguments: set[str]
) -> None:
    old_mapping = _mapping_by_argument(checkpoint_identity)
    new_mapping = _mapping_by_argument(new_identity)
    old_briefs = checkpoint_identity.get("briefs") or {}
    new_briefs = new_identity.get("briefs") or {}
    for argument_id in required_arguments:
        if argument_id not in old_mapping or argument_id not in new_mapping:
            raise ValueError("Stage 2 checkpoint does not cover every required oral-argument brief scope.")
        if old_mapping[argument_id] != new_mapping[argument_id]:
            raise ValueError("Stage 2 checkpoint run-scope mapping is incompatible for a required oral argument.")
        for brief_id in new_mapping[argument_id]:
            if old_briefs.get(brief_id) != new_briefs.get(brief_id):
                raise ValueError("Stage 2 checkpoint brief PDF or reason-tree hash is incompatible.")


def _candidate_scope_by_id(
    directory: Path, manifest: Mapping[str, Any], clean_ids: set[str]
) -> Dict[str, Dict[str, Any]]:
    relative = str(manifest.get("stage2_candidate_scope_relative_path") or "")
    if not relative:
        raise ValueError("Stage 2 checkpoint lacks the frozen per-utterance candidate scope required for reuse.")
    allowed_fields = {
        "oyez_argument_id",
        "utterance_id",
        "advocate_group",
        "eligible_brief_ids",
        "candidate_reason_ids",
    }
    by_id: Dict[str, Dict[str, Any]] = {}
    for row in read_jsonl(directory / relative):
        utterance_id = str(row.get("utterance_id") or "")
        argument_id = str(row.get("oyez_argument_id") or "")
        advocate_group = str(row.get("advocate_group") or "")
        brief_ids = row.get("eligible_brief_ids")
        reason_ids = row.get("candidate_reason_ids")
        if (
            set(row) != allowed_fields
            or not utterance_id
            or utterance_id in by_id
            or not argument_id
            or not advocate_group
            or not isinstance(brief_ids, list)
            or not brief_ids
            or any(not isinstance(value, str) or not value for value in brief_ids)
            or len(brief_ids) != len(set(brief_ids))
            or not isinstance(reason_ids, list)
            or not reason_ids
            or any(not isinstance(value, str) or not value for value in reason_ids)
            or len(reason_ids) != len(set(reason_ids))
        ):
            raise ValueError("Stage 2 checkpoint candidate-scope sidecar is malformed or duplicated.")
        by_id[utterance_id] = dict(row)
    if set(by_id) != clean_ids:
        raise ValueError("Stage 2 checkpoint candidate-scope sidecar does not exactly cover its clean final.")
    return by_id


def required_stage2_utterances(
    checkpoint_id: str, new_run_utterances: Sequence[Mapping[str, Any]]
) -> list[Dict[str, Any]]:
    directory, manifest = load_human_final_checkpoint(checkpoint_id)
    if manifest.get("stage") != "stage2":
        raise ValueError("Stage 2 routing coverage is only available from a Stage 2 human-final checkpoint.")
    require_consumer_hash(manifest, "stage2_consumer", Path(__file__))
    from . import justice_checkpoint_common
    require_consumer_hash(manifest, "checkpoint_common", Path(justice_checkpoint_common.__file__))
    relative = str(manifest.get("stage2_routing_coverage_relative_path") or "")
    if not relative:
        raise ValueError("Stage 2 checkpoint lacks the complete upstream routing coverage required for reuse.")
    coverage = read_jsonl(directory / relative)
    by_id: Dict[str, Dict[str, Any]] = {}
    coverage_fields = {
        "oyez_argument_id", "utterance_id", "turn_id", "target_justice", "text", "enters_reason_alignment"
    }
    for row in coverage:
        utterance_id = str(row.get("utterance_id") or "")
        if (
            set(row) != coverage_fields
            or not utterance_id
            or utterance_id in by_id
            or not isinstance(row.get("enters_reason_alignment"), bool)
        ):
            raise ValueError("Stage 2 checkpoint routing coverage is missing, duplicated, or invalid.")
        by_id[utterance_id] = dict(row)
    required: list[Dict[str, Any]] = []
    seen: set[str] = set()
    for row in new_run_utterances:
        utterance_id = str(row.get("utterance_id") or "")
        if not utterance_id or utterance_id in seen:
            raise ValueError("New run utterance scope contains missing or duplicate IDs.")
        seen.add(utterance_id)
        old = by_id.get(utterance_id)
        if old is None or identity_key(old) != identity_key(row):
            raise ValueError(f"Stage 2 checkpoint routing coverage does not match new-run utterance {utterance_id!r}.")
        if old["enters_reason_alignment"]:
            required.append(dict(row))
    return required


def consume_stage2_checkpoint(
    checkpoint_id: str,
    required_utterances: Sequence[Mapping[str, Any]],
    reason_nodes: Sequence[Dict[str, Any]],
    dependency_identity: Mapping[str, Any],
) -> list[Dict[str, Any]]:
    directory, manifest = load_human_final_checkpoint(checkpoint_id)
    if manifest.get("stage") != "stage2":
        raise ValueError("Only a Stage 2 human-final checkpoint can start a new run at Stage 3.")
    require_consumer_hash(manifest, "stage2_consumer", Path(__file__))
    from . import justice_checkpoint_common
    require_consumer_hash(manifest, "checkpoint_common", Path(justice_checkpoint_common.__file__))
    _validated_tree(reason_nodes)
    clean = read_jsonl(directory / str(manifest.get("clean_final_relative_path") or "clean_final.jsonl"))
    by_id: Dict[str, Dict[str, Any]] = {}
    allowed_fields = {
        "oyez_argument_id", "utterance_id", "turn_id", "target_justice", "text",
        "final_selected_reason_ids", "final_none_of_the_above",
    }
    for row in clean:
        if set(row) != allowed_fields:
            raise ValueError("Stage 2 checkpoint clean final does not match the approved clean schema.")
        utterance_id = str(row.get("utterance_id") or "")
        if not utterance_id or utterance_id in by_id:
            raise ValueError("Stage 2 checkpoint clean final contains missing or duplicate utterance IDs.")
        selected = row.get("final_selected_reason_ids")
        none = row.get("final_none_of_the_above")
        if not isinstance(selected, list) or not isinstance(none, bool):
            raise ValueError("Stage 2 checkpoint clean final has invalid selection fields.")
        by_id[utterance_id] = dict(row)
    candidate_scope = _candidate_scope_by_id(directory, manifest, set(by_id))
    required_by_id = {str(row.get("utterance_id") or ""): row for row in required_utterances}
    if any(not key for key in required_by_id) or len(required_by_id) != len(required_utterances):
        raise ValueError("New run required Stage 2 utterances contain missing or duplicate IDs.")
    missing = sorted(set(required_by_id) - set(by_id))
    if missing:
        raise ValueError(f"Stage 2 checkpoint does not cover every new-run required utterance: {missing[:10]!r}.")
    required_arguments = {str(row.get("oyez_argument_id") or "") for row in required_utterances}
    _validate_dependency_compatibility(
        manifest.get("stage2_dependency_identity") or {}, dependency_identity, required_arguments
    )
    output: list[Dict[str, Any]] = []
    for utterance_id, required in required_by_id.items():
        row = by_id[utterance_id]
        if identity_key(row) != identity_key(required):
            raise ValueError(f"Stage 2 checkpoint utterance identity/text mismatch: {utterance_id!r}.")
        selected = list(row["final_selected_reason_ids"])
        none = bool(row["final_none_of_the_above"])
        allowed_briefs = required.get("run_scope_brief_ids")
        advocate_group = str(required.get("advocate_group") or "")
        if not isinstance(allowed_briefs, list) or not allowed_briefs or not advocate_group:
            raise ValueError("New run Stage 3 input lacks the frozen brief scope or advocate_group required for Stage 2 selection validation.")
        eligible_nodes = [
            node
            for node in reason_nodes
            if node.get("brief_id") in set(allowed_briefs)
            and str(node.get("brief_group") or "") == advocate_group
        ]
        if not eligible_nodes:
            raise ValueError("New run has no exact Stage 2 candidate pool for a checkpoint utterance.")
        frozen_scope = candidate_scope[utterance_id]
        current_brief_ids = {str(node.get("brief_id") or "") for node in eligible_nodes}
        current_reason_ids = {str(node.get("reason_target_id") or "") for node in eligible_nodes}
        if (
            str(frozen_scope.get("oyez_argument_id") or "") != str(required.get("oyez_argument_id") or "")
            or str(frozen_scope.get("advocate_group") or "") != advocate_group
            or set(frozen_scope.get("eligible_brief_ids") or []) != current_brief_ids
            or set(frozen_scope.get("candidate_reason_ids") or []) != current_reason_ids
        ):
            raise ValueError("Stage 2 checkpoint candidate scope is incompatible for a required utterance.")
        bundle = _validate_and_build_bundle(utterance_id, selected, none, eligible_nodes)
        output.append(
            {
                **required,
                "final_selected_reason_ids": selected,
                "final_none_of_the_above": none,
                "final_reason_bundle": bundle,
                "stage2_resolved": True,
                "stage2_requires_human_review": False,
                "final_source": "human_final_checkpoint",
            }
        )
    return output
