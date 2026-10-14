from __future__ import annotations

import argparse
import json
import math
import re
import shutil
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple

from .common import (
    clean_text,
    estimate_tokens,
    load_config,
    model_payload,
    print_progress,
    progress_settings,
    read_jsonl,
    render_template,
    resolve_path,
    safe_name,
    write_csv,
    write_jsonl,
)
from .llm_client import chat_client_for_model
from .justice_contracts import (
    PURPOSE_LABEL_ORDER,
    build_reason_bundle as build_hierarchical_reason_bundle,
    build_selected_reason_tree,
    bundle_reason_ids,
    flatten_reason_bundle,
    locator_paths,
    normalize_purpose_label,
    normalized_verbatim_span,
    protected_fields_unchanged,
    substantive_stage1_key,
    substantive_stage2_key,
    substantive_stage3_key,
    validate_reason_tree,
    validate_stage1_output,
    validate_stage2_output,
    validate_stage3_output,
)
from .justice_runtime import (
    APPROVED_MODEL_PROFILES,
    RESPONSE_FORMAT,
    append_attempt_audit,
    configure_model_profile,
    model_manifest_entry,
    request_parameters,
    require_sealed_manifest,
    role_parameters,
    run_identity_preflight,
    seal_run_manifest,
    sha256_file,
    sha256_json,
)


RETAINED_PURPOSE_LABELS = {"Clarification", "Implications", "Support", "Criticism"}
BRIEF_GROUPS = {"pet", "res", "caa", "other"}


def annotation_run_config(config: Dict[str, Any]) -> Dict[str, Any]:
    return config.get("annotation_run", {}) or {}



LOCAL_CONTEXT_TYPE = "same_advocate_group_local_window_with_opening"


def local_context_config(config: Dict[str, Any]) -> Dict[str, Any]:
    value = annotation_run_config(config).get("local_context")
    if isinstance(value, dict):
        return value
    return {
        "enabled": True,
        "type": LOCAL_CONTEXT_TYPE,
        "before_tokens": 450,
        "after_tokens": 450,
        "long_flag_threshold": 1500,
    }


def local_context_enabled(config: Dict[str, Any]) -> bool:
    return bool(local_context_config(config).get("enabled", True))


def local_context_type(config: Dict[str, Any]) -> str:
    value = str(local_context_config(config).get("type", LOCAL_CONTEXT_TYPE))
    if value != LOCAL_CONTEXT_TYPE:
        raise ValueError(f"Unsupported local dialogue-context type: {value!r}.")
    return value


def local_context_before_tokens(config: Dict[str, Any]) -> int:
    return int(local_context_config(config).get("before_tokens", 450))


def local_context_after_tokens(config: Dict[str, Any]) -> int:
    return int(local_context_config(config).get("after_tokens", 450))


def local_context_long_flag_threshold(config: Dict[str, Any]) -> int:
    return int(local_context_config(config).get("long_flag_threshold", 1500))


def annotation_model(config: Dict[str, Any], model_id: str, stage: str = "annotation") -> Dict[str, str]:
    block = (config.get("models", {}).get(stage) or {}).get(model_id)
    if not block:
        raise ValueError(f"Model '{model_id}' is not configured under models.{stage}.")
    return model_payload(model_id, block)


def annotator_models(config: Dict[str, Any]) -> List[Dict[str, str]]:
    workflow = config.get("annotation_workflow", {}) or {}
    model_ids = workflow.get("annotator_models") or []
    if len(model_ids) != 2:
        raise ValueError("annotation_workflow.annotator_models must contain exactly two models.")
    return [annotation_model(config, str(model_id), "annotation") for model_id in model_ids]


def adjudicator_model(config: Dict[str, Any]) -> Dict[str, str]:
    workflow = config.get("annotation_workflow", {}) or {}
    model_id = workflow.get("adjudicator_model")
    if not model_id:
        raise ValueError("annotation_workflow.adjudicator_model is required.")
    return annotation_model(config, str(model_id), "adjudication")



def is_justice_row(row: Dict[str, Any]) -> bool:
    speaker_role = clean_text(row.get("speaker_role", "")).lower()
    if "scotus_justice" in speaker_role:
        return True
    speaker = clean_text(row.get("speaker", "")).lower()
    return any(
        name in speaker
        for name in [
            "roberts",
            "thomas",
            "alito",
            "sotomayor",
            "kagan",
            "gorsuch",
            "kavanaugh",
            "barrett",
            "jackson",
        ]
    )


def format_dialogue_turn(row: Dict[str, Any]) -> str:
    speaker = clean_text(row.get("speaker", ""))
    turn_id = clean_text(row.get("turn_id", ""))
    text = clean_text(row.get("text", ""))
    return f"[{turn_id}] {speaker}: {text}" if turn_id else f"{speaker}: {text}"



def context_scope_key_text(key: Tuple[str, ...]) -> str:
    return "::".join(key)


def context_id_for_parts(*parts: Any) -> str:
    return safe_name("__".join(clean_text(part) for part in parts if clean_text(part))) or "context"


def context_turn_id(turn: Dict[str, Any]) -> str:
    return clean_text(turn.get("turn_id", ""))


def selected_context_text(turns: Sequence[Dict[str, Any]]) -> str:
    return "\n".join(turn.get("_formatted_context_turn", "") for turn in turns)


def opening_context_indexes(turns: Sequence[Dict[str, Any]]) -> List[int]:
    if not turns:
        return []
    indexes = [0]
    if len(turns) > 1 and not is_justice_row(turns[1]):
        indexes.append(1)
    return indexes


def write_context_snapshot(run_dir: Optional[Path], context_id: str, context: str) -> str:
    if run_dir is None:
        return ""
    path = run_dir / "04_justice_utterances" / "dialogue_contexts" / f"{context_id}.txt"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(context, encoding="utf-8")
    return str(path)


def context_audit_row(
    utterance: Dict[str, Any],
    key: Tuple[str, ...],
    context_id: str,
    selected_turns: Sequence[Dict[str, Any]],
    all_turns: Sequence[Dict[str, Any]],
    max_tokens: int,
    context_type: str,
    text_path: str,
    truncation: Dict[str, Any],
) -> Dict[str, Any]:
    selected_ids = [context_turn_id(turn) for turn in selected_turns if context_turn_id(turn)]
    current_id = clean_text(utterance.get("turn_id", ""))
    current_index = next((index for index, turn in enumerate(all_turns) if context_turn_id(turn) == current_id), -1)
    opening_ids = [context_turn_id(all_turns[index]) for index in opening_context_indexes(all_turns)]
    preceding_count = len([turn for turn in selected_turns if current_index >= 0 and all_turns.index(turn) < current_index])
    following_count = len([turn for turn in selected_turns if current_index >= 0 and all_turns.index(turn) > current_index])
    immediate_answer_id = ""
    if current_index >= 0 and current_index + 1 < len(all_turns) and not is_justice_row(all_turns[current_index + 1]):
        immediate_answer_id = context_turn_id(all_turns[current_index + 1])
    return {
        "oyez_argument_id": utterance.get("oyez_argument_id", ""),
        "utterance_id": utterance.get("utterance_id", ""),
        "case_id": utterance.get("case_id", ""),
        "turn_id": utterance.get("turn_id", ""),
        "section_counsel": utterance.get("section_counsel", ""),
        "advocate_group": utterance.get("advocate_group", ""),
        "dialogue_context_id": context_id,
        "dialogue_context_scope_key": context_scope_key_text(key),
        "dialogue_context_type": context_type,
        "dialogue_context_max_tokens": max_tokens,
        "dialogue_context_token_estimate": estimate_tokens(selected_context_text(selected_turns)),
        "dialogue_context_truncated": truncation.get("dialogue_context_truncated", False),
        "dialogue_context_required_over_cap": truncation.get("dialogue_context_required_over_cap", False),
        "dialogue_context_opening_included": all(opening_id in selected_ids for opening_id in opening_ids if opening_id),
        "dialogue_context_current_utterance_included": current_id in selected_ids,
        "dialogue_context_immediate_answer_included": bool(immediate_answer_id and immediate_answer_id in selected_ids),
        "dialogue_context_preceding_turn_count": preceding_count,
        "dialogue_context_following_turn_count": following_count,
        "dialogue_context_source_turn_count": len(selected_turns),
        "dialogue_context_omitted_before_count": truncation.get("dialogue_context_omitted_before_count", 0),
        "dialogue_context_omitted_after_count": truncation.get("dialogue_context_omitted_after_count", 0),
        "dialogue_context_source_turn_ids": selected_ids,
        "dialogue_context_text_path": text_path,
    }


def local_context_turns(
    turns: Sequence[Dict[str, Any]],
    current_turn_id: str,
    before_tokens: int,
    after_tokens: int,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    current_index = next((index for index, turn in enumerate(turns) if context_turn_id(turn) == current_turn_id), 0)
    selected = set(opening_context_indexes(turns))
    selected.add(current_index)

    before_total = 0
    for index in range(current_index - 1, -1, -1):
        if index in selected:
            continue
        selected.add(index)
        before_total += int(turns[index].get("_token_estimate", 0))
        if before_total >= before_tokens:
            break

    after_total = 0
    for index in range(current_index + 1, len(turns)):
        if index in selected:
            continue
        selected.add(index)
        after_total += int(turns[index].get("_token_estimate", 0))
        if after_total >= after_tokens:
            break

    selected_indexes = sorted(selected)
    return [turns[index] for index in selected_indexes], {
        "dialogue_context_before_token_budget": before_tokens,
        "dialogue_context_after_token_budget": after_tokens,
        "dialogue_context_before_tokens_selected": before_total,
        "dialogue_context_after_tokens_selected": after_total,
        "dialogue_context_truncated": len(selected_indexes) < len(turns),
        "dialogue_context_required_over_cap": False,
        "dialogue_context_omitted_before_count": len([index for index in range(0, current_index) if index not in selected]),
        "dialogue_context_omitted_after_count": len([index for index in range(current_index + 1, len(turns)) if index not in selected]),
    }


def build_local_dialogue_contexts(
    transcript_rows: Sequence[Dict[str, Any]],
    utterances: Sequence[Dict[str, Any]],
    config: Dict[str, Any],
    run_dir: Optional[Path] = None,
) -> List[Dict[str, Any]]:
    context_type = local_context_type(config)
    if not local_context_enabled(config):
        return [
            {
                "local_dialogue_context": "",
                "local_dialogue_context_type": "disabled",
                "local_dialogue_context_turn_count": 0,
                "local_dialogue_context_token_estimate": 0,
                "local_dialogue_context_counsel": utterance.get("section_counsel", ""),
                "local_dialogue_context_advocate_group": utterance.get("advocate_group", ""),
                "local_dialogue_context_id": "",
                "local_dialogue_context_text_path": "",
                "local_dialogue_context_long_flag": False,
            }
            for utterance in utterances
        ]

    enriched_rows = []
    by_turn_id: Dict[str, int] = {}
    for row in transcript_rows:
        if not clean_text(row.get("text", "")):
            continue
        if not clean_text(row.get("advocate_group", "")):
            row_id = clean_text(row.get("turn_id", "")) or clean_text(row.get("utterance_id", "")) or "[unknown]"
            raise ValueError(
                f"Immutable transcript row {row_id!r} is missing advocate_group; "
                "Local dialogue context cannot consult a live case manifest."
            )
        context = {
            key: row.get(key, "")
            for key in [
                "section_counsel", "advocate_name", "advocate_description", "advocate_group",
                "advocate_side", "advocate_group_source", "advocate_case_source_ids",
            ]
        }
        enriched = {**row, **context}
        by_turn_id[clean_text(enriched.get("turn_id", ""))] = len(enriched_rows)
        enriched_rows.append(enriched)

    before_tokens = local_context_before_tokens(config)
    after_tokens = local_context_after_tokens(config)
    long_threshold = local_context_long_flag_threshold(config)
    rows = []
    audit_rows = []
    for utterance in utterances:
        turn_id = clean_text(utterance.get("turn_id", ""))
        current_index = by_turn_id.get(turn_id, -1)
        if current_index < 0:
            rows.append(
                {
                    "local_dialogue_context_id": "",
                    "local_dialogue_context": "",
                    "local_dialogue_context_type": "missing_current_turn",
                    "local_dialogue_context_turn_count": 0,
                    "local_dialogue_context_token_estimate": 0,
                    "local_dialogue_context_counsel": utterance.get("section_counsel", ""),
                    "local_dialogue_context_advocate_group": utterance.get("advocate_group", ""),
                    "local_dialogue_context_text_path": "",
                    "local_dialogue_context_long_flag": False,
                }
            )
            continue

        argument_id = clean_text(utterance.get("oyez_argument_id", ""))
        advocate_group = clean_text(utterance.get("advocate_group", ""))
        start = current_index
        while start > 0:
            candidate = enriched_rows[start - 1]
            if clean_text(candidate.get("oyez_argument_id", "")) != argument_id or clean_text(candidate.get("advocate_group", "")) != advocate_group:
                break
            start -= 1
        end = current_index
        while end + 1 < len(enriched_rows):
            candidate = enriched_rows[end + 1]
            if clean_text(candidate.get("oyez_argument_id", "")) != argument_id or clean_text(candidate.get("advocate_group", "")) != advocate_group:
                break
            end += 1

        turns = []
        for row in enriched_rows[start : end + 1]:
            formatted = format_dialogue_turn(row)
            turns.append({**row, "_formatted_context_turn": formatted, "_token_estimate": estimate_tokens(formatted)})
        selected_turns, truncation = local_context_turns(turns, turn_id, before_tokens, after_tokens)
        context = selected_context_text(selected_turns)
        context_id = context_id_for_parts(argument_id, advocate_group, turn_id, "local")
        context_path = write_context_snapshot(run_dir, context_id, context)
        long_flag = estimate_tokens(context) > long_threshold
        rows.append(
            {
                "local_dialogue_context_id": context_id,
                "local_dialogue_context": context,
                "local_dialogue_context_type": context_type,
                "local_dialogue_context_turn_count": len(selected_turns),
                "local_dialogue_context_token_estimate": estimate_tokens(context) if context else 0,
                "local_dialogue_context_counsel": utterance.get("section_counsel", ""),
                "local_dialogue_context_advocate_group": utterance.get("advocate_group", ""),
                "local_dialogue_context_text_path": context_path,
                "local_dialogue_context_long_flag": long_flag,
            }
        )
        audit_row = context_audit_row(
            utterance,
            (argument_id, advocate_group),
            context_id,
            selected_turns,
            turns,
            before_tokens + after_tokens,
            context_type,
            context_path,
            truncation,
        )
        audit_row["dialogue_context_before_token_budget"] = before_tokens
        audit_row["dialogue_context_after_token_budget"] = after_tokens
        audit_row["dialogue_context_before_tokens_selected"] = truncation.get("dialogue_context_before_tokens_selected", 0)
        audit_row["dialogue_context_after_tokens_selected"] = truncation.get("dialogue_context_after_tokens_selected", 0)
        audit_row["dialogue_context_long_flag_threshold"] = long_threshold
        audit_row["dialogue_context_long_flag"] = long_flag
        audit_rows.append(audit_row)

    if run_dir is not None:
        out_dir = run_dir / "04_justice_utterances"
        write_jsonl(out_dir / "local_dialogue_context_audit.jsonl", audit_rows)
        write_csv(out_dir / "local_dialogue_context_audit.csv", audit_rows)
    return rows


def json_for_prompt(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=2)


def local_dialogue_context_text(row: Dict[str, Any]) -> str:
    """Return exactly the dialogue-context text shown to the annotation model."""
    return str(row.get("local_dialogue_context", "") or "")


def utterance_prompt_payload(row: Dict[str, Any]) -> Dict[str, Any]:
    """Return only the current utterance fields needed by an annotation model."""
    fields = ["utterance_id", "speaker", "text"]
    return {field: row.get(field, "") for field in fields if field in row}



def tokenize(text: Any) -> List[str]:
    words = re.findall(r"[a-zA-Z][a-zA-Z0-9']+", str(text or "").lower())
    stop = {
        "the", "and", "or", "to", "of", "in", "a", "an", "is", "are", "that", "this", "for", "on", "with",
        "it", "be", "by", "as", "from", "at", "you", "your", "we", "they", "their", "not", "does", "do",
        "what", "how", "why", "if", "so", "but", "would", "could", "should",
    }
    return [word for word in words if word not in stop and len(word) > 2]


def path_sort_key(path: str) -> Tuple[Any, ...]:
    parts: List[Any] = []
    roman_order = {
        "I": 1, "II": 2, "III": 3, "IV": 4, "V": 5,
        "VI": 6, "VII": 7, "VIII": 8, "IX": 9, "X": 10,
    }
    for part in str(path or "").split("."):
        if part in roman_order:
            parts.append((1, roman_order[part]))
        elif part.isalpha() and len(part) == 1 and part.isupper():
            parts.append((2, ord(part) - ord("A") + 1))
        elif part.isdigit():
            parts.append((3, int(part)))
        elif part.isalpha() and len(part) == 1 and part.islower():
            parts.append((4, ord(part) - ord("a") + 1))
        else:
            parts.append((9, part))
    return tuple(parts)


def lexical_score(query: str, target: Dict[str, Any]) -> float:
    q_tokens = tokenize(query)
    if not q_tokens:
        return 0.0
    target_text = f"{target.get('heading_text', '')} {target.get('reason_path_text', '')}"
    t_tokens = tokenize(target_text)
    if not t_tokens:
        return 0.0
    t_counts: Dict[str, int] = {}
    for token in t_tokens:
        t_counts[token] = t_counts.get(token, 0) + 1
    overlap = sum(math.sqrt(t_counts.get(token, 0)) for token in set(q_tokens))
    phrase_bonus = 0.0
    lower_target = target_text.lower()
    for token in set(q_tokens):
        if token in lower_target:
            phrase_bonus += 0.05
    return overlap + phrase_bonus



def target_with_diagnostic_score(utterance: Dict[str, Any], target: Dict[str, Any]) -> Dict[str, Any]:
    return {**target, "lexical_score": round(lexical_score(utterance.get("text", ""), target), 4)}


def stage2_candidate_pool(
    utterance: Dict[str, Any],
    reason_targets: Sequence[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    argument_id = clean_text(utterance.get("oyez_argument_id", ""))
    allowed_briefs = utterance.get("run_scope_brief_ids")
    if not isinstance(allowed_briefs, list) or not allowed_briefs:
        raise ValueError(f"Oral argument {argument_id!r} has no authoritative run-scope brief mapping.")
    scoped = [row for row in reason_targets if row.get("brief_id") in set(allowed_briefs)]
    advocate_group = clean_text(utterance.get("advocate_group", ""))
    if advocate_group not in BRIEF_GROUPS or advocate_group == "other":
        raise ValueError(f"Cannot build Stage 2 exact target pool for advocate_group={advocate_group!r}.")

    pool = [row for row in scoped if clean_text(row.get("brief_group", "")) == advocate_group]
    if not pool:
        raise ValueError(
            f"No exact Stage 2 targets within the run-scope mapping for "
            f"oyez_argument_id={argument_id}, advocate_group={advocate_group}."
        )
    metadata = {
        "target_pool_group": advocate_group,
        "target_pool_source": "exact_advocate_group",
        "target_pool_fallback_reason": "",
    }

    scored = [target_with_diagnostic_score(utterance, target) for target in pool]
    scored.sort(key=lambda row: (row.get("brief_id", ""), path_sort_key(clean_text(row.get("heading_path", "")))))
    metadata["candidate_pool_size"] = len(scored)
    return scored, metadata


def build_candidate_rows(
    config: Dict[str, Any],
    run_dir: Path,
    purpose_rows: Sequence[Dict[str, Any]],
    reason_targets: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    retained = [row for row in purpose_rows if row.get("enters_reason_alignment") is True]
    rows = []
    for utterance in retained:
        pool, pool_metadata = stage2_candidate_pool(utterance, reason_targets)
        for index, target in enumerate(pool, start=1):
            rows.append(
                {
                    **{key: utterance.get(key, "") for key in [
                        "oyez_argument_id", "case_id", "docket_number", "case_name", "utterance_id", "turn_id",
                        "target_justice", "speaker", "text", "run_scope_brief_ids",
                        "adjudicated_purpose_label", "adjudicated_purpose_labels", "final_purpose_label", "final_purpose_labels",
                        "section_counsel", "advocate_name", "advocate_description",
                        "advocate_group", "advocate_side", "advocate_group_source",
                        "local_dialogue_context", "local_dialogue_context_type", "local_dialogue_context_turn_count",
                        "local_dialogue_context_token_estimate", "local_dialogue_context_counsel", "local_dialogue_context_advocate_group",
                    ]},
                    "candidate_rank": index,
                    **pool_metadata,
                    **compact_target(target),
                }
            )
    out_dir = run_dir / "06_reason_alignment"
    write_jsonl(out_dir / "reason_candidates.jsonl", rows)
    write_csv(out_dir / "reason_candidates.csv", rows)
    return rows


def candidate_groups(candidate_rows: Sequence[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    groups: Dict[str, List[Dict[str, Any]]] = {}
    for row in candidate_rows:
        groups.setdefault(str(row.get("utterance_id", "")), []).append(row)
    for rows in groups.values():
        rows.sort(key=lambda item: int(item.get("candidate_rank") or 0))
    return groups



def stance_bundles(
    alignment_rows: Sequence[Dict[str, Any]],
    reason_targets: Optional[Sequence[Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    seen: Set[Tuple[str, str]] = set()
    for row in alignment_rows:
        if row.get("stage2_resolved") is not True and row.get("final_source") != "human_review":
            raise ValueError(f"Stage 3 cannot accept unresolved Stage 2 utterance {row.get('utterance_id')!r}.")
        bundle = row.get("final_reason_bundle") if "final_reason_bundle" in row else row.get("adjudicated_reason_bundle")
        if not isinstance(bundle, dict):
            if row.get("final_none_of_the_above") is True or row.get("adjudicated_none_of_the_above") is True:
                continue
            raise ValueError(f"Stage 3 requires a complete hierarchical reason bundle for {row.get('utterance_id')!r}.")
        terminal_ids = bundle.get("terminal_selected_reason_ids") or []
        if not terminal_ids:
            continue
        utterance_id = str(row.get("utterance_id") or "")
        if reason_targets is not None:
            expected_bundle = build_hierarchical_reason_bundle(utterance_id, terminal_ids, reason_targets)
            def canonical_bundle(value: Dict[str, Any]) -> Dict[str, Any]:
                return {
                    "bundle_id": value.get("bundle_id"),
                    "terminal_selected_reason_ids": sorted(value.get("terminal_selected_reason_ids") or []),
                    "selected_hierarchical_paths": sorted(
                        value.get("selected_hierarchical_paths") or [],
                        key=lambda path: str(path.get("terminal_reason_id") or ""),
                    ),
                }

            if canonical_bundle(bundle) != canonical_bundle(expected_bundle):
                raise ValueError(
                    f"Stage 2 to Stage 3 data-integrity failure for utterance {utterance_id!r}: "
                    "the supplied bundle is not the deterministic bundle constructed from the sealed master hierarchy."
                )
            bundle = expected_bundle
        bundle_id = str(bundle.get("bundle_id") or "")
        key = (utterance_id, bundle_id)
        if not utterance_id or not bundle_id or key in seen:
            raise ValueError("Stage 3 bundle units require unique non-empty utterance_id × bundle_id keys.")
        seen.add(key)
        rows.append(
            {
                **{key: row.get(key, "") for key in [
                    "oyez_argument_id", "case_id", "docket_number", "case_name", "utterance_id", "turn_id",
                    "target_justice", "speaker", "text",
                    "section_counsel", "advocate_name", "advocate_description", "advocate_group", "advocate_side",
                    "local_dialogue_context", "local_dialogue_context_type", "local_dialogue_context_turn_count",
                    "local_dialogue_context_token_estimate",
                ]},
                "bundle_id": bundle_id,
                "terminal_selected_reason_ids": list(terminal_ids),
                "selected_hierarchical_paths": bundle.get("selected_hierarchical_paths", []),
                "complete_reason_bundle": bundle,
                "complete_bundle_flat_text": flatten_reason_bundle(bundle),
                "reason_tree": build_selected_reason_tree(bundle),
            }
        )
    return rows


def stance_target_payload(bundle_row: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "bundle_id": bundle_row.get("bundle_id", ""),
        "terminal_selected_reason_ids": bundle_row.get("terminal_selected_reason_ids", []),
        "reason_tree": bundle_row.get("reason_tree", []),
    }


def _stage3_review_row(row: Dict[str, Any]) -> Dict[str, Any]:
    favor = row.get("adjudicated_favor_evidence") or {}
    against = row.get("adjudicated_against_evidence") or {}
    return {
        **row,
        "student_id": "",
        "llm_dialogue_context": row.get("local_dialogue_context", ""),
        "favor_utterance_evidence": favor.get("utterance_evidence_span", ""),
        "favor_target_reason_ids": favor.get("target_reason_ids", []),
        "favor_target_reason_paths": locator_paths(row.get("complete_reason_bundle", {}), favor.get("target_reason_ids", [])),
        "against_utterance_evidence": against.get("utterance_evidence_span", ""),
        "against_target_reason_ids": against.get("target_reason_ids", []),
        "against_target_reason_paths": locator_paths(row.get("complete_reason_bundle", {}), against.get("target_reason_ids", [])),
        "student_stance_outcome": "",
        "student_notes": "",
        "professor_final_stance_outcome": "",
        "professor_notes": "",
    }


def run_stance_detection(
    config: Dict[str, Any],
    run_dir: Path,
    alignment_rows: Sequence[Dict[str, Any]],
    reason_targets: Sequence[Dict[str, Any]],
    *,
    unit_complete: Optional[
        Callable[[Dict[str, Any], Sequence[Dict[str, Any]], Dict[str, Any]], None]
    ] = None,
    write_outputs: bool = True,
) -> List[Dict[str, Any]]:
    out_dir = run_dir / "07_reason_stance"
    bundles = stance_bundles(alignment_rows, reason_targets)
    if write_outputs:
        write_jsonl(out_dir / "stance_bundles.jsonl", bundles)
        write_csv(out_dir / "stance_bundles.csv", bundles)
    models = annotator_models(config)
    adjudicator = adjudicator_model(config)
    prompt_path = resolve_path("prompts/justice_stance_prompt.txt")
    adjudication_prompt = resolve_path("prompts/justice_stance_adjudication_prompt.txt")
    model_rows: List[Dict[str, Any]] = []
    consolidated: List[Dict[str, Any]] = []
    progress_enabled, progress_every = progress_settings(config)
    if progress_enabled:
        print_progress("Stage 3", "stance bundle", 0, len(bundles), progress_every, force=True)
    for index, bundle_row in enumerate(bundles, start=1):
        unit_model_start = len(model_rows)
        unit_id = str(bundle_row["bundle_id"])
        allowed_ids = bundle_reason_ids(bundle_row["complete_reason_bundle"])
        attempts: List[Dict[str, Any]] = []
        for model in models:
            attempt = call_validated_model(
                config,
                run_dir,
                model,
                prompt_path,
                {
                    "utterance": json_for_prompt(utterance_prompt_payload(bundle_row)),
                    "dialogue_context": local_dialogue_context_text(bundle_row),
                    "target": json_for_prompt(stance_target_payload(bundle_row)),
                },
                out_dir / "model_calls",
                f"{unit_id}_{model['model_id']}",
                "stage3",
                unit_id,
                "annotator",
                lambda content, text=str(bundle_row.get("text", "")), ids=allowed_ids: validate_stage3_output(content, text, ids),
            )
            attempts.append(attempt)
            model_rows.append(
                {
                    "utterance_id": bundle_row["utterance_id"],
                    "bundle_id": unit_id,
                    "annotation_model": model["model_id"],
                    "effective_output": _effective_output(attempt),
                    "effective_raw_response": _effective_raw(attempt),
                    "valid": bool(attempt["effective_validation"].get("valid")),
                    "same_reason_conflict": bool(attempt["effective_validation"].get("same_reason_conflict")),
                    "repair_attempted": attempt.get("repair_attempted", False),
                }
            )
        valid = all(attempt["effective_validation"].get("valid") for attempt in attempts)
        any_mixed = any(_effective_output(attempt).get("stance_outcome") == "Mixed" for attempt in attempts)
        any_conflict = any(attempt["effective_validation"].get("same_reason_conflict") for attempt in attempts)
        agreement = valid and not any_mixed and not any_conflict and substantive_stage3_key(
            _effective_output(attempts[0])
        ) == substantive_stage3_key(_effective_output(attempts[1]))
        adjudicator_attempt: Optional[Dict[str, Any]] = None
        if agreement:
            decision = _effective_output(attempts[0])
            model_output_valid = True
            resolved = True
            requires_human = False
            routing = "substantive_agreement"
        else:
            annotations = [
                {
                    "model": model["model_id"],
                    "valid": bool(attempt["effective_validation"].get("valid")),
                    "same_reason_conflict": bool(attempt["effective_validation"].get("same_reason_conflict")),
                    "effective_output": _effective_output(attempt),
                }
                for model, attempt in zip(models, attempts)
            ]
            findings = {
                model["model_id"]: attempt["effective_validation"].get("validation_findings", [])
                for model, attempt in zip(models, attempts)
                if not attempt["effective_validation"].get("valid") and not attempt["effective_validation"].get("same_reason_conflict")
            }
            adjudicator_attempt = call_validated_model(
                config,
                run_dir,
                adjudicator,
                adjudication_prompt,
                {
                    "utterance": json_for_prompt(utterance_prompt_payload(bundle_row)),
                    "dialogue_context": local_dialogue_context_text(bundle_row),
                    "target": json_for_prompt(stance_target_payload(bundle_row)),
                    "annotations": json_for_prompt(annotations),
                    "validation_findings": json_for_prompt(findings),
                },
                out_dir / "adjudication_calls",
                f"{unit_id}_{adjudicator['model_id']}",
                "stage3",
                unit_id,
                "adjudicator",
                lambda content, text=str(bundle_row.get("text", "")), ids=allowed_ids: validate_stage3_output(content, text, ids),
            )
            adjudication_validation = adjudicator_attempt["effective_validation"]
            if adjudication_validation.get("valid"):
                decision = _effective_output(adjudicator_attempt)
                model_output_valid = True
                if decision["stance_outcome"] == "Mixed":
                    resolved = False
                    requires_human = True
                    routing = "legal_mixed_requires_human_review"
                else:
                    resolved = True
                    requires_human = False
                    routing = "adjudicated_valid"
            else:
                decision = {"stance_outcome": None, "favor_evidence": None, "against_evidence": None}
                model_output_valid = False
                resolved = False
                requires_human = True
                routing = (
                    "adjudicator_same_reason_conflict"
                    if adjudication_validation.get("same_reason_conflict")
                    else "adjudicator_repair_failed"
                )
        consolidated_row = {
                **bundle_row,
                "annotator_a_effective_raw_response": _effective_raw(attempts[0]),
                "annotator_b_effective_raw_response": _effective_raw(attempts[1]),
                "adjudicator_effective_raw_response": _effective_raw(adjudicator_attempt) if adjudicator_attempt else "",
                "adjudicated_stance_outcome": decision.get("stance_outcome"),
                "adjudicated_favor_evidence": decision.get("favor_evidence"),
                "adjudicated_against_evidence": decision.get("against_evidence"),
                "stage3_model_output_valid": model_output_valid,
                "stage3_resolved": resolved,
                "stage3_requires_human_review": requires_human,
                "adjudication_skipped": agreement,
                "routing": routing,
            }
        consolidated.append(consolidated_row)
        _record_stage_routing(
            out_dir,
            "stage3",
            run_dir.name,
            unit_id,
            routing,
            resolved,
            requires_human,
            model_output_valid,
        )
        if unit_complete is not None:
            unit_complete(bundle_row, model_rows[unit_model_start:], consolidated_row)
        if progress_enabled:
            print_progress("Stage 3", "stance bundle", index, len(bundles), progress_every)
    if write_outputs:
        write_jsonl(out_dir / "stance_model_annotations.jsonl", model_rows)
        write_jsonl(out_dir / "stance_adjudicated.jsonl", consolidated)
        review_rows = [_stage3_review_row(row) for row in consolidated]
        write_csv(out_dir / "stance_student_review.csv", review_rows, ["student_id", *stance_review_fields()])
        write_csv(out_dir / "stance_professor_review.csv", review_rows, ["student_id", *stance_review_fields()])
    return consolidated




def question_purpose_review_fields() -> List[str]:
    return [
        "oyez_argument_id", "case_id", "docket_number", "case_name", "utterance_id", "turn_id", "target_justice",
        "speaker", "start_time", "stop_time",
        "text", "llm_dialogue_context", "annotator_a_effective_raw_response", "annotator_b_effective_raw_response",
        "adjudicator_effective_raw_response", "adjudicated_question_function_present", "adjudicated_purposes",
        "stage1_requires_human_review", "student_question_function_present", "student_purposes", "student_notes",
        "professor_final_question_function_present", "professor_final_purposes", "professor_notes",
    ]


def alignment_review_fields() -> List[str]:
    return [
        "oyez_argument_id", "case_id", "docket_number", "case_name", "utterance_id", "turn_id", "target_justice",
        "speaker", "text", "llm_dialogue_context",
        "candidate_targets", "candidate_reason_ids", "adjudicated_reason_bundle", "selected_hierarchical_paths",
        "selected_hierarchical_paths_flat_text", "adjudicated_reason_alignments", "annotator_a_effective_traversal",
        "annotator_b_effective_traversal", "adjudicator_effective_traversal", "adjudicated_none_of_the_above",
        "stage2_requires_human_review", "student_selected_reason_ids", "student_none_of_the_above", "student_notes",
        "professor_final_selected_reason_ids", "professor_none_of_the_above", "professor_notes",
    ]


def stance_review_fields() -> List[str]:
    return [
        "oyez_argument_id", "case_id", "docket_number", "case_name", "utterance_id", "turn_id", "target_justice",
        "speaker", "text", "llm_dialogue_context",
        "bundle_id", "reason_tree", "annotator_a_effective_raw_response",
        "annotator_b_effective_raw_response", "adjudicator_effective_raw_response", "adjudicated_stance_outcome",
        "favor_utterance_evidence", "favor_target_reason_ids", "favor_target_reason_paths", "against_utterance_evidence",
        "against_target_reason_ids", "against_target_reason_paths", "stage3_requires_human_review",
        "student_stance_outcome", "student_notes", "professor_final_stance_outcome", "professor_notes",
    ]


def write_student_review_sheets(
    stage_dir: Path,
    stage_name: str,
    rows: Sequence[Dict[str, Any]],
    fields: List[str],
) -> None:
    if stage_name == "stance":
        raise RuntimeError("Stage 3 writes its approved master review files directly.")
    student_rows: List[Dict[str, Any]] = []
    for row in rows:
        payload = dict(row)
        payload["student_id"] = ""
        payload["llm_dialogue_context"] = row.get("local_dialogue_context", "")
        for field in [
            "student_question_function_present", "student_purposes", "student_selected_reason_ids",
            "student_none_of_the_above", "student_notes", "professor_final_question_function_present",
            "professor_final_purposes", "professor_final_selected_reason_ids", "professor_none_of_the_above",
            "professor_notes",
        ]:
            payload.setdefault(field, "")
        student_rows.append(payload)
    student_name = "question_purpose_student_review.csv" if stage_name == "question_purpose" else "reason_alignment_student_review.csv"
    professor_name = (
        "question_purpose_professor_adjudication.csv"
        if stage_name == "question_purpose"
        else "reason_alignment_professor_adjudication.csv"
    )
    write_csv(stage_dir / student_name, student_rows, ["student_id", *fields])
    write_csv(stage_dir.parent / "08_human_review" / professor_name, student_rows, ["student_id", *fields])



def _audit_filename(stage: str) -> str:
    return {
        "stage1": "question_purpose_attempt_audit.jsonl",
        "stage2": "reason_alignment_attempt_audit.jsonl",
        "stage3": "stance_attempt_audit.jsonl",
    }[stage]


def _stage_audit_path(out_dir: Path, stage: str) -> Path:
    stage_dir = out_dir.parent if out_dir.name in {"model_calls", "adjudication_calls"} else out_dir
    return stage_dir / _audit_filename(stage)


def _forced_finish_invalid(validation: Dict[str, Any], finish_reason: Any) -> Dict[str, Any]:
    if finish_reason == "stop":
        return validation
    updated = dict(validation)
    updated["valid"] = False
    updated["same_reason_conflict"] = False
    updated["protected_fields"] = {}
    updated["repairable_fields"] = ["complete decision output"]
    findings = list(updated.get("validation_findings", []))
    findings.append(
        f"The response finish_reason was {finish_reason!r}; only finish_reason='stop' can be accepted as a complete output."
    )
    updated["validation_findings"] = findings
    return updated


def call_json_model(
    config: Dict[str, Any],
    model: Dict[str, Any],
    prompt_path: Path,
    prompt_values: Dict[str, Any],
    out_dir: Path,
    stem: str,
    *,
    run_dir: Optional[Path] = None,
    role: str = "annotator",
    call_type: str = "initial",
) -> Dict[str, Any]:
    if run_dir is None:
        raise RuntimeError("Strict Stage 1-3 model calls require the current run_dir and sealed manifest.")
    manifest = require_sealed_manifest(run_dir, config)
    identity = model_manifest_entry(manifest, model["model_id"])
    prompt = render_template(prompt_path, prompt_values)
    safe_stem = safe_name(f"{stem}_{call_type}")
    prompt_file = out_dir / "prompts" / f"{safe_stem}.txt"
    request_file = out_dir / "request_payloads" / f"{safe_stem}.json"
    raw_file = out_dir / "raw_responses" / f"{safe_stem}.json"
    state_file = out_dir / "call_state" / f"{safe_stem}.json"
    request_identity = {
        "run_id": run_dir.name,
        "model_id": model["model_id"],
        "role": role,
        "call_type": call_type,
        "prompt": prompt,
        "requested_model_slug": identity["requested_model_slug"],
        "fixed_provider_slug": identity["fixed_provider_slug"],
        "generation_parameters": identity["generation_parameters"],
        "response_format": identity["response_format"],
    }
    fingerprint = sha256_json(request_identity)
    if state_file.exists():
        state = json.loads(state_file.read_text(encoding="utf-8"))
        if state.get("request_fingerprint") != fingerprint:
            raise RuntimeError("Same-run resume refused because the stored call request does not exactly match.")
        if state.get("status") != "complete":
            raise RuntimeError("Stored call state is incomplete and cannot be silently restarted.")
        stored_result = state.get("result") or {}
        artifact_hash_fields = {
            "prompt_path": "prompt_sha256",
            "request_payload_path": "request_payload_sha256",
            "raw_response_path": "raw_response_sha256",
            "generation_metadata_path": "generation_metadata_sha256",
        }
        for field, hash_field in artifact_hash_fields.items():
            artifact = Path(str(stored_result.get(field) or ""))
            if not artifact.is_file():
                raise RuntimeError(f"Same-run resume refused because stored call artifact {field!r} is missing.")
            if not stored_result.get(hash_field) or sha256_file(artifact) != stored_result[hash_field]:
                raise RuntimeError(f"Same-run resume refused because stored call artifact {field!r} changed.")
        if Path(stored_result["prompt_path"]).read_text(encoding="utf-8") != prompt:
            raise RuntimeError("Same-run resume refused because the stored rendered prompt changed.")
        return stored_result

    prompt_file.parent.mkdir(parents=True, exist_ok=True)
    prompt_file.write_text(prompt, encoding="utf-8")
    client = chat_client_for_model(config, model)
    result = client.chat(
        identity["requested_model_slug"],
        prompt,
        raw_file,
        out_dir / "errors.log",
        provider_only=[identity["fixed_provider_slug"]],
        require_parameters=True,
        request_parameters=request_parameters(identity["generation_parameters"]),
        response_format=identity["response_format"],
        require_generation_metadata=True,
        include_default_temperature=False,
    )
    if result.get("actual_model") != identity["actual_model"]:
        error_path = out_dir / "errors.log"
        error_path.parent.mkdir(parents=True, exist_ok=True)
        with error_path.open("a", encoding="utf-8") as handle:
            handle.write("Actual model identity did not match the sealed run manifest.\n")
        raise RuntimeError(
            f"Actual model {result.get('actual_model')!r} does not match sealed identity {identity['actual_model']!r}."
        )
    if result.get("actual_provider") != identity["actual_provider_name"]:
        error_path = out_dir / "errors.log"
        error_path.parent.mkdir(parents=True, exist_ok=True)
        with error_path.open("a", encoding="utf-8") as handle:
            handle.write("Actual provider identity did not match the sealed run manifest.\n")
        raise RuntimeError(
            f"Actual provider {result.get('actual_provider')!r} does not match sealed identity "
            f"{identity['actual_provider_name']!r}."
        )
    compact_result = {
        "run_id": run_dir.name,
        "content": result.get("content", ""),
        "finish_reason": result.get("finish_reason"),
        "prompt_path": str(prompt_file),
        "raw_response_path": str(raw_file),
        "token_usage": result.get("token_usage", {}),
        "estimated_cost": result.get("estimated_cost", ""),
        "generation_id": result.get("generation_id", ""),
        "requested_model": identity["requested_model_slug"],
        "actual_model": result.get("actual_model", ""),
        "requested_provider": identity["fixed_provider_slug"],
        "actual_provider": result.get("actual_provider", ""),
        "system_fingerprint": result.get("system_fingerprint", "not_returned"),
        "generation_metadata_attempts": result.get("generation_metadata_attempts", 0),
        "generation_metadata_query_state_path": result.get("generation_metadata_query_state_path", ""),
        "generation_metadata_query_attempt_paths": result.get("generation_metadata_query_attempt_paths", []),
        "http_attempt_count": result.get("http_attempt_count", 0),
        "http_request_state_path": result.get("http_request_state_path", ""),
        "http_attempt_paths": result.get("http_attempt_paths", []),
        "request_parameters": {
            key: value
            for key, value in (result.get("request_payload") or {}).items()
            if key != "messages"
        },
        "response_headers": result.get("response_headers", {}),
        "actual_choice_count": result.get("actual_choice_count", 0),
    }
    request_file.parent.mkdir(parents=True, exist_ok=True)
    request_file.write_text(
        json.dumps(result.get("request_payload", {}), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    compact_result["request_payload_path"] = str(request_file)
    metadata_file = out_dir / "generation_metadata" / f"{safe_stem}.json"
    metadata_file.parent.mkdir(parents=True, exist_ok=True)
    metadata_file.write_text(
        json.dumps(result.get("generation_metadata", {}), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    compact_result["generation_metadata_path"] = str(metadata_file)
    compact_result["prompt_sha256"] = sha256_file(prompt_file)
    compact_result["request_payload_sha256"] = sha256_file(request_file)
    compact_result["raw_response_sha256"] = sha256_file(raw_file)
    compact_result["generation_metadata_sha256"] = sha256_file(metadata_file)
    state_file.parent.mkdir(parents=True, exist_ok=True)
    state_file.write_text(
        json.dumps(
            {"status": "complete", "request_fingerprint": fingerprint, "request_identity": request_identity, "result": compact_result},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return compact_result


def _record_validated_attempt(
    out_dir: Path,
    stage: str,
    unit_id: str,
    model: Dict[str, Any],
    role: str,
    call_type: str,
    attempt_number: int,
    result: Dict[str, Any],
    validation: Dict[str, Any],
    routing: str,
    branch_id: str = "",
) -> None:
    append_attempt_audit(
        _stage_audit_path(out_dir, stage),
        {
            "audit_event_id": "|".join([
                stage, unit_id, role, str(model["model_id"]), call_type, str(attempt_number), branch_id or "-"
            ]),
            "run_id": result.get("run_id", ""),
            "stage": stage,
            "annotation_unit_id": unit_id,
            "branch_reason_target_id": branch_id,
            "model_role": role,
            "model_id": model["model_id"],
            "call_type": call_type,
            "attempt_number": attempt_number,
            "prompt_path": result.get("prompt_path", ""),
            "prompt_sha256": result.get("prompt_sha256", ""),
            "request_payload_path": result.get("request_payload_path", ""),
            "request_payload_sha256": result.get("request_payload_sha256", ""),
            "raw_response_path": result.get("raw_response_path", ""),
            "raw_response_sha256": result.get("raw_response_sha256", ""),
            "request_parameters": result.get("request_parameters", {}),
            "response_headers": result.get("response_headers", {}),
            "actual_choice_count": result.get("actual_choice_count", 0),
            "generation_id": result.get("generation_id", ""),
            "requested_model": result.get("requested_model", ""),
            "actual_model": result.get("actual_model", ""),
            "requested_provider": result.get("requested_provider", ""),
            "actual_provider": result.get("actual_provider", ""),
            "system_fingerprint": result.get("system_fingerprint", "not_returned"),
            "finish_reason": result.get("finish_reason"),
            "generation_metadata_attempts": result.get("generation_metadata_attempts", 0),
            "generation_metadata_path": result.get("generation_metadata_path", ""),
            "generation_metadata_sha256": result.get("generation_metadata_sha256", ""),
            "generation_metadata_query_state_path": result.get("generation_metadata_query_state_path", ""),
            "generation_metadata_query_attempt_paths": result.get("generation_metadata_query_attempt_paths", []),
            "http_attempt_count": result.get("http_attempt_count", 0),
            "http_request_state_path": result.get("http_request_state_path", ""),
            "http_attempt_paths": result.get("http_attempt_paths", []),
            "valid": validation.get("valid", False),
            "normalized_output": validation.get("normalized"),
            "same_reason_conflict": validation.get("same_reason_conflict", False),
            "validation_findings": validation.get("validation_findings", []),
            "protected_fields": validation.get("protected_fields", {}),
            "repairable_fields": validation.get("repairable_fields", []),
            "routing": routing,
        },
    )


def _record_stage_routing(
    out_dir: Path,
    stage: str,
    run_id: str,
    unit_id: str,
    routing: str,
    resolved: bool,
    requires_human_review: bool,
    model_output_valid: Optional[bool] = None,
) -> None:
    row: Dict[str, Any] = {
        "audit_event_id": "|".join([stage, unit_id, "program", "routing_summary"]),
        "run_id": run_id,
        "stage": stage,
        "annotation_unit_id": unit_id,
        "model_role": "program",
        "call_type": "routing_summary",
        "routing": routing,
        "resolved": resolved,
        "requires_human_review": requires_human_review,
        "final_effective_attempt": True,
    }
    if model_output_valid is not None:
        row["model_output_valid"] = model_output_valid
    append_attempt_audit(_stage_audit_path(out_dir, stage), row)


def call_validated_model(
    config: Dict[str, Any],
    run_dir: Path,
    model: Dict[str, Any],
    prompt_path: Path,
    prompt_values: Dict[str, Any],
    out_dir: Path,
    stem: str,
    stage: str,
    unit_id: str,
    role: str,
    validator: Callable[[Any], Dict[str, Any]],
    branch_id: str = "",
) -> Dict[str, Any]:
    initial = call_json_model(
        config,
        model,
        prompt_path,
        prompt_values,
        out_dir,
        stem,
        run_dir=run_dir,
        role=role,
        call_type="initial",
    )
    validation = _forced_finish_invalid(validator(initial.get("content", "")), initial.get("finish_reason"))
    if validation.get("valid"):
        _record_validated_attempt(
            out_dir, stage, unit_id, model, role, "initial", 1, initial, validation, "effective_valid", branch_id
        )
        return {"effective_result": initial, "effective_validation": validation, "repair_attempted": False}
    if validation.get("same_reason_conflict"):
        _record_validated_attempt(
            out_dir, stage, unit_id, model, role, "initial", 1, initial, validation, "same_reason_adjudication", branch_id
        )
        return {"effective_result": initial, "effective_validation": validation, "repair_attempted": False}

    _record_validated_attempt(
        out_dir, stage, unit_id, model, role, "initial", 1, initial, validation, "repair_required", branch_id
    )
    repair_prompt = resolve_path("prompts/justice_structured_repair_prompt.txt")
    repair_values = {
        "original_task": render_template(prompt_path, prompt_values),
        "last_output": initial.get("content", ""),
        "validation_findings": json_for_prompt(validation.get("validation_findings", [])),
        "protected_fields": json_for_prompt(validation.get("protected_fields", {})),
        "repairable_fields": json_for_prompt(validation.get("repairable_fields", [])),
    }
    repaired = call_json_model(
        config,
        model,
        repair_prompt,
        repair_values,
        out_dir,
        stem,
        run_dir=run_dir,
        role=role,
        call_type="repair",
    )
    repaired_validation = _forced_finish_invalid(validator(repaired.get("content", "")), repaired.get("finish_reason"))
    repaired_normalized = repaired_validation.get("normalized") or {}
    if repaired_validation.get("valid") and not protected_fields_unchanged(
        validation.get("protected_fields", {}), repaired_normalized
    ):
        repaired_validation = dict(repaired_validation)
        repaired_validation["valid"] = False
        repaired_validation["validation_findings"] = list(repaired_validation.get("validation_findings", [])) + [
            "The repair changed one or more protected substantive fields from the preceding output."
        ]
    routing = "effective_valid" if repaired_validation.get("valid") else "repair_failed"
    _record_validated_attempt(
        out_dir, stage, unit_id, model, role, "repair", 2, repaired, repaired_validation, routing, branch_id
    )
    return {"effective_result": repaired, "effective_validation": repaired_validation, "repair_attempted": True}


def _effective_output(attempt: Dict[str, Any]) -> Dict[str, Any]:
    return attempt.get("effective_validation", {}).get("normalized") or attempt.get("effective_validation", {}).get("parsed") or {}


def _effective_raw(attempt: Dict[str, Any]) -> str:
    return str(attempt.get("effective_result", {}).get("content", ""))


def run_question_purpose(
    config: Dict[str, Any],
    run_dir: Path,
    utterances: Sequence[Dict[str, Any]],
    *,
    unit_complete: Optional[
        Callable[[Dict[str, Any], Sequence[Dict[str, Any]], Dict[str, Any]], None]
    ] = None,
    write_outputs: bool = True,
) -> List[Dict[str, Any]]:
    out_dir = run_dir / "05_question_purpose"
    models = annotator_models(config)
    if models[0]["model_name"] == models[1]["model_name"]:
        raise ValueError("The two Stage 1-3 annotators must use different exact models.")
    adjudicator = adjudicator_model(config)
    prompt_path = resolve_path("prompts/justice_question_purpose_prompt.txt")
    adjudication_prompt = resolve_path("prompts/justice_question_purpose_adjudication_prompt.txt")
    model_rows: List[Dict[str, Any]] = []
    consolidated: List[Dict[str, Any]] = []
    progress_enabled, progress_every = progress_settings(config)
    if progress_enabled:
        print_progress("Stage 1", "justice utterance", 0, len(utterances), progress_every, force=True)
    for index, utterance in enumerate(utterances, start=1):
        unit_model_start = len(model_rows)
        utterance_id = str(utterance["utterance_id"])
        text = str(utterance.get("text", ""))
        attempts: List[Dict[str, Any]] = []
        for model in models:
            attempt = call_validated_model(
                config,
                run_dir,
                model,
                prompt_path,
                {
                    "utterance": json_for_prompt(utterance_prompt_payload(utterance)),
                    "dialogue_context": local_dialogue_context_text(utterance),
                },
                out_dir / "model_calls",
                f"{utterance_id}_{model['model_id']}",
                "stage1",
                utterance_id,
                "annotator",
                lambda content, utterance_text=text: validate_stage1_output(content, utterance_text),
            )
            attempts.append(attempt)
            model_rows.append(
                {
                    **utterance,
                    "annotation_model": model["model_id"],
                    "effective_output": _effective_output(attempt),
                    "effective_raw_response": _effective_raw(attempt),
                    "valid": bool(attempt["effective_validation"].get("valid")),
                    "repair_attempted": attempt.get("repair_attempted", False),
                }
            )
        both_valid = all(item["effective_validation"].get("valid") for item in attempts)
        agreement = both_valid and substantive_stage1_key(_effective_output(attempts[0])) == substantive_stage1_key(
            _effective_output(attempts[1])
        )
        adjudicator_attempt: Optional[Dict[str, Any]] = None
        if agreement:
            common = _effective_output(attempts[0])
            purposes = sorted(common["purposes"], key=lambda item: PURPOSE_LABEL_ORDER.index(item["label"]))
            gate: Optional[bool] = common["question_function_present"]
            resolved = True
            requires_human = False
            adjudication_skipped = True
            adjudication_route = "substantive_agreement"
            adjudicated_aux: Dict[str, Any] = {}
        else:
            annotations = [
                {
                    "model": model["model_id"],
                    "valid": bool(attempt["effective_validation"].get("valid")),
                    "effective_output": _effective_output(attempt),
                }
                for model, attempt in zip(models, attempts)
            ]
            findings = {
                model["model_id"]: attempt["effective_validation"].get("validation_findings", [])
                for model, attempt in zip(models, attempts)
                if not attempt["effective_validation"].get("valid")
            }
            adjudicator_attempt = call_validated_model(
                config,
                run_dir,
                adjudicator,
                adjudication_prompt,
                {
                    "utterance": json_for_prompt(utterance_prompt_payload(utterance)),
                    "dialogue_context": local_dialogue_context_text(utterance),
                    "annotations": json_for_prompt(annotations),
                    "validation_findings": json_for_prompt(findings),
                },
                out_dir / "adjudication_calls",
                f"{utterance_id}_{adjudicator['model_id']}",
                "stage1",
                utterance_id,
                "adjudicator",
                lambda content, utterance_text=text: validate_stage1_output(content, utterance_text, adjudicator=True),
            )
            adjudication_skipped = False
            if adjudicator_attempt["effective_validation"].get("valid"):
                decision = _effective_output(adjudicator_attempt)
                gate = decision["question_function_present"]
                purposes = sorted(decision["purposes"], key=lambda item: PURPOSE_LABEL_ORDER.index(item["label"]))
                adjudicated_aux = {
                    f"adjudicated_{key}": decision[key]
                    for key in ["rationale", "confidence", "uncertainty_notes"]
                    if key in decision
                }
                resolved = True
                requires_human = False
                adjudication_route = "adjudicated_valid"
            else:
                gate = None
                purposes = []
                adjudicated_aux = {}
                resolved = False
                requires_human = True
                adjudication_route = "adjudicator_repair_failed"
        labels = [item["label"] for item in purposes]
        consolidated_row = {
                **utterance,
                "annotator_models": [model["model_id"] for model in models],
                "annotator_a_effective_raw_response": _effective_raw(attempts[0]),
                "annotator_b_effective_raw_response": _effective_raw(attempts[1]),
                "adjudicator_model": adjudicator["model_id"],
                "adjudicator_effective_raw_response": _effective_raw(adjudicator_attempt) if adjudicator_attempt else "",
                "adjudicated_question_function_present": gate,
                "adjudicated_purposes": purposes,
                "adjudicated_purpose_labels": labels,
                **adjudicated_aux,
                "stage1_resolved": resolved,
                "stage1_requires_human_review": requires_human,
                "enters_reason_alignment": resolved and any(label in RETAINED_PURPOSE_LABELS for label in labels),
                "adjudication_skipped": adjudication_skipped,
                "routing": adjudication_route,
            }
        consolidated.append(consolidated_row)
        _record_stage_routing(
            out_dir,
            "stage1",
            run_dir.name,
            utterance_id,
            adjudication_route,
            resolved,
            requires_human,
        )
        if unit_complete is not None:
            unit_complete(utterance, model_rows[unit_model_start:], consolidated_row)
        if progress_enabled:
            print_progress("Stage 1", "justice utterance", index, len(utterances), progress_every)
    if write_outputs:
        write_jsonl(out_dir / "question_purpose_model_annotations.jsonl", model_rows)
        write_csv(out_dir / "question_purpose_model_annotations.csv", model_rows)
        write_jsonl(out_dir / "question_purpose_adjudicated.jsonl", consolidated)
        write_csv(out_dir / "question_purpose_adjudicated.csv", consolidated)
        write_student_review_sheets(out_dir, "question_purpose", consolidated, question_purpose_review_fields())
    return consolidated


def compact_target(target: Dict[str, Any]) -> Dict[str, Any]:
    fields = [
        "reason_target_id",
        "argument_unit_id",
        "brief_id",
        "brief_group",
        "side",
        "heading_path",
        "heading_level",
        "heading_text",
        "parent_reason_target_id",
        "direct_child_reason_target_ids",
        "reason_path_text",
        "lexical_score",
    ]
    return {field: target.get(field, "" if field != "direct_child_reason_target_ids" else []) for field in fields}


def alignment_target_prompt_payload(target: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "reason_target_id": target.get("reason_target_id", ""),
        "heading_path": target.get("heading_path", ""),
        "heading_text": target.get("heading_text", ""),
    }


def _root_key(root: Dict[str, Any]) -> Tuple[str, str]:
    return str(root.get("brief_id", "")), str(root.get("reason_target_id", ""))


def _stage2_roots(targets: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows = [row for row in targets if not str(row.get("parent_reason_target_id") or "")]
    rows.sort(key=lambda row: (str(row.get("brief_id", "")), path_sort_key(str(row.get("heading_path", "")))))
    return rows


def _annotator_node_context(
    annotator_results: Sequence[Dict[str, Any]], current_reason_id: str
) -> Dict[str, Any]:
    context: Dict[str, Any] = {"annotators": []}
    for result in annotator_results:
        branch = next(
            (item for item in result.get("traversal", []) if item.get("reason_target_id") == current_reason_id),
            None,
        )
        context["annotators"].append(
            {
                "model": result.get("model", ""),
                "overall_valid": result.get("valid", False),
                "terminal_reason_alignments": result.get("terminal_reason_alignments", []),
                "current_node_decision": (
                    {
                        key: branch.get(key)
                        for key in [
                            "reason_target_id", "heading_path", "current_node_relevant", "utterance_evidence_span",
                            "rationale", "valid", "validation_findings",
                        ]
                    }
                    if branch is not None
                    else "not_visited"
                ),
            }
        )
    return context


def _traverse_stage2_root(
    config: Dict[str, Any],
    run_dir: Path,
    out_dir: Path,
    utterance: Dict[str, Any],
    model: Dict[str, Any],
    root: Dict[str, Any],
    by_id: Dict[str, Dict[str, Any]],
    *,
    role: str,
    annotator_results: Sequence[Dict[str, Any]] = (),
) -> Dict[str, Any]:
    traversal: List[Dict[str, Any]] = []
    terminals: List[Dict[str, Any]] = []
    root_identity = _root_key(root)
    prompt_path = resolve_path(
        "prompts/justice_alignment_adjudication_prompt.txt"
        if role == "adjudicator"
        else "prompts/justice_alignment_prompt.txt"
    )

    def visit(current: Dict[str, Any], ancestors: List[Dict[str, Any]]) -> Tuple[bool, bool]:
        reason_id = str(current["reason_target_id"])
        values = {
            "utterance": json_for_prompt(utterance_prompt_payload(utterance)),
            "dialogue_context": local_dialogue_context_text(utterance),
            "ancestor_path": json_for_prompt([alignment_target_prompt_payload(node) for node in ancestors]),
            "current_node": json_for_prompt(alignment_target_prompt_payload(current)),
            "annotator_context": json_for_prompt(_annotator_node_context(annotator_results, reason_id)) if role == "adjudicator" else "",
        }
        attempt = call_validated_model(
            config,
            run_dir,
            model,
            prompt_path,
            values,
            out_dir,
            f"{utterance['utterance_id']}_{root_identity[0]}_{root_identity[1]}_{reason_id}_{model['model_id']}",
            "stage2",
            str(utterance["utterance_id"]),
            role,
            lambda content: validate_stage2_output(content, str(utterance.get("text", ""))),
            branch_id=reason_id,
        )
        validation = attempt["effective_validation"]
        output = _effective_output(attempt)
        row = {
            "reason_target_id": reason_id,
            "heading_path": current.get("heading_path", ""),
            "current_node_relevant": output.get("current_node_relevant") if validation.get("valid") else None,
            "utterance_evidence_span": output.get("alignment_evidence_span", "") if validation.get("valid") else "",
            "rationale": output.get("rationale") if "rationale" in output else None,
            "valid": bool(validation.get("valid")),
            "effective_raw_response": _effective_raw(attempt),
            "validation_findings": validation.get("validation_findings", []),
        }
        traversal.append(row)
        if not validation.get("valid"):
            return False, False
        if output["current_node_relevant"] is False:
            return True, False
        child_ids = current.get("direct_child_reason_target_ids") or []
        children: List[Dict[str, Any]] = []
        for child_id in child_ids:
            child = by_id.get(str(child_id))
            if child is None:
                raise ValueError(
                    f"Reason-tree data-integrity failure: relevant parent {reason_id!r} lists missing child {child_id!r}."
                )
            children.append(child)
        children.sort(key=lambda row: path_sort_key(str(row.get("heading_path", ""))))
        any_relevant_child = False
        for child in children:
            child_valid, child_relevant = visit(child, [*ancestors, current])
            if not child_valid:
                return False, True
            any_relevant_child = any_relevant_child or child_relevant
        if not any_relevant_child:
            terminals.append(
                {
                    "terminal_reason_id": reason_id,
                    "utterance_evidence_span": output["alignment_evidence_span"],
                }
            )
        return True, True

    valid, _ = visit(root, [])
    return {
        "root_identity": list(root_identity),
        "valid": valid,
        "traversal": traversal,
        "terminal_reason_alignments": terminals if valid else [],
    }


def _traverse_stage2_model(
    config: Dict[str, Any],
    run_dir: Path,
    out_dir: Path,
    utterance: Dict[str, Any],
    model: Dict[str, Any],
    targets: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    by_id = validate_reason_tree(targets)
    roots: List[Dict[str, Any]] = []
    for root in _stage2_roots(targets):
        roots.append(
            _traverse_stage2_root(config, run_dir, out_dir, utterance, model, root, by_id, role="annotator")
        )
    traversal = [item for root in roots for item in root["traversal"]]
    terminals = [item for root in roots if root["valid"] for item in root["terminal_reason_alignments"]]
    return {
        "model": model["model_id"],
        "valid": all(root["valid"] for root in roots),
        "roots": roots,
        "traversal": traversal,
        "terminal_reason_alignments": terminals,
    }


def _roots_by_identity(result: Dict[str, Any]) -> Dict[Tuple[str, str], Dict[str, Any]]:
    return {tuple(root["root_identity"]): root for root in result.get("roots", [])}


def _public_stage2_traversal(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    fields = ["reason_target_id", "heading_path", "current_node_relevant", "utterance_evidence_span", "rationale"]
    return [{field: row.get(field) for field in fields} for row in rows]


def run_reason_alignment(
    config: Dict[str, Any],
    run_dir: Path,
    purpose_rows: Sequence[Dict[str, Any]],
    candidate_rows: Sequence[Dict[str, Any]],
    *,
    unit_complete: Optional[
        Callable[[Dict[str, Any], Sequence[Dict[str, Any]], Dict[str, Any]], None]
    ] = None,
    write_outputs: bool = True,
) -> List[Dict[str, Any]]:
    out_dir = run_dir / "06_reason_alignment"
    models = annotator_models(config)
    adjudicator = adjudicator_model(config)
    purpose_by_id = {str(row["utterance_id"]): row for row in purpose_rows}
    model_rows: List[Dict[str, Any]] = []
    consolidated: List[Dict[str, Any]] = []
    groups = candidate_groups(candidate_rows)
    progress_enabled, progress_every = progress_settings(config)
    if progress_enabled:
        print_progress("Stage 2", "justice utterance", 0, len(groups), progress_every, force=True)
    for index, (utterance_id, candidates) in enumerate(groups.items(), start=1):
        utterance = purpose_by_id[utterance_id]
        unit_model_start = len(model_rows)
        candidate_targets = [compact_target(row) for row in candidates]
        validate_reason_tree(candidate_targets)
        annotator_results = [
            _traverse_stage2_model(
                config,
                run_dir,
                out_dir / "model_calls",
                utterance,
                model,
                candidate_targets,
            )
            for model in models
        ]
        for result in annotator_results:
            model_rows.append(
                {
                    **utterance,
                    "annotation_model": result["model"],
                    "valid": result["valid"],
                    "effective_traversal": _public_stage2_traversal(result["traversal"]),
                    "terminal_reason_alignments": result["terminal_reason_alignments"],
                }
            )
        left_roots = _roots_by_identity(annotator_results[0])
        right_roots = _roots_by_identity(annotator_results[1])
        identities = sorted(set(left_roots) | set(right_roots))
        common_roots: Dict[Tuple[str, str], Dict[str, Any]] = {}
        affected: List[Tuple[str, str]] = []
        for identity in identities:
            left = left_roots.get(identity)
            right = right_roots.get(identity)
            if (
                left
                and right
                and left["valid"]
                and right["valid"]
                and substantive_stage2_key(left["terminal_reason_alignments"])
                == substantive_stage2_key(right["terminal_reason_alignments"])
            ):
                common_roots[identity] = left
            else:
                affected.append(identity)
        adjudicator_roots: Dict[Tuple[str, str], Dict[str, Any]] = {}
        adjudicator_failed = False
        if affected:
            by_id = validate_reason_tree(candidate_targets)
            root_lookup = {_root_key(root): root for root in _stage2_roots(candidate_targets)}
            for identity in affected:
                root = root_lookup.get(identity)
                if root is None:
                    raise ValueError(f"Reason-tree data-integrity failure: affected root {identity!r} is missing.")
                result = _traverse_stage2_root(
                    config,
                    run_dir,
                    out_dir / "adjudication_calls",
                    utterance,
                    adjudicator,
                    root,
                    by_id,
                    role="adjudicator",
                    annotator_results=annotator_results,
                )
                adjudicator_roots[identity] = result
                if not result["valid"]:
                    adjudicator_failed = True
                    break
        if adjudicator_failed:
            resolved = False
            requires_human = True
            alignments: List[Dict[str, Any]] = []
            bundle: Optional[Dict[str, Any]] = None
            none = None
            routing = "adjudicator_repair_failed"
        else:
            final_roots = {**common_roots, **adjudicator_roots}
            alignments = [
                alignment
                for identity in identities
                for alignment in final_roots[identity]["terminal_reason_alignments"]
            ]
            terminal_ids = [item["terminal_reason_id"] for item in alignments]
            bundle = build_hierarchical_reason_bundle(utterance_id, terminal_ids, candidate_targets)
            none = not terminal_ids
            resolved = True
            requires_human = False
            routing = "adjudicated_affected_roots" if affected else "substantive_agreement"
        consolidated_row = {
                **utterance,
                "candidate_targets": candidate_targets,
                "candidate_reason_ids": [row["reason_target_id"] for row in candidate_targets],
                "annotator_a_effective_traversal": _public_stage2_traversal(annotator_results[0]["traversal"]),
                "annotator_b_effective_traversal": _public_stage2_traversal(annotator_results[1]["traversal"]),
                "adjudicator_effective_traversal": _public_stage2_traversal([
                    item for root in adjudicator_roots.values() for item in root.get("traversal", [])
                ]),
                "adjudicated_none_of_the_above": none,
                "adjudicated_reason_alignments": alignments,
                "adjudicated_reason_bundle": bundle,
                "selected_hierarchical_paths": bundle.get("selected_hierarchical_paths", []) if bundle else [],
                "selected_hierarchical_paths_flat_text": flatten_reason_bundle(bundle) if bundle else "",
                "stage2_resolved": resolved,
                "stage2_requires_human_review": requires_human,
                "adjudication_skipped": not affected,
                "routing": routing,
            }
        consolidated.append(consolidated_row)
        _record_stage_routing(
            out_dir,
            "stage2",
            run_dir.name,
            utterance_id,
            routing,
            resolved,
            requires_human,
        )
        if unit_complete is not None:
            unit_complete(utterance, model_rows[unit_model_start:], consolidated_row)
        if progress_enabled:
            print_progress("Stage 2", "justice utterance", index, len(groups), progress_every)
    if write_outputs:
        write_jsonl(out_dir / "reason_alignment_model_annotations.jsonl", model_rows)
        write_csv(out_dir / "reason_alignment_model_annotations.csv", model_rows)
        write_jsonl(out_dir / "reason_alignment_adjudicated.jsonl", consolidated)
        write_csv(out_dir / "reason_alignment_adjudicated.csv", consolidated)
        write_student_review_sheets(out_dir, "reason_alignment", consolidated, alignment_review_fields())
    return consolidated




def _load_json_object(path: Path, label: str) -> Dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must be one JSON object.")
    return payload



def _reason_nodes_for_manifest(manifest: Dict[str, Any]) -> List[Dict[str, Any]]:
    from .justice_snapshots import BRIEF_SNAPSHOT_ROOT, verify_brief_snapshot

    snapshot_id = str(manifest.get("brief_snapshot_id") or "")
    snapshot = verify_brief_snapshot(snapshot_id)
    by_brief = {str(item["brief_id"]): item for item in snapshot["briefs"]}
    selected = sorted({
        str(brief_id)
        for item in (manifest.get("run_scope") or {}).get("oral_argument_brief_scopes") or []
        for brief_id in item.get("brief_ids") or []
    })
    rows: List[Dict[str, Any]] = []
    for brief_id in selected:
        entry = by_brief.get(brief_id)
        if entry is None:
            raise ValueError(f"Run scope references a brief absent from the sealed snapshot: {brief_id!r}.")
        for row in read_jsonl(BRIEF_SNAPSHOT_ROOT / snapshot_id / "briefs" / brief_id / "reason_tree.jsonl"):
            rows.append({**row, "brief_group": entry["brief_group"]})
    validate_reason_tree(rows)
    return rows


def _stage2_dependency_identity(manifest: Dict[str, Any]) -> Dict[str, Any]:
    from .justice_snapshots import verify_brief_snapshot

    snapshot = verify_brief_snapshot(str(manifest.get("brief_snapshot_id") or ""))
    entries = {str(item["brief_id"]): item for item in snapshot["briefs"]}
    selected = sorted({
        str(brief_id)
        for item in (manifest.get("run_scope") or {}).get("oral_argument_brief_scopes") or []
        for brief_id in item.get("brief_ids") or []
    })
    return {
        "run_scope_mapping": (manifest.get("run_scope") or {}).get("oral_argument_brief_scopes") or [],
        "briefs": {
            brief_id: {
                "pdf_sha256": entries[brief_id]["pdf_sha256"],
                "reason_tree_sha256": entries[brief_id]["reason_tree_sha256"],
            }
            for brief_id in selected
        },
    }


def _prepare_new_run_data(config: Dict[str, Any], run_dir: Path, manifest: Dict[str, Any]) -> List[Dict[str, Any]]:
    from .justice_inputs import write_run_inputs

    scope = manifest["run_scope"]
    (run_dir / "run_scope.json").write_text(json.dumps(scope, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    transcript, base_utterances = write_run_inputs(
        run_dir,
        snapshot_id=str(manifest["utterance_snapshot_id"]),
        scope=scope,
        config=config,
    )
    local_contexts = build_local_dialogue_contexts(transcript, base_utterances, config, run_dir)
    utterances = [
        {**utterance, **local_context}
        for utterance, local_context in zip(base_utterances, local_contexts)
    ]
    ids = [str(row.get("utterance_id") or "") for row in utterances]
    if any(not value for value in ids) or len(ids) != len(set(ids)):
        raise ValueError("Run-derived utterance IDs must be non-empty and globally unique.")
    write_jsonl(run_dir / "04_justice_utterances" / "justice_utterances.jsonl", utterances)
    write_csv(run_dir / "04_justice_utterances" / "justice_utterances.csv", utterances)
    (run_dir / "04_justice_utterances" / "selected_utterance_ids.json").write_text(
        json.dumps(
            {"ordered_utterance_ids": ids, "ordered_utterance_ids_sha256": sha256_json(ids)},
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return utterances


def _checkpoint_clean_rows(checkpoint_id: str) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    from .justice_checkpoint_common import load_human_final_checkpoint

    directory, manifest = load_human_final_checkpoint(checkpoint_id)
    return manifest, read_jsonl(directory / str(manifest.get("clean_final_relative_path") or "clean_final.jsonl"))


def _copy_checkpoint_clean_final(checkpoint_id: str, target: Path) -> None:
    from .justice_checkpoint_common import load_human_final_checkpoint

    directory, manifest = load_human_final_checkpoint(checkpoint_id)
    source = directory / str(manifest.get("clean_final_relative_path") or "clean_final.jsonl")
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)


def create_justice_run(
    config: Dict[str, Any], brief_snapshot_id: str, utterance_snapshot_id: str,
    scope: Dict[str, Any], profile_name: str, upstream_checkpoint_id: str = "",
) -> Path:
    from .justice_checkpoint_stage1 import consume_stage1_checkpoint
    from .justice_checkpoint_stage2 import consume_stage2_checkpoint, required_stage2_utterances
    from .justice_inputs import (
        derive_processed_transcript,
        extract_target_utterances,
        select_scoped_utterances,
        target_justices_for_scope,
    )
    from .justice_lifecycle import create_run, normalize_run_scope, read_run_manifest, write_run_state
    from .justice_snapshots import verify_brief_snapshot, verify_utterance_snapshot

    config, resolved_model_roles = configure_model_profile(config, profile_name)
    brief_snapshot = verify_brief_snapshot(brief_snapshot_id)
    oral_snapshot = verify_utterance_snapshot(utterance_snapshot_id)
    scope = normalize_run_scope(scope, oral_snapshot, brief_snapshot)
    transcript = derive_processed_transcript(utterance_snapshot_id, scope)
    utterances = extract_target_utterances(
        transcript, target_justices=target_justices_for_scope(scope)
    )
    utterances = select_scoped_utterances(utterances, scope)
    if not utterances:
        raise ValueError("The approved run scope selects no non-empty Justice utterances.")
    scope_map = {
        str(item.get("oyez_argument_id")): list(item.get("brief_ids") or [])
        for item in scope.get("oral_argument_brief_scopes") or []
    }
    utterances = [{**row, "run_scope_brief_ids": scope_map.get(str(row.get("oyez_argument_id")), [])} for row in utterances]
    required_arguments = {str(row.get("oyez_argument_id") or "") for row in utterances}
    if not required_arguments <= set(scope_map):
        missing = sorted(required_arguments - set(scope_map))
        raise ValueError(
            "Every selected oral argument with target Justice utterances requires one run-scope mapping; "
            f"missing mappings for {missing!r}."
        )
    start_stage = "stage1"
    upstream_rows: List[Dict[str, Any]] = []
    upstream_manifest: Optional[Dict[str, Any]] = None
    if upstream_checkpoint_id:
        checkpoint_manifest, checkpoint_clean = _checkpoint_clean_rows(upstream_checkpoint_id)
        upstream_manifest = checkpoint_manifest
        checkpoint_stage = checkpoint_manifest.get("stage")
        if checkpoint_stage == "stage1":
            upstream_rows = consume_stage1_checkpoint(upstream_checkpoint_id, utterances)
            start_stage = "stage2"
        elif checkpoint_stage == "stage2":
            required = required_stage2_utterances(upstream_checkpoint_id, utterances)
            entries = {str(item["brief_id"]): item for item in brief_snapshot["briefs"]}
            selected_briefs = sorted({brief for item in scope["oral_argument_brief_scopes"] for brief in item["brief_ids"]})
            dependency = {
                "run_scope_mapping": scope["oral_argument_brief_scopes"],
                "briefs": {
                    brief: {"pdf_sha256": entries[brief]["pdf_sha256"], "reason_tree_sha256": entries[brief]["reason_tree_sha256"]}
                    for brief in selected_briefs
                },
            }
            from .justice_snapshots import brief_snapshot_nodes
            nodes = [
                {**row, "brief_group": entries[row["brief_id"]]["brief_group"]}
                for row in brief_snapshot_nodes(brief_snapshot_id, selected_briefs)
            ]
            upstream_rows = consume_stage2_checkpoint(upstream_checkpoint_id, required, nodes, dependency)
            start_stage = "stage3"
        else:
            raise ValueError("A Stage 3 human-final checkpoint cannot start a new annotation run.")
    run_dir = create_run(
        model_profile=profile_name,
        resolved_model_roles=resolved_model_roles,
        brief_snapshot_id=brief_snapshot_id,
        utterance_snapshot_id=utterance_snapshot_id,
        scope=scope,
        upstream_human_final_checkpoint_id=upstream_checkpoint_id,
        upstream_human_final_checkpoint_hash=(
            str(upstream_manifest.get("overall_checkpoint_sha256") or "") if upstream_manifest else ""
        ),
        upstream_human_final_checkpoint_stage=(
            str(upstream_manifest.get("stage") or "") if upstream_manifest else ""
        ),
        start_stage=start_stage,
    )
    try:
        manifest = read_run_manifest(run_dir)
        actual_utterances = _prepare_new_run_data(config, run_dir, manifest)
        if upstream_checkpoint_id:
            if start_stage == "stage2":
                upstream_rows = consume_stage1_checkpoint(upstream_checkpoint_id, actual_utterances)
                _copy_checkpoint_clean_final(
                    upstream_checkpoint_id,
                    run_dir / "05_question_purpose" / "question_purpose_final.jsonl",
                )
            else:
                required = required_stage2_utterances(upstream_checkpoint_id, actual_utterances)
                upstream_rows = consume_stage2_checkpoint(
                    upstream_checkpoint_id, required, _reason_nodes_for_manifest(manifest), _stage2_dependency_identity(manifest)
                )
                _copy_checkpoint_clean_final(
                    upstream_checkpoint_id,
                    run_dir / "06_reason_alignment" / "reason_alignment_final.jsonl",
                )
            records = {
                str(upstream_manifest["stage"]): {
                    "checkpoint_id": upstream_checkpoint_id,
                    "checkpoint_hash": upstream_manifest["overall_checkpoint_sha256"],
                    "review_checkpoint_id": upstream_manifest["review_checkpoint_id"],
                }
            }
            (run_dir / "human_final_checkpoints.json").write_text(
                json.dumps(records, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8"
            )
        run_identity_preflight(config, run_dir)
        write_run_state(run_dir, "awaiting_model_identity_confirmation", "preflight")
        return run_dir
    except Exception as exc:
        from .openrouter_client import BYOKRequiredError, RecoverableOpenRouterError

        if isinstance(exc, (BYOKRequiredError, RecoverableOpenRouterError)):
            write_run_state(run_dir, "paused_recoverable", "preflight")
            raise
        write_run_state(run_dir, "failed", "preflight")
        raise


def _review_paths(run_dir: Path, stage: str) -> Tuple[Path, Path, List[Path]]:
    if stage == "stage1":
        return (
            run_dir / "05_question_purpose" / "question_purpose_student_review.csv",
            run_dir / "08_human_review" / "question_purpose_professor_adjudication.csv",
            [run_dir / "00_run_preflight", run_dir / "01_processed_transcripts", run_dir / "04_justice_utterances", run_dir / "05_question_purpose"],
        )
    if stage == "stage2":
        return (
            run_dir / "06_reason_alignment" / "reason_alignment_student_review.csv",
            run_dir / "08_human_review" / "reason_alignment_professor_adjudication.csv",
            [run_dir / "00_run_preflight", run_dir / "01_processed_transcripts", run_dir / "04_justice_utterances", run_dir / "05_question_purpose", run_dir / "06_reason_alignment"],
        )
    return (
        run_dir / "07_reason_stance" / "stance_student_review.csv",
        run_dir / "07_reason_stance" / "stance_professor_review.csv",
        [run_dir / "00_run_preflight", run_dir / "01_processed_transcripts", run_dir / "04_justice_utterances", run_dir / "06_reason_alignment", run_dir / "07_reason_stance"],
    )


def _create_stage_review_checkpoint(run_dir: Path, stage: str) -> None:
    from .justice_checkpoints import ReviewCheckpointIntegrityError, create_review_checkpoint
    from .justice_batches import BatchIntegrityError, verify_stage_aggregate, verify_stage_progress
    from .justice_lifecycle import read_run_state, write_run_state

    progress = verify_stage_progress(run_dir, stage)
    if not progress["complete"]:
        raise RuntimeError(f"{stage} cannot create a review checkpoint before all frozen units are complete.")
    verify_stage_aggregate(run_dir, stage)
    student, professor, artifacts = _review_paths(run_dir, stage)
    retry = read_run_state(run_dir).get("status") == "review_checkpoint_pending"
    if not retry:
        write_run_state(run_dir, "review_checkpoint_pending", stage)
    try:
        create_review_checkpoint(
            run_dir,
            stage,
            student_working_path=student,
            professor_working_path=professor,
            artifact_paths=artifacts,
            require_existing_pending=retry,
        )
    except (BatchIntegrityError, ReviewCheckpointIntegrityError, FileNotFoundError):
        write_run_state(run_dir, "failed", stage)
        raise


def _stage_batch_inputs(
    config: Dict[str, Any], run_dir: Path, stage: str, manifest: Dict[str, Any]
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    from .justice_checkpoint_stage1 import consume_stage1_checkpoint
    from .justice_checkpoint_stage2 import consume_stage2_checkpoint, required_stage2_utterances

    utterances = read_jsonl(run_dir / "04_justice_utterances" / "justice_utterances.jsonl")
    if stage == "stage1":
        return utterances, {"utterances": utterances}
    records = _load_json_object(run_dir / "human_final_checkpoints.json", "Run human-final checkpoint record")
    if stage == "stage2":
        checkpoint_id = str(records["stage1"]["checkpoint_id"])
        purpose_rows = consume_stage1_checkpoint(checkpoint_id, utterances)
        nodes = _reason_nodes_for_manifest(manifest)
        candidates = build_candidate_rows(config, run_dir, purpose_rows, nodes)
        groups = candidate_groups(candidates)
        unit_rows = [
            {
                **next(row for row in purpose_rows if str(row["utterance_id"]) == utterance_id),
                "batch_candidate_rows_sha256": sha256_json(group),
            }
            for utterance_id, group in groups.items()
        ]
        return unit_rows, {
            "purpose_rows": purpose_rows,
            "candidate_rows": candidates,
            "reason_nodes": nodes,
        }
    checkpoint_id = str(records["stage2"]["checkpoint_id"])
    required = required_stage2_utterances(checkpoint_id, utterances)
    nodes = _reason_nodes_for_manifest(manifest)
    alignments = consume_stage2_checkpoint(
        checkpoint_id, required, nodes, _stage2_dependency_identity(manifest)
    )
    bundles = stance_bundles(alignments, nodes)
    return bundles, {"alignments": alignments, "reason_nodes": nodes, "bundles": bundles}


def _write_aggregated_stage_outputs(run_dir: Path, stage: str) -> None:
    from .justice_batches import (
        aggregate_rows,
        load_committed_units,
        seal_stage_aggregate,
        verify_stage_progress,
    )

    verify_stage_progress(run_dir, stage)
    model_rows, consolidated = aggregate_rows(run_dir, stage)
    if stage == "stage1":
        out_dir = run_dir / "05_question_purpose"
        write_jsonl(out_dir / "question_purpose_model_annotations.jsonl", model_rows)
        write_csv(out_dir / "question_purpose_model_annotations.csv", model_rows)
        write_jsonl(out_dir / "question_purpose_adjudicated.jsonl", consolidated)
        write_csv(out_dir / "question_purpose_adjudicated.csv", consolidated)
        write_student_review_sheets(
            out_dir, "question_purpose", consolidated, question_purpose_review_fields()
        )
    elif stage == "stage2":
        out_dir = run_dir / "06_reason_alignment"
        write_jsonl(out_dir / "reason_alignment_model_annotations.jsonl", model_rows)
        write_csv(out_dir / "reason_alignment_model_annotations.csv", model_rows)
        write_jsonl(out_dir / "reason_alignment_adjudicated.jsonl", consolidated)
        write_csv(out_dir / "reason_alignment_adjudicated.csv", consolidated)
        write_student_review_sheets(
            out_dir, "reason_alignment", consolidated, alignment_review_fields()
        )
    else:
        out_dir = run_dir / "07_reason_stance"
        committed = load_committed_units(run_dir, stage)
        bundles = [dict(item["unit_input"]) for item in committed]
        write_jsonl(out_dir / "stance_bundles.jsonl", bundles)
        write_csv(out_dir / "stance_bundles.csv", bundles)
        write_jsonl(out_dir / "stance_model_annotations.jsonl", model_rows)
        write_jsonl(out_dir / "stance_adjudicated.jsonl", consolidated)
        review_rows = [_stage3_review_row(row) for row in consolidated]
        write_csv(out_dir / "stance_student_review.csv", review_rows, ["student_id", *stance_review_fields()])
        write_csv(out_dir / "stance_professor_review.csv", review_rows, ["student_id", *stance_review_fields()])
    verify_stage_progress(run_dir, stage)
    seal_stage_aggregate(run_dir, stage)


def _run_ready_stage(config: Dict[str, Any], run_dir: Path, stage: str) -> None:
    from .justice_batches import (
        BatchIntegrityError,
        active_batch,
        choose_batch,
        commit_unit,
        ensure_unit_plan,
        finish_batch,
        stage_batch_dir,
        verify_stage_aggregate,
        verify_stage_progress,
    )
    from .justice_lifecycle import read_run_manifest, write_run_state
    from .openrouter_client import BYOKRequiredError, RecoverableOpenRouterError

    write_run_state(run_dir, "stage_in_progress", stage)
    batch: Optional[Dict[str, Any]] = None
    try:
        manifest = require_sealed_manifest(run_dir, config)
        if manifest.get("manifest_version") != 4:
            raise RuntimeError("Batch execution is only available for new scope-based Justice runs.")
        unit_rows, inputs = _stage_batch_inputs(config, run_dir, stage, manifest)
        ensure_unit_plan(run_dir, stage, unit_rows)
        existing_progress = verify_stage_progress(run_dir, stage, unit_rows)
        if not unit_rows:
            if stage == "stage2":
                write_jsonl(run_dir / "06_reason_alignment" / "reason_alignment_final.jsonl", [])
                write_jsonl(run_dir / "07_reason_stance" / "stance_final.jsonl", [])
                write_run_state(run_dir, "completed", "stage3")
            elif stage == "stage3":
                write_jsonl(run_dir / "07_reason_stance" / "stance_final.jsonl", [])
                write_run_state(run_dir, "completed", "stage3")
            return
        if existing_progress["complete"]:
            unfinished_audit_batch = active_batch(run_dir, stage)
            if unfinished_audit_batch is not None:
                finish_batch(run_dir, stage, unfinished_audit_batch)
            aggregate_root = stage_batch_dir(run_dir, stage)
            aggregate_manifest = aggregate_root / "final_aggregate_manifest.json"
            aggregate_checksum = aggregate_root / "final_aggregate_manifest.sha256"
            if aggregate_manifest.exists() or aggregate_checksum.exists():
                verify_stage_aggregate(run_dir, stage)
            else:
                _write_aggregated_stage_outputs(run_dir, stage)
            _create_stage_review_checkpoint(run_dir, stage)
            return
        batch = choose_batch(run_dir, stage)
        selected_ids = set(batch["planned_unit_ids"])
        if stage == "stage1":
            selected = [row for row in unit_rows if str(row["utterance_id"]) in selected_ids]
            run_question_purpose(
                config,
                run_dir,
                selected,
                unit_complete=lambda unit, models, final: commit_unit(
                    run_dir, stage, unit, models, final
                ),
                write_outputs=False,
            )
        elif stage == "stage2":
            selected_purpose = [
                row for row in inputs["purpose_rows"] if str(row["utterance_id"]) in selected_ids
            ]
            selected_candidates = [
                row for row in inputs["candidate_rows"] if str(row["utterance_id"]) in selected_ids
            ]
            plan_by_id = {str(row["utterance_id"]): row for row in unit_rows}
            run_reason_alignment(
                config,
                run_dir,
                selected_purpose,
                selected_candidates,
                unit_complete=lambda unit, models, final: commit_unit(
                    run_dir, stage, plan_by_id[str(unit["utterance_id"])], models, final
                ),
                write_outputs=False,
            )
        else:
            selected_utterance_ids = {
                str(row["utterance_id"]) for row in unit_rows if str(row["bundle_id"]) in selected_ids
            }
            selected_alignments = [
                row for row in inputs["alignments"]
                if str(row["utterance_id"]) in selected_utterance_ids
            ]
            run_stance_detection(
                config,
                run_dir,
                selected_alignments,
                inputs["reason_nodes"],
                unit_complete=lambda unit, models, final: commit_unit(
                    run_dir, stage, unit, models, final
                ),
                write_outputs=False,
            )
        progress = finish_batch(run_dir, stage, batch)
        if progress["complete"]:
            _write_aggregated_stage_outputs(run_dir, stage)
            verify_stage_progress(run_dir, stage, unit_rows)
            _create_stage_review_checkpoint(run_dir, stage)
        else:
            write_run_state(run_dir, f"ready_for_{stage}", stage)
    except (BYOKRequiredError, RecoverableOpenRouterError):
        if batch is not None:
            finish_batch(run_dir, stage, batch, paused=True)
        write_run_state(run_dir, "paused_recoverable", stage)
        raise
    except Exception:
        from .justice_lifecycle import read_run_state

        current = read_run_state(run_dir)
        if current.get("status") in {"stage_in_progress", "review_checkpoint_pending"}:
            if current.get("status") == "stage_in_progress":
                write_run_state(run_dir, "failed", stage)
        raise


def continue_justice_run(config: Dict[str, Any], run_id: str) -> Path:
    from .justice_lifecycle import load_run, write_run_state

    run_dir, state, manifest = load_run(run_id)
    try:
        config, resolved_model_roles = configure_model_profile(config, str(manifest.get("model_profile") or ""))
    except ValueError as exc:
        raise RuntimeError("The selected Justice model profile cannot be validated for this run.") from exc
    if resolved_model_roles != manifest.get("resolved_model_roles"):
        raise RuntimeError("The selected Justice model profile no longer matches the run manifest.")
    status = state["status"]
    if status == "paused_recoverable" and state.get("current_stage") == "preflight":
        try:
            run_identity_preflight(config, run_dir)
        except Exception as exc:
            from .openrouter_client import BYOKRequiredError, RecoverableOpenRouterError

            if isinstance(exc, (BYOKRequiredError, RecoverableOpenRouterError)):
                write_run_state(run_dir, "paused_recoverable", "preflight")
            else:
                write_run_state(run_dir, "failed", "preflight")
            raise
        write_run_state(run_dir, "awaiting_model_identity_confirmation", "preflight")
        status = "awaiting_model_identity_confirmation"
    if status == "awaiting_model_identity_confirmation":
        if manifest.get("sealed") is True:
            manifest_hash = run_dir / "00_run_preflight" / "run_manifest.sha256"
            if not manifest_hash.is_file():
                from .justice_runtime import recover_missing_manifest_checksum

                recover_missing_manifest_checksum(run_dir, config)
            else:
                require_sealed_manifest(run_dir, config)
        else:
            seal_run_manifest(config, run_dir)
        start = str(manifest.get("start_stage") or "stage1")
        write_run_state(run_dir, f"ready_for_{start}", start)
        status = f"ready_for_{start}"
    if status in {
        "ready_for_stage1", "ready_for_stage2", "ready_for_stage3",
        "paused_recoverable", "stage_in_progress",
    }:
        stage = (
            status.removeprefix("ready_for_")
            if status.startswith("ready_for_")
            else str(state.get("current_stage") or "")
        )
        if stage not in {"stage1", "stage2", "stage3"}:
            raise RuntimeError("Recoverable run state does not identify a valid current Stage.")
        if status == "stage_in_progress":
            write_run_state(run_dir, "paused_recoverable", stage)
        _run_ready_stage(config, run_dir, stage)
    elif status == "review_checkpoint_pending":
        _create_stage_review_checkpoint(run_dir, str(state["current_stage"]))
    elif status == "human_final_checkpoint_pending":
        from .justice_checkpoints import resume_human_final_checkpoint

        stage = str(state["current_stage"])
        resume_human_final_checkpoint(
            run_dir,
            reason_nodes=_reason_nodes_for_manifest(manifest) if stage == "stage2" else [],
            stage2_dependency_identity=_stage2_dependency_identity(manifest) if stage == "stage2" else None,
        )
    elif status.startswith("awaiting_stage") and status.endswith("_human"):
        print(f"Run {run_id} is waiting for a validated professor/final-authority submission.")
    elif status in {"completed", "failed", "abandoned", "preflight_in_progress"}:
        if status == "preflight_in_progress":
            write_run_state(run_dir, "failed", str(state.get("current_stage") or ""))
        raise RuntimeError(f"Run {run_id!r} cannot continue from status {status!r}.")
    return run_dir


def _select_run_interactively() -> str:
    from .justice_lifecycle import list_runs

    candidates = [row for row in list_runs() if row.get("continuable")]
    if not candidates:
        raise RuntimeError("No continuable Justice runs are available.")
    for index, row in enumerate(candidates, start=1):
        scope = row.get("scope_summary") or {}
        scope_label = (
            f"scope={scope.get('advocate_group_limits') or {}}"
            if scope
            else f"mode={row.get('mode') or '-'}"
        )
        print(
            f"[{index}] {row['run_id']} | {scope_label} | profile={row.get('model_profile')} | "
            f"{row.get('status')} | "
            f"brief={row.get('brief_snapshot_id')} | oral={row.get('utterance_snapshot_id')} | "
            f"review_checkpoint={row.get('review_checkpoint_id') or '-'} | "
            f"human_final_checkpoint={row.get('human_final_checkpoint_id') or row.get('upstream_human_final_checkpoint_id') or '-'}"
        )
    if len(candidates) == 1:
        answer = input(f"Continue {candidates[0]['run_id']}? [y/N] ").strip().lower()
        if answer != "y":
            raise RuntimeError("Run continuation cancelled.")
        return str(candidates[0]["run_id"])
    raw = input("Select run number: ").strip()
    if not raw.isdigit() or not (1 <= int(raw) <= len(candidates)):
        raise ValueError("Invalid run selection.")
    return str(candidates[int(raw) - 1]["run_id"])


def main() -> None:
    parser = argparse.ArgumentParser(description="Justice three-stage snapshot/checkpoint workflow.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    brief = subparsers.add_parser("create-brief-snapshot")
    brief.add_argument("--spec", required=True)

    oral = subparsers.add_parser("create-utterance-snapshot")
    oral.add_argument("--spec", required=True)
    oral.add_argument("--timeout", type=int, default=60)

    create = subparsers.add_parser("create-run")
    create.add_argument("--config", default="config/case_sets/justice_2024.yaml")
    create.add_argument("--profile", choices=sorted(APPROVED_MODEL_PROFILES), required=True)
    create.add_argument("--brief-snapshot-id", required=True)
    create.add_argument("--utterance-snapshot-id", required=True)
    create.add_argument("--scope", required=True)
    create.add_argument("--upstream-human-final-checkpoint", default="")

    subparsers.add_parser("list-runs")
    continuation = subparsers.add_parser("continue-run")
    continuation.add_argument("--config", default="config/case_sets/justice_2024.yaml")
    continuation.add_argument("--run-id", default="")
    subparsers.add_parser("list-checkpoints")

    submit = subparsers.add_parser("submit-final-review")
    submit.add_argument("--run-id", required=True)
    submit.add_argument("--stage", choices=["stage1", "stage2", "stage3"], required=True)
    submit.add_argument("--professor-file", required=True)
    submit.add_argument("--student-file", action="append", default=[])

    abandon = subparsers.add_parser("abandon-run")
    abandon.add_argument("--run-id", required=True)
    args = parser.parse_args()

    if args.command == "create-brief-snapshot":
        from .justice_snapshots import create_brief_snapshot

        specification = _load_json_object(Path(args.spec), "Brief snapshot specification")
        def confirm(rows: Sequence[Dict[str, Any]]) -> bool:
            print(json.dumps(rows, ensure_ascii=False, indent=2))
            return input("Confirm these brief IDs, docket numbers, groups, PDFs, and reason trees? [y/N] ").strip().lower() == "y"
        print(create_brief_snapshot(specification, confirm=confirm))
        return
    if args.command == "create-utterance-snapshot":
        from .justice_snapshots import create_utterance_snapshot

        print(create_utterance_snapshot(_load_json_object(Path(args.spec), "Oral snapshot specification"), timeout=args.timeout))
        return
    if args.command == "create-run":
        from .justice_checkpoint_common import load_human_final_checkpoint
        from .justice_lifecycle import normalize_run_scope
        from .justice_snapshots import list_snapshots, verify_brief_snapshot, verify_utterance_snapshot

        config = load_config(args.config)
        scope = _load_json_object(Path(args.scope), "Run scope")
        try:
            brief_manifest = verify_brief_snapshot(args.brief_snapshot_id)
            oral_manifest = verify_utterance_snapshot(args.utterance_snapshot_id)
            normalize_run_scope(scope, oral_manifest, brief_manifest)
            if args.upstream_human_final_checkpoint:
                load_human_final_checkpoint(args.upstream_human_final_checkpoint)
        except Exception:
            from .justice_checkpoints import list_checkpoints

            print("Available immutable brief snapshots:")
            print(json.dumps(list_snapshots("brief"), ensure_ascii=False, indent=2))
            print("Available immutable oral-argument snapshots:")
            print(json.dumps(list_snapshots("oral_argument"), ensure_ascii=False, indent=2))
            if args.upstream_human_final_checkpoint:
                print("Available checkpoints:")
                print(json.dumps(list_checkpoints(), ensure_ascii=False, indent=2))
            raise
        run_dir = create_justice_run(
            config, args.brief_snapshot_id, args.utterance_snapshot_id,
            scope, args.profile, args.upstream_human_final_checkpoint,
        )
        print(f"Created Justice run {run_dir.name}; review and confirm its OpenRouter identity probe before continuing.")
        return
    if args.command == "list-runs":
        from .justice_lifecycle import list_runs
        print(json.dumps(list_runs(), ensure_ascii=False, indent=2))
        return
    if args.command == "continue-run":
        config = load_config(args.config)
        run_id = args.run_id or _select_run_interactively()
        print(continue_justice_run(config, run_id))
        return
    if args.command == "list-checkpoints":
        from .justice_checkpoints import list_checkpoints
        print(json.dumps(list_checkpoints(), ensure_ascii=False, indent=2))
        return
    if args.command == "abandon-run":
        from .justice_lifecycle import mark_abandoned
        print(json.dumps(mark_abandoned(args.run_id), ensure_ascii=False, indent=2))
        return
    if args.command == "submit-final-review":
        from .justice_checkpoints import create_human_final_checkpoint, write_invalid_human_values
        from .justice_lifecycle import load_run, read_run_manifest, write_run_state

        run_dir, state, manifest = load_run(args.run_id)
        if state.get("status") != f"awaiting_{args.stage}_human":
            raise RuntimeError("Run is not awaiting human final for the submitted Stage.")
        records = _load_json_object(run_dir / "review_checkpoints.json", "Review checkpoint record")
        review_checkpoint_id = str(records[args.stage]["checkpoint_id"])
        nodes = _reason_nodes_for_manifest(manifest) if args.stage == "stage2" else []
        try:
            result = create_human_final_checkpoint(
                run_dir,
                review_checkpoint_id,
                Path(args.professor_file),
                [Path(value) for value in args.student_file],
                reason_nodes=nodes,
                stage2_dependency_identity=_stage2_dependency_identity(manifest) if args.stage == "stage2" else None,
            )
        except (ValueError, FileNotFoundError) as exc:
            diagnostics = write_invalid_human_values(
                run_dir, args.stage, Path(args.professor_file), [Path(value) for value in args.student_file]
            )
            write_run_state(run_dir, f"awaiting_{args.stage}_human", args.stage)
            print(f"Human final was not accepted. Raw submitted values were preserved at {diagnostics}.")
            raise
        if args.stage == "stage2":
            clean = read_jsonl(run_dir / "06_reason_alignment" / "reason_alignment_final.jsonl")
            if not any(row.get("final_selected_reason_ids") for row in clean):
                write_jsonl(run_dir / "07_reason_stance" / "stance_final.jsonl", [])
                write_run_state(run_dir, "completed", "stage3")
        print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
