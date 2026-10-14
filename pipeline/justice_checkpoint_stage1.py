from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

from .justice_checkpoint_common import identity_key, load_human_final_checkpoint, read_jsonl, require_consumer_hash
RETAINED_PURPOSES = {"Clarification", "Implications", "Support", "Criticism"}
PURPOSE_LABELS = {
    "Background", "Clarification", "Implications", "Support", "Criticism", "Communicate", "Humor"
}


def consume_stage1_checkpoint(
    checkpoint_id: str, required_utterances: Sequence[Mapping[str, Any]]
) -> list[Dict[str, Any]]:
    directory, manifest = load_human_final_checkpoint(checkpoint_id)
    if manifest.get("stage") != "stage1":
        raise ValueError("Only a Stage 1 human-final checkpoint can start a new run at Stage 2.")
    require_consumer_hash(manifest, "stage1_consumer", Path(__file__))
    from . import justice_checkpoint_common
    require_consumer_hash(manifest, "checkpoint_common", Path(justice_checkpoint_common.__file__))
    clean = read_jsonl(directory / str(manifest.get("clean_final_relative_path") or "clean_final.jsonl"))
    by_id: Dict[str, Dict[str, Any]] = {}
    allowed_fields = {
        "oyez_argument_id", "utterance_id", "turn_id", "target_justice", "text",
        "final_question_function_present", "final_purpose_labels",
    }
    for row in clean:
        if set(row) != allowed_fields:
            raise ValueError("Stage 1 checkpoint clean final does not match the approved clean schema.")
        utterance_id = str(row.get("utterance_id") or "")
        if not utterance_id or utterance_id in by_id:
            raise ValueError("Stage 1 checkpoint clean final contains missing or duplicate utterance IDs.")
        gate = row.get("final_question_function_present")
        labels = row.get("final_purpose_labels")
        if not isinstance(gate, bool) or not isinstance(labels, list) or any(label not in PURPOSE_LABELS for label in labels):
            raise ValueError("Stage 1 checkpoint clean final contains an invalid gate or purpose labels.")
        if len(labels) != len(set(labels)) or (gate is False and labels) or (gate is True and not labels):
            raise ValueError("Stage 1 checkpoint clean final contains inconsistent gate/purpose values.")
        by_id[utterance_id] = dict(row)
    required_by_id = {str(row.get("utterance_id") or ""): row for row in required_utterances}
    if any(not key for key in required_by_id) or len(required_by_id) != len(required_utterances):
        raise ValueError("New run required utterances contain missing or duplicate IDs.")
    missing = sorted(set(required_by_id) - set(by_id))
    if missing:
        raise ValueError(f"Stage 1 checkpoint does not cover every new-run required utterance: {missing[:10]!r}.")
    output: list[Dict[str, Any]] = []
    for utterance_id, required in required_by_id.items():
        row = by_id[utterance_id]
        if identity_key(row) != identity_key(required):
            raise ValueError(f"Stage 1 checkpoint utterance identity/text mismatch: {utterance_id!r}.")
        labels = list(row["final_purpose_labels"])
        output.append(
            {
                **required,
                "final_question_function_present": row["final_question_function_present"],
                "final_purpose_labels": labels,
                "enters_reason_alignment": bool(set(labels) & RETAINED_PURPOSES),
                "stage1_resolved": True,
                "stage1_requires_human_review": False,
                "final_source": "human_final_checkpoint",
            }
        )
    return output
