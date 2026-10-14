from __future__ import annotations

from typing import Any, Dict, List, Optional

from .common import clean_text, first_present, seconds


def speaker_payload(turn: Dict[str, Any]) -> Dict[str, Any]:
    speaker = turn.get("speaker") or {}
    return {"name": speaker} if isinstance(speaker, str) else speaker


def speaker_role(speaker: Dict[str, Any]) -> str:
    roles = speaker.get("roles") or []
    if isinstance(roles, dict):
        roles = [roles]
    names = []
    for role in roles:
        if isinstance(role, dict):
            names.append(str(first_present(role, ["type", "role", "name"], "")))
    return "; ".join(name for name in names if name)


def turn_text(turn: Dict[str, Any]) -> str:
    if turn.get("text"):
        return clean_text(turn["text"])
    pieces = []
    for block in turn.get("text_blocks") or turn.get("textBlocks") or []:
        if isinstance(block, dict):
            pieces.append(str(first_present(block, ["text", "content"], "")))
        else:
            pieces.append(str(block))
    return clean_text(" ".join(pieces))


def turn_time(turn: Dict[str, Any], key: str) -> Optional[float]:
    value = first_present(turn, [key, f"{key}_time", f"{key}Time"])
    blocks = turn.get("text_blocks") or turn.get("textBlocks") or []
    if value is None and blocks:
        values = []
        for block in blocks:
            if isinstance(block, dict):
                item = first_present(block, [key, f"{key}_time", f"{key}Time"])
                parsed = seconds(item)
                if parsed is not None:
                    values.append(parsed)
        if values:
            return min(values) if key == "start" else max(values)
    return seconds(value)


def text_block_id(turn: Dict[str, Any]) -> str:
    blocks = turn.get("text_blocks") or turn.get("textBlocks") or []
    ids = []
    for block in blocks:
        if isinstance(block, dict):
            value = first_present(block, ["id", "text_block_id", "textBlockId"])
            if value:
                ids.append(str(value))
    return ";".join(ids)


def is_justice(speaker: Dict[str, Any]) -> bool:
    return "scotus_justice" in speaker_role(speaker).lower()


def section_counsel_name(section: Dict[str, Any]) -> str:
    for turn in section.get("turns", []):
        speaker = speaker_payload(turn)
        name = clean_text(first_present(speaker, ["name", "display_name", "identifier"], ""))
        if name and not is_justice(speaker):
            return name
    return ""


def classify_sections(sections: List[Dict[str, Any]]) -> Dict[int, Dict[str, str]]:
    count = len(sections)
    counsel = {index: section_counsel_name(section) for index, section in enumerate(sections, start=1)}
    assignments: Dict[int, Dict[str, str]] = {}

    if count == 0:
        return assignments
    if count == 1:
        assignments[1] = {
            "phase": "unknown",
            "confidence": "low",
            "notes": "Only one Oyez section; three-phase structure cannot be inferred.",
        }
    elif count == 2:
        assignments[1] = {
            "phase": "petitioner",
            "confidence": "medium",
            "notes": "First Oyez section treated as petitioner opening.",
        }
        assignments[2] = {
            "phase": "respondent",
            "confidence": "medium",
            "notes": "Second and final Oyez section treated as respondent; no rebuttal identified.",
        }
    else:
        first_counsel = counsel.get(1, "")
        last_counsel = counsel.get(count, "")
        rebuttal_confirmed = bool(first_counsel and last_counsel and first_counsel == last_counsel)
        if rebuttal_confirmed:
            for index in range(1, count - 1):
                assignments[index] = {
                    "phase": "petitioner",
                    "confidence": "medium",
                    "notes": "Section precedes the inferred respondent section.",
                }
            assignments[count - 1] = {
                "phase": "respondent",
                "confidence": "medium",
                "notes": "Section immediately precedes confirmed petitioner rebuttal.",
            }
            assignments[count] = {
                "phase": "petitioner_rebuttal",
                "confidence": "high",
                "notes": f"Final-section counsel matches opening petitioner counsel: {first_counsel}.",
            }
        else:
            assignments[1] = {
                "phase": "petitioner",
                "confidence": "medium",
                "notes": "First Oyez section treated as petitioner opening.",
            }
            for index in range(2, count):
                assignments[index] = {
                    "phase": "unknown",
                    "confidence": "low",
                    "notes": "Intermediate section cannot be assigned reliably without counsel-side metadata.",
                }
            assignments[count] = {
                "phase": "respondent",
                "confidence": "low",
                "notes": "Final counsel does not match opening counsel; rebuttal not confirmed.",
            }

    for index, assignment in assignments.items():
        assignment["section_counsel"] = counsel.get(index, "")
    return assignments
