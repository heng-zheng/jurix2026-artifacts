from __future__ import annotations

import csv
import json
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import yaml

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover
    load_dotenv = None


REPO_ROOT = Path(__file__).resolve().parents[1]
_PROGRESS_STARTED_AT: Dict[Tuple[str, str], float] = {}


def load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def load_config(config_path: str) -> Dict[str, Any]:
    path = resolve_path(config_path)
    config = load_yaml(path)
    config["_config_path"] = str(path)
    if load_dotenv is not None:
        load_dotenv(REPO_ROOT / ".env")
    return config


def resolve_path(path: str) -> Path:
    value = Path(path)
    return value if value.is_absolute() else REPO_ROOT / value


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing required input: {path}")
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_csv(path: Path, rows: List[Dict[str, Any]], fieldnames: Optional[List[str]] = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = []
        for row in rows:
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: stringify_cell(row.get(key, "")) for key in fieldnames})


def stringify_cell(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (list, dict)):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def estimate_tokens(text: str) -> int:
    return max(1, int(len(text.split()) / 0.75))


def clean_text(text: Any) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def first_present(payload: Dict[str, Any], keys: List[str], default: Any = None) -> Any:
    for key in keys:
        if key in payload and payload[key] not in (None, ""):
            return payload[key]
    return default


def model_payload(model_id: str, block: Dict[str, Any]) -> Dict[str, Any]:
    payload = {
        "model_id": model_id,
        "provider": block.get("provider", "openrouter"),
        "model_name": block.get("model_name", model_id),
    }
    for key in ["fixed_provider", "expected_actual_model", "expected_provider_name"]:
        if key in block:
            payload[key] = block.get(key)
    return payload


def normalize_person_name(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(value or "").lower()).strip()


def render_template(path: Path, values: Dict[str, Any]) -> str:
    rendered = path.read_text(encoding="utf-8")
    for key, value in values.items():
        rendered = rendered.replace("{{" + key + "}}", stringify_cell(value))
    return rendered


def append_log(path: Path, message: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().isoformat(timespec="seconds")
    with path.open("a", encoding="utf-8") as handle:
        handle.write(f"[{timestamp}] {message}\n")


def progress_settings(config: Dict[str, Any]) -> Tuple[bool, int]:
    progress = config.get("progress", {})
    return bool(progress.get("enabled", False)), max(1, int(progress.get("every_n_calls", 10)))


def print_progress(
    module: str,
    label: str,
    current: int,
    total: int,
    every_n: int,
    force: bool = False,
) -> None:
    if total <= 0 or (not force and current % every_n != 0 and current != total):
        return
    key = (module, label)
    now = time.monotonic()
    if current == 0 or key not in _PROGRESS_STARTED_AT:
        _PROGRESS_STARTED_AT[key] = now
    elapsed = max(0.0, now - _PROGRESS_STARTED_AT[key])
    eta = (elapsed / current) * (total - current) if current > 0 else None

    def duration(value: Optional[float]) -> str:
        if value is None:
            return "--:--:--"
        seconds_value = max(0, int(round(value)))
        hours, remainder = divmod(seconds_value, 3600)
        minutes, seconds_value = divmod(remainder, 60)
        return f"{hours:02d}:{minutes:02d}:{seconds_value:02d}"

    pct = (current / total) * 100
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(
        f"[{timestamp}] [{module}] {label} {current}/{total} ({pct:.1f}%) "
        f"elapsed={duration(elapsed)} eta={duration(eta)}",
        flush=True,
    )


def seconds(value: Any) -> Optional[float]:
    if value in (None, ""):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    if ":" in text:
        total = 0.0
        for part in [float(item) for item in text.split(":")]:
            total = total * 60 + part
        return total
    try:
        return float(text)
    except ValueError:
        return None


def safe_name(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", value).strip("_")[:160]


def sleep_retry(attempt: int) -> None:
    time.sleep(min(2**attempt, 8))
