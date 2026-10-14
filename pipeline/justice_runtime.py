from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
import tempfile
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import requests

from .common import REPO_ROOT, read_jsonl, write_jsonl
from .justice_contracts import contract_manifest, strict_json_object
from .llm_client import chat_client_for_model


ANNOTATOR_PARAMETERS = {
    "max_completion_tokens": 4096,
    "reasoning": {"exclude": True},
}
ADJUDICATOR_PARAMETERS = {
    "max_completion_tokens": 8192,
    "reasoning": {"exclude": True},
}
RESPONSE_FORMAT = {"type": "json_object"}
REQUIRED_PARAMETERS = {"response_format", "reasoning"}
NOT_SENT_PARAMETER_POLICY = {
    "temperature_status": "not_sent_by_policy",
    "top_p_status": "not_sent_by_policy",
    "seed_status": "not_sent_by_policy",
    "n_status": "not_sent_by_policy",
    "reasoning_effort_status": "not_sent_by_policy",
    "reasoning_max_tokens_status": "not_sent_by_policy",
    "reasoning_enabled_status": "not_sent_by_policy",
}
RELEVANT_PACKAGES = {
    "numpy",
    "openai",
    "openpyxl",
    "pandas",
    "python-dotenv",
    "pypdf",
    "pyyaml",
    "requests",
    "pdfservices-sdk",
}
PROBE_PROMPT = (
    "Return exactly one JSON object with this content and no other text: "
    '{"probe":"ok","rationale":"capability probe","confidence":1,"uncertainty_notes":""}'
)

APPROVED_MODEL_PROFILES = {
    "frontier": {
        "annotator_models": ["gpt_5_2", "gemini_2_5_pro"],
        "adjudicator_model": "claude_opus_4_5",
    },
    "balanced": {
        "annotator_models": ["gpt_5_6_luna", "gemini_3_5_flash"],
        "adjudicator_model": "claude_sonnet_5",
    },
    "economy": {
        "annotator_models": ["gpt_5_4_mini", "gemini_3_flash_preview"],
        "adjudicator_model": "claude_sonnet_5",
    },
}

APPROVED_PROFILE_MODELS = {
    "gpt_5_2": ("annotation", "openai/gpt-5.2", "openai"),
    "gemini_2_5_pro": ("annotation", "google/gemini-2.5-pro", "google-ai-studio"),
    "claude_opus_4_5": ("adjudication", "anthropic/claude-opus-4.5", "anthropic"),
    "gpt_5_6_luna": ("annotation", "openai/gpt-5.6-luna", "openai"),
    "gemini_3_5_flash": ("annotation", "google/gemini-3.5-flash", "google-ai-studio"),
    "claude_sonnet_5": ("adjudication", "anthropic/claude-sonnet-5", "anthropic"),
    "gpt_5_4_mini": ("annotation", "openai/gpt-5.4-mini", "openai"),
    "gemini_3_flash_preview": ("annotation", "google/gemini-3-flash-preview", "google-ai-studio"),
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(payload: Any) -> str:
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def role_parameters(role: str, capability: Dict[str, Any]) -> Dict[str, Any]:
    base = dict(ADJUDICATOR_PARAMETERS if role == "adjudicator" else ANNOTATOR_PARAMETERS)
    requested_limit = int(base.pop("max_completion_tokens"))
    supported = set(capability.get("supported_parameters") or [])
    missing = sorted(REQUIRED_PARAMETERS - supported)
    if missing:
        raise RuntimeError(f"The fixed model/provider does not explicitly support required parameters: {missing!r}.")
    if "max_completion_tokens" in supported:
        token_limit_parameter = "max_completion_tokens"
    elif "max_tokens" in supported:
        token_limit_parameter = "max_tokens"
    else:
        raise RuntimeError(
            "The fixed model/provider does not explicitly support max_tokens or max_completion_tokens."
        )
    limit = capability.get("max_completion_tokens")
    if not isinstance(limit, int) or limit < requested_limit:
        raise RuntimeError(
            f"The fixed model/provider does not explicitly support max_completion_tokens={requested_limit}; "
            f"reported limit={limit!r}."
        )
    base[token_limit_parameter] = requested_limit
    base.update(NOT_SENT_PARAMETER_POLICY)
    return base


def request_parameters(parameters: Dict[str, Any]) -> Dict[str, Any]:
    return {
        key: value
        for key, value in parameters.items()
        if not key.endswith("_status")
    }


def canonical_capability(metadata: Dict[str, Any], provider_slug: str) -> Dict[str, Any]:
    model = metadata.get("model_raw", {}).get("data") or {}
    endpoint = metadata.get("selected_endpoint") or {}
    endpoint_supported = endpoint.get("supported_parameters")
    supported = sorted(set(endpoint_supported)) if isinstance(endpoint_supported, list) else []
    return {
        "model_id": model.get("id", ""),
        "canonical_slug": model.get("canonical_slug", ""),
        "provider_slug": provider_slug,
        "endpoint_tag": endpoint.get("tag") or "",
        "provider_name": endpoint.get("provider_name") or endpoint.get("provider") or endpoint.get("name") or "",
        "supported_parameters": supported,
        "max_completion_tokens": endpoint.get("max_completion_tokens"),
    }


def preflight_dir(run_dir: Path) -> Path:
    return run_dir / "00_run_preflight"


def sealed_manifest_path(run_dir: Path) -> Path:
    return preflight_dir(run_dir) / "run_manifest.json"


def _atomic_write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _validate_sealed_manifest_contents(
    run_dir: Path, manifest: Dict[str, Any], config: Optional[Dict[str, Any]] = None
) -> None:
    if manifest.get("sealed") is not True:
        raise RuntimeError("Stage 1-3 cannot start until the run manifest is sealed.")
    if manifest.get("run_id") != run_dir.name:
        raise RuntimeError("The sealed run manifest identity does not match the run directory.")
    verify_manifest_files(manifest)
    verify_manifest_environment(manifest)
    if manifest.get("brief_snapshot_id") and manifest.get("utterance_snapshot_id"):
        from .justice_snapshots import verify_brief_snapshot, verify_utterance_snapshot

        brief = verify_brief_snapshot(str(manifest["brief_snapshot_id"]))
        oral = verify_utterance_snapshot(str(manifest["utterance_snapshot_id"]))
        if brief.get("overall_snapshot_sha256") != manifest.get("brief_snapshot_hash"):
            raise RuntimeError("The sealed run's brief snapshot is missing or changed.")
        if oral.get("overall_snapshot_sha256") != manifest.get("utterance_snapshot_hash"):
            raise RuntimeError("The sealed run's oral-argument snapshot is missing or changed.")
    _verify_upstream_checkpoint(manifest)
    report = read_json(preflight_dir(run_dir) / "preflight_report.json")
    confirmation = read_json(preflight_dir(run_dir) / "preflight_confirmation.json")
    _validate_preflight_confirmation(report, confirmation)
    if report != manifest.get("preflight"):
        raise RuntimeError("The preflight report no longer matches the sealed run manifest.")
    if config is not None:
        current_config = _without_secrets({key: value for key, value in config.items() if key != "_config_path"})
        if current_config != manifest.get("config_without_secrets"):
            raise RuntimeError("The current effective configuration does not match the sealed run manifest.")


def require_sealed_manifest(run_dir: Path, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    path = sealed_manifest_path(run_dir)
    checksum_path = preflight_dir(run_dir) / "run_manifest.sha256"
    if not path.exists() or not checksum_path.exists():
        raise RuntimeError("Stage 1-3 cannot start until this run's identity probe is confirmed and manifest is sealed.")
    expected = checksum_path.read_text(encoding="utf-8").strip()
    actual = sha256_file(path)
    if actual != expected:
        raise RuntimeError("The sealed run manifest checksum does not match; resume is refused.")
    manifest = read_json(path)
    _validate_sealed_manifest_contents(run_dir, manifest, config)
    return manifest


def recover_missing_manifest_checksum(run_dir: Path, config: Dict[str, Any]) -> Dict[str, Any]:
    path = sealed_manifest_path(run_dir)
    checksum_path = preflight_dir(run_dir) / "run_manifest.sha256"
    if checksum_path.exists():
        raise RuntimeError("Manifest checksum recovery is only allowed when the checksum file is missing.")
    if not path.is_file():
        raise RuntimeError("Manifest checksum recovery requires an existing sealed run manifest.")
    manifest = read_json(path)
    _validate_sealed_manifest_contents(run_dir, manifest, config)
    _atomic_write_text(checksum_path, sha256_file(path))
    return require_sealed_manifest(run_dir, config)


def _verify_upstream_checkpoint(manifest: Dict[str, Any]) -> None:
    checkpoint_id = str(manifest.get("upstream_human_final_checkpoint_id") or "")
    if not checkpoint_id:
        return
    from .justice_checkpoint_common import load_human_final_checkpoint

    _, checkpoint = load_human_final_checkpoint(checkpoint_id)
    if (
        checkpoint.get("overall_checkpoint_sha256") != manifest.get("upstream_human_final_checkpoint_hash")
        or checkpoint.get("stage") != manifest.get("upstream_human_final_checkpoint_stage")
    ):
        raise RuntimeError("The sealed run's upstream human-final checkpoint identity changed or is incompatible.")


def _model_blocks(config: Dict[str, Any]) -> List[Tuple[str, Dict[str, Any]]]:
    workflow = config.get("annotation_workflow", {}) or {}
    rows: List[Tuple[str, Dict[str, Any]]] = []
    for model_id in workflow.get("annotator_models") or []:
        block = (config.get("models", {}).get("annotation", {}) or {}).get(model_id) or {}
        rows.append(("annotator", {"model_id": model_id, **block}))
    adjudicator_id = workflow.get("adjudicator_model")
    block = (config.get("models", {}).get("adjudication", {}) or {}).get(adjudicator_id) or {}
    rows.append(("adjudicator", {"model_id": adjudicator_id, **block}))
    return rows


def configure_model_profile(
    config: Dict[str, Any], profile_name: str
) -> Tuple[Dict[str, Any], List[Dict[str, str]]]:
    expected_profile = APPROVED_MODEL_PROFILES.get(profile_name)
    if expected_profile is None:
        raise ValueError(
            f"Unknown Justice model profile {profile_name!r}; choose exactly one of "
            f"{sorted(APPROVED_MODEL_PROFILES)!r}."
        )
    workflow = config.get("annotation_workflow", {}) or {}
    configured_profiles = workflow.get("profiles")
    if not isinstance(configured_profiles, dict):
        raise ValueError("annotation_workflow.profiles must define all approved Justice model profiles.")
    configured_profile = configured_profiles.get(profile_name)
    if configured_profile != expected_profile:
        raise ValueError(
            f"Justice model profile {profile_name!r} does not exactly match the approved role mapping."
        )

    resolved: List[Dict[str, str]] = []
    role_specs = [
        ("annotator_a", "annotation", expected_profile["annotator_models"][0]),
        ("annotator_b", "annotation", expected_profile["annotator_models"][1]),
        ("adjudicator", "adjudication", expected_profile["adjudicator_model"]),
    ]
    for role, stage, model_id in role_specs:
        expected_stage, expected_slug, expected_provider = APPROVED_PROFILE_MODELS[model_id]
        if expected_stage != stage:
            raise ValueError(f"Approved profile model {model_id!r} is assigned to the wrong role.")
        block = ((config.get("models", {}) or {}).get(stage, {}) or {}).get(model_id)
        if not isinstance(block, dict):
            raise ValueError(f"Justice profile model {model_id!r} is missing from models.{stage}.")
        identity = {
            "role": role,
            "model_id": model_id,
            "provider": str(block.get("provider") or ""),
            "model_name": str(block.get("model_name") or ""),
            "fixed_provider": str(block.get("fixed_provider") or ""),
        }
        expected_identity = {
            "role": role,
            "model_id": model_id,
            "provider": "openrouter",
            "model_name": expected_slug,
            "fixed_provider": expected_provider,
        }
        if identity != expected_identity:
            raise ValueError(
                f"Justice profile model {model_id!r} must exactly match its approved model and provider identity."
            )
        resolved.append(identity)

    effective = deepcopy(config)
    effective_workflow = effective.setdefault("annotation_workflow", {})
    effective_workflow["annotator_models"] = list(expected_profile["annotator_models"])
    effective_workflow["adjudicator_model"] = expected_profile["adjudicator_model"]
    effective_workflow["selected_profile"] = profile_name
    return effective, resolved


def run_identity_preflight(config: Dict[str, Any], run_dir: Path) -> Dict[str, Any]:
    out = preflight_dir(run_dir)
    out.mkdir(parents=True, exist_ok=True)
    report: Dict[str, Any] = {"status": "awaiting_human_confirmation", "models": []}
    model_blocks = _model_blocks(config)
    annotator_names = [str(model.get("model_name") or "") for role, model in model_blocks if role == "annotator"]
    if len(annotator_names) != 2 or not all(annotator_names) or annotator_names[0] == annotator_names[1]:
        raise RuntimeError("The two Stage 1-3 annotators must use different exact models.")
    for role, model in model_blocks:
        if model.get("provider", "openrouter") != "openrouter":
            raise RuntimeError("Justice Stage 1-3 identity preflight requires OpenRouter models.")
        model_name = str(model.get("model_name") or "")
        provider_slug = str(model.get("fixed_provider") or "")
        if not provider_slug:
            raise RuntimeError(
                f"Model {model.get('model_id')!r} must configure one exact fixed_provider before preflight."
            )
        client = chat_client_for_model(config, model)
        stem = f"{role}_{model['model_id']}"
        raw_capability = client.capability_metadata(
            model_name,
            provider_slug,
            out / "errors.log",
            state_dir=out / "capability_query_state" / stem,
        )
        capability = canonical_capability(raw_capability, provider_slug)
        parameters = role_parameters(role, capability)
        write_json(out / "capabilities" / f"{stem}_raw.json", raw_capability)
        probe_prompt = PROBE_PROMPT
        probe_path = out / "identity_probes" / f"{stem}.json"
        probe = client.chat(
            model_name,
            probe_prompt,
            probe_path,
            out / "errors.log",
            provider_only=[provider_slug],
            require_parameters=True,
            request_parameters=request_parameters(parameters),
            response_format=RESPONSE_FORMAT,
            require_generation_metadata=True,
            include_default_temperature=False,
        )
        probe_request_path = out / "identity_probes" / f"{stem}_request.json"
        write_json(probe_request_path, probe.get("request_payload", {}))
        if probe.get("finish_reason") != "stop":
            raise RuntimeError(f"Identity probe for {stem} did not finish with finish_reason=stop.")
        probe_object, probe_findings = strict_json_object(probe.get("content", ""))
        if probe_object is None or probe_findings:
            raise RuntimeError(f"Identity probe for {stem} did not return one strict complete JSON object.")
        model_report = {
            "role": role,
            "model_id": model["model_id"],
            "requested_model_slug": model_name,
            "actual_model": probe.get("actual_model", ""),
            "fixed_provider_slug": provider_slug,
            "actual_provider_name": probe.get("actual_provider", ""),
            "generation_id": probe.get("generation_id", ""),
            "request_payload_path": str(probe_request_path),
            "request_payload_sha256": sha256_file(probe_request_path),
            "raw_response_path": str(probe_path),
            "raw_response_sha256": sha256_file(probe_path),
            "actual_choice_count": probe.get("actual_choice_count", 0),
            "response_headers": probe.get("response_headers", {}),
            "system_fingerprint": probe.get("system_fingerprint", "not_returned"),
            "capability": capability,
            "generation_parameters": parameters,
            "response_format": RESPONSE_FORMAT,
        }
        report["models"].append(model_report)
    actual_annotators = [
        str(item.get("actual_model") or "")
        for item in report["models"]
        if item.get("role") == "annotator"
    ]
    if len(actual_annotators) != 2 or not all(actual_annotators) or actual_annotators[0] == actual_annotators[1]:
        raise RuntimeError("The two Stage 1-3 annotators must resolve to two different exact actual models.")
    write_json(out / "preflight_report.json", report)
    confirmation = {
        "confirmed": False,
        "models": [
            {
                "model_id": item["model_id"],
                "requested_model_slug": item["requested_model_slug"],
                "expected_actual_model": item["actual_model"],
                "fixed_provider_slug": item["fixed_provider_slug"],
                "expected_provider_name": item["actual_provider_name"],
            }
            for item in report["models"]
        ],
    }
    write_json(out / "preflight_confirmation.json", confirmation)
    return report


def _git_identity() -> Dict[str, Any]:
    def run(*args: str) -> str:
        result = subprocess.run(["git", *args], cwd=REPO_ROOT, check=False, text=True, capture_output=True)
        return result.stdout.strip()
    return {"commit": run("rev-parse", "HEAD"), "dirty_status": run("status", "--short")}


def _package_versions() -> Dict[str, str]:
    try:
        from importlib.metadata import distributions
        versions: Dict[str, str] = {}
        for dist in distributions():
            name = str(dist.metadata.get("Name") or "")
            if name.lower().replace("_", "-") in RELEVANT_PACKAGES:
                versions[name] = dist.version
        return versions
    except Exception:
        return {}


def _without_secrets(value: Any) -> Any:
    if isinstance(value, dict):
        result: Dict[str, Any] = {}
        for key, item in value.items():
            lowered = str(key).lower()
            if lowered.endswith("_env"):
                result[key] = item
            elif any(marker in lowered for marker in ["api_key", "password", "credential", "secret", "access_token"]):
                result[key] = "[REDACTED]"
            else:
                result[key] = _without_secrets(item)
        return result
    if isinstance(value, list):
        return [_without_secrets(item) for item in value]
    return value


def controlled_files(config: Dict[str, Any], run_dir: Path) -> List[Path]:
    files: List[Path] = []
    for relative in [
        "pipeline/justice_workflow.py",
        "pipeline/justice_contracts.py",
        "pipeline/justice_runtime.py",
        "pipeline/justice_snapshots.py",
        "pipeline/justice_lifecycle.py",
        "pipeline/justice_inputs.py",
        "pipeline/justice_batches.py",
        "pipeline/justice_checkpoints.py",
        "pipeline/justice_checkpoint_common.py",
        "pipeline/justice_checkpoint_stage1.py",
        "pipeline/justice_checkpoint_stage2.py",
        "pipeline/preprocess_oyez.py",
        "pipeline/openrouter_client.py",
        "pipeline/llm_client.py",
        "pipeline/common.py",
        "prompts/justice_question_purpose_prompt.txt",
        "prompts/justice_question_purpose_adjudication_prompt.txt",
        "prompts/justice_alignment_prompt.txt",
        "prompts/justice_alignment_adjudication_prompt.txt",
        "prompts/justice_stance_prompt.txt",
        "prompts/justice_stance_adjudication_prompt.txt",
        "prompts/justice_structured_repair_prompt.txt",
        "requirements.txt",
    ]:
        path = REPO_ROOT / relative
        if path.exists():
            files.append(path)
    config_path = Path(str(config.get("_config_path") or ""))
    if str(config.get("_config_path") or "") and config_path.is_file():
        files.append(config_path)
    for relative in [
        "run_scope.json",
        "01_processed_transcripts/processed_transcripts.jsonl",
        "04_justice_utterances/justice_utterances.jsonl",
        "04_justice_utterances/selected_utterance_ids.json",
        "04_justice_utterances/local_dialogue_context_audit.jsonl",
        "05_question_purpose/question_purpose_final.jsonl",
        "06_reason_alignment/reason_alignment_final.jsonl",
        "00_run_preflight/preflight_report.json",
        "00_run_preflight/preflight_confirmation.json",
    ]:
        path = run_dir / relative
        if path.exists():
            files.append(path)
    for path in sorted((run_dir / "01_processed_transcripts").glob("*.jsonl")):
        files.append(path)
    for pattern in [
        "00_run_preflight/identity_probes/*.json",
        "00_run_preflight/capabilities/*.json",
        "00_run_preflight/capability_query_state/**/*.json",
    ]:
        for path in sorted(run_dir.glob(pattern)):
            files.append(path)
    return files


def _validate_preflight_confirmation(report: Dict[str, Any], confirmation: Dict[str, Any]) -> None:
    if confirmation.get("confirmed") is not True:
        raise RuntimeError("preflight_confirmation.json must set confirmed to true after human identity review.")
    report_models = report.get("models", [])
    if not isinstance(report_models, list) or any(not isinstance(item, dict) for item in report_models):
        raise RuntimeError("Preflight report models must be a complete JSON list.")
    report_by_id = {item.get("model_id"): item for item in report_models}
    if len(report_by_id) != len(report_models) or None in report_by_id:
        raise RuntimeError("Preflight report contains missing or duplicate model identities.")
    confirmation_models = confirmation.get("models", [])
    if not isinstance(confirmation_models, list) or any(not isinstance(item, dict) for item in confirmation_models):
        raise RuntimeError("Preflight confirmation models must be a complete JSON list.")
    confirmation_ids = [item.get("model_id") for item in confirmation_models]
    if len(confirmation_ids) != len(set(confirmation_ids)) or set(confirmation_ids) != set(report_by_id):
        raise RuntimeError("Preflight confirmation must cover every probed model identity exactly once.")
    for item in confirmation_models:
        expected = report_by_id.get(item.get("model_id"))
        comparisons = {
            "requested_model_slug": expected["requested_model_slug"],
            "expected_actual_model": expected["actual_model"],
            "fixed_provider_slug": expected["fixed_provider_slug"],
            "expected_provider_name": expected["actual_provider_name"],
        }
        for field, value in comparisons.items():
            if item.get(field) != value:
                raise RuntimeError(f"Preflight confirmation field {field!r} does not exactly match the probe result.")


def _validate_current_model_configuration(config: Dict[str, Any], report: Dict[str, Any]) -> None:
    current = _model_blocks(config)
    reported = report.get("models", [])
    if not isinstance(reported, list) or len(current) != len(reported):
        raise RuntimeError("Current model-role configuration no longer matches the identity preflight.")
    for index, ((role, model), expected) in enumerate(zip(current, reported)):
        if not isinstance(expected, dict):
            raise RuntimeError("Identity preflight contains an invalid model-role entry.")
        current_identity = {
            "role": role,
            "model_id": model.get("model_id"),
            "requested_model_slug": str(model.get("model_name") or ""),
            "fixed_provider_slug": str(model.get("fixed_provider") or ""),
        }
        expected_identity = {
            "role": expected.get("role"),
            "model_id": expected.get("model_id"),
            "requested_model_slug": expected.get("requested_model_slug"),
            "fixed_provider_slug": expected.get("fixed_provider_slug"),
        }
        if current_identity != expected_identity or model.get("provider", "openrouter") != "openrouter":
            raise RuntimeError(
                f"Current model-role configuration at position {index} no longer matches the identity preflight."
            )
        capability = expected.get("capability")
        if not isinstance(capability, dict):
            raise RuntimeError("Identity preflight lacks the frozen endpoint capability needed for seal validation.")
        if role_parameters(role, capability) != expected.get("generation_parameters"):
            raise RuntimeError("Current role generation parameters no longer match the identity preflight.")
        if expected.get("response_format") != RESPONSE_FORMAT:
            raise RuntimeError("Current response format no longer matches the identity preflight.")


def seal_run_manifest(config: Dict[str, Any], run_dir: Path) -> Dict[str, Any]:
    out = preflight_dir(run_dir)
    initial_path = sealed_manifest_path(run_dir)
    if not initial_path.exists():
        raise RuntimeError("Run creation must write the initial run manifest before model preflight.")
    initial = read_json(initial_path)
    try:
        _, resolved_model_roles = configure_model_profile(config, str(initial.get("model_profile") or ""))
    except ValueError as exc:
        raise RuntimeError("The selected Justice model profile cannot be validated before manifest seal.") from exc
    if resolved_model_roles != initial.get("resolved_model_roles"):
        raise RuntimeError("The selected Justice model profile no longer matches the initial run manifest.")
    report = read_json(out / "preflight_report.json")
    confirmation = read_json(out / "preflight_confirmation.json")
    _validate_preflight_confirmation(report, confirmation)
    _validate_current_model_configuration(config, report)
    if initial.get("sealed") is True:
        raise RuntimeError("The run manifest is already sealed and cannot be modified.")
    if initial.get("run_id") != run_dir.name:
        raise RuntimeError("Initial run manifest identity does not match the run directory.")
    _verify_upstream_checkpoint(initial)
    from .justice_lifecycle import normalize_run_scope
    from .justice_snapshots import brief_snapshot_nodes, verify_brief_snapshot, verify_utterance_snapshot

    brief_manifest = verify_brief_snapshot(str(initial.get("brief_snapshot_id") or ""))
    oral_manifest = verify_utterance_snapshot(str(initial.get("utterance_snapshot_id") or ""))
    if brief_manifest.get("overall_snapshot_sha256") != initial.get("brief_snapshot_hash"):
        raise RuntimeError("Brief snapshot changed before run manifest seal.")
    if oral_manifest.get("overall_snapshot_sha256") != initial.get("utterance_snapshot_hash"):
        raise RuntimeError("Oral-argument snapshot changed before run manifest seal.")
    normalized_scope = normalize_run_scope(initial.get("run_scope") or {}, oral_manifest, brief_manifest)
    if normalized_scope != initial.get("run_scope") or sha256_json(normalized_scope) != initial.get("run_scope_sha256"):
        raise RuntimeError("Run scope changed or failed deterministic validation before manifest seal.")
    selected_briefs = sorted({
        brief_id
        for item in (initial.get("run_scope") or {}).get("oral_argument_brief_scopes") or []
        for brief_id in item.get("brief_ids") or []
    })
    brief_snapshot_nodes(str(initial.get("brief_snapshot_id") or ""), selected_briefs)
    utterance_rows = read_jsonl(run_dir / "04_justice_utterances" / "justice_utterances.jsonl")
    required_identity = ["oyez_argument_id", "utterance_id", "turn_id", "target_justice", "text"]
    utterance_ids = [str(row.get("utterance_id") or "") for row in utterance_rows]
    if (
        not utterance_rows
        or any(any(not str(row.get(field) or "") for field in required_identity) for row in utterance_rows)
        or len(utterance_ids) != len(set(utterance_ids))
    ):
        raise RuntimeError("Run-derived utterances have missing required identity fields or duplicate IDs.")
    selection_path = run_dir / "04_justice_utterances" / "selected_utterance_ids.json"
    if not selection_path.is_file():
        raise RuntimeError("Run-derived selected utterance identity file is missing.")
    selection = read_json(selection_path)
    if selection != {
        "ordered_utterance_ids": utterance_ids,
        "ordered_utterance_ids_sha256": sha256_json(utterance_ids),
    }:
        raise RuntimeError("Run-derived selected utterance identity file is inconsistent.")
    files = controlled_files(config, run_dir)
    file_hashes = {str(path.relative_to(REPO_ROOT) if path.is_relative_to(REPO_ROOT) else path): sha256_file(path) for path in files}
    selected_entries = {
        str(item.get("brief_id")): item for item in brief_manifest.get("briefs") or []
        if str(item.get("brief_id")) in selected_briefs
    }
    derived_hashes = {}
    for relative in [
        "01_processed_transcripts/processed_transcripts.jsonl",
        "04_justice_utterances/justice_utterances.jsonl",
        "04_justice_utterances/selected_utterance_ids.json",
        "04_justice_utterances/local_dialogue_context_audit.jsonl",
    ]:
        path = run_dir / relative
        if not path.is_file():
            raise RuntimeError(f"Run-derived input required before manifest seal is missing: {relative!r}.")
        derived_hashes[relative] = sha256_file(path)
    manifest = {
        **initial,
        "manifest_version": 4,
        "sealed": True,
        "run_id": run_dir.name,
        "preflight": report,
        "probe_prompt": PROBE_PROMPT,
        "contracts": contract_manifest(),
        "contracts_sha256": sha256_json(contract_manifest()),
        "selected_utterance_ids": utterance_ids,
        "selected_utterance_ids_sha256": sha256_json(utterance_ids),
        "selected_brief_ids": selected_briefs,
        "selected_briefs": {
            brief_id: {
                "pdf_sha256": selected_entries[brief_id]["pdf_sha256"],
                "reason_tree_sha256": selected_entries[brief_id]["reason_tree_sha256"],
            }
            for brief_id in selected_briefs
        },
        "selected_reason_trees_aggregate_sha256": sha256_json(
            {brief_id: selected_entries[brief_id]["reason_tree_sha256"] for brief_id in selected_briefs}
        ),
        "derived_file_hashes": derived_hashes,
        "file_hashes": file_hashes,
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "packages": _package_versions(),
            "git": _git_identity(),
            "openrouter_base_url": (config.get("openrouter", {}) or {}).get("base_url", "https://openrouter.ai/api/v1"),
            "openrouter_timeout_seconds": (config.get("openrouter", {}) or {}).get("timeout_seconds", 120),
            "openrouter_max_retries": (config.get("openrouter", {}) or {}).get("max_retries", 2),
        },
        "config_without_secrets": _without_secrets(
            {key: value for key, value in config.items() if key not in {"_config_path"}}
        ),
    }
    path = sealed_manifest_path(run_dir)
    temporary = path.with_suffix(".json.tmp")
    write_json(temporary, manifest)
    os.replace(temporary, path)
    _atomic_write_text(out / "run_manifest.sha256", sha256_file(path))
    return manifest


def verify_manifest_files(manifest: Dict[str, Any]) -> None:
    for label, expected in (manifest.get("file_hashes") or {}).items():
        path = Path(label)
        if not path.is_absolute():
            path = REPO_ROOT / path
        if not path.exists() or sha256_file(path) != expected:
            raise RuntimeError(f"Sealed run content changed or cannot be verified: {label!r}.")


def verify_manifest_environment(manifest: Dict[str, Any]) -> None:
    expected = manifest.get("environment") or {}
    current = {
        "python": sys.version,
        "platform": platform.platform(),
        "packages": _package_versions(),
    }
    for field, value in current.items():
        if expected.get(field) != value:
            raise RuntimeError(f"The current runtime {field} does not match the sealed run manifest.")


def model_manifest_entry(manifest: Dict[str, Any], model_id: str) -> Dict[str, Any]:
    matches = [item for item in (manifest.get("preflight", {}) or {}).get("models", []) if item.get("model_id") == model_id]
    if len(matches) != 1:
        raise RuntimeError(f"Sealed manifest does not contain exactly one identity for model {model_id!r}.")
    return matches[0]


def append_attempt_audit(path: Path, row: Dict[str, Any]) -> None:
    existing = read_jsonl(path) if path.exists() else []
    event_id = str(row.get("audit_event_id") or "")
    if event_id:
        matches = [item for item in existing if str(item.get("audit_event_id") or "") == event_id]
        if matches:
            if len(matches) != 1 or matches[0] != row:
                raise RuntimeError(f"Attempt-audit event {event_id!r} already exists with different content.")
            return
    existing.append(row)
    _atomic_write_text(
        path,
        "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in existing),
    )
