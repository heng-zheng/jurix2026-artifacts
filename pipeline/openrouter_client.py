from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import quote

import requests

from .common import append_log, sleep_retry


class BYOKRequiredError(RuntimeError):
    pass


class RecoverableOpenRouterError(RuntimeError):
    """An external OpenRouter condition can be retried later without changing the request."""

    pass


class ResponseContractError(RuntimeError):
    """A completed response cannot be accepted and must not be retried as an API failure."""

    pass


class OpenRouterClient:
    def __init__(self, config: Dict[str, Any]) -> None:
        openrouter = config.get("openrouter", {})
        self.base_url = openrouter.get("base_url", "https://openrouter.ai/api/v1").rstrip("/")
        api_key_env = openrouter.get("api_key_env", "OPENROUTER_API_KEY")
        self.api_key = os.environ.get(api_key_env)
        self.api_key_env = api_key_env
        self.max_retries = int(openrouter.get("max_retries", 2))
        self.timeout = int(openrouter.get("timeout_seconds", 120))

    def validate(self) -> None:
        if not self.api_key:
            raise RecoverableOpenRouterError(
                f"Missing OpenRouter API key. Set {self.api_key_env} in .env before running LLM modules."
            )

    @staticmethod
    def _immediate_pause_error(message: str, status_code: Optional[int] = None) -> bool:
        text = message.lower()
        markers = [
            "insufficient credit", "insufficient balance", "out of credit", "balance",
            "quota exceeded", "quota exhausted", "invalid api key", "authentication",
            "unauthorized", "forbidden", "byok",
        ]
        return status_code in {401, 402, 403} or any(marker in text for marker in markers)

    def chat(
        self,
        model_name: str,
        prompt: str,
        raw_response_path: Path,
        error_log_path: Path,
        require_byok: bool = False,
        provider_order: Optional[List[str]] = None,
        response_format: Optional[Dict[str, Any]] = None,
        provider_only: Optional[List[str]] = None,
        require_parameters: bool = False,
        request_parameters: Optional[Dict[str, Any]] = None,
        require_generation_metadata: bool = False,
        include_default_temperature: bool = True,
    ) -> Dict[str, Any]:
        self.validate()
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://local.scotus-oralarg-stance-detection",
            "X-Title": "SCOTUS oral argument stance detection",
            "X-OpenRouter-Metadata": "enabled",
        }
        payload = {
            "model": model_name,
            "messages": [{"role": "user", "content": prompt}],
        }
        if include_default_temperature:
            payload["temperature"] = 0
        payload.update(request_parameters or {})
        if response_format:
            payload["response_format"] = response_format
        if provider_only:
            if len(provider_only) != 1:
                raise ValueError("Stage annotation calls must lock exactly one OpenRouter provider endpoint.")
            payload["provider"] = {
                "only": list(provider_only),
                "allow_fallbacks": False,
                "require_parameters": bool(require_parameters),
            }
        elif provider_order:
            payload["provider"] = {
                "order": provider_order,
                "allow_fallbacks": False,
            }
        pending_path = raw_response_path.with_name(f"{raw_response_path.stem}.pending_metadata.json")
        metadata_state_path = raw_response_path.with_name(f"{raw_response_path.stem}.metadata_query_state.json")
        if require_generation_metadata and pending_path.exists():
            pending = json.loads(pending_path.read_text(encoding="utf-8"))
            if pending.get("request_payload") != payload:
                raise ResponseContractError("Same-run metadata resume refused because the request payload changed.")
            if pending.get("status") == "complete":
                return pending["result"]
            if pending.get("status") == "hard_failure":
                raise ResponseContractError(str(pending.get("error") or "Generation metadata hard failure."))
            result = pending.get("result") or {}
            generation_id = str(result.get("generation_id") or "")
            if not generation_id:
                raise ResponseContractError("Pending response has no generation ID; the model response cannot be reused.")
            try:
                metadata, metadata_attempts = self.generation_metadata(
                    generation_id,
                    error_log_path,
                    state_path=metadata_state_path,
                )
            except RecoverableOpenRouterError:
                raise
            result["generation_metadata"] = metadata
            result["generation_metadata_attempts"] = metadata_attempts
            result["generation_metadata_query_state_path"] = str(metadata_state_path)
            result["generation_metadata_query_attempt_paths"] = [
                str(path) for path in sorted(
                    metadata_state_path.parent.glob(f"{metadata_state_path.stem}.attempt_*.json")
                )
            ] + [
                str(path) for path in sorted(
                    metadata_state_path.parent.glob(f"{metadata_state_path.stem}.episode_*.attempt_*.json")
                )
            ]
            result["http_request_state_path"] = str(
                raw_response_path.with_name(f"{raw_response_path.stem}.http_request_state.json")
            )
            result["http_attempt_paths"] = [
                str(path) for path in sorted(
                    raw_response_path.parent.glob(f"{raw_response_path.stem}.http_*attempt_*{raw_response_path.suffix}")
                )
            ]
            result["actual_provider"] = (metadata.get("data") or {}).get("provider_name", "")
            result["actual_model"] = (metadata.get("data") or {}).get("model") or result.get("actual_model", "")
            if not result["actual_provider"]:
                pending_path.write_text(
                    json.dumps({"status": "hard_failure", "request_payload": payload,
                                "error": "OpenRouter generation metadata did not provide actual provider identity."}, indent=2),
                    encoding="utf-8",
                )
                raise ResponseContractError("OpenRouter generation metadata did not provide actual provider identity.")
            pending_path.write_text(
                json.dumps({"status": "complete", "request_payload": payload, "result": result}, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            return result
        http_state_path = raw_response_path.with_name(f"{raw_response_path.stem}.http_request_state.json")
        http_state_path.parent.mkdir(parents=True, exist_ok=True)
        http_state: Dict[str, Any] = {}
        if require_generation_metadata and http_state_path.exists():
            http_state = json.loads(http_state_path.read_text(encoding="utf-8"))
            if http_state.get("request_payload") != payload:
                raise ResponseContractError("Same-run HTTP resume refused because the request payload changed.")
            if http_state.get("status") == "hard_failure":
                raise ResponseContractError(str(http_state.get("error") or "Response contract hard failure."))
            if http_state.get("status") == "api_failed":
                previous_episode = int(http_state.get("episode_number", 1))
                episodes = list(http_state.get("episodes") or [])
                episodes.append(
                    {
                        "episode_number": previous_episode,
                        "attempts_used": int(http_state.get("attempts_used", 0)),
                        "error": str(http_state.get("error") or ""),
                    }
                )
                http_state = {
                    "status": "retry_episode_started",
                    "request_payload": payload,
                    "attempts_used": 0,
                    "episode_number": previous_episode + 1,
                    "episodes": episodes,
                }
                http_state_path.write_text(
                    json.dumps(http_state, ensure_ascii=False, indent=2), encoding="utf-8"
                )
        attempts_used = int(http_state.get("attempts_used", 0)) if require_generation_metadata else 0
        episode_number = int(http_state.get("episode_number", 1)) if require_generation_metadata else 1
        attempt_limit = self.max_retries + 1
        if attempts_used >= attempt_limit:
            raise RecoverableOpenRouterError("OpenRouter API retries were already exhausted in this episode.")
        last_error = None
        for attempt in range(attempts_used, attempt_limit):
            attempt_path = (
                raw_response_path.with_name(
                    (
                        f"{raw_response_path.stem}.http_attempt_{attempt + 1:02d}"
                        if episode_number == 1
                        else f"{raw_response_path.stem}.http_episode_{episode_number:03d}_attempt_{attempt + 1:02d}"
                    )
                    + raw_response_path.suffix
                )
                if require_generation_metadata
                else None
            )
            if attempt_path is not None:
                attempt_path.parent.mkdir(parents=True, exist_ok=True)
                attempt_path.write_text(
                    json.dumps(
                        {
                            "status": "request_started",
                            "attempt_number": attempt + 1,
                            "episode_number": episode_number,
                            "model": model_name,
                            "fixed_provider": list(provider_only or []),
                        },
                        ensure_ascii=False,
                        indent=2,
                    ),
                    encoding="utf-8",
                )
            try:
                if require_generation_metadata:
                    http_state_path.write_text(
                        json.dumps({"status": "requesting", "request_payload": payload,
                                    "attempts_used": attempt + 1, "episode_number": episode_number,
                                    "episodes": http_state.get("episodes", [])}, ensure_ascii=False, indent=2),
                        encoding="utf-8",
                    )
                response = requests.post(
                    f"{self.base_url}/chat/completions",
                    headers=headers,
                    json=payload,
                    timeout=self.timeout,
                )
                raw_text = response.text
                if not require_generation_metadata:
                    raw_response_path.parent.mkdir(parents=True, exist_ok=True)
                    raw_response_path.write_text(raw_text, encoding="utf-8")
                try:
                    response.raise_for_status()
                except Exception:
                    if attempt_path is not None:
                        attempt_path.write_text(raw_text, encoding="utf-8")
                    status_code = getattr(response, "status_code", None)
                    if self._immediate_pause_error(raw_text, status_code):
                        raise RecoverableOpenRouterError(
                            f"OpenRouter account or authentication error (HTTP {status_code}): {raw_text}"
                        )
                    if isinstance(status_code, int) and 400 <= status_code < 500 and status_code != 429:
                        raise ResponseContractError(
                            f"OpenRouter non-retryable HTTP {status_code}: {raw_text}"
                        )
                    raise
                data = response.json()
                if data.get("error"):
                    if attempt_path is not None:
                        attempt_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
                    error = data["error"]
                    provider_message = (
                        f"OpenRouter provider error {error.get('code', '')}: {error.get('message', error)}"
                    )
                    if self._immediate_pause_error(provider_message):
                        raise RecoverableOpenRouterError(provider_message)
                    raise RuntimeError(provider_message)
                choices = data.get("choices")
                unexpected_reasoning: List[str] = []
                if require_generation_metadata and isinstance(choices, list):
                    for choice_index, candidate in enumerate(choices):
                        if not isinstance(candidate, dict):
                            continue
                        message = candidate.get("message") if isinstance(candidate.get("message"), dict) else {}
                        reasoning_fields = [
                            field
                            for field in list(message)
                            if "reasoning" in str(field).lower()
                            or str(field).lower() in {"thinking", "thoughts", "analysis"}
                        ]
                        for field in reasoning_fields:
                            if field in message and message.get(field) not in (None, "", []):
                                unexpected_reasoning.append(f"choices[{choice_index}].message.{field}")
                                message.pop(field, None)
                        candidate["message"] = message
                if attempt_path is not None:
                    attempt_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
                if require_generation_metadata:
                    raw_response_path.parent.mkdir(parents=True, exist_ok=True)
                    raw_response_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
                if require_generation_metadata and (not isinstance(choices, list) or len(choices) != 1):
                    raise ResponseContractError(
                        f"OpenRouter response contract failure: expected exactly one choice, received "
                        f"{len(choices) if isinstance(choices, list) else 0}."
                    )
                choice = choices[0] if isinstance(choices, list) and choices else {}
                finish_reason = choice.get("finish_reason")
                if require_generation_metadata and finish_reason == "error":
                    raise RuntimeError("OpenRouter returned finish_reason=error.")
                message = choice.get("message") if isinstance(choice.get("message"), dict) else {}
                content = message.get("content", "")
                usage = data.get("usage", {}) or {}
                if require_byok and usage.get("is_byok") is not True:
                    message = (
                        f"BYOK is required, but OpenRouter reported is_byok={usage.get('is_byok')!r}. "
                        "Enable 'Always use for this provider' for the OpenAI BYOK key before retrying."
                    )
                    append_log(error_log_path, message)
                    raise BYOKRequiredError(message)
                result = {
                    "content": content,
                    "raw_response": data,
                    "token_usage": usage,
                    "estimated_cost": data.get("estimated_cost") or data.get("cost"),
                    "finish_reason": finish_reason,
                    "generation_id": data.get("id") or response.headers.get("X-Generation-Id", ""),
                    "actual_model": data.get("model", ""),
                    "system_fingerprint": data.get("system_fingerprint", "not_returned") or "not_returned",
                    "response_headers": {
                        key: value
                        for key, value in response.headers.items()
                        if key.lower() not in {"authorization", "set-cookie"}
                    },
                    "actual_choice_count": len(choices) if isinstance(choices, list) else 0,
                    "request_payload": payload,
                    "http_attempt_count": attempt + 1,
                }
                if unexpected_reasoning:
                    result["removed_reasoning_fields"] = unexpected_reasoning
                if require_generation_metadata:
                    generation_id = str(result["generation_id"] or "")
                    if not generation_id:
                        raise ResponseContractError("OpenRouter response did not provide a generation ID required for audit.")
                    pending_path.write_text(
                        json.dumps({"status": "metadata_pending", "request_payload": payload, "result": result},
                                   ensure_ascii=False, indent=2),
                        encoding="utf-8",
                    )
                    response_state = {
                        "status": "response_received",
                        "request_payload": payload,
                        "attempts_used": attempt + 1,
                        "episode_number": episode_number,
                        "episodes": http_state.get("episodes", []),
                    }
                    if unexpected_reasoning:
                        response_state["removed_reasoning_fields"] = unexpected_reasoning
                    http_state_path.write_text(
                        json.dumps(response_state, ensure_ascii=False, indent=2),
                        encoding="utf-8",
                    )
                    try:
                        metadata, metadata_attempts = self.generation_metadata(
                            generation_id,
                            error_log_path,
                            state_path=metadata_state_path,
                        )
                    except RecoverableOpenRouterError:
                        raise
                    result["generation_metadata"] = metadata
                    result["generation_metadata_attempts"] = metadata_attempts
                    result["generation_metadata_query_state_path"] = str(metadata_state_path)
                    result["generation_metadata_query_attempt_paths"] = [
                        str(path) for path in sorted(
                            metadata_state_path.parent.glob(f"{metadata_state_path.stem}.attempt_*.json")
                        )
                    ] + [
                        str(path) for path in sorted(
                            metadata_state_path.parent.glob(
                                f"{metadata_state_path.stem}.episode_*.attempt_*.json"
                            )
                        )
                    ]
                    result["http_request_state_path"] = str(http_state_path)
                    result["http_attempt_paths"] = [
                        str(path) for path in sorted(
                            raw_response_path.parent.glob(
                                f"{raw_response_path.stem}.http_*attempt_*{raw_response_path.suffix}"
                            )
                        )
                    ]
                    result["actual_provider"] = (metadata.get("data") or {}).get("provider_name", "")
                    if not result["actual_provider"]:
                        pending_path.write_text(
                            json.dumps({"status": "hard_failure", "request_payload": payload,
                                        "error": "OpenRouter generation metadata did not provide actual provider identity."}, indent=2),
                            encoding="utf-8",
                        )
                        raise ResponseContractError("OpenRouter generation metadata did not provide actual provider identity.")
                    result["actual_model"] = (metadata.get("data") or {}).get("model") or result["actual_model"]
                    pending_path.write_text(
                        json.dumps({"status": "complete", "request_payload": payload, "result": result},
                                   ensure_ascii=False, indent=2),
                        encoding="utf-8",
                    )
                return result
            except BYOKRequiredError as exc:
                if require_generation_metadata:
                    http_state_path.write_text(
                        json.dumps(
                            {
                                "status": "api_failed",
                                "request_payload": payload,
                                "attempts_used": attempt + 1,
                                "episode_number": episode_number,
                                "episodes": http_state.get("episodes", []),
                                "error": str(exc),
                            },
                            ensure_ascii=False,
                            indent=2,
                        ),
                        encoding="utf-8",
                    )
                raise
            except ResponseContractError as exc:
                if require_generation_metadata:
                    http_state_path.write_text(
                        json.dumps({"status": "hard_failure", "request_payload": payload,
                                    "attempts_used": attempt + 1, "error": str(exc)}, ensure_ascii=False, indent=2),
                        encoding="utf-8",
                    )
                raise
            except Exception as exc:
                last_error = exc
                if attempt_path is not None and not attempt_path.exists():
                    attempt_path.write_text(
                        json.dumps({"request_attempt_error": str(exc)}, ensure_ascii=False, indent=2),
                        encoding="utf-8",
                    )
                append_log(error_log_path, f"OpenRouter call failed on attempt {attempt + 1}: {exc}")
                if require_generation_metadata:
                    immediate = isinstance(exc, RecoverableOpenRouterError) and self._immediate_pause_error(str(exc))
                    status = "api_failed" if immediate or attempt + 1 >= attempt_limit else "retry_pending"
                    http_state_path.write_text(
                        json.dumps({"status": status, "request_payload": payload,
                                    "attempts_used": attempt + 1, "episode_number": episode_number,
                                    "episodes": http_state.get("episodes", []),
                                    "error": str(exc)}, ensure_ascii=False, indent=2),
                        encoding="utf-8",
                    )
                if immediate:
                    raise RecoverableOpenRouterError(str(exc)) from exc
                if attempt + 1 < attempt_limit:
                    if require_generation_metadata:
                        time.sleep([5, 15, 30][min(attempt, 2)])
                    else:
                        sleep_retry(attempt)
        raw_response_path.parent.mkdir(parents=True, exist_ok=True)
        if not raw_response_path.exists():
            raw_response_path.write_text(json.dumps({"error": str(last_error)}, indent=2), encoding="utf-8")
        raise RecoverableOpenRouterError(f"OpenRouter call failed after retries: {last_error}")

    def _auth_headers(self) -> Dict[str, str]:
        self.validate()
        return {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}

    def _get_with_metadata_retries(
        self,
        url: str,
        error_log_path: Path,
        *,
        params: Optional[Dict[str, Any]] = None,
        label: str,
        state_path: Optional[Path] = None,
    ) -> tuple[Dict[str, Any], int]:
        waits = [5, 15, 30]
        last_error: Optional[Exception] = None
        state: Dict[str, Any] = {}
        if state_path is not None and state_path.exists():
            state = json.loads(state_path.read_text(encoding="utf-8"))
            if state.get("url") != url or state.get("params") != (params or {}):
                raise RuntimeError(f"{label} resume state does not match the current query.")
            if state.get("status") == "complete":
                return state["data"], int(state.get("attempts_used", 0))
            if state.get("status") == "query_failed":
                previous_episode = int(state.get("episode_number", 1))
                episodes = list(state.get("episodes") or [])
                episodes.append(
                    {
                        "episode_number": previous_episode,
                        "attempts_used": int(state.get("attempts_used", 0)),
                        "error": str(state.get("error") or ""),
                    }
                )
                state = {
                    "status": "retry_episode_started",
                    "url": url,
                    "params": params or {},
                    "attempts_used": 0,
                    "episode_number": previous_episode + 1,
                    "episodes": episodes,
                }
        attempts_used = int(state.get("attempts_used", 0))
        episode_number = int(state.get("episode_number", 1))
        if attempts_used >= 4:
            raise RecoverableOpenRouterError(f"{label} query already exhausted four attempts in this episode.")
        for attempt in range(attempts_used, 4):
            attempt_path = (
                state_path.with_name(
                    (
                        f"{state_path.stem}.attempt_{attempt + 1:02d}.json"
                        if episode_number == 1
                        else f"{state_path.stem}.episode_{episode_number:03d}.attempt_{attempt + 1:02d}.json"
                    )
                )
                if state_path is not None
                else None
            )
            response: Optional[requests.Response] = None
            try:
                if attempt_path is not None:
                    attempt_path.parent.mkdir(parents=True, exist_ok=True)
                    attempt_path.write_text(
                        json.dumps(
                            {
                                "status": "request_started",
                                "url": url,
                                "params": params or {},
                                "attempt_number": attempt + 1,
                                "episode_number": episode_number,
                            },
                            ensure_ascii=False,
                            indent=2,
                        ),
                        encoding="utf-8",
                    )
                if state_path is not None:
                    state_path.parent.mkdir(parents=True, exist_ok=True)
                    state_path.write_text(
                        json.dumps({"status": "querying", "url": url, "params": params or {},
                                    "attempts_used": attempt + 1, "episode_number": episode_number,
                                    "episodes": state.get("episodes", [])}, indent=2),
                        encoding="utf-8",
                    )
                response = requests.get(url, headers=self._auth_headers(), params=params, timeout=self.timeout)
                response.raise_for_status()
                data = response.json()
                if attempt_path is not None:
                    attempt_path.write_text(
                        json.dumps(
                            {
                                "status": "response_received",
                                "url": url,
                                "params": params or {},
                                "http_status": getattr(response, "status_code", None),
                                "response_headers": {
                                    key: value
                                    for key, value in response.headers.items()
                                    if key.lower() not in {"authorization", "set-cookie"}
                                },
                                "response": data,
                            },
                            ensure_ascii=False,
                            indent=2,
                        ),
                        encoding="utf-8",
                    )
                if data.get("error"):
                    raise RuntimeError(str(data["error"]))
                if state_path is not None:
                    state_path.write_text(
                        json.dumps({"status": "complete", "url": url, "params": params or {},
                                    "attempts_used": attempt + 1, "episode_number": episode_number,
                                    "episodes": state.get("episodes", []), "data": data}, ensure_ascii=False, indent=2),
                        encoding="utf-8",
                    )
                return data, attempt + 1
            except Exception as exc:
                last_error = exc
                if attempt_path is not None:
                    error_record: Dict[str, Any] = {
                        "status": "query_error",
                        "url": url,
                        "params": params or {},
                        "error": str(exc),
                    }
                    if response is not None:
                        error_record.update(
                            {
                                "http_status": getattr(response, "status_code", None),
                                "response_headers": {
                                    key: value
                                    for key, value in response.headers.items()
                                    if key.lower() not in {"authorization", "set-cookie"}
                                },
                                "raw_response_text": response.text,
                            }
                        )
                    attempt_path.write_text(
                        json.dumps(error_record, ensure_ascii=False, indent=2),
                        encoding="utf-8",
                    )
                append_log(error_log_path, f"{label} query failed on attempt {attempt + 1}: {exc}")
                if self._immediate_pause_error(
                    str(exc), getattr(response, "status_code", None) if response is not None else None
                ):
                    if state_path is not None:
                        state_path.write_text(
                            json.dumps(
                                {
                                    "status": "query_failed", "url": url, "params": params or {},
                                    "attempts_used": attempt + 1, "episode_number": episode_number,
                                    "episodes": state.get("episodes", []), "error": str(exc),
                                },
                                ensure_ascii=False,
                                indent=2,
                            ),
                            encoding="utf-8",
                        )
                    raise RecoverableOpenRouterError(str(exc)) from exc
                if attempt < 3:
                    time.sleep(waits[attempt])
        if state_path is not None:
            state_path.write_text(
                json.dumps(
                    {
                        "status": "query_failed", "url": url, "params": params or {},
                        "attempts_used": 4, "episode_number": episode_number,
                        "episodes": state.get("episodes", []), "error": str(last_error),
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
        raise RecoverableOpenRouterError(f"{label} query failed after four attempts: {last_error}")

    def generation_metadata(
        self,
        generation_id: str,
        error_log_path: Path,
        state_path: Optional[Path] = None,
    ) -> tuple[Dict[str, Any], int]:
        return self._get_with_metadata_retries(
            f"{self.base_url}/generation",
            error_log_path,
            params={"id": generation_id},
            label="OpenRouter generation metadata",
            state_path=state_path,
        )

    def capability_metadata(
        self,
        model_name: str,
        provider_slug: str,
        error_log_path: Path,
        state_dir: Optional[Path] = None,
    ) -> Dict[str, Any]:
        if "/" not in model_name:
            raise ValueError("OpenRouter model names must be exact author/model slugs.")
        author, slug = model_name.split("/", 1)
        model_data, model_attempts = self._get_with_metadata_retries(
            f"{self.base_url}/model/{quote(author, safe='')}/{quote(slug, safe=':')}",
            error_log_path,
            label="OpenRouter model capability metadata",
            state_path=(state_dir / "model_capability_query_state.json") if state_dir is not None else None,
        )
        endpoints_data, endpoint_attempts = self._get_with_metadata_retries(
            f"{self.base_url}/models/{quote(author, safe='')}/{quote(slug, safe=':')}/endpoints",
            error_log_path,
            label="OpenRouter endpoint capability metadata",
            state_path=(state_dir / "endpoint_capability_query_state.json") if state_dir is not None else None,
        )
        endpoints = (endpoints_data.get("data") or {}).get("endpoints") or endpoints_data.get("data") or []
        if not isinstance(endpoints, list):
            endpoints = []
        matching = [
            endpoint for endpoint in endpoints
            if str(endpoint.get("tag") or "") == provider_slug
        ]
        if len(matching) != 1:
            raise RuntimeError(
                "OpenRouter capability metadata must contain exactly one endpoint whose tag "
                f"equals {provider_slug!r} for model {model_name!r}; found {len(matching)}."
            )
        return {
            "model_raw": model_data,
            "endpoints_raw": endpoints_data,
            "model_query_attempts": model_attempts,
            "endpoint_query_attempts": endpoint_attempts,
            "model_query_attempt_paths": (
                [str(path) for path in sorted(state_dir.glob("model_capability_query_state.attempt_*.json"))]
                if state_dir is not None
                else []
            ),
            "endpoint_query_attempt_paths": (
                [str(path) for path in sorted(state_dir.glob("endpoint_capability_query_state.attempt_*.json"))]
                if state_dir is not None
                else []
            ),
            "selected_endpoint": matching[0],
        }
