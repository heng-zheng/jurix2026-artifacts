import json
import os
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from pipeline.common import load_config
from pipeline.justice_runtime import (
    APPROVED_MODEL_PROFILES,
    ADJUDICATOR_PARAMETERS,
    ANNOTATOR_PARAMETERS,
    RESPONSE_FORMAT,
    append_attempt_audit,
    _without_secrets,
    canonical_capability,
    configure_model_profile,
    recover_missing_manifest_checksum,
    request_parameters,
    role_parameters,
    run_identity_preflight,
    seal_run_manifest,
    require_sealed_manifest,
    sha256_file,
    sha256_json,
)
from pipeline.openrouter_client import (
    OpenRouterClient,
    RecoverableOpenRouterError,
    ResponseContractError,
)


class FakeResponse:
    def __init__(self, payload, status=200, headers=None):
        self.payload = payload
        self.status = status
        self.headers = headers or {}
        self.text = json.dumps(payload)

    def raise_for_status(self):
        if self.status >= 400:
            raise RuntimeError(f"HTTP {self.status}")

    def json(self):
        return self.payload


def capability(*, output_parameter="max_tokens", limit=8192, efforts=None):
    result = {
        "supported_parameters": ["reasoning", "response_format", output_parameter],
        "max_completion_tokens": limit,
    }
    if efforts is not None:
        result["supported_reasoning_efforts"] = efforts
    return result


def frontier_config():
    return {
        "annotation_workflow": {
            "profiles": APPROVED_MODEL_PROFILES,
            "annotator_models": ["gpt_5_2", "gemini_2_5_pro"],
            "adjudicator_model": "claude_opus_4_5",
            "selected_profile": "frontier",
        },
        "models": {
            "annotation": {
                "gpt_5_2": {
                    "provider": "openrouter", "model_name": "openai/gpt-5.2", "fixed_provider": "openai"
                },
                "gemini_2_5_pro": {
                    "provider": "openrouter", "model_name": "google/gemini-2.5-pro",
                    "fixed_provider": "google-ai-studio",
                },
            },
            "adjudication": {
                "claude_opus_4_5": {
                    "provider": "openrouter", "model_name": "anthropic/claude-opus-4.5",
                    "fixed_provider": "anthropic",
                },
            },
        },
    }


class RuntimeParameterTests(unittest.TestCase):
    def _new_manifest_fixture(self, run_dir, config):
        preflight = run_dir / "00_run_preflight"
        preflight.mkdir(parents=True)
        config.update(frontier_config())
        _, resolved_model_roles = configure_model_profile(config, "frontier")
        models = []
        for role, model_id, slug, provider in [
            ("annotator", "gpt_5_2", "openai/gpt-5.2", "openai"),
            ("annotator", "gemini_2_5_pro", "google/gemini-2.5-pro", "google-ai-studio"),
            ("adjudicator", "claude_opus_4_5", "anthropic/claude-opus-4.5", "anthropic"),
        ]:
            model_capability = capability()
            models.append({
                "role": role,
                "model_id": model_id,
                "requested_model_slug": slug,
                "actual_model": f"actual-{model_id}",
                "fixed_provider_slug": provider,
                "actual_provider_name": f"Provider {model_id}",
                "capability": model_capability,
                "generation_parameters": role_parameters(role, model_capability),
                "response_format": RESPONSE_FORMAT,
            })
        (preflight / "preflight_report.json").write_text(json.dumps({"models": models}))
        (preflight / "preflight_confirmation.json").write_text(json.dumps({
            "confirmed": True,
            "models": [
                {
                    "model_id": item["model_id"],
                    "requested_model_slug": item["requested_model_slug"],
                    "expected_actual_model": item["actual_model"],
                    "fixed_provider_slug": item["fixed_provider_slug"],
                    "expected_provider_name": item["actual_provider_name"],
                }
                for item in models
            ],
        }))
        scope = {
            "selected_oyez_argument_ids": ["25500"],
            "oral_argument_brief_scopes": [{"oyez_argument_id": "25500", "brief_ids": ["b1"]}],
            "phase_times_by_argument": {},
            "annotation_scope": {
                "target_justices": ["Justice Test"],
                "advocate_group_limits": {"pet": "all"},
            },
        }
        (preflight / "run_manifest.json").write_text(json.dumps({
            "manifest_version": 4, "sealed": False, "run_id": run_dir.name,
            "model_profile": "frontier", "resolved_model_roles": resolved_model_roles,
            "brief_snapshot_id": "brief-snapshot", "brief_snapshot_hash": "brief-hash",
            "utterance_snapshot_id": "oral-snapshot", "utterance_snapshot_hash": "oral-hash",
            "run_scope": scope, "run_scope_sha256": sha256_json(scope), "start_stage": "stage1",
        }))
        utterance_dir = run_dir / "04_justice_utterances"
        utterance_dir.mkdir(parents=True)
        (utterance_dir / "justice_utterances.jsonl").write_text(
            json.dumps({
                "oyez_argument_id": "25500", "utterance_id": "u1", "turn_id": "25500__turn_000001",
                "target_justice": "Justice Test", "text": "Question",
            }) + "\n"
        )
        (utterance_dir / "selected_utterance_ids.json").write_text(json.dumps({
            "ordered_utterance_ids": ["u1"],
            "ordered_utterance_ids_sha256": sha256_json(["u1"]),
        }))
        (utterance_dir / "local_dialogue_context_audit.jsonl").write_text("")
        processed = run_dir / "01_processed_transcripts" / "processed_transcripts.jsonl"
        processed.parent.mkdir(parents=True)
        processed.write_text("{}\n")
        (run_dir / "run_scope.json").write_text(json.dumps(scope))
        brief_manifest = {
            "overall_snapshot_sha256": "brief-hash",
            "briefs": [{"brief_id": "b1", "brief_group": "pet", "pdf_sha256": "pdf", "reason_tree_sha256": "tree"}],
        }
        oral_manifest = {
            "overall_snapshot_sha256": "oral-hash",
            "oral_arguments": [{"oyez_argument_id": "25500"}],
        }
        tree = [{
            "brief_id": "b1", "reason_target_id": "r1", "heading_path": "I", "heading_text": "Root",
            "heading_level": 1, "parent_reason_target_id": "", "direct_child_reason_target_ids": [],
        }]
        return brief_manifest, oral_manifest, tree

    def test_role_parameters_are_frozen_by_role(self):
        annotator = role_parameters("annotator", capability())
        adjudicator = role_parameters("adjudicator", capability())
        self.assertEqual(annotator["max_tokens"], 4096)
        self.assertNotIn("max_completion_tokens", annotator)
        self.assertEqual(annotator["reasoning"], {"exclude": True})
        self.assertEqual(adjudicator["max_tokens"], 8192)
        self.assertNotIn("max_completion_tokens", adjudicator)
        self.assertEqual(adjudicator["reasoning"], {"exclude": True})
        for field in [
            "temperature", "top_p", "seed", "n", "reasoning_effort", "reasoning_max_tokens",
            "reasoning_enabled",
        ]:
            self.assertEqual(annotator[f"{field}_status"], "not_sent_by_policy")
            self.assertNotIn(field, request_parameters(annotator))

    def test_supported_efforts_seed_and_n_do_not_change_request_policy(self):
        parameters = role_parameters("annotator", {
            **capability(efforts=["high", "medium", "low"]),
            "supported_parameters": [
                "reasoning", "response_format", "max_tokens", "seed", "n", "reasoning_effort",
            ],
        })
        request = request_parameters(parameters)
        self.assertEqual(request, {"max_tokens": 4096, "reasoning": {"exclude": True}})

    def test_missing_endpoint_parameter_or_token_limit_hard_fails(self):
        with self.assertRaises(RuntimeError):
            role_parameters("annotator", {"supported_parameters": [], "max_completion_tokens": 4096})
        with self.assertRaises(RuntimeError):
            role_parameters("annotator", {"supported_parameters": ["reasoning", "max_tokens"], "max_completion_tokens": 4096})
        with self.assertRaises(RuntimeError):
            role_parameters("annotator", {"supported_parameters": ["reasoning", "response_format"], "max_completion_tokens": 4096})
        with self.assertRaises(RuntimeError):
            role_parameters("adjudicator", capability(limit=4096))
        with self.assertRaises(RuntimeError):
            role_parameters("annotator", capability(limit=None))

        max_tokens_parameters = role_parameters("annotator", capability(output_parameter="max_tokens"))
        self.assertEqual(max_tokens_parameters["max_tokens"], 4096)
        self.assertNotIn("max_completion_tokens", max_tokens_parameters)

        max_completion_parameters = role_parameters(
            "annotator", capability(output_parameter="max_completion_tokens")
        )
        self.assertEqual(max_completion_parameters["max_completion_tokens"], 4096)
        self.assertNotIn("max_tokens", max_completion_parameters)

        both_capability = capability(output_parameter="max_tokens")
        both_capability["supported_parameters"].append("max_completion_tokens")
        both_parameters = role_parameters("annotator", both_capability)
        self.assertEqual(both_parameters["max_completion_tokens"], 4096)
        self.assertNotIn("max_tokens", both_parameters)

    def test_canonical_capability_uses_endpoint_only(self):
        metadata = {
            "model_raw": {"data": {"id": "m", "canonical_slug": "a/m", "supported_parameters": ["temperature", "seed", "reasoning"]}},
            "selected_endpoint": {
                "tag": "provider-slug", "provider_name": "Provider",
                "supported_parameters": ["max_tokens", "response_format", "reasoning"],
                "supported_reasoning_efforts": ["medium"], "max_completion_tokens": 4096,
            },
        }
        result = canonical_capability(metadata, "provider-slug")
        self.assertEqual(result["supported_parameters"], ["max_tokens", "reasoning", "response_format"])
        self.assertEqual(result["provider_slug"], "provider-slug")
        self.assertEqual(result["endpoint_tag"], "provider-slug")
        self.assertNotIn("supported_reasoning_efforts", result)
        metadata["selected_endpoint"] = {"provider_name": "Provider"}
        result = canonical_capability(metadata, "provider-slug")
        self.assertEqual(result["supported_parameters"], [])
        self.assertIsNone(result["max_completion_tokens"])

    def test_all_approved_profiles_resolve_exact_role_identities(self):
        config = frontier_config()
        config["models"]["annotation"].update({
            "gpt_5_6_luna": {
                "provider": "openrouter", "model_name": "openai/gpt-5.6-luna", "fixed_provider": "openai"
            },
            "gemini_3_5_flash": {
                "provider": "openrouter", "model_name": "google/gemini-3.5-flash",
                "fixed_provider": "google-ai-studio",
            },
            "gpt_5_4_mini": {
                "provider": "openrouter", "model_name": "openai/gpt-5.4-mini", "fixed_provider": "openai"
            },
            "gemini_3_flash_preview": {
                "provider": "openrouter", "model_name": "google/gemini-3-flash-preview",
                "fixed_provider": "google-ai-studio",
            },
        })
        config["models"]["adjudication"]["claude_sonnet_5"] = {
            "provider": "openrouter", "model_name": "anthropic/claude-sonnet-5", "fixed_provider": "anthropic"
        }
        for profile_name, expected in APPROVED_MODEL_PROFILES.items():
            effective, resolved = configure_model_profile(config, profile_name)
            self.assertEqual(effective["annotation_workflow"]["annotator_models"], expected["annotator_models"])
            self.assertEqual(effective["annotation_workflow"]["adjudicator_model"], expected["adjudicator_model"])
            self.assertEqual([row["role"] for row in resolved], ["annotator_a", "annotator_b", "adjudicator"])
        with self.assertRaises(ValueError):
            configure_model_profile(config, "unknown")

    def test_deposit_config_defines_every_approved_profile(self):
        config = load_config("config/case_sets/justice_2024.yaml")
        for profile_name in APPROVED_MODEL_PROFILES:
            effective, resolved = configure_model_profile(config, profile_name)
            self.assertEqual(effective["annotation_workflow"]["selected_profile"], profile_name)
            self.assertEqual(len(resolved), 3)

    def test_each_profile_preflight_uses_default_reasoning_policy(self):
        class FakeClient:
            def __init__(self, model):
                self.model = model

            def capability_metadata(self, model_name, provider_slug, error_log_path, state_dir=None):
                return {
                    "model_raw": {"data": {"id": model_name, "canonical_slug": model_name}},
                    "selected_endpoint": {
                        "tag": provider_slug,
                        "provider_name": provider_slug,
                        "supported_parameters": ["reasoning", "response_format", "max_tokens"],
                        "max_completion_tokens": 128000,
                    },
                }

            def chat(self, model_name, prompt, raw_response_path, error_log_path, **kwargs):
                raw_response_path.parent.mkdir(parents=True, exist_ok=True)
                raw_response_path.write_text('{"probe":"ok"}')
                calls.append(kwargs)
                return {
                    "content": '{"probe":"ok"}', "finish_reason": "stop", "actual_model": model_name,
                    "actual_provider": kwargs["provider_only"][0], "generation_id": "g",
                    "actual_choice_count": 1, "response_headers": {}, "request_payload": {},
                }

        config_source = load_config("config/case_sets/justice_2024.yaml")
        for profile_name in APPROVED_MODEL_PROFILES:
            calls = []
            config, _ = configure_model_profile(config_source, profile_name)
            with TemporaryDirectory() as tmp, patch(
                "pipeline.justice_runtime.chat_client_for_model", side_effect=lambda cfg, model: FakeClient(model)
            ):
                report = run_identity_preflight(config, Path(tmp))
            self.assertEqual(len(report["models"]), 3)
            for model_report, call, expected_limit in zip(report["models"], calls, [4096, 4096, 8192]):
                self.assertFalse(call["include_default_temperature"])
                self.assertEqual(call["request_parameters"]["reasoning"], {"exclude": True})
                self.assertEqual(call["request_parameters"]["max_tokens"], expected_limit)
                self.assertNotIn("max_completion_tokens", call["request_parameters"])
                self.assertEqual(model_report["generation_parameters"]["max_tokens"], expected_limit)
                self.assertNotIn("max_completion_tokens", model_report["generation_parameters"])
                self.assertTrue(call["require_parameters"])
                self.assertEqual(call["provider_only"], [model_report["fixed_provider_slug"]])
                self.assertEqual(call["response_format"], RESPONSE_FORMAT)
                for forbidden in ["temperature", "top_p", "seed", "n"]:
                    self.assertNotIn(forbidden, call["request_parameters"])
                self.assertEqual(model_report["generation_parameters"]["n_status"], "not_sent_by_policy")

    def test_manifest_secret_filter_and_hash_are_deterministic(self):
        value = {"api_key": "secret", "api_key_env": "OPENROUTER_API_KEY", "nested": {"password": "secret", "x": 1}}
        filtered = _without_secrets(value)
        self.assertEqual(filtered["api_key"], "[REDACTED]")
        self.assertEqual(filtered["api_key_env"], "OPENROUTER_API_KEY")
        self.assertEqual(filtered["nested"]["password"], "[REDACTED]")
        self.assertEqual(sha256_json({"a": 1, "b": 2}), sha256_json({"b": 2, "a": 1}))

    def test_manifest_is_immutable_redacted_and_detects_input_change(self):
        with TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run-1"
            config = {"openrouter": {"api_key": "secret"}}
            brief_manifest, oral_manifest, tree = self._new_manifest_fixture(run_dir, config)
            utterances = run_dir / "04_justice_utterances" / "justice_utterances.jsonl"
            checkpoint_records = run_dir / "human_final_checkpoints.json"
            checkpoint_records.write_text(json.dumps({"stage1": {"checkpoint_id": "upstream"}}))
            with patch("pipeline.justice_snapshots.verify_brief_snapshot", return_value=brief_manifest), patch(
                "pipeline.justice_snapshots.verify_utterance_snapshot", return_value=oral_manifest
            ), patch("pipeline.justice_snapshots.brief_snapshot_nodes", return_value=tree):
                manifest = seal_run_manifest(config, run_dir)
                require_sealed_manifest(run_dir)
                self.assertTrue(manifest["sealed"])
                self.assertEqual(manifest["model_profile"], "frontier")
                self.assertEqual([row["role"] for row in manifest["resolved_model_roles"]], [
                    "annotator_a", "annotator_b", "adjudicator",
                ])
                self.assertEqual(manifest["config_without_secrets"]["openrouter"]["api_key"], "[REDACTED]")
                self.assertIn("contracts_sha256", manifest)
                self.assertEqual(manifest["manifest_version"], 4)
                self.assertEqual(
                    set(manifest["derived_file_hashes"]),
                    {
                        "01_processed_transcripts/processed_transcripts.jsonl",
                        "04_justice_utterances/justice_utterances.jsonl",
                        "04_justice_utterances/selected_utterance_ids.json",
                        "04_justice_utterances/local_dialogue_context_audit.jsonl",
                    },
                )
                checkpoint_records.write_text(json.dumps({"stage1": {"checkpoint_id": "upstream"}, "stage2": {"checkpoint_id": "new"}}))
                require_sealed_manifest(run_dir)
                with self.assertRaises(RuntimeError):
                    seal_run_manifest({}, run_dir)
                utterances.write_text('{"utterance_id":"changed"}\n')
                with self.assertRaises(RuntimeError):
                    require_sealed_manifest(run_dir)

    def test_new_manifest_hashes_oyez_preprocessing_code(self):
        with TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run-1"
            config = {}
            brief_manifest, oral_manifest, tree = self._new_manifest_fixture(run_dir, config)
            with patch("pipeline.justice_snapshots.verify_brief_snapshot", return_value=brief_manifest), patch(
                "pipeline.justice_snapshots.verify_utterance_snapshot", return_value=oral_manifest
            ), patch("pipeline.justice_snapshots.brief_snapshot_nodes", return_value=tree):
                manifest = seal_run_manifest(config, run_dir)
            preprocess_path = Path(__file__).resolve().parents[1] / "pipeline" / "preprocess_oyez.py"
            self.assertEqual(
                manifest["file_hashes"]["pipeline/preprocess_oyez.py"],
                sha256_file(preprocess_path),
            )

    def test_manifest_rejects_changed_effective_config_or_runtime(self):
        with TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run-1"
            config = {"annotation_run": {"local_context": {"enabled": True}}}
            brief_manifest, oral_manifest, tree = self._new_manifest_fixture(run_dir, config)
            with patch("pipeline.justice_snapshots.verify_brief_snapshot", return_value=brief_manifest), patch(
                "pipeline.justice_snapshots.verify_utterance_snapshot", return_value=oral_manifest
            ), patch("pipeline.justice_snapshots.brief_snapshot_nodes", return_value=tree):
                manifest = seal_run_manifest(config, run_dir)
                require_sealed_manifest(run_dir, config)
                with self.assertRaises(RuntimeError):
                    require_sealed_manifest(run_dir, {"annotation_run": {"local_context": {"enabled": False}}})
                manifest["environment"]["python"] = "different"
                manifest_path = run_dir / "00_run_preflight" / "run_manifest.json"
                manifest_path.write_text(json.dumps(manifest, sort_keys=True))
                (run_dir / "00_run_preflight" / "run_manifest.sha256").write_text(
                    __import__("hashlib").sha256(manifest_path.read_bytes()).hexdigest()
                )
                with self.assertRaises(RuntimeError):
                    require_sealed_manifest(run_dir, config)

    def test_manifest_rechecks_snapshot_identity_before_stage_use(self):
        with TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run-1"
            config = {}
            brief_manifest, oral_manifest, tree = self._new_manifest_fixture(run_dir, config)
            with patch("pipeline.justice_snapshots.verify_brief_snapshot", return_value=brief_manifest), patch(
                "pipeline.justice_snapshots.verify_utterance_snapshot", return_value=oral_manifest
            ), patch("pipeline.justice_snapshots.brief_snapshot_nodes", return_value=tree):
                seal_run_manifest(config, run_dir)
                require_sealed_manifest(run_dir, config)

            changed_brief = dict(brief_manifest)
            changed_brief["overall_snapshot_sha256"] = "changed-brief-hash"
            with patch("pipeline.justice_snapshots.verify_brief_snapshot", return_value=changed_brief), patch(
                "pipeline.justice_snapshots.verify_utterance_snapshot", return_value=oral_manifest
            ):
                with self.assertRaises(RuntimeError):
                    require_sealed_manifest(run_dir, config)

    def test_seal_rejects_model_configuration_changed_after_probe(self):
        with TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run-1"
            config = {}
            brief_manifest, oral_manifest, tree = self._new_manifest_fixture(run_dir, config)
            config["models"]["annotation"]["gpt_5_2"]["fixed_provider"] = "changed-provider"
            with patch("pipeline.justice_snapshots.verify_brief_snapshot", return_value=brief_manifest), patch(
                "pipeline.justice_snapshots.verify_utterance_snapshot", return_value=oral_manifest
            ), patch("pipeline.justice_snapshots.brief_snapshot_nodes", return_value=tree):
                with self.assertRaises(RuntimeError):
                    seal_run_manifest(config, run_dir)
            self.assertFalse((run_dir / "00_run_preflight/run_manifest.sha256").exists())

    def test_seal_rejects_role_order_or_generation_contract_changed_after_probe(self):
        for mutation in ["role_order", "generation"]:
            with self.subTest(mutation=mutation), TemporaryDirectory() as tmp:
                run_dir = Path(tmp) / "run-1"
                config = {}
                brief_manifest, oral_manifest, tree = self._new_manifest_fixture(run_dir, config)
                if mutation == "role_order":
                    config["annotation_workflow"]["annotator_models"] = ["gemini_2_5_pro", "gpt_5_2"]
                    context = patch("pipeline.justice_runtime.ANNOTATOR_PARAMETERS", ANNOTATOR_PARAMETERS)
                else:
                    changed_parameters = {
                        **ANNOTATOR_PARAMETERS,
                        "max_completion_tokens": 4095,
                    }
                    context = patch("pipeline.justice_runtime.ANNOTATOR_PARAMETERS", changed_parameters)
                with context, patch(
                    "pipeline.justice_snapshots.verify_brief_snapshot", return_value=brief_manifest
                ), patch(
                    "pipeline.justice_snapshots.verify_utterance_snapshot", return_value=oral_manifest
                ), patch("pipeline.justice_snapshots.brief_snapshot_nodes", return_value=tree):
                    with self.assertRaises(RuntimeError):
                        seal_run_manifest(config, run_dir)

    def test_missing_manifest_checksum_recovers_only_after_full_validation(self):
        with TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run-1"
            config = {}
            brief_manifest, oral_manifest, tree = self._new_manifest_fixture(run_dir, config)
            with patch("pipeline.justice_snapshots.verify_brief_snapshot", return_value=brief_manifest), patch(
                "pipeline.justice_snapshots.verify_utterance_snapshot", return_value=oral_manifest
            ), patch("pipeline.justice_snapshots.brief_snapshot_nodes", return_value=tree):
                seal_run_manifest(config, run_dir)
                checksum = run_dir / "00_run_preflight/run_manifest.sha256"
                checksum.unlink()
                recovered = recover_missing_manifest_checksum(run_dir, config)
                self.assertTrue(recovered["sealed"])
                self.assertTrue(checksum.is_file())
                checksum.unlink()
                changed = json.loads(json.dumps(config))
                changed["models"]["annotation"]["gpt_5_2"]["model_name"] = "openai/changed"
                with self.assertRaises(RuntimeError):
                    recover_missing_manifest_checksum(run_dir, changed)
                self.assertFalse(checksum.exists())

    def test_attempt_audit_resume_is_idempotent_and_never_overwrites(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "audit.jsonl"
            row = {"audit_event_id": "stage|unit|attempt", "valid": True}
            append_attempt_audit(path, row)
            append_attempt_audit(path, row)
            self.assertEqual(len(json.loads("[" + ",".join(path.read_text().splitlines()) + "]")), 1)
            with self.assertRaises(RuntimeError):
                append_attempt_audit(path, {"audit_event_id": "stage|unit|attempt", "valid": False})


class OpenRouterContractTests(unittest.TestCase):
    def client(self, max_retries=0):
        return OpenRouterClient({
            "openrouter": {
                "api_key_env": "TEST_OPENROUTER_KEY",
                "max_retries": max_retries,
                "timeout_seconds": 1,
            }
        })

    def test_locked_provider_payload_and_single_choice(self):
        captured = {}

        def fake_post(url, headers, json, timeout):
            captured.update(json)
            return FakeResponse({
                "id": "gen-1", "model": "actual-model", "choices": [{"finish_reason": "stop", "message": {"content": "{}"}}]
            })

        with TemporaryDirectory() as tmp, patch.dict(os.environ, {"TEST_OPENROUTER_KEY": "key"}), patch(
            "pipeline.openrouter_client.requests.post", side_effect=fake_post
        ):
            result = self.client().chat(
                "requested-model", "prompt", Path(tmp) / "raw.json", Path(tmp) / "errors.log",
                provider_only=["provider"], require_parameters=True,
                request_parameters=request_parameters(role_parameters("annotator", capability())),
                response_format=RESPONSE_FORMAT, include_default_temperature=False,
            )
        self.assertEqual(captured["model"], "requested-model")
        self.assertEqual(captured["provider"], {"only": ["provider"], "allow_fallbacks": False, "require_parameters": True})
        for field in ["temperature", "top_p", "seed", "n"]:
            self.assertNotIn(field, captured)
        self.assertEqual(captured["reasoning"], {"exclude": True})
        self.assertEqual(captured["max_tokens"], 4096)
        self.assertNotIn("max_completion_tokens", captured)
        self.assertEqual(result["finish_reason"], "stop")
        self.assertEqual(result["actual_choice_count"], 1)
        self.assertEqual(result["request_payload"]["provider"]["only"], ["provider"])

    def test_capability_endpoint_requires_one_exact_tag(self):
        model = {"data": {"id": "author/model"}}

        def query(endpoints):
            with TemporaryDirectory() as tmp:
                client = self.client()
                with patch.object(
                    client, "_get_with_metadata_retries",
                    side_effect=[(model, 1), ({"data": {"endpoints": endpoints}}, 1)],
                ):
                    return client.capability_metadata(
                        "author/model", "provider-slug", Path(tmp) / "errors.log"
                    )

        selected = query([
            {"provider_name": "Display", "tag": "provider-slug"},
            {"provider_name": "provider-slug", "tag": "different"},
        ])
        self.assertEqual(selected["selected_endpoint"]["tag"], "provider-slug")
        with self.assertRaises(RuntimeError):
            query([{"provider_name": "provider-slug", "provider_slug": "provider-slug", "slug": "provider-slug"}])
        with self.assertRaises(RuntimeError):
            query([{"tag": "provider-slug"}, {"tag": "provider-slug"}])

    def test_capability_metadata_creates_fresh_nested_state_directory(self):
        model = {"data": {"id": "author/model"}}
        endpoints = {
            "data": {
                "endpoints": [
                    {"provider_name": "Provider", "tag": "provider-slug"},
                ]
            }
        }
        with TemporaryDirectory() as tmp, patch.dict(
            os.environ, {"TEST_OPENROUTER_KEY": "key"}
        ), patch(
            "pipeline.openrouter_client.requests.get",
            side_effect=[FakeResponse(model), FakeResponse(endpoints)],
        ):
            state_dir = Path(tmp) / "fresh" / "nested"
            result = self.client().capability_metadata(
                "author/model",
                "provider-slug",
                Path(tmp) / "errors.log",
                state_dir=state_dir,
            )

            self.assertEqual(result["selected_endpoint"]["tag"], "provider-slug")
            for stem in ["model_capability_query_state", "endpoint_capability_query_state"]:
                state_path = state_dir / f"{stem}.json"
                attempt_path = state_dir / f"{stem}.attempt_01.json"
                self.assertTrue(state_path.is_file())
                self.assertTrue(attempt_path.is_file())
                self.assertEqual(json.loads(state_path.read_text())["status"], "complete")
                self.assertEqual(json.loads(attempt_path.read_text())["status"], "response_received")

    def test_default_openrouter_chat_sends_zero_temperature(self):
        captured = {}

        def fake_post(url, headers, json, timeout):
            captured.update(json)
            return FakeResponse({
                "id": "gen-default", "model": "actual-model",
                "choices": [{"finish_reason": "stop", "message": {"content": "{}"}}],
            })

        with TemporaryDirectory() as tmp, patch.dict(os.environ, {"TEST_OPENROUTER_KEY": "key"}), patch(
            "pipeline.openrouter_client.requests.post", side_effect=fake_post
        ):
            self.client().chat("model", "prompt", Path(tmp) / "raw.json", Path(tmp) / "errors.log")
        self.assertEqual(captured["temperature"], 0)

    def test_multiple_choices_hard_fail_without_selecting_first(self):
        payload = {"choices": [
            {"finish_reason": "stop", "message": {"content": "{}"}},
            {"finish_reason": "stop", "message": {"content": "{}"}},
        ]}
        with TemporaryDirectory() as tmp, patch.dict(os.environ, {"TEST_OPENROUTER_KEY": "key"}), patch(
            "pipeline.openrouter_client.requests.post", return_value=FakeResponse(payload)
        ):
            raw = Path(tmp) / "raw.json"
            with self.assertRaises(ResponseContractError):
                self.client().chat(
                    "m", "p", raw, Path(tmp) / "errors.log", require_generation_metadata=True
                )
            self.assertEqual(len(json.loads(raw.read_text())["choices"]), 2)

    def test_unexpected_reasoning_is_removed_and_audited_without_failure(self):
        payload = {
            "id": "gen-reasoning",
            "model": "m",
            "choices": [{
                "finish_reason": "stop",
                "message": {
                    "content": "{}",
                    "reasoning": "private trace",
                    "reasoning_details": [{"x": 1}],
                    "thinking": "also private",
                },
            }],
        }
        removed_fields = [
            "choices[0].message.reasoning",
            "choices[0].message.reasoning_details",
            "choices[0].message.thinking",
        ]
        with TemporaryDirectory() as tmp, patch.dict(
            os.environ, {"TEST_OPENROUTER_KEY": "key"}
        ), patch(
            "pipeline.openrouter_client.requests.post", return_value=FakeResponse(payload)
        ) as post, patch(
            "pipeline.openrouter_client.requests.get",
            return_value=FakeResponse({"data": {"provider_name": "Provider", "model": "m"}}),
        ):
            raw = Path(tmp) / "raw.json"
            result = self.client().chat(
                "m", "p", raw, Path(tmp) / "errors.log", require_generation_metadata=True
            )
            attempt = json.loads(
                raw.with_name("raw.http_attempt_01.json").read_text(encoding="utf-8")
            )
            pending = json.loads(
                raw.with_name("raw.pending_metadata.json").read_text(encoding="utf-8")
            )
            http_state = json.loads(
                raw.with_name("raw.http_request_state.json").read_text(encoding="utf-8")
            )

            self.assertEqual(post.call_count, 1)
            self.assertEqual(result["content"], "{}")
            self.assertEqual(result["actual_provider"], "Provider")
            self.assertEqual(result["removed_reasoning_fields"], removed_fields)
            self.assertEqual(pending["status"], "complete")
            self.assertEqual(pending["result"]["removed_reasoning_fields"], removed_fields)
            self.assertEqual(http_state["status"], "response_received")
            self.assertEqual(http_state["removed_reasoning_fields"], removed_fields)
            for response in [
                json.loads(raw.read_text(encoding="utf-8")),
                attempt,
                result["raw_response"],
                pending["result"]["raw_response"],
            ]:
                message = response["choices"][0]["message"]
                self.assertNotIn("reasoning", message)
                self.assertNotIn("reasoning_details", message)
                self.assertNotIn("thinking", message)

    def test_non_strict_openrouter_call_preserves_reasoning_trace(self):
        payload = {"choices": [{
            "finish_reason": "stop",
            "message": {"content": "{}", "reasoning": "trace field"},
        }]}
        with TemporaryDirectory() as tmp, patch.dict(os.environ, {"TEST_OPENROUTER_KEY": "key"}), patch(
            "pipeline.openrouter_client.requests.post", return_value=FakeResponse(payload)
        ):
            raw = Path(tmp) / "raw.json"
            result = self.client().chat("m", "p", raw, Path(tmp) / "errors.log")
            self.assertEqual(result["content"], "{}")
            self.assertIn("trace field", raw.read_text())

    def test_strict_chat_retries_use_5_15_then_complete(self):
        success = {
            "id": "gen-retry",
            "model": "actual-model",
            "choices": [{"finish_reason": "stop", "message": {"content": "{}"}}],
        }
        with TemporaryDirectory() as tmp, patch.dict(
            os.environ, {"TEST_OPENROUTER_KEY": "key"}
        ), patch(
            "pipeline.openrouter_client.requests.post",
            side_effect=[
                FakeResponse({"error": "unavailable"}, status=503),
                FakeResponse({"error": "unavailable"}, status=503),
                FakeResponse(success),
            ],
        ) as post, patch(
            "pipeline.openrouter_client.requests.get",
            return_value=FakeResponse({
                "data": {"provider_name": "Provider", "model": "actual-model"}
            }),
        ), patch("pipeline.openrouter_client.time.sleep") as sleep:
            raw = Path(tmp) / "raw.json"
            result = self.client(max_retries=2).chat(
                "requested-model",
                "prompt",
                raw,
                Path(tmp) / "errors.log",
                provider_only=["provider"],
                require_parameters=True,
                request_parameters={"max_tokens": 4096, "reasoning": {"exclude": True}},
                response_format=RESPONSE_FORMAT,
                require_generation_metadata=True,
                include_default_temperature=False,
            )

            payloads = [call.kwargs["json"] for call in post.call_args_list]
            self.assertEqual(post.call_count, 3)
            self.assertEqual(payloads, [payloads[0], payloads[0], payloads[0]])
            self.assertEqual([call.args[0] for call in sleep.call_args_list], [5, 15])
            self.assertEqual(len(list(Path(tmp).glob("raw.http_attempt_*.json"))), 3)
            self.assertEqual(result["actual_provider"], "Provider")
            self.assertEqual(result["actual_model"], "actual-model")
            self.assertEqual(
                json.loads(
                    raw.with_name("raw.http_request_state.json").read_text(encoding="utf-8")
                )["status"],
                "response_received",
            )

    def test_strict_chat_exhausted_retries_remain_api_failed(self):
        with TemporaryDirectory() as tmp, patch.dict(
            os.environ, {"TEST_OPENROUTER_KEY": "key"}
        ), patch(
            "pipeline.openrouter_client.requests.post",
            return_value=FakeResponse({"error": "unavailable"}, status=503),
        ) as post, patch("pipeline.openrouter_client.time.sleep") as sleep:
            raw = Path(tmp) / "raw.json"
            with self.assertRaisesRegex(RuntimeError, "OpenRouter call failed after retries: HTTP 503"):
                self.client(max_retries=2).chat(
                    "requested-model",
                    "prompt",
                    raw,
                    Path(tmp) / "errors.log",
                    provider_only=["provider"],
                    require_parameters=True,
                    request_parameters={"max_tokens": 4096, "reasoning": {"exclude": True}},
                    response_format=RESPONSE_FORMAT,
                    require_generation_metadata=True,
                    include_default_temperature=False,
                )

            state = json.loads(
                raw.with_name("raw.http_request_state.json").read_text(encoding="utf-8")
            )
            self.assertEqual(post.call_count, 3)
            self.assertEqual([call.args[0] for call in sleep.call_args_list], [5, 15])
            self.assertEqual(len(list(Path(tmp).glob("raw.http_attempt_*.json"))), 3)
            self.assertEqual(state["status"], "api_failed")
            self.assertEqual(state["attempts_used"], 3)

    def test_strict_chat_resume_opens_new_retry_episode_without_overwriting_attempts(self):
        success = {
            "id": "gen-recovered",
            "model": "actual-model",
            "choices": [{"finish_reason": "stop", "message": {"content": "{}"}}],
        }
        with TemporaryDirectory() as tmp, patch.dict(
            os.environ, {"TEST_OPENROUTER_KEY": "key"}
        ), patch(
            "pipeline.openrouter_client.requests.post",
            side_effect=[
                FakeResponse({"error": "unavailable"}, status=503),
                FakeResponse({"error": "unavailable"}, status=503),
                FakeResponse({"error": "unavailable"}, status=503),
                FakeResponse(success),
            ],
        ) as post, patch(
            "pipeline.openrouter_client.requests.get",
            return_value=FakeResponse({
                "data": {"provider_name": "Provider", "model": "actual-model"}
            }),
        ), patch("pipeline.openrouter_client.time.sleep"):
            raw = Path(tmp) / "raw.json"
            kwargs = {
                "provider_only": ["provider"],
                "require_parameters": True,
                "request_parameters": {"max_tokens": 4096, "reasoning": {"exclude": True}},
                "response_format": RESPONSE_FORMAT,
                "require_generation_metadata": True,
                "include_default_temperature": False,
            }
            with self.assertRaises(RecoverableOpenRouterError):
                self.client(max_retries=2).chat(
                    "requested-model", "prompt", raw, Path(tmp) / "errors.log", **kwargs
                )
            result = self.client(max_retries=2).chat(
                "requested-model", "prompt", raw, Path(tmp) / "errors.log", **kwargs
            )
            self.assertEqual(post.call_count, 4)
            self.assertEqual(len(list(Path(tmp).glob("raw.http_attempt_*.json"))), 3)
            self.assertTrue((Path(tmp) / "raw.http_episode_002_attempt_01.json").is_file())
            state = json.loads((Path(tmp) / "raw.http_request_state.json").read_text())
            self.assertEqual(state["episode_number"], 2)
            self.assertEqual(state["episodes"][0]["episode_number"], 1)
            self.assertEqual(result["actual_provider"], "Provider")

    def test_balance_or_authentication_error_pauses_without_repeating(self):
        with TemporaryDirectory() as tmp, patch.dict(
            os.environ, {"TEST_OPENROUTER_KEY": "key"}
        ), patch(
            "pipeline.openrouter_client.requests.post",
            return_value=FakeResponse({"error": "insufficient balance"}, status=402),
        ) as post, patch("pipeline.openrouter_client.time.sleep") as sleep:
            with self.assertRaises(RecoverableOpenRouterError):
                self.client(max_retries=2).chat(
                    "requested-model",
                    "prompt",
                    Path(tmp) / "raw.json",
                    Path(tmp) / "errors.log",
                    require_generation_metadata=True,
                )
            self.assertEqual(post.call_count, 1)
            sleep.assert_not_called()

    def test_generation_metadata_retries_use_5_15_30(self):
        responses = [RuntimeError("x"), RuntimeError("x"), RuntimeError("x"), FakeResponse({"data": {"provider_name": "P"}})]

        def fake_get(*args, **kwargs):
            item = responses.pop(0)
            if isinstance(item, Exception):
                raise item
            return item

        with TemporaryDirectory() as tmp, patch.dict(os.environ, {"TEST_OPENROUTER_KEY": "key"}), patch(
            "pipeline.openrouter_client.requests.get", side_effect=fake_get
        ), patch("pipeline.openrouter_client.time.sleep") as sleep:
            state_path = Path(tmp) / "metadata_query_state.json"
            data, attempts = self.client().generation_metadata("g", Path(tmp) / "errors.log", state_path=state_path)
            self.assertEqual(attempts, 4)
            self.assertEqual(data["data"]["provider_name"], "P")
            self.assertEqual([call.args[0] for call in sleep.call_args_list], [5, 15, 30])
            attempt_files = sorted(Path(tmp).glob("metadata_query_state.attempt_*.json"))
            self.assertEqual(len(attempt_files), 4)
            self.assertEqual(json.loads(attempt_files[-1].read_text())["status"], "response_received")

    def test_interrupted_metadata_resume_does_not_recall_model_or_reset_count(self):
        chat_payload = {
            "id": "gen-1", "model": "actual-model",
            "choices": [{"finish_reason": "stop", "message": {"content": "{}"}}],
        }
        metadata_payload = {"data": {"provider_name": "Provider", "model": "actual-model"}}
        with TemporaryDirectory() as tmp, patch.dict(os.environ, {"TEST_OPENROUTER_KEY": "key"}), patch(
            "pipeline.openrouter_client.requests.post", return_value=FakeResponse(chat_payload)
        ) as post, patch(
            "pipeline.openrouter_client.requests.get", side_effect=[KeyboardInterrupt(), FakeResponse(metadata_payload)]
        ) as get:
            client = self.client()
            raw = Path(tmp) / "raw.json"
            with self.assertRaises(KeyboardInterrupt):
                client.chat(
                    "requested-model", "prompt", raw, Path(tmp) / "errors.log",
                    provider_only=["provider"], require_parameters=True,
                    request_parameters=ANNOTATOR_PARAMETERS, response_format=RESPONSE_FORMAT,
                    require_generation_metadata=True, include_default_temperature=False,
                )
            result = client.chat(
                "requested-model", "prompt", raw, Path(tmp) / "errors.log",
                provider_only=["provider"], require_parameters=True,
                request_parameters=ANNOTATOR_PARAMETERS, response_format=RESPONSE_FORMAT,
                require_generation_metadata=True, include_default_temperature=False,
            )
        self.assertEqual(post.call_count, 1)
        self.assertEqual(get.call_count, 2)
        self.assertEqual(result["generation_metadata_attempts"], 2)
        self.assertEqual(result["actual_provider"], "Provider")

    def test_exhausted_metadata_resume_does_not_recall_model(self):
        chat_payload = {
            "id": "gen-metadata-retry", "model": "actual-model",
            "choices": [{"finish_reason": "stop", "message": {"content": "{}"}}],
        }
        metadata_payload = {"data": {"provider_name": "Provider", "model": "actual-model"}}
        with TemporaryDirectory() as tmp, patch.dict(
            os.environ, {"TEST_OPENROUTER_KEY": "key"}
        ), patch(
            "pipeline.openrouter_client.requests.post", return_value=FakeResponse(chat_payload)
        ) as post, patch(
            "pipeline.openrouter_client.requests.get",
            side_effect=[
                RuntimeError("temporary"), RuntimeError("temporary"),
                RuntimeError("temporary"), RuntimeError("temporary"),
                FakeResponse(metadata_payload),
            ],
        ), patch("pipeline.openrouter_client.time.sleep"):
            client = self.client()
            raw = Path(tmp) / "raw.json"
            kwargs = {"require_generation_metadata": True}
            with self.assertRaises(RecoverableOpenRouterError):
                client.chat("requested-model", "prompt", raw, Path(tmp) / "errors.log", **kwargs)
            result = client.chat(
                "requested-model", "prompt", raw, Path(tmp) / "errors.log", **kwargs
            )
            self.assertEqual(post.call_count, 1)
            self.assertEqual(result["actual_provider"], "Provider")
            self.assertTrue(
                (Path(tmp) / "raw.metadata_query_state.episode_002.attempt_01.json").is_file()
            )
        self.assertNotIn("removed_reasoning_fields", result)


if __name__ == "__main__":
    unittest.main()
