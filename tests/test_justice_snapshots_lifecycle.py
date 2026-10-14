import json
import math
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from pipeline.common import read_jsonl, write_jsonl
from pipeline.justice_inputs import derive_processed_transcript, extract_target_utterances
from pipeline.justice_lifecycle import (
    create_run,
    list_runs,
    load_run,
    mark_abandoned,
    normalize_run_scope,
    read_run_state,
    write_run_state,
)
from pipeline.justice_snapshots import (
    create_brief_snapshot,
    create_utterance_snapshot,
    verify_brief_snapshot,
    verify_utterance_snapshot,
)
from pipeline.justice_workflow import _run_ready_stage, build_local_dialogue_contexts


def reason_tree(brief_id, prefix="r"):
    return [
        {
            "brief_id": brief_id,
            "reason_target_id": f"{prefix}1",
            "parent_reason_target_id": "",
            "direct_child_reason_target_ids": [f"{prefix}1a"],
            "heading_level": 1,
            "heading_path": "I",
            "heading_text": "Root",
        },
        {
            "brief_id": brief_id,
            "reason_target_id": f"{prefix}1a",
            "parent_reason_target_id": f"{prefix}1",
            "direct_child_reason_target_ids": [],
            "heading_level": 2,
            "heading_path": "I.A",
            "heading_text": "Child",
        },
    ]


class DownloadResponse:
    def __init__(self, payload, url):
        self.content = json.dumps(payload, separators=(",", ":")).encode()
        self.url = url

    def raise_for_status(self):
        return None


class SnapshotTests(unittest.TestCase):
    def test_multibrief_snapshot_is_atomic_and_detects_tampering(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            snapshot_root = root / "snapshots" / "briefs"
            specs = []
            for brief_id, prefix in [("22-451-pet", "p"), ("22-451-res", "s")]:
                pdf = root / f"{brief_id}.pdf"
                tree = root / f"{brief_id}.jsonl"
                pdf.write_bytes(f"PDF {brief_id}".encode())
                write_jsonl(tree, reason_tree(brief_id, prefix))
                specs.append(
                    {
                        "brief_id": brief_id,
                        "docket_numbers": ["22-451"],
                        "brief_group": "pet" if brief_id.endswith("pet") else "res",
                        "pdf_path": str(pdf),
                        "reason_tree_path": str(tree),
                    }
                )
            with patch("pipeline.justice_snapshots.BRIEF_SNAPSHOT_ROOT", snapshot_root):
                directory = create_brief_snapshot(
                    {"briefs": specs}, confirm=lambda rows: len(rows) == 2, snapshot_id="brief-test"
                )
                manifest = verify_brief_snapshot("brief-test")
                self.assertEqual({row["brief_id"] for row in manifest["briefs"]}, {"22-451-pet", "22-451-res"})
                self.assertTrue((directory / "briefs/22-451-pet/22-451-pet.pdf").is_file())
                (directory / "briefs/22-451-pet/22-451-pet.pdf").write_bytes(b"changed")
                with self.assertRaises(RuntimeError):
                    verify_brief_snapshot("brief-test")

    def test_brief_snapshot_rejects_duplicate_reason_ids_without_partial_commit(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            snapshot_root = root / "snapshots" / "briefs"
            specs = []
            for brief_id in ["a", "b"]:
                pdf = root / f"{brief_id}.pdf"
                tree = root / f"{brief_id}.jsonl"
                pdf.write_bytes(b"pdf")
                write_jsonl(tree, reason_tree(brief_id, "same"))
                specs.append(
                    {
                        "brief_id": brief_id,
                        "docket_numbers": ["1"],
                        "brief_group": "pet",
                        "pdf_path": str(pdf),
                        "reason_tree_path": str(tree),
                    }
                )
            with patch("pipeline.justice_snapshots.BRIEF_SNAPSHOT_ROOT", snapshot_root):
                with self.assertRaises(ValueError):
                    create_brief_snapshot({"briefs": specs}, confirm=lambda rows: True, snapshot_id="bad")
                self.assertFalse((snapshot_root / "bad").exists())

    def _oral_sources(self):
        case = {
            "ID": 99,
            "advocates": [
                {"name": "Counsel Pet", "description": "for the Petitioner"},
                {"name": "Counsel Res", "description": "for the Respondent"},
            ],
        }
        argument = {
            "id": 25500,
            "transcript": {
                "sections": [
                    {
                        "turns": [
                            {"speaker": {"name": "Counsel Pet", "roles": []}, "text": "Opening", "start": 0},
                            {"speaker": {"name": "Justice Test", "roles": [{"type": "scotus_justice"}]}, "text": "", "start": 1},
                            {"speaker": {"name": "Justice Test", "roles": [{"type": "scotus_justice"}]}, "text": "Question", "start": 2},
                        ]
                    },
                    {"turns": [{"speaker": {"name": "Counsel Res", "roles": []}, "text": "Answer", "start": 3}]},
                ]
            },
        }
        return case, argument

    def test_oral_snapshot_preserves_raw_bytes_and_stable_ids(self):
        with TemporaryDirectory() as tmp:
            oral_root = Path(tmp) / "snapshots" / "oral_arguments"
            case, argument = self._oral_sources()
            payloads = {
                "https://oyez/case": DownloadResponse(case, "https://final/case"),
                "https://oyez/argument": DownloadResponse(argument, "https://final/argument"),
            }
            specification = {
                "case_sources": [{"source_url": "https://oyez/case"}],
                "oral_arguments": [
                    {"source_url": "https://oyez/argument", "case_source_ids": ["99"]}
                ],
            }
            with patch("pipeline.justice_snapshots.ORAL_SNAPSHOT_ROOT", oral_root), patch(
                "pipeline.justice_inputs.ORAL_SNAPSHOT_ROOT", oral_root
            ), patch("pipeline.justice_snapshots.requests.get", side_effect=lambda url, timeout: payloads[url]):
                create_utterance_snapshot(specification, snapshot_id="oral-test")
                manifest = verify_utterance_snapshot("oral-test")
                self.assertEqual(manifest["oral_arguments"][0]["oyez_argument_id"], "25500")
                scope = {
                    "selected_oyez_argument_ids": ["25500"],
                    "oral_argument_brief_scopes": [{"oyez_argument_id": "25500", "brief_ids": ["b"]}],
                    "phase_times_by_argument": {},
                }
                transcript = derive_processed_transcript("oral-test", scope)
                self.assertEqual(
                    [row["turn_id"] for row in transcript],
                    ["25500__turn_000001", "25500__turn_000002", "25500__turn_000003", "25500__turn_000004"],
                )
                utterances = extract_target_utterances(transcript, target_justices=["Justice Test"])
                self.assertEqual([row["utterance_id"] for row in utterances], ["25500__turn_000003__utterance"])
                self.assertEqual(transcript[2]["advocate_group"], "pet")

    def test_conflicting_case_source_advocate_groups_hard_stop(self):
        with TemporaryDirectory() as tmp:
            oral_root = Path(tmp) / "snapshots" / "oral_arguments"
            _, argument = self._oral_sources()
            case_a = {"ID": 1, "advocates": [{"name": "Counsel Pet", "description": "for the Petitioner"}]}
            case_b = {"ID": 2, "advocates": [{"name": "Counsel Pet", "description": "for the Respondent"}]}
            payloads = {
                "case-a": DownloadResponse(case_a, "case-a"),
                "case-b": DownloadResponse(case_b, "case-b"),
                "argument": DownloadResponse(argument, "argument"),
            }
            specification = {
                "case_sources": [{"source_url": "case-a"}, {"source_url": "case-b"}],
                "oral_arguments": [{"source_url": "argument", "case_source_ids": ["1", "2"]}],
            }
            with patch("pipeline.justice_snapshots.ORAL_SNAPSHOT_ROOT", oral_root), patch(
                "pipeline.justice_inputs.ORAL_SNAPSHOT_ROOT", oral_root
            ), patch("pipeline.justice_snapshots.requests.get", side_effect=lambda url, timeout: payloads[url]):
                create_utterance_snapshot(specification, snapshot_id="oral-conflict")
                scope = {
                    "selected_oyez_argument_ids": ["25500"],
                    "oral_argument_brief_scopes": [{"oyez_argument_id": "25500", "brief_ids": ["b"]}],
                    "phase_times_by_argument": {},
                }
                with self.assertRaises(ValueError):
                    derive_processed_transcript("oral-conflict", scope)


class LifecycleTests(unittest.TestCase):
    def manifests(self):
        return (
            {
                "overall_snapshot_sha256": "brief-hash",
                "briefs": [{"brief_id": "b1", "brief_group": "pet"}],
            },
            {
                "overall_snapshot_sha256": "oral-hash",
                "oral_arguments": [{"oyez_argument_id": "25500"}],
            },
        )

    def scope(self):
        return {
            "selected_oyez_argument_ids": ["25500"],
            "oral_argument_brief_scopes": [{"oyez_argument_id": "25500", "brief_ids": ["b1"]}],
            "phase_times_by_argument": {},
            "annotation_scope": {
                "target_justices": ["Justice Test"],
                "advocate_group_limits": {"pet": "all"},
            },
        }

    def test_run_identity_state_and_terminal_rules(self):
        with TemporaryDirectory() as tmp:
            run_root = Path(tmp) / "runs"
            brief, oral = self.manifests()
            with patch("pipeline.justice_lifecycle.RUN_ROOT", run_root), patch(
                "pipeline.justice_lifecycle.verify_brief_snapshot", return_value=brief
            ), patch("pipeline.justice_lifecycle.verify_utterance_snapshot", return_value=oral):
                run_dir = create_run(
                    model_profile="frontier",
                    resolved_model_roles=[{"role": "annotator_a"}, {"role": "annotator_b"}, {"role": "adjudicator"}],
                    brief_snapshot_id="bs", utterance_snapshot_id="os",
                    scope=self.scope(), run_id="program-id",
                )
                _, state, manifest = load_run("program-id")
                self.assertEqual(state["status"], "preflight_in_progress")
                self.assertEqual(manifest["run_id"], "program-id")
                self.assertEqual(manifest["model_profile"], "frontier")
                self.assertEqual(manifest["manifest_version"], 4)
                self.assertNotIn("run_mode", manifest)
                rows = list_runs()
                self.assertEqual(rows[0]["status"], "failed")
                self.assertEqual(rows[0]["model_profile"], "frontier")
                with self.assertRaises(RuntimeError):
                    mark_abandoned("program-id")
                manifest_path = run_dir / "00_run_preflight/run_manifest.json"
                manifest["manifest_version"] = 3
                manifest_path.write_text(json.dumps(manifest))
                with self.assertRaisesRegex(RuntimeError, "manifest schema is unsupported"):
                    load_run("program-id")
                manifest["manifest_version"] = 4
                manifest["run_id"] = "different"
                manifest_path.write_text(json.dumps(manifest))
                with self.assertRaises(RuntimeError):
                    load_run("program-id")

    def test_list_runs_ignores_directories_without_current_identity_files(self):
        with TemporaryDirectory() as tmp:
            run_root = Path(tmp) / "runs"
            (run_root / "unrecognized-run").mkdir(parents=True)
            (run_root / "unrecognized-run" / "unrelated_output.jsonl").write_text("{}\n")
            with patch("pipeline.justice_lifecycle.RUN_ROOT", run_root):
                self.assertEqual(list_runs(), [])


    def test_invalid_scope_and_nonfinite_phase_do_not_create_run(self):
        with TemporaryDirectory() as tmp:
            run_root = Path(tmp) / "runs"
            brief, oral = self.manifests()
            scope = self.scope()
            scope["phase_times_by_argument"] = {
                "25500": {"petitioner": {"start_time": 0, "end_time": math.nan}}
            }
            with patch("pipeline.justice_lifecycle.RUN_ROOT", run_root), patch(
                "pipeline.justice_lifecycle.verify_brief_snapshot", return_value=brief
            ), patch("pipeline.justice_lifecycle.verify_utterance_snapshot", return_value=oral):
                with self.assertRaises(ValueError):
                    create_run(
                        model_profile="frontier",
                        resolved_model_roles=[{"role": "annotator_a"}, {"role": "annotator_b"}, {"role": "adjudicator"}],
                        brief_snapshot_id="bs", utterance_snapshot_id="os",
                        scope=scope, run_id="never-created",
                    )
                self.assertFalse((run_root / "never-created").exists())

    def test_selected_argument_without_target_mapping_can_remain_unmapped(self):
        brief = {"briefs": [{"brief_id": "b1", "brief_group": "pet"}]}
        oral = {"oral_arguments": [{"oyez_argument_id": "a1"}, {"oyez_argument_id": "a2"}]}
        scope = normalize_run_scope(
            {
                "selected_oyez_argument_ids": ["a1", "a2"],
                "oral_argument_brief_scopes": [{"oyez_argument_id": "a1", "brief_ids": ["b1"]}],
                "phase_times_by_argument": {},
                "annotation_scope": {
                    "target_justices": ["Justice Test"],
                    "advocate_group_limits": {"pet": "all"},
                },
            },
            oral,
            brief,
        )
        self.assertEqual(scope["selected_oyez_argument_ids"], ["a1", "a2"])
        self.assertEqual([row["oyez_argument_id"] for row in scope["oral_argument_brief_scopes"]], ["a1"])


class ContextIsolationTests(unittest.TestCase):
    def test_local_context_never_crosses_oral_argument_boundary(self):
        transcript = [
            {
                "oyez_argument_id": "a1", "turn_id": "a1__turn_000001", "speaker": "Justice Test",
                "speaker_role": "scotus_justice", "text": "Question one", "section_counsel": "Counsel", "advocate_group": "pet",
            },
            {
                "oyez_argument_id": "a1", "turn_id": "a1__turn_000002", "speaker": "Counsel",
                "text": "Answer one", "section_counsel": "Counsel", "advocate_group": "pet",
            },
            {
                "oyez_argument_id": "a2", "turn_id": "a2__turn_000001", "speaker": "Justice Test",
                "speaker_role": "scotus_justice", "text": "Question two", "section_counsel": "Counsel", "advocate_group": "pet",
            },
            {
                "oyez_argument_id": "a2", "turn_id": "a2__turn_000002", "speaker": "Counsel",
                "text": "Answer two", "section_counsel": "Counsel", "advocate_group": "pet",
            },
        ]
        utterances = [
            {**transcript[0], "utterance_id": "u1", "target_justice": "Justice Test"},
            {**transcript[2], "utterance_id": "u2", "target_justice": "Justice Test"},
        ]
        contexts = build_local_dialogue_contexts(transcript, utterances, {})
        self.assertIn("Question one", contexts[0]["local_dialogue_context"])
        self.assertIn("Answer one", contexts[0]["local_dialogue_context"])
        self.assertNotIn("Question two", contexts[0]["local_dialogue_context"])
        self.assertIn("a1", contexts[0]["local_dialogue_context_id"])
        self.assertNotIn("Question one", contexts[1]["local_dialogue_context"])

    def test_blank_turn_is_never_added_to_dialogue_context(self):
        transcript = [
            {
                "oyez_argument_id": "a1", "turn_id": "a1__turn_000001", "speaker": "Justice Test",
                "speaker_role": "scotus_justice", "text": "Question", "section_counsel": "Counsel", "advocate_group": "pet",
            },
            {
                "oyez_argument_id": "a1", "turn_id": "a1__turn_000002", "speaker": "Unknown",
                "text": "   ", "section_counsel": "Counsel", "advocate_group": "pet",
            },
        ]
        utterances = [{**transcript[0], "utterance_id": "u1", "target_justice": "Justice Test"}]
        context = build_local_dialogue_contexts(transcript, utterances, {})[0]["local_dialogue_context"]
        self.assertIn("a1__turn_000001", context)
        self.assertNotIn("a1__turn_000002", context)


class EmptyDownstreamRoutingTests(unittest.TestCase):
    def test_zero_stage2_expected_rows_complete_without_empty_review_checkpoint(self):
        with TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run-empty-stage2"
            run_dir.mkdir()
            write_run_state(run_dir, "ready_for_stage2", "stage2")
            utterance_path = run_dir / "04_justice_utterances/justice_utterances.jsonl"
            write_jsonl(utterance_path, [identity_row := {
                "oyez_argument_id": "25500", "utterance_id": "u1", "turn_id": "25500__turn_000001",
                "target_justice": "Justice Test", "text": "Question",
            }])
            (run_dir / "human_final_checkpoints.json").write_text(
                json.dumps({"stage1": {"checkpoint_id": "stage1-final"}})
            )
            with patch("pipeline.justice_workflow.require_sealed_manifest", return_value={"manifest_version": 4}), patch(
                "pipeline.justice_checkpoint_stage1.consume_stage1_checkpoint",
                return_value=[{**identity_row, "enters_reason_alignment": False}],
            ), patch("pipeline.justice_workflow._reason_nodes_for_manifest", return_value=[]), patch(
                "pipeline.justice_workflow.build_candidate_rows", return_value=[]
            ):
                _run_ready_stage({}, run_dir, "stage2")
            self.assertEqual(read_run_state(run_dir)["status"], "completed")
            self.assertEqual(read_jsonl(run_dir / "06_reason_alignment/reason_alignment_final.jsonl"), [])
            self.assertEqual(read_jsonl(run_dir / "07_reason_stance/stance_final.jsonl"), [])
            self.assertFalse((run_dir / "06_reason_alignment/reason_alignment_student_review.csv").exists())


if __name__ == "__main__":
    unittest.main()
