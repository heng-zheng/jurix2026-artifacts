import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from pipeline.common import write_jsonl
from pipeline.justice_batches import (
    BatchIntegrityError,
    aggregate_rows,
    choose_batch,
    commit_unit,
    ensure_unit_plan,
    finish_batch,
    load_committed_units,
    seal_stage_aggregate,
    verify_stage_aggregate,
    verify_stage_progress,
)
from pipeline.justice_inputs import select_scoped_utterances
from pipeline.justice_lifecycle import normalize_run_scope
from pipeline.justice_lifecycle import read_run_state
from pipeline.justice_workflow import _run_ready_stage


class ScopedSelectionTests(unittest.TestCase):
    def manifests(self):
        return (
            {
                "briefs": [
                    {"brief_id": "pet-brief", "brief_group": "pet"},
                    {"brief_id": "res-brief", "brief_group": "res"},
                    {"brief_id": "caa-brief", "brief_group": "caa"},
                    {"brief_id": "old-rsp", "brief_group": "rsp"},
                ]
            },
            {"oral_arguments": [{"oyez_argument_id": "a1"}, {"oyez_argument_id": "a2"}]},
        )

    def scope(self):
        return {
            "selected_oyez_argument_ids": ["a1", "a2"],
            "oral_argument_brief_scopes": [
                {"oyez_argument_id": "a1", "brief_ids": ["pet-brief", "res-brief", "caa-brief"]}
            ],
            "phase_times_by_argument": {},
            "annotation_scope": {
                "target_justices": ["Justice One", "Justice Two"],
                "advocate_group_limits": {"pet": 2, "res": "all"},
            },
        }

    def test_scope_contract_accepts_only_approved_groups_and_limits(self):
        brief, oral = self.manifests()
        normalized = normalize_run_scope(self.scope(), oral, brief)
        self.assertEqual(normalized["annotation_scope"]["advocate_group_limits"], {"pet": 2, "res": "all"})
        for group in ["rsp", "other"]:
            invalid = self.scope()
            invalid["annotation_scope"]["advocate_group_limits"] = {group: "all"}
            with self.assertRaises(ValueError):
                normalize_run_scope(invalid, oral, brief)
        invalid = self.scope()
        invalid["oral_argument_brief_scopes"][0]["brief_ids"] = ["old-rsp"]
        with self.assertRaises(ValueError):
            normalize_run_scope(invalid, oral, brief)
        for value in [0, -1, True, "ALL", "3"]:
            invalid = self.scope()
            invalid["annotation_scope"]["advocate_group_limits"] = {"pet": value}
            with self.assertRaises(ValueError):
                normalize_run_scope(invalid, oral, brief)

    def test_selection_is_per_argument_justice_group_and_preserves_turn_order(self):
        rows = []
        for argument in ["a1", "a2"]:
            for justice in ["Justice One", "Justice Two"]:
                for group in ["pet", "res"]:
                    for number in range(1, 4):
                        rows.append(
                            {
                                "oyez_argument_id": argument,
                                "target_justice": justice,
                                "advocate_group": group,
                                "utterance_id": f"{argument}-{justice}-{group}-{number}",
                            }
                        )
        selected = select_scoped_utterances(rows, self.scope())
        pet = [row for row in selected if row["advocate_group"] == "pet"]
        res = [row for row in selected if row["advocate_group"] == "res"]
        self.assertEqual(len(pet), 2 * 2 * 2)
        self.assertEqual(len(res), 2 * 2 * 3)
        self.assertTrue(all(not row["utterance_id"].endswith("-3") for row in pet))
        self.assertEqual([row["utterance_id"] for row in selected], [
            row["utterance_id"]
            for row in rows
            if row["advocate_group"] == "res" or not row["utterance_id"].endswith("-3")
        ])


class BatchPersistenceTests(unittest.TestCase):
    def _unit_rows(self):
        return [{"utterance_id": f"u{number}", "text": f"Question {number}"} for number in range(1, 4)]

    def _route(self, run_dir, unit_id):
        path = run_dir / "05_question_purpose/question_purpose_attempt_audit.jsonl"
        existing = [
            json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
        ] if path.exists() else []
        write_jsonl(
            path,
            [
                *existing,
                {
                    "audit_event_id": f"stage1|{unit_id}|program|routing_summary",
                    "annotation_unit_id": unit_id,
                    "call_type": "routing_summary",
                    "routing": "substantive_agreement",
                },
            ],
        )

    def _commit(self, run_dir, row):
        self._route(run_dir, row["utterance_id"])
        commit_unit(
            run_dir,
            "stage1",
            row,
            [{"utterance_id": row["utterance_id"], "annotation_model": "a"}],
            {"utterance_id": row["utterance_id"], "routing": "substantive_agreement"},
        )

    def test_multiple_batches_form_one_continuous_ordered_result(self):
        with TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            rows = self._unit_rows()
            ensure_unit_plan(run_dir, "stage1", rows)
            first = choose_batch(run_dir, "stage1", input_func=lambda _: "2")
            self.assertEqual(first["planned_unit_ids"], ["u1", "u2"])
            for row in rows[:2]:
                self._commit(run_dir, row)
            progress = finish_batch(run_dir, "stage1", first)
            self.assertFalse(progress["complete"])
            second = choose_batch(run_dir, "stage1", input_func=lambda _: "all")
            self.assertEqual(second["planned_unit_ids"], ["u3"])
            self._commit(run_dir, rows[2])
            progress = finish_batch(run_dir, "stage1", second)
            self.assertTrue(progress["complete"])
            models, final = aggregate_rows(run_dir, "stage1")
            self.assertEqual([row["utterance_id"] for row in models], ["u1", "u2", "u3"])
            self.assertEqual([row["utterance_id"] for row in final], ["u1", "u2", "u3"])

    def test_interrupted_batch_can_close_and_rebatch_without_rerunning_completed_unit(self):
        with TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            rows = self._unit_rows()
            ensure_unit_plan(run_dir, "stage1", rows)
            original = choose_batch(run_dir, "stage1", input_func=lambda _: "all")
            self._commit(run_dir, rows[0])
            finish_batch(run_dir, "stage1", original, paused=True)
            answers = iter(["new", "1"])
            replacement = choose_batch(run_dir, "stage1", input_func=lambda _: next(answers))
            self.assertEqual(replacement["planned_unit_ids"], ["u2"])
            self.assertEqual(verify_stage_progress(run_dir, "stage1")["completed_unit_ids"], ["u1"])

    def test_stage3_plan_records_utterance_and_bundle_identity(self):
        with TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            plan = ensure_unit_plan(
                run_dir,
                "stage3",
                [{"utterance_id": "u1", "bundle_id": "u1__reason_bundle", "text": "Question"}],
            )
            self.assertEqual(plan["unit_key"], "bundle_id")
            self.assertEqual(plan["units"][0]["utterance_id"], "u1")
            self.assertEqual(plan["units"][0]["bundle_id"], "u1__reason_bundle")

    def test_manual_changes_or_deletions_hard_stop(self):
        with TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            row = self._unit_rows()[0]
            ensure_unit_plan(run_dir, "stage1", [row])
            self._commit(run_dir, row)
            result = next((run_dir / "05_question_purpose/batch_execution/units").glob("*.json"))
            payload = json.loads(result.read_text(encoding="utf-8"))
            payload["consolidated_row"]["routing"] = "manually_changed"
            result.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaises(BatchIntegrityError):
                load_committed_units(run_dir, "stage1")

        with TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            ensure_unit_plan(run_dir, "stage1", self._unit_rows())
            batch = choose_batch(run_dir, "stage1", input_func=lambda _: "1")
            finish_batch(run_dir, "stage1", batch)
            audit = run_dir / "05_question_purpose/batch_execution/batch_execution_audit.jsonl"
            audit.write_text(audit.read_text(encoding="utf-8").splitlines()[0] + "\n")
            with self.assertRaises(BatchIntegrityError):
                choose_batch(run_dir, "stage1", input_func=lambda _: "1")

        with TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            ensure_unit_plan(run_dir, "stage1", self._unit_rows())
            audit_root = run_dir / "05_question_purpose/batch_execution"
            (audit_root / "batch_execution_audit.jsonl").unlink()
            (audit_root / "batch_execution_audit_head.json").unlink()
            with self.assertRaises(BatchIntegrityError):
                choose_batch(run_dir, "stage1", input_func=lambda _: "1")

    def test_final_aggregate_manifest_rejects_late_manual_changes(self):
        with TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            row = self._unit_rows()[0]
            ensure_unit_plan(run_dir, "stage1", [row])
            batch = choose_batch(run_dir, "stage1", input_func=lambda _: "all")
            self._commit(run_dir, row)
            finish_batch(run_dir, "stage1", batch)
            official = run_dir / "05_question_purpose/question_purpose_adjudicated.jsonl"
            write_jsonl(official, [{"utterance_id": "u1"}])
            seal_stage_aggregate(run_dir, "stage1")
            verify_stage_aggregate(run_dir, "stage1")
            official.write_text('{"utterance_id":"changed"}\n', encoding="utf-8")
            with self.assertRaises(BatchIntegrityError):
                verify_stage_aggregate(run_dir, "stage1")

    def test_one_stage_invocation_processes_only_one_selected_batch(self):
        with TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            run_dir.mkdir()
            rows = self._unit_rows()
            write_jsonl(run_dir / "04_justice_utterances/justice_utterances.jsonl", rows)

            def fake_stage(config, current_run, selected, *, unit_complete, write_outputs):
                self.assertFalse(write_outputs)
                for row in selected:
                    self._route(current_run, row["utterance_id"])
                    unit_complete(
                        row,
                        [{"utterance_id": row["utterance_id"], "annotation_model": "a"}],
                        {"utterance_id": row["utterance_id"], "routing": "substantive_agreement"},
                    )

            with patch(
                "pipeline.justice_workflow.require_sealed_manifest",
                return_value={"manifest_version": 4},
            ), patch(
                "pipeline.justice_workflow.run_question_purpose",
                side_effect=fake_stage,
            ) as runner, patch("builtins.input", return_value="2"):
                _run_ready_stage({}, run_dir, "stage1")
            self.assertEqual(runner.call_count, 1)
            self.assertEqual(
                verify_stage_progress(run_dir, "stage1")["completed_unit_ids"],
                ["u1", "u2"],
            )
            self.assertEqual(read_run_state(run_dir)["status"], "ready_for_stage1")


if __name__ == "__main__":
    unittest.main()
