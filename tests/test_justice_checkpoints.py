import json
import unittest
from contextlib import ExitStack
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from pipeline.common import read_jsonl
from pipeline.justice_checkpoint_stage1 import consume_stage1_checkpoint
from pipeline.justice_checkpoint_stage2 import consume_stage2_checkpoint, required_stage2_utterances
from pipeline.justice_checkpoints import (
    _read_csv,
    _write_csv,
    create_human_final_checkpoint,
    create_review_checkpoint,
    load_review_checkpoint,
    resume_human_final_checkpoint,
)
from pipeline.justice_lifecycle import read_run_state, write_run_state
from pipeline.justice_workflow import _copy_checkpoint_clean_final


def identity(utterance_id="u1"):
    return {
        "oyez_argument_id": "25500",
        "utterance_id": utterance_id,
        "turn_id": f"25500__turn_00000{1 if utterance_id == 'u1' else 2}",
        "target_justice": "Justice Test",
        "text": f"Question {utterance_id}",
    }


def tree():
    return [
        {
            "brief_id": "b1", "brief_group": "pet", "reason_target_id": "r1",
            "parent_reason_target_id": "", "direct_child_reason_target_ids": ["r1a"],
            "heading_level": 1, "heading_path": "I", "heading_text": "Root",
        },
        {
            "brief_id": "b1", "brief_group": "pet", "reason_target_id": "r1a",
            "parent_reason_target_id": "r1", "direct_child_reason_target_ids": [],
            "heading_level": 2, "heading_path": "I.A", "heading_text": "Child",
        },
        {
            "brief_id": "b2", "brief_group": "res", "reason_target_id": "r2",
            "parent_reason_target_id": "", "direct_child_reason_target_ids": [],
            "heading_level": 1, "heading_path": "I", "heading_text": "Other",
        },
    ]


class CheckpointTests(unittest.TestCase):
    def roots(self, base):
        review = base / "checkpoints/review"
        human = base / "checkpoints/human_final"
        pending = base / "checkpoints/.pending"
        stack = ExitStack()
        stack.enter_context(patch("pipeline.justice_checkpoints.REVIEW_CHECKPOINT_ROOT", review))
        stack.enter_context(patch("pipeline.justice_checkpoints.HUMAN_FINAL_CHECKPOINT_ROOT", human))
        stack.enter_context(patch("pipeline.justice_checkpoints.PENDING_CHECKPOINT_ROOT", pending))
        stack.enter_context(patch("pipeline.justice_checkpoint_common.HUMAN_FINAL_CHECKPOINT_ROOT", human))
        return stack

    def stage1_review(self, run_dir, rows=None):
        rows = rows or [identity()]
        student = run_dir / "05_question_purpose/question_purpose_student_review.csv"
        professor = run_dir / "08_human_review/question_purpose_professor_adjudication.csv"
        fields = [
            "oyez_argument_id", "utterance_id", "turn_id", "target_justice", "text", "frozen_material",
            "student_id", "student_question_function_present", "student_purposes", "student_notes",
            "professor_final_question_function_present", "professor_final_purposes", "professor_notes",
        ]
        payload = [
            {
                **row, "frozen_material": "LLM output", "student_id": "",
                "student_question_function_present": "", "student_purposes": "", "student_notes": "",
                "professor_final_question_function_present": "", "professor_final_purposes": "", "professor_notes": "",
            }
            for row in rows
        ]
        _write_csv(student, fields, payload)
        _write_csv(professor, fields, payload)
        return student, professor

    def test_review_and_human_final_are_immutable_and_clean(self):
        with TemporaryDirectory() as tmp, self.roots(Path(tmp)):
            run_dir = Path(tmp) / "run-a"
            run_dir.mkdir()
            write_run_state(run_dir, "review_checkpoint_pending", "stage1")
            student, professor = self.stage1_review(run_dir)
            artifact = run_dir / "05_question_purpose/attempt.json"
            artifact.write_text('{"raw":"response"}')
            review = create_review_checkpoint(
                run_dir, "stage1", student_working_path=student, professor_working_path=professor,
                artifact_paths=[run_dir / "05_question_purpose"], checkpoint_id="review-a",
            )
            self.assertEqual(read_run_state(run_dir)["status"], "awaiting_stage1_human")
            directory, loaded = load_review_checkpoint("review-a")
            self.assertEqual(review["overall_checkpoint_sha256"], loaded["overall_checkpoint_sha256"])
            self.assertTrue((directory / "artifacts/05_question_purpose/attempt.json").is_file())
            fields, rows = _read_csv(professor)
            rows[0]["professor_final_question_function_present"] = "true"
            rows[0]["professor_final_purposes"] = '["Clarification"]'
            _write_csv(professor, fields, rows)
            final = create_human_final_checkpoint(
                run_dir, "review-a", professor, [], checkpoint_id="human-a"
            )
            self.assertEqual(read_run_state(run_dir)["status"], "ready_for_stage2")
            clean = read_jsonl(Path(tmp) / "checkpoints/human_final/human-a/clean_final.jsonl")
            self.assertEqual(clean[0]["final_purpose_labels"], ["Clarification"])
            self.assertNotIn("frozen_material", clean[0])
            self.assertEqual(
                (run_dir / "05_question_purpose/question_purpose_final.jsonl").read_bytes(),
                (Path(tmp) / "checkpoints/human_final/human-a/clean_final.jsonl").read_bytes(),
            )
            imported = Path(tmp) / "new-run/05_question_purpose/question_purpose_final.jsonl"
            _copy_checkpoint_clean_final("human-a", imported)
            self.assertEqual(
                imported.read_bytes(),
                (Path(tmp) / "checkpoints/human_final/human-a/clean_final.jsonl").read_bytes(),
            )
            self.assertNotIn("enters_reason_alignment", read_jsonl(imported)[0])
            consumed = consume_stage1_checkpoint("human-a", [{**identity(), "local_dialogue_context": "new"}])
            self.assertTrue(consumed[0]["enters_reason_alignment"])
            self.assertEqual(final["coverage"], 1)

    def test_checkpoint_consumers_use_only_the_narrow_common_boundary(self):
        import pipeline.justice_checkpoint_common as common
        import pipeline.justice_checkpoint_stage1 as stage1
        import pipeline.justice_checkpoint_stage2 as stage2

        self.assertNotIn("justice_runtime", Path(common.__file__).read_text())
        self.assertNotIn("justice_lifecycle", Path(common.__file__).read_text())
        self.assertNotIn("from .common", Path(stage1.__file__).read_text())
        self.assertNotIn("from .common", Path(stage2.__file__).read_text())

    def test_invalid_frozen_edit_does_not_change_awaiting_state(self):
        with TemporaryDirectory() as tmp, self.roots(Path(tmp)):
            run_dir = Path(tmp) / "run-b"
            run_dir.mkdir()
            write_run_state(run_dir, "review_checkpoint_pending", "stage1")
            student, professor = self.stage1_review(run_dir)
            create_review_checkpoint(
                run_dir, "stage1", student_working_path=student, professor_working_path=professor,
                artifact_paths=[], checkpoint_id="review-b",
            )
            fields, rows = _read_csv(professor)
            rows[0]["frozen_material"] = "tampered"
            rows[0]["professor_final_question_function_present"] = "true"
            rows[0]["professor_final_purposes"] = '["Clarification"]'
            _write_csv(professor, fields, rows)
            with self.assertRaises(ValueError):
                create_human_final_checkpoint(run_dir, "review-b", professor, [])
            self.assertEqual(read_run_state(run_dir)["status"], "awaiting_stage1_human")

    def test_pending_review_checkpoint_rebuilds_without_model_work(self):
        with TemporaryDirectory() as tmp, self.roots(Path(tmp)):
            run_dir = Path(tmp) / "run-review-pending"
            run_dir.mkdir()
            write_run_state(run_dir, "review_checkpoint_pending", "stage1")
            student, professor = self.stage1_review(run_dir)
            with patch("pipeline.justice_checkpoints._manifest_write", side_effect=RuntimeError("interrupt")):
                with self.assertRaises(RuntimeError):
                    create_review_checkpoint(
                        run_dir, "stage1", student_working_path=student, professor_working_path=professor,
                        artifact_paths=[], checkpoint_id="review-pending",
                    )
            self.assertEqual(read_run_state(run_dir)["status"], "review_checkpoint_pending")
            create_review_checkpoint(
                run_dir, "stage1", student_working_path=student, professor_working_path=professor,
                artifact_paths=[], require_existing_pending=True,
            )
            self.assertEqual(read_run_state(run_dir)["status"], "awaiting_stage1_human")
            load_review_checkpoint("review-pending")

    def test_pending_review_checkpoint_rejects_changed_source_set_or_content(self):
        with TemporaryDirectory() as tmp, self.roots(Path(tmp)):
            run_dir = Path(tmp) / "run-review-drift"
            run_dir.mkdir()
            write_run_state(run_dir, "review_checkpoint_pending", "stage1")
            student, professor = self.stage1_review(run_dir)
            artifact_dir = run_dir / "audit"
            artifact_dir.mkdir()
            (artifact_dir / "attempt.json").write_text('{"attempt":1}')
            with patch("pipeline.justice_checkpoints._manifest_write", side_effect=RuntimeError("interrupt")):
                with self.assertRaises(RuntimeError):
                    create_review_checkpoint(
                        run_dir, "stage1", student_working_path=student,
                        professor_working_path=professor, artifact_paths=[artifact_dir],
                        checkpoint_id="review-drift",
                    )
            (artifact_dir / "attempt.json").write_text('{"attempt":2}')
            with self.assertRaises(RuntimeError):
                create_review_checkpoint(
                    run_dir, "stage1", student_working_path=student,
                    professor_working_path=professor, artifact_paths=[artifact_dir],
                    require_existing_pending=True,
                )
            self.assertEqual(read_run_state(run_dir)["status"], "review_checkpoint_pending")

    def test_pending_review_checkpoint_rejects_working_csv_or_added_artifact(self):
        for mutation in ["working_csv", "added_artifact"]:
            with self.subTest(mutation=mutation), TemporaryDirectory() as tmp, self.roots(Path(tmp)):
                run_dir = Path(tmp) / "run-review-drift"
                run_dir.mkdir()
                write_run_state(run_dir, "review_checkpoint_pending", "stage1")
                student, professor = self.stage1_review(run_dir)
                artifact_dir = run_dir / "audit"
                artifact_dir.mkdir()
                (artifact_dir / "attempt.json").write_text('{"attempt":1}')
                with patch("pipeline.justice_checkpoints._manifest_write", side_effect=RuntimeError("interrupt")):
                    with self.assertRaises(RuntimeError):
                        create_review_checkpoint(
                            run_dir, "stage1", student_working_path=student,
                            professor_working_path=professor, artifact_paths=[artifact_dir],
                            checkpoint_id=f"review-{mutation}",
                        )
                if mutation == "working_csv":
                    professor.write_text(professor.read_text() + "\n")
                else:
                    (artifact_dir / "extra.json").write_text("{}")
                with self.assertRaises(RuntimeError):
                    create_review_checkpoint(
                        run_dir, "stage1", student_working_path=student,
                        professor_working_path=professor, artifact_paths=[artifact_dir],
                        require_existing_pending=True,
                    )

    def test_pending_human_final_rebuild_uses_staged_returns(self):
        with TemporaryDirectory() as tmp, self.roots(Path(tmp)):
            run_dir = Path(tmp) / "run-c"
            run_dir.mkdir()
            write_run_state(run_dir, "review_checkpoint_pending", "stage1")
            student, professor = self.stage1_review(run_dir)
            create_review_checkpoint(
                run_dir, "stage1", student_working_path=student, professor_working_path=professor,
                artifact_paths=[], checkpoint_id="review-c",
            )
            fields, rows = _read_csv(professor)
            rows[0]["professor_final_question_function_present"] = "true"
            rows[0]["professor_final_purposes"] = '["Support"]'
            _write_csv(professor, fields, rows)
            with patch("pipeline.justice_checkpoints._manifest_write", side_effect=RuntimeError("interrupt")):
                with self.assertRaises(RuntimeError):
                    create_human_final_checkpoint(
                        run_dir, "review-c", professor, [], checkpoint_id="human-c"
                    )
            self.assertEqual(read_run_state(run_dir)["status"], "human_final_checkpoint_pending")
            professor.unlink()
            resume_human_final_checkpoint(run_dir)
            self.assertEqual(read_run_state(run_dir)["status"], "ready_for_stage2")
            self.assertEqual(
                read_jsonl(Path(tmp) / "checkpoints/human_final/human-c/clean_final.jsonl")[0]["final_purpose_labels"],
                ["Support"],
            )

    def test_partial_overlapping_student_returns_are_audit_only(self):
        with TemporaryDirectory() as tmp, self.roots(Path(tmp)):
            run_dir = Path(tmp) / "run-students"
            run_dir.mkdir()
            write_run_state(run_dir, "review_checkpoint_pending", "stage1")
            student, professor = self.stage1_review(run_dir, [identity("u1"), identity("u2")])
            create_review_checkpoint(
                run_dir, "stage1", student_working_path=student, professor_working_path=professor,
                artifact_paths=[], checkpoint_id="review-students",
            )
            student_fields, master_rows = _read_csv(student)
            returns = []
            for student_id in ["A", "B"]:
                row = dict(master_rows[0])
                row["student_id"] = student_id
                row["student_question_function_present"] = "true"
                row["student_purposes"] = '["Clarification"]'
                path = run_dir / f"student-{student_id}.csv"
                _write_csv(path, student_fields, [row])
                returns.append(path)
            professor_fields, professor_rows = _read_csv(professor)
            for row in professor_rows:
                row["professor_final_question_function_present"] = "true"
                row["professor_final_purposes"] = '["Clarification"]'
            _write_csv(professor, professor_fields, professor_rows)
            create_human_final_checkpoint(
                run_dir, "review-students", professor, returns, checkpoint_id="human-students"
            )
            checkpoint = Path(tmp) / "checkpoints/human_final/human-students"
            self.assertTrue((checkpoint / "human_returns/student_0001.csv").is_file())
            self.assertTrue((checkpoint / "human_returns/student_0002.csv").is_file())
            self.assertEqual(len(read_jsonl(checkpoint / "clean_final.jsonl")), 2)

    def test_stage2_checkpoint_reuse_accepts_extra_old_scope_but_rejects_hash_change(self):
        with TemporaryDirectory() as tmp, self.roots(Path(tmp)):
            run_dir = Path(tmp) / "run-d"
            run_dir.mkdir()
            utterance_source = run_dir / "04_justice_utterances/justice_utterances.jsonl"
            utterance_source.parent.mkdir(parents=True)
            utterance_source.write_text(json.dumps(identity()) + "\n")
            write_run_state(run_dir, "review_checkpoint_pending", "stage1")
            stage1_student, stage1_professor = self.stage1_review(run_dir)
            create_review_checkpoint(
                run_dir, "stage1", student_working_path=stage1_student,
                professor_working_path=stage1_professor, artifact_paths=[], checkpoint_id="review-d-stage1",
            )
            stage1_fields, stage1_rows = _read_csv(stage1_professor)
            stage1_rows[0]["professor_final_question_function_present"] = "true"
            stage1_rows[0]["professor_final_purposes"] = '["Clarification"]'
            _write_csv(stage1_professor, stage1_fields, stage1_rows)
            create_human_final_checkpoint(
                run_dir, "review-d-stage1", stage1_professor, [], checkpoint_id="human-d-stage1"
            )
            (run_dir / "05_question_purpose/question_purpose_final.jsonl").write_text('{"tampered":true}\n')
            write_run_state(run_dir, "review_checkpoint_pending", "stage2")
            student = run_dir / "06_reason_alignment/reason_alignment_student_review.csv"
            professor = run_dir / "08_human_review/reason_alignment_professor_adjudication.csv"
            fields = [
                "oyez_argument_id", "utterance_id", "turn_id", "target_justice", "text",
                "candidate_targets", "candidate_reason_ids",
                "student_id", "student_selected_reason_ids", "student_none_of_the_above", "student_notes",
                "professor_final_selected_reason_ids", "professor_none_of_the_above", "professor_notes",
            ]
            row = {
                **identity(),
                "candidate_targets": json.dumps([row for row in tree() if row["brief_id"] == "b1"]),
                "candidate_reason_ids": '["r1","r1a"]', "student_id": "",
                "student_selected_reason_ids": "", "student_none_of_the_above": "", "student_notes": "",
                "professor_final_selected_reason_ids": "", "professor_none_of_the_above": "", "professor_notes": "",
            }
            _write_csv(student, fields, [row])
            _write_csv(professor, fields, [row])
            create_review_checkpoint(
                run_dir, "stage2", student_working_path=student, professor_working_path=professor,
                artifact_paths=[], checkpoint_id="review-d",
            )
            fields, rows = _read_csv(professor)
            rows[0]["professor_final_selected_reason_ids"] = '[]'
            rows[0]["professor_none_of_the_above"] = "true"
            _write_csv(professor, fields, rows)
            old_dependency = {
                "run_scope_mapping": [
                    {"oyez_argument_id": "25500", "brief_ids": ["b1"]},
                    {"oyez_argument_id": "extra", "brief_ids": ["b2"]},
                ],
                "briefs": {
                    "b1": {"pdf_sha256": "p1", "reason_tree_sha256": "t1"},
                    "b2": {"pdf_sha256": "p2", "reason_tree_sha256": "t2"},
                },
            }
            create_human_final_checkpoint(
                run_dir, "review-d", professor, [], reason_nodes=tree(),
                stage2_dependency_identity=old_dependency, checkpoint_id="human-d",
            )
            required = [{**identity(), "advocate_group": "pet", "run_scope_brief_ids": ["b1"]}]
            new_dependency = {
                "run_scope_mapping": [{"oyez_argument_id": "25500", "brief_ids": ["b1"]}],
                "briefs": {"b1": {"pdf_sha256": "p1", "reason_tree_sha256": "t1"}},
            }
            consumed = consume_stage2_checkpoint("human-d", required, tree(), new_dependency)
            self.assertEqual(consumed[0]["final_reason_bundle"]["terminal_selected_reason_ids"], [])
            self.assertEqual(
                consume_stage2_checkpoint("human-d", required, list(reversed(tree())), new_dependency)[0]
                ["final_reason_bundle"]["terminal_selected_reason_ids"],
                [],
            )
            self.assertEqual(len(required_stage2_utterances("human-d", required)), 1)
            with self.assertRaises(ValueError):
                required_stage2_utterances("human-d", [*required, identity("u2")])
            changed = json.loads(json.dumps(new_dependency))
            changed["briefs"]["b1"]["reason_tree_sha256"] = "changed"
            with self.assertRaises(ValueError):
                consume_stage2_checkpoint("human-d", required, tree(), changed)
            changed_group = [{**required[0], "advocate_group": "res"}]
            with self.assertRaises(ValueError):
                consume_stage2_checkpoint("human-d", changed_group, tree(), new_dependency)
            expanded_tree = [
                *tree(),
                {
                    "brief_id": "b1", "brief_group": "pet", "reason_target_id": "r-new",
                    "parent_reason_target_id": "", "direct_child_reason_target_ids": [],
                    "heading_level": 1, "heading_path": "II", "heading_text": "New root",
                },
            ]
            with self.assertRaises(ValueError):
                consume_stage2_checkpoint("human-d", required, expanded_tree, new_dependency)


if __name__ == "__main__":
    unittest.main()
