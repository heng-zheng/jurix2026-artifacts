import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from pipeline.justice_contracts import build_reason_bundle, validate_stage2_output, validate_stage3_output
from pipeline.justice_workflow import (
    _traverse_stage2_model,
    _forced_finish_invalid,
    call_validated_model,
    run_question_purpose,
    run_stance_detection,
    stance_bundles,
)


def tree():
    return [
        {"reason_target_id": "r1", "brief_id": "b", "heading_path": "1", "heading_text": "Root one", "heading_level": 1,
         "parent_reason_target_id": "", "direct_child_reason_target_ids": ["r1a", "r1b"]},
        {"reason_target_id": "r1a", "brief_id": "b", "heading_path": "1.A", "heading_text": "Child A", "heading_level": 2,
         "parent_reason_target_id": "r1", "direct_child_reason_target_ids": []},
        {"reason_target_id": "r1b", "brief_id": "b", "heading_path": "1.B", "heading_text": "Child B", "heading_level": 2,
         "parent_reason_target_id": "r1", "direct_child_reason_target_ids": []},
        {"reason_target_id": "r2", "brief_id": "b", "heading_path": "2", "heading_text": "Root two", "heading_level": 1,
         "parent_reason_target_id": "", "direct_child_reason_target_ids": []},
    ]


def attempt(output, validation, raw=None):
    return {
        "effective_result": {"content": raw if raw is not None else json.dumps(output)},
        "effective_validation": {"normalized": output, "parsed": output, **validation},
        "repair_attempted": False,
    }


class RepairRoutingTests(unittest.TestCase):
    def test_stage1_adjudicator_model_schema_is_mapped_only_at_consolidation(self):
        valid = {"valid": True, "validation_findings": [], "same_reason_conflict": False}
        annotator_a = {
            "question_function_present": True,
            "purposes": [{"label": "Clarification", "evidence_span": "what is the rule"}],
        }
        annotator_b = {
            "question_function_present": True,
            "purposes": [{"label": "Implications", "evidence_span": "what is the rule"}],
        }
        adjudicator = {
            "question_function_present": True,
            "purposes": [{"label": "Support", "evidence_span": "what is the rule"}],
        }
        with TemporaryDirectory() as tmp, patch(
            "pipeline.justice_workflow.annotator_models",
            return_value=[
                {"model_id": "a", "model_name": "author/a"},
                {"model_id": "b", "model_name": "author/b"},
            ],
        ), patch(
            "pipeline.justice_workflow.adjudicator_model",
            return_value={"model_id": "j", "model_name": "author/j"},
        ), patch(
            "pipeline.justice_workflow.call_validated_model",
            side_effect=[attempt(annotator_a, valid), attempt(annotator_b, valid), attempt(adjudicator, valid)],
        ) as call:
            rows = run_question_purpose(
                {},
                Path(tmp),
                [{
                    "utterance_id": "u1",
                    "text": "what is the rule",
                    "local_dialogue_context": "SHARED LOCAL CONTEXT",
                }],
            )
        self.assertEqual(rows[0]["adjudicated_question_function_present"], True)
        self.assertEqual(rows[0]["adjudicated_purposes"], adjudicator["purposes"])
        self.assertNotIn("question_function_present", rows[0])
        self.assertEqual(
            [item.args[4]["dialogue_context"] for item in call.call_args_list],
            ["SHARED LOCAL CONTEXT"] * 3,
        )

    def test_only_stop_finish_reason_can_remain_valid(self):
        base = {"valid": True, "validation_findings": [], "same_reason_conflict": False}
        self.assertTrue(_forced_finish_invalid(base, "stop")["valid"])
        for reason in ["length", "content_filter", "tool_calls", None, "unknown"]:
            result = _forced_finish_invalid(base, reason)
            self.assertFalse(result["valid"])
            self.assertTrue(result["validation_findings"])

    def test_ordinary_invalid_repairs_once_and_preserves_boolean(self):
        initial = {"run_id": "run", "content": '{"current_node_relevant":true,"alignment_evidence_span":"paraphrase"}', "finish_reason": "stop"}
        repaired = {"run_id": "run", "content": '{"current_node_relevant":true,"alignment_evidence_span":"verbatim"}', "finish_reason": "stop"}
        with TemporaryDirectory() as tmp, patch(
            "pipeline.justice_workflow.call_json_model", side_effect=[initial, repaired]
        ) as call:
            result = call_validated_model(
                {}, Path(tmp), {"model_id": "m"}, Path("prompts/justice_alignment_prompt.txt"), {
                    "utterance": "{}", "dialogue_context": "", "ancestor_path": "[]", "current_node": "{}"
                }, Path(tmp) / "calls", "stem",
                "stage2", "u1", "annotator", lambda content: validate_stage2_output(content, "verbatim"), branch_id="r1",
            )
        self.assertEqual(call.call_count, 2)
        self.assertTrue(result["repair_attempted"])
        self.assertTrue(result["effective_validation"]["valid"])

    def test_stage3_duplicate_key_repairs_once_without_protected_decision(self):
        initial = {
            "run_id": "run",
            "content": (
                '{"stance_outcome":"Favor","stance_outcome":"Favor",'
                '"favor_evidence":{"utterance_evidence_span":"favor","target_reason_ids":["r1"]},'
                '"against_evidence":null}'
            ),
            "finish_reason": "stop",
        }
        repaired = {
            "run_id": "run",
            "content": json.dumps({
                "stance_outcome": "Favor",
                "favor_evidence": {"utterance_evidence_span": "favor", "target_reason_ids": ["r1"]},
                "against_evidence": None,
            }),
            "finish_reason": "stop",
        }
        with TemporaryDirectory() as tmp, patch(
            "pipeline.justice_workflow.call_json_model", side_effect=[initial, repaired]
        ) as call:
            result = call_validated_model(
                {}, Path(tmp), {"model_id": "m"}, Path("prompts/justice_stance_prompt.txt"), {
                    "utterance": "{}", "dialogue_context": "", "target": "{}"
                }, Path(tmp) / "calls", "stem",
                "stage3", "bundle", "annotator", lambda content: validate_stage3_output(content, "favor", {"r1"}),
            )

        repair_values = call.call_args_list[1].args[3]
        self.assertEqual(call.call_count, 2)
        self.assertTrue(result["repair_attempted"])
        self.assertTrue(result["effective_validation"]["valid"])
        self.assertIn("duplicate JSON object key 'stance_outcome'", repair_values["validation_findings"])
        self.assertEqual(repair_values["protected_fields"], "{}")

    def test_stage3_same_reason_skips_repair(self):
        raw = json.dumps({
            "stance_outcome": "Mixed",
            "favor_evidence": {"utterance_evidence_span": "favor", "target_reason_ids": ["r1"]},
            "against_evidence": {"utterance_evidence_span": "against", "target_reason_ids": ["r1"]},
        })
        initial = {"run_id": "run", "content": raw, "finish_reason": "stop"}
        with TemporaryDirectory() as tmp, patch(
            "pipeline.justice_workflow.call_json_model", return_value=initial
        ) as call:
            result = call_validated_model(
                {}, Path(tmp), {"model_id": "m"}, Path("prompt"), {}, Path(tmp) / "calls", "stem",
                "stage3", "bundle", "annotator", lambda content: validate_stage3_output(content, "favor against", {"r1"}),
            )
        self.assertEqual(call.call_count, 1)
        self.assertFalse(result["repair_attempted"])
        self.assertTrue(result["effective_validation"]["same_reason_conflict"])


class PreflightAndHandoffTests(unittest.TestCase):





    def test_relevant_parent_visits_all_children_and_selects_relevant_leaf(self):
        decisions = {
            "r1": {"current_node_relevant": True, "alignment_evidence_span": "root words"},
            "r1a": {"current_node_relevant": True, "alignment_evidence_span": "child a words"},
            "r1b": {"current_node_relevant": False, "alignment_evidence_span": ""},
            "r2": {"current_node_relevant": False, "alignment_evidence_span": ""},
        }

        def fake_call(*args, **kwargs):
            self.assertEqual(args[4]["dialogue_context"], "SHARED LOCAL CONTEXT")
            current = json.loads(args[4]["current_node"])["reason_target_id"]
            return attempt(decisions[current], {"valid": True, "validation_findings": [], "same_reason_conflict": False})

        with TemporaryDirectory() as tmp, patch("pipeline.justice_workflow.call_validated_model", side_effect=fake_call):
            result = _traverse_stage2_model(
                {},
                Path(tmp),
                Path(tmp),
                {
                    "utterance_id": "u",
                    "text": "root words child a words",
                    "local_dialogue_context": "SHARED LOCAL CONTEXT",
                },
                {"model_id": "m"}, tree(),
            )
        self.assertTrue(result["valid"])
        self.assertEqual([x["reason_target_id"] for x in result["traversal"]], ["r1", "r1a", "r1b", "r2"])
        self.assertEqual(result["terminal_reason_alignments"], [{"terminal_reason_id": "r1a", "utterance_evidence_span": "child a words"}])

    def test_invalid_node_stops_its_root_but_other_root_continues(self):
        decisions = {
            "r1": attempt({"current_node_relevant": True, "alignment_evidence_span": "root"}, {"valid": True, "validation_findings": [], "same_reason_conflict": False}),
            "r1a": attempt({}, {"valid": False, "validation_findings": ["invalid"], "same_reason_conflict": False}),
            "r2": attempt({"current_node_relevant": True, "alignment_evidence_span": "second"}, {"valid": True, "validation_findings": [], "same_reason_conflict": False}),
        }

        def fake_call(*args, **kwargs):
            current = json.loads(args[4]["current_node"])["reason_target_id"]
            return decisions[current]

        with TemporaryDirectory() as tmp, patch("pipeline.justice_workflow.call_validated_model", side_effect=fake_call):
            result = _traverse_stage2_model(
                {}, Path(tmp), Path(tmp), {"utterance_id": "u", "text": "root second"}, {"model_id": "m"}, tree(),
            )
        self.assertFalse(result["valid"])
        self.assertEqual([x["reason_target_id"] for x in result["traversal"]], ["r1", "r1a", "r2"])
        self.assertEqual(result["terminal_reason_alignments"], [{"terminal_reason_id": "r2", "utterance_evidence_span": "second"}])


class BundleStageRoutingTests(unittest.TestCase):
    def alignment_row(self):
        bundle = build_reason_bundle("u1", ["r1a", "r2"], tree())
        return {
            "case_id": "c", "utterance_id": "u1", "text": "favor words but against words",
            "local_dialogue_context": "context", "stage2_resolved": True,
            "adjudicated_none_of_the_above": False, "adjudicated_reason_bundle": bundle,
        }

    def test_stage3_bundle_integrity_comparison_ignores_internal_path_order(self):
        row = self.alignment_row()
        reordered_tree = [tree()[3], tree()[0], tree()[1], tree()[2]]
        bundles = stance_bundles([row], reordered_tree)
        self.assertEqual(len(bundles), 1)
        self.assertEqual(set(bundles[0]["terminal_selected_reason_ids"]), {"r1a", "r2"})

    def test_legal_mixed_always_adjudicates_and_blocks_final(self):
        mixed = {
            "stance_outcome": "Mixed",
            "favor_evidence": {"utterance_evidence_span": "favor words", "target_reason_ids": ["r1a"]},
            "against_evidence": {"utterance_evidence_span": "against words", "target_reason_ids": ["r2"]},
        }
        valid = {"valid": True, "validation_findings": [], "same_reason_conflict": False}
        with TemporaryDirectory() as tmp, patch(
            "pipeline.justice_workflow.annotator_models", return_value=[{"model_id": "a"}, {"model_id": "b"}]
        ), patch("pipeline.justice_workflow.adjudicator_model", return_value={"model_id": "adj"}), patch(
            "pipeline.justice_workflow.call_validated_model",
            side_effect=[attempt(mixed, valid), attempt(mixed, valid), attempt(mixed, valid)],
        ) as call:
            out = run_stance_detection({}, Path(tmp), [self.alignment_row()], tree())
            self.assertEqual(call.call_count, 3)
            self.assertEqual(out[0]["adjudicated_stance_outcome"], "Mixed")
            self.assertTrue(out[0]["stage3_model_output_valid"])
            self.assertFalse(out[0]["stage3_resolved"])
            self.assertTrue(out[0]["stage3_requires_human_review"])
            stage_dir = Path(tmp) / "07_reason_stance"
            self.assertFalse((stage_dir / "stance_final.jsonl").exists())
            self.assertFalse((stage_dir / "stance_pairs.jsonl").exists())
            self.assertTrue((stage_dir / "stance_bundles.jsonl").exists())
            self.assertTrue((stage_dir / "stance_student_review.csv").exists())
            self.assertTrue((stage_dir / "stance_professor_review.csv").exists())
            self.assertEqual(
                [item.args[4]["dialogue_context"] for item in call.call_args_list],
                ["context"] * 3,
            )

    def test_valid_nonmixed_agreement_still_waits_for_human_final(self):
        favor = {
            "stance_outcome": "Favor",
            "favor_evidence": {"utterance_evidence_span": "favor words", "target_reason_ids": ["r1", "r1a"]},
            "against_evidence": None,
        }
        valid = {"valid": True, "validation_findings": [], "same_reason_conflict": False}
        with TemporaryDirectory() as tmp, patch(
            "pipeline.justice_workflow.annotator_models", return_value=[{"model_id": "a"}, {"model_id": "b"}]
        ), patch("pipeline.justice_workflow.adjudicator_model", return_value={"model_id": "adj"}), patch(
            "pipeline.justice_workflow.call_validated_model", side_effect=[attempt(favor, valid), attempt(favor, valid)]
        ):
            out = run_stance_detection({}, Path(tmp), [self.alignment_row()], tree())
            stage_dir = Path(tmp) / "07_reason_stance"
            final_exists = (stage_dir / "stance_final.jsonl").exists()
            student_exists = (stage_dir / "stance_student_review.csv").exists()
            professor_exists = (stage_dir / "stance_professor_review.csv").exists()
        self.assertTrue(out[0]["stage3_resolved"])
        self.assertFalse(final_exists)
        self.assertTrue(student_exists)
        self.assertTrue(professor_exists)

    def test_stage3_batch_callback_keeps_existing_bundle_decision_and_writes_no_partial_outputs(self):
        favor = {
            "stance_outcome": "Favor",
            "favor_evidence": {"utterance_evidence_span": "favor words", "target_reason_ids": ["r1", "r1a"]},
            "against_evidence": None,
        }
        valid = {"valid": True, "validation_findings": [], "same_reason_conflict": False}
        captured = []
        with TemporaryDirectory() as tmp, patch(
            "pipeline.justice_workflow.annotator_models", return_value=[{"model_id": "a"}, {"model_id": "b"}]
        ), patch("pipeline.justice_workflow.adjudicator_model", return_value={"model_id": "adj"}), patch(
            "pipeline.justice_workflow.call_validated_model",
            side_effect=[attempt(favor, valid), attempt(favor, valid)],
        ):
            out = run_stance_detection(
                {},
                Path(tmp),
                [self.alignment_row()],
                tree(),
                unit_complete=lambda unit, models, final: captured.append((unit, list(models), final)),
                write_outputs=False,
            )
            stage_dir = Path(tmp) / "07_reason_stance"
            self.assertEqual(captured[0][0]["bundle_id"], "u1__reason_bundle")
            self.assertEqual(captured[0][0]["utterance_id"], "u1")
            self.assertEqual(captured[0][2], out[0])
            self.assertFalse((stage_dir / "stance_bundles.jsonl").exists())
            self.assertFalse((stage_dir / "stance_adjudicated.jsonl").exists())



if __name__ == "__main__":
    unittest.main()
