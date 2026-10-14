import csv
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from pipeline.justice_contracts import STANCE_OUTCOMES
from pipeline.justice_workflow import (
    adjudicator_model,
    annotator_models,
    build_local_dialogue_contexts,
    alignment_review_fields,
    alignment_target_prompt_payload,
    local_dialogue_context_text,
    question_purpose_review_fields,
    run_reason_alignment,
    stage2_candidate_pool,
    stance_review_fields,
    stance_bundles,
    stance_target_payload,
    utterance_prompt_payload,
    write_student_review_sheets,
)
from pipeline.common import load_config
from pipeline.justice_runtime import configure_model_profile


class JusticeWorkflowTests(unittest.TestCase):








    def test_local_context_uses_opening_and_neighboring_turns(self):
        transcript_rows = [
            {"case_id": "24-1", "turn_id": "a0", "speaker": "Pet Counsel", "text": "Opening argument starts here.", "section_counsel": "Pet Counsel", "phase": "petitioner"},
            {"case_id": "24-1", "turn_id": "j0", "speaker": "Elena Kagan", "speaker_role": "scotus_justice", "text": "First justice question.", "section_counsel": "Pet Counsel", "phase": "petitioner"},
            {"case_id": "24-1", "turn_id": "a1", "speaker": "Pet Counsel", "text": "First answer from counsel.", "section_counsel": "Pet Counsel", "phase": "petitioner"},
            {"case_id": "24-1", "turn_id": "j1", "speaker": "Amy Coney Barrett", "speaker_role": "scotus_justice", "text": "Current utterance.", "section_counsel": "Pet Counsel", "phase": "petitioner"},
            {"case_id": "24-1", "turn_id": "a2", "speaker": "Pet Counsel", "text": "Immediate answer after the current utterance.", "section_counsel": "Pet Counsel", "phase": "petitioner"},
            {"case_id": "24-1", "turn_id": "j2", "speaker": "Neil Gorsuch", "speaker_role": "scotus_justice", "text": "Another justice follows up.", "section_counsel": "Pet Counsel", "phase": "petitioner"},
            {"case_id": "24-1", "turn_id": "a3", "speaker": "Pet Counsel", "text": "Follow-up answer.", "section_counsel": "Pet Counsel", "phase": "petitioner"},
            {"case_id": "24-1", "turn_id": "r0", "speaker": "Resp Counsel", "text": "Respondent opening.", "section_counsel": "Resp Counsel", "phase": "respondent"},
        ]
        for row in transcript_rows:
            row["advocate_group"] = "res" if row["turn_id"] == "r0" else "pet"
        utterances = [{
            "case_id": "24-1",
            "turn_id": "j1",
            "target_justice": "Amy Coney Barrett",
            "section_counsel": "Pet Counsel",
            "advocate_group": "pet",
            "text": "Current utterance.",
        }]
        enriched = build_local_dialogue_contexts(
            transcript_rows,
            utterances,
            {"annotation_run": {"local_context": {"enabled": True, "type": "same_advocate_group_local_window_with_opening", "before_tokens": 120, "after_tokens": 120, "long_flag_threshold": 1500}}},
        )
        context = enriched[0]["local_dialogue_context"]
        self.assertIn("Opening argument starts here.", context)
        self.assertIn("First justice question.", context)
        self.assertIn("Current utterance.", context)
        self.assertIn("Immediate answer after the current utterance.", context)
        self.assertIn("Another justice follows up.", context)
        self.assertNotIn("Respondent opening.", context)

    def test_local_context_does_not_read_a_live_case_manifest(self):
        transcript_rows = [{
            "case_id": "24-1",
            "oyez_argument_id": "25500",
            "turn_id": "j1",
            "speaker": "Justice Test",
            "speaker_role": "scotus_justice",
            "text": "Current utterance.",
            "section_counsel": "Pet Counsel",
            "advocate_group": "pet",
        }]
        utterances = [{
            **transcript_rows[0],
            "utterance_id": "u1",
            "target_justice": "Justice Test",
        }]
        with patch("pipeline.common.load_yaml", side_effect=AssertionError("live manifest read")):
            enriched = build_local_dialogue_contexts(
                transcript_rows,
                utterances,
                {"paths": {"case_manifest": "must-not-be-read.yaml"}},
            )
        self.assertEqual(enriched[0]["local_dialogue_context_advocate_group"], "pet")

    def test_local_context_rejects_transcript_without_advocate_group(self):
        transcript_rows = [{
            "oyez_argument_id": "25500",
            "turn_id": "j1",
            "speaker": "Justice Test",
            "text": "Current utterance.",
        }]
        utterances = [{
            **transcript_rows[0],
            "utterance_id": "u1",
            "target_justice": "Justice Test",
        }]
        with self.assertRaisesRegex(ValueError, "missing advocate_group"):
            build_local_dialogue_contexts(transcript_rows, utterances, {})

    def test_local_context_disabled_returns_empty_common_fields(self):
        rows = build_local_dialogue_contexts(
            [],
            [{"utterance_id": "u1", "section_counsel": "Counsel", "advocate_group": "pet"}],
            {"annotation_run": {"local_context": {"enabled": False}}},
        )
        self.assertEqual(rows[0]["local_dialogue_context"], "")
        self.assertEqual(rows[0]["local_dialogue_context_type"], "disabled")
        self.assertFalse(rows[0]["local_dialogue_context_long_flag"])

    def test_local_context_records_long_flag_and_rejects_unsupported_type(self):
        transcript = [{
            "oyez_argument_id": "a1",
            "turn_id": "t1",
            "speaker": "Justice Test",
            "speaker_role": "scotus_justice",
            "text": "A sufficiently long current utterance.",
            "section_counsel": "Counsel",
            "advocate_group": "pet",
        }]
        utterances = [{**transcript[0], "utterance_id": "u1", "target_justice": "Justice Test"}]
        rows = build_local_dialogue_contexts(
            transcript,
            utterances,
            {"annotation_run": {"local_context": {"long_flag_threshold": 1}}},
        )
        self.assertTrue(rows[0]["local_dialogue_context_long_flag"])
        with self.assertRaisesRegex(ValueError, "Unsupported local dialogue-context type"):
            build_local_dialogue_contexts(
                transcript,
                utterances,
                {"annotation_run": {"local_context": {"type": "unsupported"}}},
            )

    def test_local_context_writes_one_stage_neutral_audit(self):
        transcript = [{
            "oyez_argument_id": "a1",
            "turn_id": "t1",
            "speaker": "Justice Test",
            "speaker_role": "scotus_justice",
            "text": "Current utterance.",
            "section_counsel": "Counsel",
            "advocate_group": "pet",
        }]
        utterances = [{**transcript[0], "utterance_id": "u1", "target_justice": "Justice Test"}]
        with TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            build_local_dialogue_contexts(transcript, utterances, {}, run_dir)
            audit_path = run_dir / "04_justice_utterances/local_dialogue_context_audit.jsonl"
            audit = json.loads(audit_path.read_text(encoding="utf-8").strip())
            self.assertEqual(audit["dialogue_context_scope_key"], "a1::pet")
            self.assertEqual(
                audit["dialogue_context_type"],
                "same_advocate_group_local_window_with_opening",
            )

    def test_justice_prompts_include_dialogue_context_guardrails(self):
        prompt_paths = [
            "prompts/justice_question_purpose_prompt.txt",
            "prompts/justice_question_purpose_adjudication_prompt.txt",
            "prompts/justice_alignment_prompt.txt",
            "prompts/justice_alignment_adjudication_prompt.txt",
            "prompts/justice_stance_prompt.txt",
            "prompts/justice_stance_adjudication_prompt.txt",
        ]
        for path in prompt_paths:
            text = Path(path).read_text(encoding="utf-8")
            self.assertIn("{{dialogue_context}}", text)
            self.assertIn("Justice utterance", text)
        stance_text = Path("prompts/justice_stance_prompt.txt").read_text(encoding="utf-8")
        adjudication_text = Path("prompts/justice_stance_adjudication_prompt.txt").read_text(encoding="utf-8")
        tree_explanation = (
            "The `reason_tree` contains only the Stage 2 human-final selected bundle; `children` are included "
            "descendants within that bundle, and `selected_terminal=true` marks a selected terminal reason."
        )
        self.assertIn("Stance evidence must come from the current Justice utterance", stance_text)
        self.assertIn("Do not treat advocate answers as the current Justice's stance", adjudication_text)
        self.assertIn(tree_explanation, stance_text)
        self.assertIn(tree_explanation, adjudication_text)
        purpose_text = Path("prompts/justice_question_purpose_prompt.txt").read_text(encoding="utf-8")
        self.assertIn("substantive analytical meaning", purpose_text)
        purpose_adjudication = Path(
            "prompts/justice_question_purpose_adjudication_prompt.txt"
        ).read_text(encoding="utf-8")
        for required in [
            "Pure fillers, apologies, acknowledgments, and incomplete fragments",
            "Background: Questions seeking factual or procedural information",
            "Clarification: questions seeking to get an advocate",
            "Implications: questions about 1) the outer limits",
            "Support: Questions offering support",
            "Criticism: Questions criticizing",
            "Communicate: Questions designed primarily to communicate",
            "Humor: Questions designed to interject humor",
        ]:
            self.assertIn(required, purpose_adjudication)
        self.assertNotIn("same purpose definitions and substantive standard", purpose_adjudication)
        self.assertIn("expresses one or more oral-argument question purposes", purpose_text)

    def test_justice_stance_directions_and_bundle_outcome_use_exact_definitions(self):
        expected_labels = {"Favor", "Against", "Neutral", "Mixed"}
        self.assertEqual(set(STANCE_OUTCOMES), expected_labels)

        definitions = [
            "Favor: A justice's utterance can be labeled as Favor if it can reasonably be inferred that the Justice conveys support for or agreement with the target legal proposition, either explicitly or implicitly.",
            "Against: A justice's utterance can be labeled as Against if it can reasonably be inferred that the Justice conveys opposition to or disagreement with the target legal proposition, either explicitly or implicitly.",
            "Neutral: A justice's utterance can be labeled as Neutral if it cannot reasonably be inferred that the Justice conveys either support for or opposition to the target legal proposition, either explicitly or implicitly.",
        ]
        prompt_text = Path("prompts/justice_stance_prompt.txt").read_text(encoding="utf-8")
        adjudication_text = Path("prompts/justice_stance_adjudication_prompt.txt").read_text(encoding="utf-8")
        for definition in definitions:
            self.assertIn(f"- {definition}", prompt_text)
            self.assertIn(f"- {definition}", adjudication_text)
        self.assertIn("Favor | Against | Neutral | Mixed", prompt_text)
        self.assertIn("Favor | Against | Neutral | Mixed", adjudication_text)



    def test_stage2_candidate_pool_uses_exact_advocate_group(self):
        utterance = {
            "oyez_argument_id": "25500", "text": "Question", "advocate_group": "caa",
            "advocate_side": "neutral", "run_scope_brief_ids": ["pet", "rsp", "caa"],
        }
        targets = [
            {"case_id": "22-7466", "reason_target_id": "pet1", "brief_group": "pet", "brief_id": "pet", "heading_path": "I"},
            {"case_id": "22-7466", "reason_target_id": "rsp1", "brief_group": "rsp", "brief_id": "rsp", "heading_path": "I"},
            {"case_id": "22-7466", "reason_target_id": "caa1", "brief_group": "caa", "brief_id": "caa", "heading_path": "I"},
            {"case_id": "23-477", "reason_target_id": "other", "brief_group": "caa", "brief_id": "other", "heading_path": "I"},
        ]
        pool, metadata = stage2_candidate_pool(utterance, targets)
        self.assertEqual([row["reason_target_id"] for row in pool], ["caa1"])
        self.assertEqual(metadata["target_pool_source"], "exact_advocate_group")
        self.assertEqual(metadata["candidate_pool_size"], 1)

    def test_stage2_candidate_pool_raises_without_exact_group(self):
        targets = [
            {"case_id": "24-1", "reason_target_id": "pet1", "brief_group": "pet", "brief_id": "pet", "heading_path": "I"},
            {"case_id": "24-1", "reason_target_id": "res1", "brief_group": "res", "brief_id": "res", "heading_path": "I"},
        ]
        with self.assertRaises(ValueError):
            stage2_candidate_pool(
                {"oyez_argument_id": "1", "text": "Question", "advocate_group": "rsp", "advocate_side": "petitioner", "run_scope_brief_ids": ["pet", "res"]},
                targets,
            )
        with self.assertRaises(ValueError):
            stage2_candidate_pool(
                {"oyez_argument_id": "1", "text": "Question", "advocate_group": "other", "advocate_side": "", "run_scope_brief_ids": ["pet", "res"]},
                targets,
            )

    def test_llm_only_alignment_sets_final_bundle_to_adjudicated_bundle(self):
        with TemporaryDirectory() as tmp, \
                patch("pipeline.justice_workflow.annotator_models", return_value=[
                    {"model_id": "m1", "model_name": "m1"}, {"model_id": "m2", "model_name": "m2"}
                ]), \
                patch("pipeline.justice_workflow.adjudicator_model", return_value={"model_id": "adj", "model_name": "adj"}), \
                patch("pipeline.justice_workflow._traverse_stage2_model") as traverse:
            traverse.side_effect = [
                {
                    "model": model,
                    "valid": True,
                    "roots": [{
                        "root_identity": ["b1", "r1"], "valid": True,
                        "traversal": [{"reason_target_id": "r1", "heading_path": "I", "current_node_relevant": True,
                                       "utterance_evidence_span": "the rule", "rationale": "", "valid": True}],
                        "terminal_reason_alignments": [{"terminal_reason_id": "r1", "utterance_evidence_span": "the rule"}],
                    }],
                    "traversal": [{"reason_target_id": "r1", "heading_path": "I", "current_node_relevant": True,
                                   "utterance_evidence_span": "the rule", "rationale": "", "valid": True}],
                    "terminal_reason_alignments": [{"terminal_reason_id": "r1", "utterance_evidence_span": "the rule"}],
                }
                for model in ["m1", "m2"]
            ]
            rows = run_reason_alignment(
                {"progress": {"enabled": False}},
                Path(tmp),
                [{"utterance_id": "u1", "speaker": "Justice", "text": "What is the rule? the rule"}],
                [{
                    "utterance_id": "u1",
                    "candidate_rank": 1,
                    "reason_target_id": "r1",
                    "brief_id": "b1",
                    "heading_path": "I",
                    "heading_text": "Selected",
                    "heading_level": 1,
                    "parent_reason_target_id": "",
                    "direct_child_reason_target_ids": [],
                }],
            )
        self.assertTrue(rows[0]["stage2_resolved"])
        self.assertEqual(rows[0]["adjudicated_reason_bundle"]["terminal_selected_reason_ids"], ["r1"])

    def test_alignment_target_prompt_payload_is_minimal(self):
        payload = alignment_target_prompt_payload({
            "reason_target_id": "r1",
            "brief_id": "b1",
            "side": "petitioner",
            "heading_path": "I.A",
            "heading_text": "Reason",
            "reason_path_text": "I. Top / I.A. Reason",
            "parent_heading_path": "I",
            "direct_child_paths": ["I.A.1"],
            "lexical_score": 0.8,
        })
        self.assertEqual(set(payload), {
            "reason_target_id", "heading_path", "heading_text",
        })

    def test_review_fields_do_not_include_review_priority_flag(self):
        field = "needs" + "_human_review"
        self.assertNotIn(field, question_purpose_review_fields())
        self.assertNotIn(field, alignment_review_fields())
        self.assertNotIn(field, stance_review_fields())

    def test_stage3_review_fields_use_complete_bundle(self):
        fields = stance_review_fields()
        self.assertIn("reason_tree", fields)
        self.assertNotIn("complete_reason_bundle", fields)
        self.assertNotIn("complete_bundle_flat_text", fields)
        self.assertIn("favor_target_reason_paths", fields)
        self.assertNotIn("stance_pair_id", fields)
        self.assertIn("llm_dialogue_context", fields)

    def test_all_stage_review_fields_include_full_llm_context(self):
        self.assertIn("llm_dialogue_context", question_purpose_review_fields())
        self.assertIn("llm_dialogue_context", alignment_review_fields())
        self.assertIn("llm_dialogue_context", stance_review_fields())

    def test_prompt_utterance_payload_is_strictly_minimal(self):
        payload = utterance_prompt_payload({
            "utterance_id": "u1",
            "speaker": "JUSTICE TEST",
            "text": "Question",
            "case_id": "24-1",
            "adjudicated_purpose_labels": ["Clarification"],
            "local_dialogue_context": "LOCAL",
        })
        self.assertEqual(payload, {
            "utterance_id": "u1",
            "speaker": "JUSTICE TEST",
            "text": "Question",
        })

    def test_prompt_dialogue_context_is_exact_raw_text(self):
        row = {
            "local_dialogue_context": "First turn.\nSecond turn.",
            "local_dialogue_context_type": "metadata must not be included",
        }
        self.assertEqual(local_dialogue_context_text(row), "First turn.\nSecond turn.")

    def test_review_sheet_copies_full_stage_context_exactly(self):
        with TemporaryDirectory() as tmp:
            stage_dir = Path(tmp)
            full_context = "First turn.\nSecond turn.\nThird turn."
            write_student_review_sheets(
                stage_dir,
                "question_purpose",
                [{"utterance_id": "u1", "text": "Question", "local_dialogue_context": full_context}],
                question_purpose_review_fields(),
            )
            with (stage_dir / "question_purpose_student_review.csv").open(
                "r", encoding="utf-8", newline=""
            ) as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(rows[0]["llm_dialogue_context"], full_context)

    def test_justice_config_uses_cross_company_three_model_workflow(self):
        config = load_config("config/case_sets/justice_2024.yaml")
        config, _ = configure_model_profile(config, "frontier")
        models = annotator_models(config)
        adjudicator = adjudicator_model(config)
        self.assertEqual([model["model_id"] for model in models], ["gpt_5_2", "gemini_2_5_pro"])
        self.assertEqual([model["model_name"] for model in models], ["openai/gpt-5.2", "google/gemini-2.5-pro"])
        self.assertEqual(adjudicator["model_id"], "claude_opus_4_5")
        self.assertEqual(adjudicator["model_name"], "anthropic/claude-opus-4.5")
        self.assertTrue(config["annotation_run"]["local_context"]["enabled"])
        self.assertEqual(config["annotation_run"]["local_context"]["before_tokens"], 450)
        self.assertEqual(config["annotation_run"]["local_context"]["after_tokens"], 450)
        self.assertEqual(
            config["annotation_run"]["local_context"]["type"],
            "same_advocate_group_local_window_with_opening",
        )









    def test_stage3_uses_one_complete_bundle_per_utterance(self):
        alignment_rows = [{
            "case_id": "24-1",
            "docket_number": "24-1",
            "case_name": "Demo",
            "utterance_id": "u1",
            "turn_id": "t1",
            "speaker": "Amy Coney Barrett",
            "text": "Question",
            "local_dialogue_context": "LOCAL CONTEXT",
            "local_dialogue_context_type": "same_advocate_group_local_window_with_opening",
            "local_dialogue_context_turn_count": 3,
            "local_dialogue_context_token_estimate": 30,
            "stage2_resolved": True,
            "adjudicated_none_of_the_above": False,
            "adjudicated_reason_bundle": {
                "bundle_id": "u1__reason_bundle",
                "terminal_selected_reason_ids": ["r2"],
                "selected_hierarchical_paths": [{
                    "terminal_reason_id": "r2",
                    "brief_id": "b1",
                    "hierarchical_reason_node_ids": ["r1", "r2"],
                    "hierarchical_reason_nodes": [
                        {"reason_target_id": "r1", "heading_path": "I", "heading_text": "First"},
                        {"reason_target_id": "r2", "heading_path": "I.A", "heading_text": "Selected"},
                    ],
                    "reason_path_text": "I. First / I.A. Selected",
                }],
            },
        }]
        bundles = stance_bundles(alignment_rows)
        self.assertEqual(len(bundles), 1)
        self.assertEqual(bundles[0]["bundle_id"], "u1__reason_bundle")
        self.assertEqual(bundles[0]["complete_bundle_flat_text"], "I. First / I.A. Selected")
        self.assertEqual(
            bundles[0]["reason_tree"],
            [{
                "brief_id": "b1",
                "roots": [{
                    "reason_target_id": "r1",
                    "heading_path": "I",
                    "heading_text": "First",
                    "selected_terminal": False,
                    "children": [{
                        "reason_target_id": "r2",
                        "heading_path": "I.A",
                        "heading_text": "Selected",
                        "selected_terminal": True,
                        "children": [],
                    }],
                }],
            }],
        )
        self.assertEqual(bundles[0]["local_dialogue_context"], "LOCAL CONTEXT")

    def test_stage3_target_payload_uses_complete_bundle(self):
        bundle = {
            "bundle_id": "u1__reason_bundle",
            "terminal_selected_reason_ids": ["r2"],
            "selected_hierarchical_paths": [{"terminal_reason_id": "r2"}],
            "complete_bundle_flat_text": "I. First / I.A. Selected",
            "reason_tree": [{"brief_id": "b1", "roots": []}],
        }
        payload = stance_target_payload(bundle)
        self.assertEqual(payload, {
            "bundle_id": "u1__reason_bundle",
            "terminal_selected_reason_ids": ["r2"],
            "reason_tree": [{"brief_id": "b1", "roots": []}],
        })
        self.assertNotIn("selected_hierarchical_paths", payload)
        self.assertNotIn("complete_bundle_flat_text", payload)










if __name__ == "__main__":
    unittest.main()
