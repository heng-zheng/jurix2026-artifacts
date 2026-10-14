import json
import unittest

from pipeline.justice_contracts import (
    ancestor_path,
    build_reason_bundle,
    build_selected_reason_tree,
    bundle_reason_ids,
    contract_manifest,
    flatten_reason_bundle,
    locator_paths,
    normalize_whitespace,
    protected_fields_unchanged,
    strict_json_object,
    substantive_stage1_key,
    substantive_stage2_key,
    substantive_stage3_key,
    validate_reason_tree,
    validate_stage1_output,
    validate_stage2_output,
    validate_stage3_output,
)


def tree():
    return [
        {
            "reason_target_id": "r_1",
            "brief_id": "brief_1",
            "heading_path": "1",
            "heading_text": "Statutory argument",
            "heading_level": 1,
            "parent_reason_target_id": "",
            "direct_child_reason_target_ids": ["r_1B"],
        },
        {
            "reason_target_id": "r_1B",
            "brief_id": "brief_1",
            "heading_path": "1.B",
            "heading_text": "The exception limits authority",
            "heading_level": 2,
            "parent_reason_target_id": "r_1",
            "direct_child_reason_target_ids": ["r_1B1"],
        },
        {
            "reason_target_id": "r_1B1",
            "brief_id": "brief_1",
            "heading_path": "1.B.1",
            "heading_text": "The exception is narrow",
            "heading_level": 3,
            "parent_reason_target_id": "r_1B",
            "direct_child_reason_target_ids": [],
        },
        {
            "reason_target_id": "r_2",
            "brief_id": "brief_1",
            "heading_path": "A",
            "heading_text": "Constitutional argument",
            "heading_level": 1,
            "parent_reason_target_id": "",
            "direct_child_reason_target_ids": [],
        },
    ]


class StrictJsonAndEvidenceTests(unittest.TestCase):
    def test_strict_json_accepts_only_complete_object(self):
        self.assertEqual(strict_json_object('{"a":1}')[0], {"a": 1})
        for raw in [
            "```json\n{\"a\":1}\n```",
            "before {\"a\":1}",
            '{"a":1} after',
            "[]",
            '{"a":1}{"b":2}',
            '{"a":NaN}',
            '{"a":Infinity}',
            '{"a":-Infinity}',
        ]:
            with self.subTest(raw=raw):
                parsed, findings = strict_json_object(raw)
                self.assertIsNone(parsed)
                self.assertTrue(findings)

    def test_strict_json_rejects_duplicate_keys_at_any_depth(self):
        cases = [
            ('{"a":1,"a":1}', "a"),
            (
                '{"favor_evidence":{"target_reason_ids":["r1"],"target_reason_ids":["r1"]}}',
                "target_reason_ids",
            ),
            ('{"rationale":{"note":"same","note":"same"}}', "note"),
        ]
        for raw, duplicate_key in cases:
            with self.subTest(duplicate_key=duplicate_key):
                parsed, findings = strict_json_object(raw)
                self.assertIsNone(parsed)
                self.assertEqual(len(findings), 1)
                self.assertIn("duplicate JSON object key", findings[0])
                self.assertIn(repr(duplicate_key), findings[0])
                self.assertIn("Field-level checks were not performed", findings[0])

    def test_contract_manifest_seals_duplicate_key_rule(self):
        manifest = contract_manifest()
        self.assertEqual(manifest["contract_version"], 2)
        self.assertEqual(
            manifest["common"]["json"],
            "one_complete_object_no_fences_no_surrounding_text_no_unknown_fields_no_duplicate_object_keys",
        )

    def test_whitespace_normalization_is_only_whitespace(self):
        self.assertEqual(normalize_whitespace("  A\n\t B  "), "A B")
        self.assertNotEqual(normalize_whitespace("A, B"), normalize_whitespace("A B"))
        self.assertNotEqual(normalize_whitespace("Favor"), normalize_whitespace("favor"))


class StageContractTests(unittest.TestCase):
    def test_stage1_valid_positive_and_nonvalidating_auxiliary_fields(self):
        payload = {
            "question_function_present": True,
            "purposes": [{"label": " clarification ", "evidence_span": "what\n is the rule"}],
            "confidence": {"unexpected": "type"},
        }
        result = validate_stage1_output(json.dumps(payload), "Please tell me what is the rule today.")
        self.assertTrue(result["valid"])
        self.assertEqual(result["normalized"]["purposes"][0]["label"], "Clarification")
        self.assertNotIn("rationale", result["normalized"])
        self.assertEqual(result["normalized"]["confidence"], {"unexpected": "type"})

    def test_stage1_adjudicator_uses_unified_auxiliary_field_names(self):
        payload = {
            "question_function_present": True,
            "purposes": [{"label": "Clarification", "evidence_span": "what is the rule"}],
            "rationale": "short",
            "confidence": 0.8,
            "uncertainty_notes": "",
        }
        result = validate_stage1_output(json.dumps(payload), "Please explain what is the rule.", adjudicator=True)
        self.assertTrue(result["valid"])
        prefixed = dict(payload)
        prefixed["adjudicated_question_function_present"] = prefixed.pop("question_function_present")
        prefixed["adjudicated_purposes"] = prefixed.pop("purposes")
        self.assertFalse(validate_stage1_output(json.dumps(prefixed), "Please explain what is the rule.", adjudicator=True)["valid"])

    def test_stage1_rejects_silent_negative_paths(self):
        invalid = [
            {"purposes": []},
            {"question_function_present": "false", "purposes": []},
            {"question_function_present": True, "purposes": []},
            {"question_function_present": False, "purposes": [{"label": "Humor", "evidence_span": "joke"}]},
            {"question_function_present": True, "purposes": [{"label": "Clarify", "evidence_span": "rule"}]},
            {"question_function_present": True, "purposes": [{"label": "Humor", "evidence_span": ""}]},
            {"question_function_present": False, "purposes": [], "purpose_1": "Humor"},
        ]
        for payload in invalid:
            with self.subTest(payload=payload):
                result = validate_stage1_output(json.dumps(payload), "rule joke")
                self.assertFalse(result["valid"])
                self.assertTrue(result["validation_findings"])

    def test_stage1_duplicate_after_normalization_is_invalid(self):
        payload = {
            "question_function_present": True,
            "purposes": [
                {"label": "Humor", "evidence_span": "that is funny"},
                {"label": " humor ", "evidence_span": "that is funny"},
            ],
        }
        result = validate_stage1_output(json.dumps(payload), "that is funny")
        self.assertFalse(result["valid"])

    def test_stage1_agreement_is_unordered_but_evidence_sensitive(self):
        left = validate_stage1_output(json.dumps({
            "question_function_present": True,
            "purposes": [
                {"label": "Support", "evidence_span": "text grants authority"},
                {"label": "Clarification", "evidence_span": "what does it mean"},
            ],
        }), "what does it mean when the text grants authority")
        right = validate_stage1_output(json.dumps({
            "question_function_present": True,
            "purposes": [
                {"label": "Clarification", "evidence_span": "what  does it mean"},
                {"label": "Support", "evidence_span": "text grants authority"},
            ],
            "rationale": "different",
        }), "what does it mean when the text grants authority")
        self.assertEqual(substantive_stage1_key(left["normalized"]), substantive_stage1_key(right["normalized"]))
        right["normalized"]["purposes"][0]["evidence_span"] = "what does it mean when"
        self.assertNotEqual(substantive_stage1_key(left["normalized"]), substantive_stage1_key(right["normalized"]))

    def test_stage2_boolean_and_evidence_contract(self):
        valid_true = validate_stage2_output(
            '{"current_node_relevant":true,"alignment_evidence_span":"the exception is narrow"}',
            "I think the exception is narrow here.",
        )
        valid_false = validate_stage2_output(
            '{"current_node_relevant":false,"alignment_evidence_span":""}',
            "Anything",
        )
        self.assertTrue(valid_true["valid"])
        self.assertTrue(valid_false["valid"])
        for raw in [
            '{"current_node_relevant":"false","alignment_evidence_span":""}',
            '{"current_node_relevant":true,"alignment_evidence_span":""}',
            '{"current_node_relevant":false,"alignment_evidence_span":"text"}',
            '{"current_node_relevant":false,"alignment_evidence_span":"","none_of_the_above":true}',
            '{"current_node_relevant":true,"alignment_evidence_span":"text","relevant_child_ids":[]}',
        ]:
            self.assertFalse(validate_stage2_output(raw, "text")["valid"])

    def test_stage2_agreement_uses_terminal_evidence(self):
        left = [{"terminal_reason_id": "r2", "utterance_evidence_span": "same\nspan"}]
        right = [{"terminal_reason_id": "r2", "utterance_evidence_span": "same span"}]
        self.assertEqual(substantive_stage2_key(left), substantive_stage2_key(right))
        right[0]["utterance_evidence_span"] = "different span"
        self.assertNotEqual(substantive_stage2_key(left), substantive_stage2_key(right))

    def test_stage3_topologies_and_auxiliary_fields(self):
        utterance = "the text grants authority but the exception is narrow"
        ids = {"r_1", "r_1B", "r_1B1"}
        valid = {
            "Favor": {"stance_outcome": "Favor", "favor_evidence": {"utterance_evidence_span": "text grants authority", "target_reason_ids": ["r_1"]}, "against_evidence": None},
            "Against": {"stance_outcome": "Against", "favor_evidence": None, "against_evidence": {"utterance_evidence_span": "exception is narrow", "target_reason_ids": ["r_1B1"]}},
            "Neutral": {"stance_outcome": "Neutral", "favor_evidence": None, "against_evidence": None},
            "Mixed": {"stance_outcome": "Mixed", "favor_evidence": {"utterance_evidence_span": "text grants authority", "target_reason_ids": ["r_1"]}, "against_evidence": {"utterance_evidence_span": "exception is narrow", "target_reason_ids": ["r_1B1"]}},
        }
        for outcome, payload in valid.items():
            payload["confidence"] = ["nonvalidating"]
            result = validate_stage3_output(json.dumps(payload), utterance, ids)
            self.assertTrue(result["valid"], outcome)

    def test_stage3_rejects_invalid_neutral_mapping_and_bundle_external_id(self):
        invalid = [
            {"stance_outcome": "Neutral", "favor_evidence": {"utterance_evidence_span": "text", "target_reason_ids": ["r_1"]}, "against_evidence": None},
            {"stance_outcome": "Favor", "favor_evidence": {"utterance_evidence_span": "text", "target_reason_ids": ["fake"]}, "against_evidence": None},
            {"stance_outcome": "Favor", "favor_evidence": {"utterance_evidence_span": "text", "target_reason_ids": []}, "against_evidence": None},
            {"stance_outcome": "Favor", "favor_evidence": {"utterance_evidence_span": "text", "target_reason_ids": ["r_1", "r_1"]}, "against_evidence": None},
        ]
        for payload in invalid:
            result = validate_stage3_output(json.dumps(payload), "text", {"r_1"})
            self.assertFalse(result["valid"])
            self.assertFalse(result["same_reason_conflict"])

    def test_stage3_same_reason_requires_legal_exact_intersection(self):
        conflict = {
            "stance_outcome": "Mixed",
            "favor_evidence": {"utterance_evidence_span": "favor words", "target_reason_ids": ["r_1", "r_2"]},
            "against_evidence": {"utterance_evidence_span": "against words", "target_reason_ids": ["r_2", "r_3"]},
        }
        result = validate_stage3_output(json.dumps(conflict), "favor words and against words", {"r_1", "r_2", "r_3"})
        self.assertFalse(result["valid"])
        self.assertTrue(result["same_reason_conflict"])
        self.assertTrue(result["validation_findings"])
        conflict["favor_evidence"]["target_reason_ids"] = ["fake"]
        conflict["against_evidence"]["target_reason_ids"] = ["fake"]
        result = validate_stage3_output(json.dumps(conflict), "favor words and against words", {"r_1"})
        self.assertFalse(result["same_reason_conflict"])
        self.assertTrue(result["validation_findings"])

    def test_stage3_agreement_ignores_id_order_and_auxiliary_fields(self):
        left = {
            "stance_outcome": "Favor",
            "favor_evidence": {"utterance_evidence_span": "same span", "target_reason_ids": ["r2", "r1"]},
            "against_evidence": None,
            "rationale": "one",
        }
        right = {
            "stance_outcome": "Favor",
            "favor_evidence": {"utterance_evidence_span": "same\nspan", "target_reason_ids": ["r1", "r2"]},
            "against_evidence": None,
            "rationale": "two",
        }
        self.assertEqual(substantive_stage3_key(left), substantive_stage3_key(right))

    def test_protected_fields_detect_repair_changes(self):
        validation = validate_stage2_output(
            '{"current_node_relevant":true,"alignment_evidence_span":"paraphrase"}',
            "verbatim only",
        )
        self.assertEqual(validation["protected_fields"], {"current_node_relevant": True})
        self.assertTrue(protected_fields_unchanged(validation["protected_fields"], {
            "current_node_relevant": True, "alignment_evidence_span": "verbatim only"
        }))
        self.assertFalse(protected_fields_unchanged(validation["protected_fields"], {
            "current_node_relevant": False, "alignment_evidence_span": ""
        }))

    def test_unknown_field_repair_protects_valid_decision_content(self):
        stage2 = validate_stage2_output(
            '{"current_node_relevant":true,"alignment_evidence_span":"text","relevant_child_ids":[]}',
            "text",
        )
        self.assertFalse(stage2["valid"])
        self.assertEqual(stage2["protected_fields"]["current_node_relevant"], True)
        self.assertEqual(stage2["protected_fields"]["alignment_evidence_span"], "text")
        self.assertTrue(any("remove undefined" in item for item in stage2["repairable_fields"]))

        stage3 = validate_stage3_output(
            json.dumps({
                "stance_outcome": "Favor",
                "favor_evidence": {
                    "utterance_evidence_span": "text",
                    "target_reason_ids": ["r_1"],
                    "reason_text": "not allowed",
                },
                "against_evidence": None,
            }),
            "text",
            {"r_1"},
        )
        self.assertFalse(stage3["valid"])
        self.assertEqual(stage3["protected_fields"]["stance_outcome"], "Favor")
        self.assertEqual(stage3["protected_fields"]["favor_evidence.target_reason_ids"], ["r_1"])
        self.assertTrue(any("remove undefined" in item for item in stage3["repairable_fields"]))


class HierarchyContractTests(unittest.TestCase):
    def test_valid_hierarchy_and_complete_bundle(self):
        by_id = validate_reason_tree(tree())
        self.assertEqual([row["reason_target_id"] for row in ancestor_path("r_1B1", by_id)], ["r_1", "r_1B", "r_1B1"])
        bundle = build_reason_bundle("u1", ["r_1B1", "r_2"], tree())
        self.assertEqual(bundle["bundle_id"], "u1__reason_bundle")
        self.assertEqual(bundle["terminal_selected_reason_ids"], ["r_1B1", "r_2"])
        self.assertEqual(
            bundle["selected_hierarchical_paths"][0]["reason_path_text"],
            "1. Statutory argument / 1.B. The exception limits authority / 1.B.1. The exception is narrow",
        )
        self.assertEqual(bundle_reason_ids(bundle), {"r_1", "r_1B", "r_1B1", "r_2"})
        self.assertIn(" || ", flatten_reason_bundle(bundle))
        self.assertEqual(locator_paths(bundle, ["r_1B"]), ["1. Statutory argument / 1.B. The exception limits authority"])
        selected_tree = build_selected_reason_tree(bundle)
        self.assertEqual([root["reason_target_id"] for root in selected_tree[0]["roots"]], ["r_1", "r_2"])
        self.assertEqual(
            selected_tree[0]["roots"][0]["children"][0]["children"][0]["reason_target_id"],
            "r_1B1",
        )
        self.assertTrue(selected_tree[0]["roots"][0]["children"][0]["children"][0]["selected_terminal"])
        self.assertTrue(selected_tree[0]["roots"][1]["selected_terminal"])
        reordered = build_reason_bundle("u1", ["r_2", "r_1B1"], tree())
        self.assertEqual(reordered, bundle)

    def test_selected_reason_tree_deduplicates_shared_parents_and_preserves_distinct_ids(self):
        rows = [
            {
                "reason_target_id": "b1_root", "brief_id": "brief_1", "heading_path": "I",
                "heading_text": "Shared parent", "heading_level": 1, "parent_reason_target_id": "",
                "direct_child_reason_target_ids": ["b1_a", "b1_b"],
            },
            {
                "reason_target_id": "b1_a", "brief_id": "brief_1", "heading_path": "I.A",
                "heading_text": "Same proposition", "heading_level": 2, "parent_reason_target_id": "b1_root",
                "direct_child_reason_target_ids": [],
            },
            {
                "reason_target_id": "b1_b", "brief_id": "brief_1", "heading_path": "I.B",
                "heading_text": "Second proposition", "heading_level": 2, "parent_reason_target_id": "b1_root",
                "direct_child_reason_target_ids": [],
            },
            {
                "reason_target_id": "b2_root", "brief_id": "brief_2", "heading_path": "I",
                "heading_text": "Other brief", "heading_level": 1, "parent_reason_target_id": "",
                "direct_child_reason_target_ids": ["b2_a"],
            },
            {
                "reason_target_id": "b2_a", "brief_id": "brief_2", "heading_path": "I.A",
                "heading_text": "Same proposition", "heading_level": 2, "parent_reason_target_id": "b2_root",
                "direct_child_reason_target_ids": [],
            },
        ]
        bundle = build_reason_bundle("u1", ["b1_a", "b1_b", "b2_a"], rows)
        selected_tree = build_selected_reason_tree(bundle)
        self.assertEqual([item["brief_id"] for item in selected_tree], ["brief_1", "brief_2"])
        self.assertEqual(len(selected_tree[0]["roots"]), 1)
        self.assertEqual(
            [item["reason_target_id"] for item in selected_tree[0]["roots"][0]["children"]],
            ["b1_a", "b1_b"],
        )
        self.assertEqual(selected_tree[0]["roots"][0]["heading_text"], "Shared parent")
        self.assertEqual(selected_tree[0]["roots"][0]["children"][0]["heading_text"], "Same proposition")
        self.assertEqual(selected_tree[1]["roots"][0]["children"][0]["heading_text"], "Same proposition")

        def reason_ids(nodes):
            result = set()
            for node in nodes:
                result.add(node["reason_target_id"])
                result.update(reason_ids(node["children"]))
            return result

        rendered_ids = {
            reason_id
            for brief in selected_tree
            for reason_id in reason_ids(brief["roots"])
        }
        self.assertEqual(rendered_ids, bundle_reason_ids(bundle))

    def test_selected_reason_tree_rejects_inconsistent_bundle_paths(self):
        rows = [
            {
                "reason_target_id": "root", "brief_id": "brief", "heading_path": "I",
                "heading_text": "Root", "heading_level": 1, "parent_reason_target_id": "",
                "direct_child_reason_target_ids": ["a", "b"],
            },
            {
                "reason_target_id": "a", "brief_id": "brief", "heading_path": "I.A",
                "heading_text": "A", "heading_level": 2, "parent_reason_target_id": "root",
                "direct_child_reason_target_ids": [],
            },
            {
                "reason_target_id": "b", "brief_id": "brief", "heading_path": "I.B",
                "heading_text": "B", "heading_level": 2, "parent_reason_target_id": "root",
                "direct_child_reason_target_ids": [],
            },
        ]
        bundle = build_reason_bundle("u1", ["a", "b"], rows)

        metadata_conflict = json.loads(json.dumps(bundle))
        metadata_conflict["selected_hierarchical_paths"][1]["hierarchical_reason_nodes"][0]["heading_text"] = "Changed"

        terminal_mismatch = json.loads(json.dumps(bundle))
        terminal_mismatch["selected_hierarchical_paths"].reverse()

        multiple_parent = {
            "bundle_id": "u1__reason_bundle",
            "terminal_selected_reason_ids": ["a", "b"],
            "selected_hierarchical_paths": [
                {
                    "terminal_reason_id": "a", "brief_id": "brief",
                    "hierarchical_reason_node_ids": ["root_1", "shared", "a"],
                    "hierarchical_reason_nodes": [
                        {"reason_target_id": "root_1", "heading_path": "I", "heading_text": "Root 1"},
                        {"reason_target_id": "shared", "heading_path": "I.A", "heading_text": "Shared"},
                        {"reason_target_id": "a", "heading_path": "I.A.1", "heading_text": "A"},
                    ],
                },
                {
                    "terminal_reason_id": "b", "brief_id": "brief",
                    "hierarchical_reason_node_ids": ["root_2", "shared", "b"],
                    "hierarchical_reason_nodes": [
                        {"reason_target_id": "root_2", "heading_path": "II", "heading_text": "Root 2"},
                        {"reason_target_id": "shared", "heading_path": "I.A", "heading_text": "Shared"},
                        {"reason_target_id": "b", "heading_path": "I.A.2", "heading_text": "B"},
                    ],
                },
            ],
        }

        for invalid in [metadata_conflict, terminal_mismatch, multiple_parent]:
            with self.subTest(invalid=invalid):
                with self.assertRaises(ValueError):
                    build_selected_reason_tree(invalid)

        self.assertEqual(build_selected_reason_tree({
            "bundle_id": "u1__reason_bundle",
            "terminal_selected_reason_ids": [],
            "selected_hierarchical_paths": [],
        }), [])

    def test_hierarchy_rejects_integrity_failures(self):
        mutations = []
        duplicate = tree() + [dict(tree()[0])]
        mutations.append(duplicate)
        missing_parent = tree(); missing_parent[2] = dict(missing_parent[2], parent_reason_target_id="missing")
        mutations.append(missing_parent)
        cross_brief = tree(); cross_brief[1] = dict(cross_brief[1], brief_id="brief_2")
        mutations.append(cross_brief)
        broken_reciprocal = tree(); broken_reciprocal[0] = dict(broken_reciprocal[0], direct_child_reason_target_ids=[])
        mutations.append(broken_reciprocal)
        wrong_level = tree(); wrong_level[1] = dict(wrong_level[1], heading_level=3)
        mutations.append(wrong_level)
        cycle = tree(); cycle[0] = dict(cycle[0], parent_reason_target_id="r_1B1", heading_level=4); cycle[2] = dict(cycle[2], direct_child_reason_target_ids=["r_1"])
        mutations.append(cycle)
        for rows in mutations:
            with self.subTest(rows=rows):
                with self.assertRaises(ValueError):
                    validate_reason_tree(rows)

    def test_same_heading_path_is_allowed_across_briefs(self):
        rows = tree()
        rows.append({
            "reason_target_id": "other_root", "brief_id": "brief_2", "heading_path": "1",
            "heading_text": "Other", "heading_level": 1, "parent_reason_target_id": "",
            "direct_child_reason_target_ids": [],
        })
        validate_reason_tree(rows)

    def test_terminal_ancestor_and_descendant_cannot_both_be_selected(self):
        with self.assertRaises(ValueError):
            build_reason_bundle("u1", ["r_1B", "r_1B1"], tree())


if __name__ == "__main__":
    unittest.main()
