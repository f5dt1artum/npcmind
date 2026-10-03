import json
import unittest

from npcmind.service import Service


def condition(node_id, op, key, value=None):
    node = {"id": node_id, "type": "condition", "op": op, "key": key}
    if op in ("equals", "not_equals"):
        node["value"] = value
    return node


def action(node_id, op, key=None, value=None, status=None):
    node = {"id": node_id, "type": "action", "op": op}
    if key is not None:
        node["key"] = key
    if op == "set":
        node["value"] = value
    if op == "status":
        node["status"] = status
    return node


class EvaluateBehaviorTest(unittest.TestCase):
    def setUp(self):
        self.service = Service()

    def evaluate(self, tree, blackboard=None):
        request = {"tree": tree}
        if blackboard is not None:
            request["blackboard"] = blackboard
        return self.service.evaluate_behavior(request)

    def test_sequence_all_success(self):
        tree = {
            "id": "root",
            "type": "sequence",
            "children": [
                action("a1", "set", key="x", value=1),
                action("a2", "set", key="y", value=[True, None]),
            ],
        }
        result = self.evaluate(tree)
        self.assertEqual(result["status"], "SUCCESS")
        self.assertEqual(result["blackboard"], {"x": 1, "y": [True, None]})
        self.assertEqual(
            result["trace"],
            [
                {"id": "a1", "type": "action", "status": "SUCCESS"},
                {"id": "a2", "type": "action", "status": "SUCCESS"},
                {"id": "root", "type": "sequence", "status": "SUCCESS"},
            ],
        )

    def test_sequence_short_circuits_on_failure(self):
        tree = {
            "id": "root",
            "type": "sequence",
            "children": [
                action("a1", "status", status="FAILURE"),
                action("a2", "set", key="x", value=1),
            ],
        }
        result = self.evaluate(tree)
        self.assertEqual(result["status"], "FAILURE")
        self.assertEqual(result["blackboard"], {})
        self.assertEqual([t["id"] for t in result["trace"]], ["a1", "root"])

    def test_sequence_stops_on_running(self):
        tree = {
            "id": "root",
            "type": "sequence",
            "children": [
                action("a1", "set", key="x", value=1),
                action("a2", "status", status="RUNNING"),
                action("a3", "set", key="y", value=2),
            ],
        }
        result = self.evaluate(tree)
        self.assertEqual(result["status"], "RUNNING")
        self.assertEqual(result["blackboard"], {"x": 1})
        self.assertEqual([t["id"] for t in result["trace"]], ["a1", "a2", "root"])

    def test_selector_short_circuits_on_success(self):
        tree = {
            "id": "root",
            "type": "selector",
            "children": [
                condition("c1", "exists", "missing"),
                action("a1", "set", key="x", value=1),
                action("a2", "set", key="y", value=2),
            ],
        }
        result = self.evaluate(tree)
        self.assertEqual(result["status"], "SUCCESS")
        self.assertEqual(result["blackboard"], {"x": 1})
        self.assertEqual([t["id"] for t in result["trace"]], ["c1", "a1", "root"])

    def test_selector_all_failure(self):
        tree = {
            "id": "root",
            "type": "selector",
            "children": [
                condition("c1", "exists", "nope"),
                action("a1", "status", status="FAILURE"),
            ],
        }
        result = self.evaluate(tree)
        self.assertEqual(result["status"], "FAILURE")
        self.assertEqual([t["id"] for t in result["trace"]], ["c1", "a1", "root"])

    def test_empty_composites(self):
        result = self.evaluate({"id": "root", "type": "sequence"})
        self.assertEqual(result["status"], "SUCCESS")
        self.assertEqual(result["trace"], [{"id": "root", "type": "sequence", "status": "SUCCESS"}])
        result = self.evaluate({"id": "root", "type": "selector", "children": []})
        self.assertEqual(result["status"], "FAILURE")

    def test_conditions(self):
        self.assertEqual(self.evaluate(condition("c", "exists", "k"), {"k": None})["status"], "SUCCESS")
        self.assertEqual(self.evaluate(condition("c", "exists", "k"))["status"], "FAILURE")
        self.assertEqual(self.evaluate(condition("c", "equals", "k", 1), {"k": 1})["status"], "SUCCESS")
        # JSON type-sensitive: true != 1
        self.assertEqual(self.evaluate(condition("c", "equals", "k", 1), {"k": True})["status"], "FAILURE")
        self.assertEqual(self.evaluate(condition("c", "equals", "k", 1.0), {"k": 1})["status"], "SUCCESS")
        self.assertEqual(self.evaluate(condition("c", "not_equals", "k", 1), {"k": 2})["status"], "SUCCESS")
        # missing key fails even for not_equals
        self.assertEqual(self.evaluate(condition("c", "not_equals", "k", 1))["status"], "FAILURE")

    def test_condition_does_not_modify_blackboard(self):
        result = self.evaluate(condition("c", "equals", "k", 1), {"k": 1})
        self.assertEqual(result["blackboard"], {"k": 1})

    def test_action_delete_missing_key_succeeds(self):
        result = self.evaluate(action("a", "delete", key="ghost"))
        self.assertEqual(result["status"], "SUCCESS")
        self.assertEqual(result["blackboard"], {})

    def test_action_delete_removes_key(self):
        result = self.evaluate(action("a", "delete", key="k"), {"k": 1, "keep": 2})
        self.assertEqual(result["blackboard"], {"keep": 2})

    def test_action_status_running(self):
        result = self.evaluate(action("a", "status", status="RUNNING"))
        self.assertEqual(result["status"], "RUNNING")

    def test_default_blackboard_is_empty_object(self):
        result = self.evaluate(condition("c", "exists", "k"))
        self.assertEqual(result["blackboard"], {})

    def test_input_blackboard_not_mutated(self):
        board = {"k": 1}
        tree = {
            "id": "root",
            "type": "sequence",
            "children": [action("a", "set", key="k", value=2)],
        }
        self.evaluate(tree, board)
        self.assertEqual(board, {"k": 1})

    def test_nested_trace_order(self):
        tree = {
            "id": "root",
            "type": "selector",
            "children": [
                {
                    "id": "seq",
                    "type": "sequence",
                    "children": [
                        condition("c1", "exists", "nope"),
                        action("a1", "set", key="x", value=1),
                    ],
                },
                action("a2", "set", key="y", value=2),
            ],
        }
        result = self.evaluate(tree)
        self.assertEqual(result["status"], "SUCCESS")
        self.assertEqual(result["blackboard"], {"y": 2})
        self.assertEqual([t["id"] for t in result["trace"]], ["c1", "seq", "a2", "root"])


class EvaluateBehaviorErrorsTest(unittest.TestCase):
    def setUp(self):
        self.service = Service()

    def assert_invalid(self, request, *fragments):
        with self.assertRaises(ValueError) as ctx:
            self.service.evaluate_behavior(request)
        for fragment in fragments:
            self.assertIn(fragment, str(ctx.exception))

    def test_missing_tree(self):
        self.assert_invalid({}, "tree")

    def test_request_not_object(self):
        self.assert_invalid([1, 2], "object")

    def test_blackboard_not_object(self):
        self.assert_invalid({"tree": {"id": "r", "type": "sequence"}, "blackboard": []}, "blackboard")

    def test_blackboard_invalid_key(self):
        self.assert_invalid(
            {"tree": {"id": "r", "type": "sequence"}, "blackboard": {"": 1}}, "blackboard"
        )

    def test_tree_not_object(self):
        self.assert_invalid({"tree": 42}, "root")

    def test_missing_id_reports_path(self):
        tree = {"id": "root", "type": "sequence", "children": [{"type": "action", "op": "delete", "key": "k"}]}
        self.assert_invalid({"tree": tree}, "root.children[0]")

    def test_empty_id(self):
        self.assert_invalid({"tree": {"id": "", "type": "sequence"}}, "id")

    def test_duplicate_id(self):
        tree = {
            "id": "root",
            "type": "sequence",
            "children": [
                action("dup", "delete", key="k"),
                action("dup", "delete", key="k"),
            ],
        }
        self.assert_invalid({"tree": tree}, "dup")

    def test_unknown_node_type(self):
        self.assert_invalid({"tree": {"id": "n1", "type": "parallel"}}, "n1")

    def test_unknown_condition_op(self):
        self.assert_invalid({"tree": condition("c", "gt", "k", 1)}, "c")

    def test_condition_missing_value(self):
        self.assert_invalid({"tree": {"id": "c", "type": "condition", "op": "equals", "key": "k"}}, "value")

    def test_condition_missing_key(self):
        self.assert_invalid({"tree": {"id": "c", "type": "condition", "op": "exists"}}, "key")

    def test_unknown_action_op(self):
        self.assert_invalid({"tree": {"id": "a", "type": "action", "op": "print"}}, "a")

    def test_action_set_missing_value(self):
        self.assert_invalid({"tree": {"id": "a", "type": "action", "op": "set", "key": "k"}}, "value")

    def test_action_status_illegal(self):
        self.assert_invalid({"tree": action("a", "status", status="BROKEN")}, "BROKEN")

    def test_children_not_list(self):
        self.assert_invalid({"tree": {"id": "r", "type": "sequence", "children": {}}}, "children")


class ExportBehaviorTreeTest(unittest.TestCase):
    def setUp(self):
        self.service = Service()

    def tree(self):
        return {
            "id": "root",
            "type": "sequence",
            "children": [
                condition("c1", "exists", "x"),
                {
                    "id": "fallback",
                    "type": "selector",
                    "children": [
                        action("a1", "set", key="y", value=1),
                        action("a2", "status", status="RUNNING"),
                    ],
                },
                action("a3", "delete", key="z"),
            ],
        }

    def export(self, tree, trace=None):
        request = {"tree": tree}
        if trace is not None:
            request["trace"] = trace
        return self.service.export_behavior_tree(request)

    def assert_invalid(self, request, fragment):
        with self.assertRaises(ValueError) as caught:
            self.service.export_behavior_tree(request)
        self.assertIn(fragment, str(caught.exception))

    def test_full_tree_without_trace(self):
        result = self.export(self.tree())
        self.assertEqual(result["status"], "EXPORTED")
        self.assertEqual(
            result["nodes"],
            [
                {"id": "root", "type": "sequence", "depth": 0, "visited": False, "status": None},
                {"id": "c1", "type": "condition", "depth": 1, "visited": False, "status": None},
                {"id": "fallback", "type": "selector", "depth": 1, "visited": False, "status": None},
                {"id": "a1", "type": "action", "depth": 2, "visited": False, "status": None},
                {"id": "a2", "type": "action", "depth": 2, "visited": False, "status": None},
                {"id": "a3", "type": "action", "depth": 1, "visited": False, "status": None},
            ],
        )
        self.assertEqual(
            result["edges"],
            [
                {"from": "root", "to": "c1", "index": 0},
                {"from": "root", "to": "fallback", "index": 1},
                {"from": "root", "to": "a3", "index": 2},
                {"from": "fallback", "to": "a1", "index": 0},
                {"from": "fallback", "to": "a2", "index": 1},
            ],
        )
        self.assertEqual(result["trace_order"], [])

    def test_partial_trace_marks_visited(self):
        trace = [
            {"id": "c1", "type": "condition", "status": "FAILURE"},
            {"id": "root", "type": "sequence", "status": "FAILURE"},
        ]
        result = self.export(self.tree(), trace)
        self.assertEqual(result["status"], "EXPORTED")
        by_id = {node["id"]: node for node in result["nodes"]}
        self.assertEqual(by_id["c1"]["visited"], True)
        self.assertEqual(by_id["c1"]["status"], "FAILURE")
        self.assertEqual(by_id["root"]["visited"], True)
        self.assertEqual(by_id["root"]["status"], "FAILURE")
        for node_id in ("fallback", "a1", "a2", "a3"):
            self.assertEqual(by_id[node_id]["visited"], False)
            self.assertIsNone(by_id[node_id]["status"])
        self.assertEqual(result["trace_order"], ["c1", "root"])

    def test_selector_short_circuit_trace(self):
        tree = {
            "id": "root",
            "type": "selector",
            "children": [
                action("a1", "status", status="FAILURE"),
                action("a2", "status", status="SUCCESS"),
                action("a3", "status", status="SUCCESS"),
            ],
        }
        evaluated = self.service.evaluate_behavior({"tree": tree})
        result = self.export(tree, evaluated["trace"])
        self.assertEqual(
            result["trace_order"], ["a1", "a2", "root"]
        )
        statuses = {node["id"]: node["status"] for node in result["nodes"]}
        self.assertEqual(
            statuses,
            {"root": "SUCCESS", "a1": "FAILURE", "a2": "SUCCESS", "a3": None},
        )

    def test_single_node_tree(self):
        tree = action("only", "status", status="RUNNING")
        result = self.export(tree, [{"id": "only", "type": "action", "status": "RUNNING"}])
        self.assertEqual(result["status"], "EXPORTED")
        self.assertEqual(
            result["nodes"],
            [{"id": "only", "type": "action", "depth": 0, "visited": True, "status": "RUNNING"}],
        )
        self.assertEqual(result["edges"], [])
        self.assertEqual(result["trace_order"], ["only"])

    def test_empty_children_produce_no_edges(self):
        tree = {"id": "root", "type": "sequence", "children": []}
        result = self.export(tree)
        self.assertEqual(len(result["nodes"]), 1)
        self.assertEqual(result["edges"], [])

    def test_request_is_not_mutated(self):
        tree = self.tree()
        trace = [{"id": "c1", "type": "condition", "status": "SUCCESS"}]
        request = {"tree": tree, "trace": trace}
        snapshot = json.loads(json.dumps(request))
        self.service.export_behavior_tree(request)
        self.assertEqual(request, snapshot)

    def test_no_state_between_calls(self):
        trace = [{"id": "c1", "type": "condition", "status": "SUCCESS"}]
        first = self.export(self.tree(), trace)
        second = self.export(self.tree())
        self.assertEqual(first["trace_order"], ["c1"])
        self.assertEqual(second["trace_order"], [])
        visited = {node["id"]: node["visited"] for node in second["nodes"]}
        self.assertFalse(any(visited.values()))

    def test_request_not_object(self):
        self.assert_invalid([1, 2], "object")

    def test_missing_tree(self):
        self.assert_invalid({}, "tree")

    def test_invalid_tree_rules_still_apply(self):
        self.assert_invalid({"tree": {"id": "r", "type": "parallel"}}, "r")
        self.assert_invalid(
            {"tree": {"id": "r", "type": "sequence", "children": [{"id": "r", "type": "sequence"}]}},
            "duplicate",
        )

    def test_trace_not_list(self):
        self.assert_invalid({"tree": self.tree(), "trace": {}}, "trace")

    def test_trace_entry_not_object(self):
        self.assert_invalid({"tree": self.tree(), "trace": ["c1"]}, "object")

    def test_trace_id_not_non_empty_string(self):
        self.assert_invalid({"tree": self.tree(), "trace": [{"id": "", "type": "condition", "status": "SUCCESS"}]}, "id")
        self.assert_invalid({"tree": self.tree(), "trace": [{"type": "condition", "status": "SUCCESS"}]}, "id")

    def test_trace_duplicate_id(self):
        trace = [
            {"id": "c1", "type": "condition", "status": "SUCCESS"},
            {"id": "c1", "type": "condition", "status": "FAILURE"},
        ]
        self.assert_invalid({"tree": self.tree(), "trace": trace}, "duplicate")

    def test_trace_unknown_id(self):
        trace = [{"id": "ghost", "type": "condition", "status": "SUCCESS"}]
        self.assert_invalid({"tree": self.tree(), "trace": trace}, "ghost")

    def test_trace_type_mismatch(self):
        trace = [{"id": "c1", "type": "action", "status": "SUCCESS"}]
        self.assert_invalid({"tree": self.tree(), "trace": trace}, "type")

    def test_trace_illegal_status(self):
        trace = [{"id": "c1", "type": "condition", "status": "BROKEN"}]
        self.assert_invalid({"tree": self.tree(), "trace": trace}, "BROKEN")


if __name__ == "__main__":
    unittest.main()
