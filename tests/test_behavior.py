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


if __name__ == "__main__":
    unittest.main()
