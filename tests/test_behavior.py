import json
import unittest

from npcmind.service import Service


def make_service():
    return Service()


class SequenceSelectorTest(unittest.TestCase):
    def test_sequence_all_success(self):
        tree = {
            "id": "root", "type": "sequence", "children": [
                {"id": "a", "type": "action", "op": "set", "key": "x", "value": 1},
                {"id": "b", "type": "action", "op": "set", "key": "y", "value": 2},
            ]
        }
        result = make_service().evaluate_behavior({"tree": tree})
        self.assertEqual(result["status"], "SUCCESS")
        self.assertEqual(result["blackboard"], {"x": 1, "y": 2})
        self.assertEqual(
            [(t["id"], t["status"]) for t in result["trace"]],
            [("a", "SUCCESS"), ("b", "SUCCESS"), ("root", "SUCCESS")],
        )

    def test_sequence_short_circuits_on_failure(self):
        tree = {
            "id": "root", "type": "sequence", "children": [
                {"id": "a", "type": "action", "op": "status", "status": "FAILURE"},
                {"id": "b", "type": "action", "op": "set", "key": "x", "value": 1},
            ]
        }
        result = make_service().evaluate_behavior({"tree": tree, "blackboard": {}})
        self.assertEqual(result["status"], "FAILURE")
        self.assertEqual(result["blackboard"], {})
        self.assertEqual([t["id"] for t in result["trace"]], ["a", "root"])

    def test_sequence_stops_on_running(self):
        tree = {
            "id": "root", "type": "sequence", "children": [
                {"id": "a", "type": "action", "op": "status", "status": "RUNNING"},
                {"id": "b", "type": "action", "op": "set", "key": "x", "value": 1},
            ]
        }
        result = make_service().evaluate_behavior({"tree": tree})
        self.assertEqual(result["status"], "RUNNING")
        self.assertEqual(result["blackboard"], {})

    def test_empty_sequence_succeeds_empty_selector_fails(self):
        svc = make_service()
        seq = svc.evaluate_behavior({"tree": {"id": "s", "type": "sequence", "children": []}})
        sel = svc.evaluate_behavior({"tree": {"id": "x", "type": "selector", "children": []}})
        self.assertEqual(seq["status"], "SUCCESS")
        self.assertEqual(sel["status"], "FAILURE")

    def test_selector_short_circuits_on_success(self):
        tree = {
            "id": "root", "type": "selector", "children": [
                {"id": "a", "type": "action", "op": "status", "status": "FAILURE"},
                {"id": "b", "type": "action", "op": "set", "key": "x", "value": 1},
                {"id": "c", "type": "action", "op": "set", "key": "y", "value": 2},
            ]
        }
        result = make_service().evaluate_behavior({"tree": tree})
        self.assertEqual(result["status"], "SUCCESS")
        self.assertEqual(result["blackboard"], {"x": 1})
        self.assertEqual([t["id"] for t in result["trace"]], ["a", "b", "root"])

    def test_selector_all_failure(self):
        tree = {
            "id": "root", "type": "selector", "children": [
                {"id": "a", "type": "action", "op": "status", "status": "FAILURE"},
            ]
        }
        result = make_service().evaluate_behavior({"tree": tree})
        self.assertEqual(result["status"], "FAILURE")


class ConditionTest(unittest.TestCase):
    def evaluate(self, node, blackboard):
        return make_service().evaluate_behavior({"tree": node, "blackboard": blackboard})

    def test_exists(self):
        node = {"id": "c", "type": "condition", "key": "k", "op": "exists"}
        self.assertEqual(self.evaluate(node, {"k": None})["status"], "SUCCESS")
        self.assertEqual(self.evaluate(node, {})["status"], "FAILURE")

    def test_equals_type_sensitive(self):
        node = {"id": "c", "type": "condition", "key": "k", "op": "equals", "value": 1}
        self.assertEqual(self.evaluate(node, {"k": 1})["status"], "SUCCESS")
        self.assertEqual(self.evaluate(node, {"k": 1.0})["status"], "SUCCESS")
        self.assertEqual(self.evaluate(node, {"k": True})["status"], "FAILURE")
        self.assertEqual(self.evaluate(node, {"k": "1"})["status"], "FAILURE")

    def test_equals_nested_values(self):
        node = {"id": "c", "type": "condition", "key": "k", "op": "equals",
                "value": {"a": [1, True], "b": None}}
        self.assertEqual(self.evaluate(node, {"k": {"b": None, "a": [1.0, True]}})["status"], "SUCCESS")
        self.assertEqual(self.evaluate(node, {"k": {"a": [1, 1], "b": None}})["status"], "FAILURE")

    def test_not_equals_and_missing_key(self):
        node = {"id": "c", "type": "condition", "key": "k", "op": "not_equals", "value": 1}
        self.assertEqual(self.evaluate(node, {"k": 2})["status"], "SUCCESS")
        self.assertEqual(self.evaluate(node, {"k": 1})["status"], "FAILURE")
        self.assertEqual(self.evaluate(node, {})["status"], "FAILURE")

    def test_equals_missing_key_fails(self):
        node = {"id": "c", "type": "condition", "key": "k", "op": "equals", "value": None}
        self.assertEqual(self.evaluate(node, {})["status"], "FAILURE")


class ActionTest(unittest.TestCase):
    def test_set_delete(self):
        tree = {
            "id": "root", "type": "sequence", "children": [
                {"id": "s", "type": "action", "op": "set", "key": "a", "value": [1, 2]},
                {"id": "d1", "type": "action", "op": "delete", "key": "b"},
                {"id": "d2", "type": "action", "op": "delete", "key": "missing"},
            ]
        }
        result = make_service().evaluate_behavior({"tree": tree, "blackboard": {"b": 5}})
        self.assertEqual(result["status"], "SUCCESS")
        self.assertEqual(result["blackboard"], {"a": [1, 2]})

    def test_status_action(self):
        for status in ("SUCCESS", "FAILURE", "RUNNING"):
            node = {"id": "a", "type": "action", "op": "status", "status": status}
            result = make_service().evaluate_behavior({"tree": node})
            self.assertEqual(result["status"], status)

    def test_caller_blackboard_not_mutated(self):
        blackboard = {"k": 1}
        node = {"id": "a", "type": "action", "op": "set", "key": "x", "value": 9}
        make_service().evaluate_behavior({"tree": node, "blackboard": blackboard})
        self.assertEqual(blackboard, {"k": 1})


class ValidationTest(unittest.TestCase):
    def assert_invalid(self, request, *fragments):
        with self.assertRaises(ValueError) as ctx:
            make_service().evaluate_behavior(request)
        message = str(ctx.exception)
        for fragment in fragments:
            self.assertIn(fragment, message)

    def test_missing_tree(self):
        self.assert_invalid({}, "tree")

    def test_blackboard_not_object(self):
        self.assert_invalid({"tree": {"id": "a", "type": "action", "op": "status", "status": "SUCCESS"},
                             "blackboard": []}, "blackboard")

    def test_invalid_blackboard_key(self):
        self.assert_invalid({"tree": {"id": "a", "type": "action", "op": "status", "status": "SUCCESS"},
                             "blackboard": {"": 1}}, "key")

    def test_duplicate_ids(self):
        tree = {"id": "r", "type": "sequence", "children": [
            {"id": "dup", "type": "action", "op": "status", "status": "SUCCESS"},
            {"id": "dup", "type": "action", "op": "status", "status": "SUCCESS"},
        ]}
        self.assert_invalid({"tree": tree}, "dup")

    def test_empty_and_nonstring_id_report_path(self):
        tree = {"id": "r", "type": "sequence", "children": [
            {"id": "ok", "type": "action", "op": "status", "status": "SUCCESS"},
            {"id": "", "type": "action", "op": "status", "status": "SUCCESS"},
        ]}
        self.assert_invalid({"tree": tree}, "children[1]")

    def test_unknown_type_includes_node_id(self):
        self.assert_invalid({"tree": {"id": "n1", "type": "parallel"}}, "n1")

    def test_unknown_ops(self):
        self.assert_invalid({"tree": {"id": "c", "type": "condition", "key": "k", "op": "gt"}}, "c")
        self.assert_invalid({"tree": {"id": "a", "type": "action", "op": "inc", "key": "k"}}, "a")

    def test_missing_operation_fields(self):
        self.assert_invalid({"tree": {"id": "c", "type": "condition", "key": "k", "op": "equals"}}, "value")
        self.assert_invalid({"tree": {"id": "a", "type": "action", "op": "set", "key": "k"}}, "value")
        self.assert_invalid({"tree": {"id": "a", "type": "action", "op": "set", "value": 1}}, "key")
        self.assert_invalid({"tree": {"id": "r", "type": "sequence"}}, "children")

    def test_invalid_status_value(self):
        self.assert_invalid({"tree": {"id": "a", "type": "action", "op": "status", "status": "DONE"}}, "a")

    def test_no_partial_result_on_invalid_tree(self):
        tree = {"id": "r", "type": "sequence", "children": [
            {"id": "a", "type": "action", "op": "set", "key": "x", "value": 1},
            {"id": "b", "type": "bogus"},
        ]}
        blackboard = {}
        with self.assertRaises(ValueError):
            make_service().evaluate_behavior({"tree": tree, "blackboard": blackboard})
        self.assertEqual(blackboard, {})


if __name__ == "__main__":
    unittest.main()
