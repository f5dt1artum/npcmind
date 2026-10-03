import copy
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


class ExportBehaviorTreeTest(unittest.TestCase):
    def setUp(self):
        self.service = Service()

    def export(self, tree, trace=None):
        request = {"tree": tree}
        if trace is not None:
            request["trace"] = trace
        return self.service.export_behavior_tree(request)

    def test_single_leaf_without_trace(self):
        result = self.export(action("a", "status", status="SUCCESS"))
        self.assertEqual(result["status"], "EXPORTED")
        self.assertEqual(
            result["nodes"],
            [{"id": "a", "type": "action", "depth": 0, "visited": False, "status": None}],
        )
        self.assertEqual(result["edges"], [])
        self.assertEqual(result["trace_order"], [])

    def test_explicit_empty_trace_matches_omitted(self):
        tree = {"id": "root", "type": "selector", "children": [condition("c", "exists", "k")]}
        omitted = self.export(tree)
        explicit = self.export(tree, [])
        self.assertEqual(omitted, explicit)
        self.assertEqual([n["visited"] for n in omitted["nodes"]], [False, False])
        self.assertEqual([n["status"] for n in omitted["nodes"]], [None, None])
        self.assertEqual(omitted["trace_order"], [])

    def test_nodes_are_preorder_with_depth(self):
        tree = {
            "id": "root",
            "type": "sequence",
            "children": [
                {
                    "id": "sel",
                    "type": "selector",
                    "children": [condition("c1", "exists", "x"), action("a1", "delete", key="k")],
                },
                action("a2", "set", key="y", value=1),
            ],
        }
        result = self.export(tree)
        self.assertEqual(
            [(n["id"], n["type"], n["depth"]) for n in result["nodes"]],
            [
                ("root", "sequence", 0),
                ("sel", "selector", 1),
                ("c1", "condition", 2),
                ("a1", "action", 2),
                ("a2", "action", 1),
            ],
        )

    def test_edges_follow_parent_preorder_then_child_order(self):
        tree = {
            "id": "root",
            "type": "sequence",
            "children": [
                {
                    "id": "sel",
                    "type": "selector",
                    "children": [condition("c1", "exists", "x"), action("a1", "delete", key="k")],
                },
                action("a2", "delete", key="k"),
            ],
        }
        result = self.export(tree)
        self.assertEqual(
            result["edges"],
            [
                {"from": "root", "to": "sel", "index": 0},
                {"from": "root", "to": "a2", "index": 1},
                {"from": "sel", "to": "c1", "index": 2},
                {"from": "sel", "to": "a1", "index": 3},
            ],
        )

    def test_empty_composite_and_leaf_emit_no_edges(self):
        result = self.export({"id": "root", "type": "sequence"})
        self.assertEqual(result["edges"], [])
        self.assertEqual(
            result["nodes"],
            [{"id": "root", "type": "sequence", "depth": 0, "visited": False, "status": None}],
        )

    def test_full_trace_marks_every_node(self):
        tree = {
            "id": "root",
            "type": "sequence",
            "children": [
                action("a1", "set", key="x", value=1),
                action("a2", "set", key="y", value=2),
            ],
        }
        evaluated = self.service.evaluate_behavior({"tree": tree})
        result = self.export(tree, evaluated["trace"])
        self.assertTrue(all(n["visited"] for n in result["nodes"]))
        statuses = {n["id"]: n["status"] for n in result["nodes"]}
        self.assertEqual(statuses, {"a1": "SUCCESS", "a2": "SUCCESS", "root": "SUCCESS"})
        self.assertEqual(result["trace_order"], ["a1", "a2", "root"])

    def test_sequence_short_circuit_partial_trace(self):
        tree = {
            "id": "root",
            "type": "sequence",
            "children": [
                action("a1", "status", status="FAILURE"),
                action("a2", "set", key="x", value=1),
            ],
        }
        evaluated = self.service.evaluate_behavior({"tree": tree})
        result = self.export(tree, evaluated["trace"])
        by_id = {n["id"]: n for n in result["nodes"]}
        self.assertEqual(
            (by_id["root"]["visited"], by_id["root"]["status"]), (True, "FAILURE")
        )
        self.assertEqual((by_id["a1"]["visited"], by_id["a1"]["status"]), (True, "FAILURE"))
        self.assertEqual((by_id["a2"]["visited"], by_id["a2"]["status"]), (False, None))
        self.assertEqual(result["trace_order"], ["a1", "root"])

    def test_selector_short_circuit_partial_trace(self):
        tree = {
            "id": "root",
            "type": "selector",
            "children": [
                condition("c1", "exists", "missing"),
                action("a1", "set", key="x", value=1),
                action("a2", "set", key="y", value=2),
            ],
        }
        evaluated = self.service.evaluate_behavior({"tree": tree})
        result = self.export(tree, evaluated["trace"])
        by_id = {n["id"]: n for n in result["nodes"]}
        self.assertEqual(
            (by_id["c1"]["visited"], by_id["a1"]["visited"], by_id["a2"]["visited"]),
            (True, True, False),
        )
        self.assertEqual(by_id["a1"]["status"], "SUCCESS")
        self.assertIsNone(by_id["a2"]["status"])
        self.assertEqual(result["trace_order"], ["c1", "a1", "root"])

    def test_trace_order_only_lists_ids(self):
        tree = {"id": "root", "type": "sequence", "children": [action("a", "delete", key="k")]}
        result = self.export(tree, [
            {"id": "a", "type": "action", "status": "SUCCESS"},
            {"id": "root", "type": "sequence", "status": "SUCCESS"},
        ])
        self.assertEqual(result["trace_order"], ["a", "root"])
        for entry in result["nodes"]:
            self.assertEqual(set(entry.keys()), {"id", "type", "depth", "visited", "status"})
        for edge in result["edges"]:
            self.assertEqual(set(edge.keys()), {"from", "to", "index"})

    def test_does_not_execute_nodes_or_mutate_inputs(self):
        tree = {
            "id": "root",
            "type": "sequence",
            "children": [
                action("a1", "set", key="x", value=[1, 2]),
                action("a2", "delete", key="y"),
                action("a3", "status", status="RUNNING"),
            ],
        }
        trace = [
            {"id": "a1", "type": "action", "status": "SUCCESS"},
            {"id": "root", "type": "sequence", "status": "RUNNING"},
        ]
        tree_snapshot = copy.deepcopy(tree)
        trace_snapshot = copy.deepcopy(trace)
        self.export(tree, trace)
        self.assertEqual(tree, tree_snapshot)
        self.assertEqual(trace, trace_snapshot)

    def test_stateless_between_calls(self):
        tree_a = {"id": "root", "type": "sequence", "children": [action("a", "delete", key="k")]}
        tree_b = action("solo", "status", status="FAILURE")
        trace = [
            {"id": "a", "type": "action", "status": "SUCCESS"},
            {"id": "root", "type": "sequence", "status": "SUCCESS"},
        ]
        first = self.export(tree_a, trace)
        second = self.export(tree_b, [])
        third = self.export(tree_a)
        self.assertEqual([n["id"] for n in first["nodes"]], ["root", "a"])
        self.assertEqual([n["id"] for n in second["nodes"]], ["solo"])
        self.assertFalse(third["nodes"][0]["visited"])
        self.assertEqual(third["trace_order"], [])


class ExportBehaviorTreeErrorsTest(unittest.TestCase):
    def setUp(self):
        self.service = Service()

    def assert_invalid(self, request):
        with self.assertRaises(ValueError):
            self.service.export_behavior_tree(request)

    def test_request_not_object(self):
        self.assert_invalid([1, 2])
        self.assert_invalid("nope")

    def test_missing_tree(self):
        self.assert_invalid({})

    def test_tree_violations_use_existing_rules(self):
        bad_trees = [
            42,
            {"type": "sequence"},
            {"id": "", "type": "sequence"},
            {"id": "root", "type": "parallel"},
            {"id": "root", "type": "sequence", "children": {}},
            {"id": "root", "type": "sequence",
             "children": [{"id": "dup", "type": "action", "op": "delete", "key": "k"},
                          {"id": "dup", "type": "action", "op": "delete", "key": "k"}]},
            condition("c", "gt", "k", 1),
            {"id": "c", "type": "condition", "op": "equals", "key": "k"},
            {"id": "a", "type": "action", "op": "set", "key": "k"},
            action("a", "status", status="BROKEN"),
        ]
        for tree in bad_trees:
            self.assert_invalid({"tree": tree})

    def test_trace_not_array(self):
        request = {"tree": action("a", "delete", key="k"), "trace": {"id": "a"}}
        self.assert_invalid(request)

    def test_trace_entry_not_object(self):
        self.assert_invalid({"tree": action("a", "delete", key="k"), "trace": [5]})

    def test_trace_id_not_non_empty_string(self):
        tree = action("a", "delete", key="k")
        for entry in ({"type": "action", "status": "SUCCESS"},
                      {"id": "", "type": "action", "status": "SUCCESS"},
                      {"id": 7, "type": "action", "status": "SUCCESS"}):
            self.assert_invalid({"tree": tree, "trace": [entry]})

    def test_duplicate_trace_id(self):
        tree = {"id": "root", "type": "sequence", "children": [action("a", "delete", key="k")]}
        trace = [
            {"id": "a", "type": "action", "status": "SUCCESS"},
            {"id": "a", "type": "action", "status": "FAILURE"},
        ]
        self.assert_invalid({"tree": tree, "trace": trace})

    def test_trace_id_unknown(self):
        tree = action("a", "delete", key="k")
        trace = [{"id": "ghost", "type": "action", "status": "SUCCESS"}]
        self.assert_invalid({"tree": tree, "trace": trace})

    def test_trace_type_mismatch(self):
        tree = action("a", "delete", key="k")
        trace = [{"id": "a", "type": "condition", "status": "SUCCESS"}]
        self.assert_invalid({"tree": tree, "trace": trace})

    def test_trace_status_illegal(self):
        tree = action("a", "delete", key="k")
        for status in ("WAT", None, 1):
            trace = [{"id": "a", "type": "action", "status": status}]
            self.assert_invalid({"tree": tree, "trace": trace})

    def test_invalid_trace_does_not_mutate_tree_or_trace(self):
        tree = {"id": "root", "type": "sequence", "children": [action("a", "delete", key="k")]}
        trace = [{"id": "ghost", "type": "action", "status": "SUCCESS"}]
        tree_snapshot = copy.deepcopy(tree)
        trace_snapshot = copy.deepcopy(trace)
        with self.assertRaises(ValueError):
            self.service.export_behavior_tree({"tree": tree, "trace": trace})
        self.assertEqual(tree, tree_snapshot)
        self.assertEqual(trace, trace_snapshot)


if __name__ == "__main__":
    unittest.main()
