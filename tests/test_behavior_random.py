import copy
import hashlib
import http.client
import json
import threading
import unittest
from http.server import ThreadingHTTPServer

from npcmind.server import Handler
from npcmind.service import Service

MAX_SEED = 18446744073709551615


def action(node_id, op, key=None, value=None, status=None):
    node = {"id": node_id, "type": "action", "op": op}
    if key is not None:
        node["key"] = key
    if op == "set":
        node["value"] = value
    if op == "status":
        node["status"] = status
    return node


def expected_child_index(node_id, weights, seed):
    digest = hashlib.sha256(f"{seed}:{node_id}".encode("utf-8")).digest()
    remainder = int.from_bytes(digest[:8], "big") % sum(weights)
    cumulative = 0
    for index, weight in enumerate(weights):
        cumulative += weight
        if remainder < cumulative:
            return index
    raise AssertionError("unreachable")


class EvaluateRandomSelectorTest(unittest.TestCase):
    def setUp(self):
        self.service = Service()

    def tree(self, weights=None):
        node = {
            "id": "pick",
            "type": "random_selector",
            "children": [
                action("a0", "set", key="choice", value=0),
                action("a1", "set", key="choice", value=1),
                action("a2", "set", key="choice", value=2),
            ],
        }
        if weights is not None:
            node["weights"] = weights
        return node

    def test_default_weights_select_exactly_one_child(self):
        seed = 42
        result = self.service.evaluate_behavior({"tree": self.tree(), "seed": seed})
        index = expected_child_index("pick", [1, 1, 1], seed)
        self.assertEqual(result["status"], "SUCCESS")
        self.assertEqual(result["blackboard"], {"choice": index})
        self.assertEqual(
            result["trace"],
            [
                {"id": f"a{index}", "type": "action", "status": "SUCCESS"},
                {"id": "pick", "type": "random_selector", "status": "SUCCESS"},
            ],
        )

    def test_weighted_selection_matches_reference(self):
        weights = [2, 3, 5]
        for seed in (0, 1, 7, 123456789, MAX_SEED):
            result = self.service.evaluate_behavior(
                {"tree": self.tree(weights), "seed": seed}
            )
            index = expected_child_index("pick", weights, seed)
            self.assertEqual(result["blackboard"], {"choice": index}, f"seed={seed}")
            self.assertEqual([t["id"] for t in result["trace"]], [f"a{index}", "pick"])

    def test_same_seed_is_deterministic_across_calls(self):
        request = {"tree": self.tree([4, 1, 2]), "blackboard": {"k": 1}, "seed": 99}
        first = self.service.evaluate_behavior(copy.deepcopy(request))
        second = self.service.evaluate_behavior(copy.deepcopy(request))
        self.assertEqual(first, second)

    def test_node_status_equals_child_status(self):
        tree = {
            "id": "pick",
            "type": "random_selector",
            "children": [
                action("f", "status", status="FAILURE"),
                action("r", "status", status="RUNNING"),
            ],
        }
        for seed in range(10):
            result = self.service.evaluate_behavior({"tree": tree, "seed": seed})
            chosen = result["trace"][0]
            self.assertEqual(result["status"], chosen["status"])
            self.assertEqual(result["trace"][1]["status"], chosen["status"])
        statuses = set()
        for seed in range(50):
            result = self.service.evaluate_behavior({"tree": tree, "seed": seed})
            statuses.add(result["trace"][0]["id"])
        self.assertEqual(statuses, {"f", "r"})

    def test_unselected_branch_never_executes(self):
        tree = {
            "id": "root",
            "type": "sequence",
            "children": [
                {
                    "id": "pick",
                    "type": "random_selector",
                    "children": [
                        action("set_x", "set", key="x", value=1),
                        action("set_y", "set", key="y", value=1),
                    ],
                }
            ],
        }
        for seed in range(20):
            result = self.service.evaluate_behavior({"tree": tree, "seed": seed})
            self.assertEqual(len(result["blackboard"]), 1)
            self.assertNotIn("set_x" if "y" in result["blackboard"] else "set_y",
                             [t["id"] for t in result["trace"]])

    def test_nested_random_selectors_use_their_own_ids(self):
        tree = {
            "id": "outer",
            "type": "random_selector",
            "children": [
                {
                    "id": "inner",
                    "type": "random_selector",
                    "weights": [3, 1],
                    "children": [
                        action("deep_a", "set", key="leaf", value="a"),
                        action("deep_b", "set", key="leaf", value="b"),
                    ],
                },
                action("shallow", "set", key="leaf", value="s"),
            ],
        }
        seed = 2024
        result = self.service.evaluate_behavior({"tree": tree, "seed": seed})
        outer_index = expected_child_index("outer", [1, 1], seed)
        if outer_index == 0:
            inner_index = expected_child_index("inner", [3, 1], seed)
            self.assertEqual(result["blackboard"], {"leaf": "ab"[inner_index]})
            self.assertEqual(
                [t["id"] for t in result["trace"]],
                [f"deep_{'ab'[inner_index]}", "inner", "outer"],
            )
        else:
            self.assertEqual(result["blackboard"], {"leaf": "s"})
            self.assertEqual([t["id"] for t in result["trace"]], ["shallow", "outer"])

    def test_seed_zero_and_max_are_accepted(self):
        for seed in (0, MAX_SEED):
            result = self.service.evaluate_behavior({"tree": self.tree(), "seed": seed})
            self.assertEqual(result["status"], "SUCCESS")

    def test_seed_ignored_without_random_selector(self):
        tree = {"id": "root", "type": "sequence", "children": [action("a", "set", key="x", value=1)]}
        plain = self.service.evaluate_behavior({"tree": tree})
        with_seed = self.service.evaluate_behavior({"tree": tree, "seed": 7})
        self.assertEqual(plain, with_seed)

    def test_request_and_tree_not_mutated(self):
        tree = self.tree([1, 2, 3])
        snapshot = copy.deepcopy(tree)
        board = {"k": [1, 2]}
        self.service.evaluate_behavior({"tree": tree, "blackboard": board, "seed": 5})
        self.assertEqual(tree, snapshot)
        self.assertEqual(board, {"k": [1, 2]})


class EvaluateRandomSelectorErrorsTest(unittest.TestCase):
    def setUp(self):
        self.service = Service()

    def assert_invalid(self, request, *fragments):
        with self.assertRaises(ValueError) as ctx:
            self.service.evaluate_behavior(request)
        for fragment in fragments:
            self.assertIn(fragment, str(ctx.exception))

    def tree(self, **overrides):
        node = {
            "id": "pick",
            "type": "random_selector",
            "children": [action("a", "set", key="x", value=1)],
        }
        node.update(overrides)
        return node

    def test_missing_children(self):
        node = self.tree()
        del node["children"]
        self.assert_invalid({"tree": node, "seed": 1}, "children")

    def test_empty_children(self):
        self.assert_invalid({"tree": self.tree(children=[]), "seed": 1}, "children")

    def test_children_not_list(self):
        self.assert_invalid({"tree": self.tree(children={}), "seed": 1}, "children")

    def test_weights_wrong_length(self):
        self.assert_invalid({"tree": self.tree(weights=[1, 2]), "seed": 1}, "weights")

    def test_weights_not_list(self):
        self.assert_invalid({"tree": self.tree(weights="1"), "seed": 1}, "weights")

    def test_weights_null(self):
        self.assert_invalid({"tree": self.tree(weights=None), "seed": 1}, "weights")

    def test_weights_zero_or_negative(self):
        self.assert_invalid({"tree": self.tree(weights=[0]), "seed": 1}, "weights")
        self.assert_invalid({"tree": self.tree(weights=[-3]), "seed": 1}, "weights")

    def test_weights_non_integer(self):
        self.assert_invalid({"tree": self.tree(weights=[1.5]), "seed": 1}, "weights")

    def test_weights_boolean(self):
        self.assert_invalid({"tree": self.tree(weights=[True]), "seed": 1}, "weights")

    def test_missing_seed(self):
        self.assert_invalid({"tree": self.tree()}, "seed")

    def test_seed_not_integer(self):
        self.assert_invalid({"tree": self.tree(), "seed": "1"}, "seed")
        self.assert_invalid({"tree": self.tree(), "seed": 1.5}, "seed")

    def test_seed_boolean(self):
        self.assert_invalid({"tree": self.tree(), "seed": True}, "seed")

    def test_seed_out_of_range(self):
        self.assert_invalid({"tree": self.tree(), "seed": -1}, "seed")
        self.assert_invalid({"tree": self.tree(), "seed": MAX_SEED + 1}, "seed")

    def test_nested_random_selector_also_requires_seed(self):
        tree = {
            "id": "root",
            "type": "selector",
            "children": [self.tree()],
        }
        self.assert_invalid({"tree": tree}, "seed")

    def test_invalid_weights_inside_untaken_branch_still_rejected(self):
        tree = {
            "id": "root",
            "type": "sequence",
            "children": [
                action("a", "set", key="x", value=1),
                self.tree(weights=[]),
            ],
        }
        self.assert_invalid({"tree": tree, "seed": 1}, "weights")


class ExportRandomSelectorTest(unittest.TestCase):
    def setUp(self):
        self.service = Service()

    def tree(self):
        return {
            "id": "pick",
            "type": "random_selector",
            "weights": [2, 1],
            "children": [
                action("a0", "set", key="x", value=0),
                action("a1", "set", key="x", value=1),
            ],
        }

    def test_export_includes_random_selector_without_seed(self):
        result = self.service.export_behavior_tree({"tree": self.tree()})
        self.assertEqual(result["status"], "EXPORTED")
        self.assertEqual(
            [(n["id"], n["type"], n["depth"]) for n in result["nodes"]],
            [("pick", "random_selector", 0), ("a0", "action", 1), ("a1", "action", 1)],
        )
        self.assertEqual(
            result["edges"],
            [
                {"from": "pick", "to": "a0", "index": 0},
                {"from": "pick", "to": "a1", "index": 1},
            ],
        )

    def test_export_marks_trace_from_evaluation(self):
        tree = self.tree()
        seed = 3
        evaluated = self.service.evaluate_behavior({"tree": copy.deepcopy(tree), "seed": seed})
        result = self.service.export_behavior_tree(
            {"tree": tree, "trace": evaluated["trace"]}
        )
        chosen = evaluated["trace"][0]["id"]
        self.assertEqual(result["trace_order"], [chosen, "pick"])
        by_id = {n["id"]: n for n in result["nodes"]}
        self.assertTrue(by_id["pick"]["visited"])
        self.assertTrue(by_id[chosen]["visited"])
        unchosen = "a1" if chosen == "a0" else "a0"
        self.assertFalse(by_id[unchosen]["visited"])
        self.assertIsNone(by_id[unchosen]["status"])

    def test_export_rejects_invalid_random_selector(self):
        tree = self.tree()
        tree["weights"] = [1]
        with self.assertRaises(ValueError):
            self.service.export_behavior_tree({"tree": tree})
        del tree["weights"]
        tree["children"] = []
        with self.assertRaises(ValueError):
            self.service.export_behavior_tree({"tree": tree})


class RandomSelectorHttpTest(unittest.TestCase):
    def setUp(self):
        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.port = self.httpd.server_address[1]
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.thread.start()

    def tearDown(self):
        self.httpd.shutdown()
        self.httpd.server_close()
        self.thread.join()

    def post_raw(self, path, raw_body):
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=5)
        conn.request("POST", path, raw_body, {"Content-Type": "application/json"})
        response = conn.getresponse()
        body = response.read().decode("utf-8")
        conn.close()
        return response.status, json.loads(body)

    def post(self, path, payload):
        return self.post_raw(path, json.dumps(payload))

    def tree(self):
        return {
            "id": "pick",
            "type": "random_selector",
            "children": [
                {"id": "a0", "type": "action", "op": "set", "key": "x", "value": 0},
                {"id": "a1", "type": "action", "op": "set", "key": "x", "value": 1},
            ],
        }

    def test_evaluate_ok(self):
        status, body = self.post(
            "/v1/behavior-trees/evaluate", {"tree": self.tree(), "seed": 11}
        )
        self.assertEqual(status, 200)
        self.assertEqual(body["status"], "SUCCESS")
        self.assertEqual(body["trace"][-1]["type"], "random_selector")

    def test_evaluate_missing_seed_is_422_invalid_tree(self):
        status, body = self.post("/v1/behavior-trees/evaluate", {"tree": self.tree()})
        self.assertEqual(status, 422)
        self.assertEqual(body["error"]["code"], "invalid_tree")

    def test_evaluate_boolean_seed_is_422(self):
        status, body = self.post(
            "/v1/behavior-trees/evaluate", {"tree": self.tree(), "seed": True}
        )
        self.assertEqual(status, 422)
        self.assertEqual(body["error"]["code"], "invalid_tree")

    def test_evaluate_bad_weights_is_422(self):
        tree = self.tree()
        tree["weights"] = [1, 0]
        status, body = self.post(
            "/v1/behavior-trees/evaluate", {"tree": tree, "seed": 1}
        )
        self.assertEqual(status, 422)
        self.assertEqual(body["error"]["code"], "invalid_tree")

    def test_evaluate_unparseable_json_is_400(self):
        status, body = self.post_raw("/v1/behavior-trees/evaluate", "{not json")
        self.assertEqual(status, 400)
        self.assertEqual(body["error"]["code"], "invalid_json")

    def test_visualize_accepts_random_selector_without_seed(self):
        status, body = self.post("/v1/behavior-trees/visualize", {"tree": self.tree()})
        self.assertEqual(status, 200)
        self.assertEqual(body["status"], "EXPORTED")
        self.assertEqual([n["id"] for n in body["nodes"]], ["pick", "a0", "a1"])

    def test_visualize_invalid_random_selector_is_422(self):
        tree = self.tree()
        tree["children"] = []
        status, body = self.post("/v1/behavior-trees/visualize", {"tree": tree})
        self.assertEqual(status, 422)
        self.assertEqual(body["error"]["code"], "invalid_behavior_visualization")


if __name__ == "__main__":
    unittest.main()
