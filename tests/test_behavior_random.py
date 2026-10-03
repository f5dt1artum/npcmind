import copy
import hashlib
import http.client
import json
import subprocess
import sys
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


def expected_index(node_id, seed, weights):
    digest = hashlib.sha256(f"{seed}:{node_id}".encode("utf-8")).digest()
    remainder = int.from_bytes(digest[:8], "big") % sum(weights)
    cumulative = 0
    for index, weight in enumerate(weights):
        cumulative += weight
        if remainder < cumulative:
            return index
    raise AssertionError("unreachable")


def pick_tree(node_id="pick", weights=None, child_prefix="c", count=3):
    node = {
        "id": node_id,
        "type": "random_selector",
        "children": [
            action(f"{child_prefix}{i}", "set", key=f"k{i}", value=i) for i in range(count)
        ],
    }
    if weights is not None:
        node["weights"] = weights
    return node


class RandomSelectorEvaluateTest(unittest.TestCase):
    def setUp(self):
        self.service = Service()

    def evaluate(self, tree, seed=None, blackboard=None):
        request = {"tree": tree}
        if seed is not None:
            request["seed"] = seed
        if blackboard is not None:
            request["blackboard"] = blackboard
        return self.service.evaluate_behavior(request)

    def test_runs_exactly_one_child_and_mirrors_its_status(self):
        tree = pick_tree()
        result = self.evaluate(tree, seed=42)
        # Known-answer vector: sha256("42:pick") -> index 2 for weights [1, 1, 1].
        self.assertEqual(result["status"], "SUCCESS")
        self.assertEqual(result["blackboard"], {"k2": 2})
        self.assertEqual(
            result["trace"],
            [
                {"id": "c2", "type": "action", "status": "SUCCESS"},
                {"id": "pick", "type": "random_selector", "status": "SUCCESS"},
            ],
        )

    def test_selection_matches_specified_hash_for_default_weights(self):
        tree = pick_tree(count=4)
        for seed in (0, 1, 7, 123456789, MAX_SEED):
            result = self.evaluate(tree, seed=seed)
            index = expected_index("pick", seed, [1, 1, 1, 1])
            self.assertEqual(result["blackboard"], {f"k{index}": index})
            self.assertEqual([t["id"] for t in result["trace"]], [f"c{index}", "pick"])

    def test_selection_matches_specified_hash_for_weighted_children(self):
        weights = [2, 3, 5]
        tree = pick_tree(weights=weights)
        for seed in (0, 42, 999, MAX_SEED):
            result = self.evaluate(tree, seed=seed)
            index = expected_index("pick", seed, weights)
            self.assertEqual(result["blackboard"], {f"k{index}": index})
            self.assertEqual([t["id"] for t in result["trace"]], [f"c{index}", "pick"])

    def test_known_weighted_vector(self):
        # sha256("42:pick") mod 10 == 9, which falls in the third interval
        # [5, 10) of weights [2, 3, 5].
        result = self.evaluate(pick_tree(weights=[2, 3, 5]), seed=42)
        self.assertEqual(result["blackboard"], {"k2": 2})

    def test_seed_boundaries_are_accepted(self):
        tree = pick_tree(node_id="r", count=2)
        # Known-answer vectors for id "r" with weights [1, 1].
        self.assertEqual(self.evaluate(tree, seed=0)["blackboard"], {"k1": 1})
        self.assertEqual(self.evaluate(tree, seed=MAX_SEED)["blackboard"], {"k0": 0})

    def test_status_mirrors_chosen_child(self):
        for child_status in ("SUCCESS", "FAILURE", "RUNNING"):
            tree = {
                "id": "pick",
                "type": "random_selector",
                "children": [action("only", "status", status=child_status)],
            }
            result = self.evaluate(tree, seed=5)
            self.assertEqual(result["status"], child_status)
            self.assertEqual(result["trace"][-1]["status"], child_status)

    def test_nested_selectors_use_their_own_ids(self):
        tree = {
            "id": "outer",
            "type": "random_selector",
            "children": [
                {
                    "id": "inner",
                    "type": "random_selector",
                    "children": [
                        action("x0", "set", key="x", value=0),
                        action("x1", "set", key="x", value=1),
                    ],
                },
                action("y", "set", key="y", value=9),
            ],
        }
        # Known-answer vectors: seed 7 picks child 0 of "outer", then child 1
        # of "inner".
        result = self.evaluate(tree, seed=7)
        self.assertEqual(result["blackboard"], {"x": 1})
        self.assertEqual(
            [t["id"] for t in result["trace"]], ["x1", "inner", "outer"]
        )

    def test_unchosen_branches_never_run(self):
        tree = {
            "id": "pick",
            "type": "random_selector",
            "weights": [1, 1, 1],
            "children": [
                action("a", "set", key="a", value=1),
                action("b", "set", key="b", value=1),
                action("c", "set", key="c", value=1),
            ],
        }
        for seed in range(20):
            result = self.evaluate(tree, seed=seed)
            self.assertEqual(len(result["blackboard"]), 1)
            self.assertEqual(len(result["trace"]), 2)

    def test_deterministic_across_calls(self):
        tree = pick_tree(weights=[3, 1, 4])
        first = self.evaluate(tree, seed=123, blackboard={"keep": True})
        second = self.evaluate(tree, seed=123, blackboard={"keep": True})
        self.assertEqual(first, second)

    def test_deterministic_across_processes(self):
        tree = pick_tree(weights=[3, 1, 4])
        request = {"tree": tree, "seed": 123, "blackboard": {"keep": True}}
        here = self.service.evaluate_behavior(copy.deepcopy(request))
        script = (
            "import json, sys;"
            "from npcmind.service import Service;"
            "print(json.dumps(Service().evaluate_behavior(json.loads(sys.stdin.read()))))"
        )
        proc = subprocess.run(
            [sys.executable, "-c", script],
            input=json.dumps(request),
            capture_output=True,
            text=True,
            check=True,
        )
        self.assertEqual(json.loads(proc.stdout), json.loads(json.dumps(here)))

    def test_seed_ignored_when_tree_has_no_random_selector(self):
        tree = {
            "id": "root",
            "type": "sequence",
            "children": [action("a", "set", key="x", value=1)],
        }
        baseline = self.evaluate(tree)
        for seed in (0, 1, MAX_SEED):
            self.assertEqual(self.evaluate(tree, seed=seed), baseline)

    def test_request_and_inputs_not_mutated(self):
        tree = pick_tree(weights=[2, 3, 5])
        blackboard = {"keep": [1, {"nested": True}]}
        request = {"tree": tree, "seed": 42, "blackboard": blackboard}
        snapshot = copy.deepcopy(request)
        self.service.evaluate_behavior(request)
        self.assertEqual(request, snapshot)


class RandomSelectorValidationTest(unittest.TestCase):
    def setUp(self):
        self.service = Service()

    def assert_invalid(self, request, *fragments):
        with self.assertRaises(ValueError) as ctx:
            self.service.evaluate_behavior(request)
        for fragment in fragments:
            self.assertIn(fragment, str(ctx.exception))

    def test_missing_children(self):
        self.assert_invalid(
            {"tree": {"id": "r", "type": "random_selector"}, "seed": 1}, "children"
        )

    def test_empty_children(self):
        self.assert_invalid(
            {"tree": {"id": "r", "type": "random_selector", "children": []}, "seed": 1},
            "children",
        )

    def test_children_not_list(self):
        self.assert_invalid(
            {"tree": {"id": "r", "type": "random_selector", "children": {}}, "seed": 1},
            "children",
        )

    def test_weights_wrong_length(self):
        tree = pick_tree(weights=[1, 2])
        self.assert_invalid({"tree": tree, "seed": 1}, "weights")

    def test_weights_not_list(self):
        tree = pick_tree(weights="1,1,1")
        self.assert_invalid({"tree": tree, "seed": 1}, "weights")

    def test_weights_non_positive_or_non_integer(self):
        for bad in ([0, 1, 1], [-1, 1, 1], [1.5, 1, 1], [True, 1, 1], ["1", 1, 1]):
            tree = pick_tree(weights=bad)
            self.assert_invalid({"tree": tree, "seed": 1}, "weights")

    def test_missing_seed(self):
        with self.assertRaises(ValueError) as ctx:
            self.service.evaluate_behavior({"tree": pick_tree()})
        self.assertIn("seed", str(ctx.exception))

    def test_seed_not_integer(self):
        for bad_seed in (True, False, 1.5, "42", None, [1]):
            self.assert_invalid({"tree": pick_tree(), "seed": bad_seed}, "seed")

    def test_seed_out_of_range(self):
        self.assert_invalid({"tree": pick_tree(), "seed": -1}, "seed")
        self.assert_invalid({"tree": pick_tree(), "seed": MAX_SEED + 1}, "seed")

    def test_invalid_seed_rejected_before_any_effect(self):
        tree = pick_tree()
        for bad_request in (
            {"tree": tree},
            {"tree": tree, "seed": True},
            {"tree": tree, "seed": MAX_SEED + 1},
        ):
            with self.assertRaises(ValueError):
                self.service.evaluate_behavior(bad_request)

    def test_nested_random_selector_also_requires_seed(self):
        tree = {
            "id": "root",
            "type": "sequence",
            "children": [pick_tree()],
        }
        self.assert_invalid({"tree": tree}, "seed")
        result = self.service.evaluate_behavior({"tree": tree, "seed": 3})
        self.assertEqual(result["trace"][-1]["id"], "root")


class RandomSelectorExportTest(unittest.TestCase):
    def setUp(self):
        self.service = Service()

    def test_export_accepts_random_selector_without_seed(self):
        tree = pick_tree(weights=[2, 3, 5])
        result = self.service.export_behavior_tree({"tree": tree})
        self.assertEqual(result["status"], "EXPORTED")
        self.assertEqual(
            [(n["id"], n["type"], n["depth"]) for n in result["nodes"]],
            [
                ("pick", "random_selector", 0),
                ("c0", "action", 1),
                ("c1", "action", 1),
                ("c2", "action", 1),
            ],
        )
        self.assertEqual(
            result["edges"],
            [
                {"from": "pick", "to": "c0", "index": 0},
                {"from": "pick", "to": "c1", "index": 1},
                {"from": "pick", "to": "c2", "index": 2},
            ],
        )
        self.assertEqual(result["trace_order"], [])

    def test_export_marks_trace_from_evaluation(self):
        tree = pick_tree()
        evaluated = self.service.evaluate_behavior({"tree": tree, "seed": 42})
        result = self.service.export_behavior_tree(
            {"tree": tree, "trace": evaluated["trace"]}
        )
        self.assertEqual(result["trace_order"], ["c2", "pick"])
        by_id = {n["id"]: n for n in result["nodes"]}
        self.assertTrue(by_id["c2"]["visited"])
        self.assertTrue(by_id["pick"]["visited"])
        self.assertFalse(by_id["c0"]["visited"])
        self.assertFalse(by_id["c1"]["visited"])

    def test_export_rejects_invalid_random_selector(self):
        bad_trees = [
            {"id": "r", "type": "random_selector", "children": []},
            pick_tree(weights=[1, 1]),
            pick_tree(weights=[0, 1, 1]),
            pick_tree(weights=[True, 1, 1]),
        ]
        for tree in bad_trees:
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

    def test_evaluate_ok(self):
        status, body = self.post(
            "/v1/behavior-trees/evaluate", {"tree": pick_tree(), "seed": 42}
        )
        self.assertEqual(status, 200)
        self.assertEqual(body["status"], "SUCCESS")
        self.assertEqual(body["blackboard"], {"k2": 2})
        self.assertEqual([t["id"] for t in body["trace"]], ["c2", "pick"])

    def test_evaluate_invalid_returns_422_invalid_tree(self):
        bad_bodies = [
            {"tree": pick_tree()},  # missing seed
            {"tree": pick_tree(), "seed": True},
            {"tree": pick_tree(), "seed": -1},
            {"tree": pick_tree(), "seed": MAX_SEED + 1},
            {"tree": pick_tree(weights=[1, 1]), "seed": 1},
            {"tree": pick_tree(weights=[0, 1, 1]), "seed": 1},
            {"tree": {"id": "r", "type": "random_selector", "children": []}, "seed": 1},
        ]
        for payload in bad_bodies:
            status, body = self.post("/v1/behavior-trees/evaluate", payload)
            self.assertEqual(status, 422, payload)
            self.assertEqual(body["error"]["code"], "invalid_tree")

    def test_evaluate_unparseable_json_returns_400(self):
        status, body = self.post_raw("/v1/behavior-trees/evaluate", "{not json")
        self.assertEqual(status, 400)
        self.assertEqual(body["error"]["code"], "invalid_json")

    def test_visualize_accepts_random_selector_without_seed(self):
        status, body = self.post("/v1/behavior-trees/visualize", {"tree": pick_tree()})
        self.assertEqual(status, 200)
        self.assertEqual(body["status"], "EXPORTED")
        self.assertEqual(body["nodes"][0]["type"], "random_selector")
        self.assertEqual(len(body["edges"]), 3)

    def test_visualize_invalid_tree_returns_422(self):
        status, body = self.post(
            "/v1/behavior-trees/visualize",
            {"tree": {"id": "r", "type": "random_selector", "children": []}},
        )
        self.assertEqual(status, 422)
        self.assertEqual(body["error"]["code"], "invalid_behavior_visualization")


if __name__ == "__main__":
    unittest.main()
