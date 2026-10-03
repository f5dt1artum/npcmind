import http.client
import json
import threading
import unittest
from http.server import ThreadingHTTPServer

from npcmind.server import Handler


class BehaviorVisualizationHttpTest(unittest.TestCase):
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
            "id": "root",
            "type": "sequence",
            "children": [
                {"id": "a1", "type": "action", "op": "status", "status": "FAILURE"},
                {"id": "a2", "type": "action", "op": "set", "key": "x", "value": 1},
            ],
        }

    def test_export_without_trace(self):
        status, body = self.post("/v1/behavior-trees/visualize", {"tree": self.tree()})
        self.assertEqual(status, 200)
        self.assertEqual(body["status"], "EXPORTED")
        self.assertEqual([n["id"] for n in body["nodes"]], ["root", "a1", "a2"])
        self.assertEqual(
            [(n["depth"], n["visited"], n["status"]) for n in body["nodes"]],
            [(0, False, None), (1, False, None), (1, False, None)],
        )
        self.assertEqual(
            body["edges"],
            [
                {"from": "root", "to": "a1", "index": 0},
                {"from": "root", "to": "a2", "index": 1},
            ],
        )
        self.assertEqual(body["trace_order"], [])

    def test_export_partial_trace_from_evaluate(self):
        tree = self.tree()
        ev_status, ev_body = self.post("/v1/behavior-trees/evaluate", {"tree": tree})
        self.assertEqual(ev_status, 200)
        status, body = self.post(
            "/v1/behavior-trees/visualize", {"tree": tree, "trace": ev_body["trace"]}
        )
        self.assertEqual(status, 200)
        self.assertEqual(body["trace_order"], ["a1", "root"])
        by_id = {n["id"]: n for n in body["nodes"]}
        self.assertTrue(by_id["a1"]["visited"])
        self.assertEqual(by_id["a1"]["status"], "FAILURE")
        self.assertTrue(by_id["root"]["visited"])
        self.assertEqual(by_id["root"]["status"], "FAILURE")
        self.assertFalse(by_id["a2"]["visited"])
        self.assertIsNone(by_id["a2"]["status"])

    def test_invalid_requests_return_422_invalid_behavior_visualization(self):
        good = self.tree()
        bad_bodies = [
            [1, 2],
            {},
            {"tree": {"id": "r", "type": "parallel"}},
            {"tree": good, "trace": {}},
            {"tree": good, "trace": [42]},
            {"tree": good, "trace": [{"type": "action", "status": "SUCCESS"}]},
            {"tree": good, "trace": [{"id": "a1", "type": "action", "status": "SUCCESS"},
                                     {"id": "a1", "type": "action", "status": "FAILURE"}]},
            {"tree": good, "trace": [{"id": "ghost", "type": "action", "status": "SUCCESS"}]},
            {"tree": good, "trace": [{"id": "a1", "type": "condition", "status": "FAILURE"}]},
            {"tree": good, "trace": [{"id": "a1", "type": "action", "status": "MAYBE"}]},
        ]
        for bad in bad_bodies:
            status, body = self.post("/v1/behavior-trees/visualize", bad)
            self.assertEqual(status, 422, bad)
            self.assertEqual(body["error"]["code"], "invalid_behavior_visualization")
            self.assertNotIn("nodes", body)

    def test_unparseable_json_returns_400_invalid_json(self):
        status, body = self.post_raw("/v1/behavior-trees/visualize", "{broken")
        self.assertEqual(status, 400)
        self.assertEqual(body["error"]["code"], "invalid_json")

    def test_non_finite_constants_are_rejected(self):
        # A bare NaN token (not a quoted string) is rejected as invalid JSON.
        raw = json.dumps({"tree": self.tree()}).replace('"x", "value": 1', '"x", "value": NaN')
        status, body = self.post_raw("/v1/behavior-trees/visualize", raw)
        self.assertEqual(status, 400)
        self.assertEqual(body["error"]["code"], "invalid_json")

    def test_evaluate_endpoint_unchanged(self):
        status, body = self.post("/v1/behavior-trees/evaluate", {"tree": self.tree()})
        self.assertEqual(status, 200)
        self.assertEqual(body["status"], "FAILURE")
        self.assertEqual([t["id"] for t in body["trace"]], ["a1", "root"])

    def test_unknown_route_returns_404(self):
        status, body = self.post("/v1/unknown", {})
        self.assertEqual(status, 404)
        self.assertEqual(body["error"]["code"], "not_found")

    def test_stateless_across_requests(self):
        tree = {"id": "solo", "type": "action", "op": "delete", "key": "k"}
        status, first = self.post(
            "/v1/behavior-trees/visualize",
            {"tree": tree, "trace": [{"id": "solo", "type": "action", "status": "SUCCESS"}]},
        )
        self.assertEqual(status, 200)
        self.assertTrue(first["nodes"][0]["visited"])
        status, second = self.post("/v1/behavior-trees/visualize", {"tree": tree})
        self.assertEqual(status, 200)
        self.assertFalse(second["nodes"][0]["visited"])
        self.assertIsNone(second["nodes"][0]["status"])
        self.assertEqual(second["trace_order"], [])


if __name__ == "__main__":
    unittest.main()
