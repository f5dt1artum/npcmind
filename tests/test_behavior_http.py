import http.client
import json
import threading
import unittest
from http.server import ThreadingHTTPServer

from npcmind.server import Handler


class BehaviorVisualizeHttpTest(unittest.TestCase):
    def setUp(self):
        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.port = self.httpd.server_address[1]
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.thread.start()

    def tearDown(self):
        self.httpd.shutdown()
        self.httpd.server_close()
        self.thread.join()

    def post(self, path, raw_body):
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=5)
        conn.request("POST", path, raw_body, {"Content-Type": "application/json"})
        response = conn.getresponse()
        body = response.read().decode("utf-8")
        conn.close()
        return response.status, json.loads(body)

    def tree(self):
        return {
            "id": "root",
            "type": "sequence",
            "children": [
                {"id": "c1", "type": "condition", "op": "exists", "key": "x"},
                {"id": "a1", "type": "action", "op": "set", "key": "y", "value": 1},
            ],
        }

    def test_export_round_trip(self):
        payload = {
            "tree": self.tree(),
            "trace": [
                {"id": "c1", "type": "condition", "status": "FAILURE"},
                {"id": "root", "type": "sequence", "status": "FAILURE"},
            ],
        }
        status, body = self.post("/v1/behavior-trees/visualize", json.dumps(payload))
        self.assertEqual(status, 200)
        self.assertEqual(body["status"], "EXPORTED")
        self.assertEqual(
            body["nodes"],
            [
                {"id": "root", "type": "sequence", "depth": 0, "visited": True, "status": "FAILURE"},
                {"id": "c1", "type": "condition", "depth": 1, "visited": True, "status": "FAILURE"},
                {"id": "a1", "type": "action", "depth": 1, "visited": False, "status": None},
            ],
        )
        self.assertEqual(
            body["edges"],
            [
                {"from": "root", "to": "c1", "index": 0},
                {"from": "root", "to": "a1", "index": 1},
            ],
        )
        self.assertEqual(body["trace_order"], ["c1", "root"])

    def test_trace_defaults_to_empty(self):
        status, body = self.post(
            "/v1/behavior-trees/visualize", json.dumps({"tree": self.tree()})
        )
        self.assertEqual(status, 200)
        self.assertEqual(body["status"], "EXPORTED")
        self.assertEqual(body["trace_order"], [])
        self.assertTrue(all(not node["visited"] for node in body["nodes"]))

    def test_invalid_request_returns_422_invalid_behavior_visualization(self):
        tree = self.tree()
        for bad in (
            {"not": "a visualization request"},
            {},
            {"tree": 42},
            {"tree": {"id": "r", "type": "parallel"}},
            {"tree": tree, "trace": {}},
            {"tree": tree, "trace": ["c1"]},
            {"tree": tree, "trace": [{"id": "", "type": "condition", "status": "SUCCESS"}]},
            {
                "tree": tree,
                "trace": [
                    {"id": "c1", "type": "condition", "status": "SUCCESS"},
                    {"id": "c1", "type": "condition", "status": "FAILURE"},
                ],
            },
            {"tree": tree, "trace": [{"id": "ghost", "type": "action", "status": "SUCCESS"}]},
            {"tree": tree, "trace": [{"id": "c1", "type": "action", "status": "SUCCESS"}]},
            {"tree": tree, "trace": [{"id": "c1", "type": "condition", "status": "BROKEN"}]},
        ):
            status, body = self.post("/v1/behavior-trees/visualize", json.dumps(bad))
            self.assertEqual(status, 422, bad)
            self.assertEqual(body["error"]["code"], "invalid_behavior_visualization")

    def test_unparseable_json_returns_400_invalid_json(self):
        status, body = self.post("/v1/behavior-trees/visualize", "{not valid json")
        self.assertEqual(status, 400)
        self.assertEqual(body["error"]["code"], "invalid_json")

    def test_evaluate_route_still_served(self):
        payload = {
            "tree": {"id": "a", "type": "action", "op": "set", "key": "x", "value": 1}
        }
        status, body = self.post("/v1/behavior-trees/evaluate", json.dumps(payload))
        self.assertEqual(status, 200)
        self.assertEqual(body["status"], "SUCCESS")
        self.assertEqual(body["blackboard"], {"x": 1})

    def test_healthz_unchanged(self):
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=5)
        conn.request("GET", "/healthz")
        response = conn.getresponse()
        body = json.loads(response.read().decode("utf-8"))
        conn.close()
        self.assertEqual(response.status, 200)
        self.assertEqual(body["status"], "ok")


if __name__ == "__main__":
    unittest.main()
