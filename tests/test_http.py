import json
import unittest
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer

from npcmind.server import Handler


class HttpEndpointTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        cls.port = cls.server.server_address[1]
        import threading
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()

    def request(self, method, path, body=None, raw=None):
        data = raw if raw is not None else (json.dumps(body).encode("utf-8") if body is not None else None)
        req = urllib.request.Request(
            f"http://127.0.0.1:{self.port}{path}", data=data, method=method,
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req) as resp:
                return resp.status, json.loads(resp.read())
        except urllib.error.HTTPError as exc:
            payload = json.loads(exc.read())
            exc.close()
            return exc.code, payload

    def test_healthz_still_works(self):
        status, payload = self.request("GET", "/healthz")
        self.assertEqual(status, 200)
        self.assertEqual(payload["status"], "ok")

    def test_unknown_route(self):
        status, payload = self.request("GET", "/nope")
        self.assertEqual(status, 404)
        self.assertEqual(payload["error"]["code"], "not_found")
        status, payload = self.request("POST", "/nope", body={})
        self.assertEqual(status, 404)
        self.assertEqual(payload["error"]["code"], "not_found")

    def test_evaluate_success(self):
        body = {
            "tree": {"id": "root", "type": "sequence", "children": [
                {"id": "a", "type": "action", "op": "set", "key": "hp", "value": 10},
                {"id": "c", "type": "condition", "key": "hp", "op": "equals", "value": 10},
            ]},
        }
        status, payload = self.request("POST", "/v1/behavior-trees/evaluate", body=body)
        self.assertEqual(status, 200)
        self.assertEqual(payload["status"], "SUCCESS")
        self.assertEqual(payload["blackboard"], {"hp": 10})
        self.assertEqual([t["id"] for t in payload["trace"]], ["a", "c", "root"])

    def test_evaluate_default_blackboard(self):
        body = {"tree": {"id": "a", "type": "action", "op": "status", "status": "RUNNING"}}
        status, payload = self.request("POST", "/v1/behavior-trees/evaluate", body=body)
        self.assertEqual(status, 200)
        self.assertEqual(payload["status"], "RUNNING")
        self.assertEqual(payload["blackboard"], {})

    def test_invalid_json_returns_400(self):
        status, payload = self.request("POST", "/v1/behavior-trees/evaluate", raw=b"{not json")
        self.assertEqual(status, 400)
        self.assertEqual(payload["error"]["code"], "invalid_json")

    def test_invalid_tree_returns_422(self):
        status, payload = self.request("POST", "/v1/behavior-trees/evaluate", body={"tree": {"id": "x"}})
        self.assertEqual(status, 422)
        self.assertEqual(payload["error"]["code"], "invalid_tree")
        self.assertIn("x", payload["error"]["message"])

    def test_missing_tree_returns_422(self):
        status, payload = self.request("POST", "/v1/behavior-trees/evaluate", body={})
        self.assertEqual(status, 422)
        self.assertEqual(payload["error"]["code"], "invalid_tree")


if __name__ == "__main__":
    unittest.main()
