import http.client
import json
import threading
import unittest
from http.server import ThreadingHTTPServer

from npcmind.server import Handler


class PerceptionHttpTest(unittest.TestCase):
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

    def test_success_round_trip(self):
        status, body = self.post(
            "/v1/perception/memory",
            json.dumps(
                {
                    "now": 10,
                    "retention": 5,
                    "memory": [
                        {"id": "a", "kind": "npc", "last_seen": 6, "confidence": 0.4,
                         "position": {"x": 1, "y": 1}},
                        {"id": "gone", "kind": "item", "last_seen": 0, "confidence": 1,
                         "position": {"x": 0, "y": 0}},
                    ],
                    "observations": [
                        {"id": "a", "kind": "enemy", "confidence": 0.9,
                         "position": {"x": 2, "y": 2}, "data": {"hp": 10}},
                        {"id": "b", "kind": "npc", "confidence": 0.2,
                         "position": {"x": 3, "y": 3}},
                    ],
                }
            ),
        )
        self.assertEqual(status, 200)
        self.assertEqual(body["status"], "UPDATED")
        self.assertEqual([r["id"] for r in body["memory"]], ["a", "b"])
        self.assertEqual(body["seen"], ["a", "b"])
        self.assertEqual(body["forgotten"], ["gone"])
        record_a = body["memory"][0]
        self.assertEqual(record_a["kind"], "enemy")
        self.assertEqual(record_a["last_seen"], 10)
        self.assertEqual(record_a["data"], {"hp": 10})
        self.assertEqual(body["memory"][1]["data"], {})

    def test_invalid_request_returns_422_invalid_perception(self):
        status, body = self.post(
            "/v1/perception/memory",
            json.dumps({"now": 10, "retention": 0, "memory": [], "observations": []}),
        )
        self.assertEqual(status, 422)
        self.assertEqual(body["error"]["code"], "invalid_perception")

    def test_duplicate_ids_returns_422(self):
        status, body = self.post(
            "/v1/perception/memory",
            json.dumps(
                {
                    "now": 1,
                    "retention": 1,
                    "observations": [
                        {"id": "a", "kind": "k", "confidence": 0, "position": {"x": 0, "y": 0}},
                        {"id": "a", "kind": "k", "confidence": 0, "position": {"x": 0, "y": 0}},
                    ],
                }
            ),
        )
        self.assertEqual(status, 422)
        self.assertEqual(body["error"]["code"], "invalid_perception")

    def test_unparseable_json_returns_400_invalid_json(self):
        status, body = self.post("/v1/perception/memory", "{not valid json")
        self.assertEqual(status, 400)
        self.assertEqual(body["error"]["code"], "invalid_json")

    def test_unknown_route_returns_404(self):
        status, body = self.post("/v1/unknown", "{}")
        self.assertEqual(status, 404)
        self.assertEqual(body["error"]["code"], "not_found")

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
