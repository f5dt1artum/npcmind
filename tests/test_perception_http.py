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

    def test_update_round_trip(self):
        status, body = self.post(
            "/v1/perception/memory",
            json.dumps(
                {
                    "now": 10.0,
                    "retention": 5.0,
                    "memory": [
                        {"id": "old", "kind": "a", "last_seen": 2.0, "confidence": 0.4,
                         "position": {"x": 0, "y": 0}},
                        {"id": "keep", "kind": "b", "last_seen": 9.0, "confidence": 0.4,
                         "position": {"x": 0, "y": 0}},
                    ],
                    "observations": [
                        {"id": "old", "kind": "a2", "confidence": 0.9,
                         "position": {"x": 3, "y": 4}, "data": {"z": True}},
                        {"id": "new", "kind": "c", "confidence": 0.5,
                         "position": {"x": 1, "y": 1}},
                    ],
                }
            ),
        )
        self.assertEqual(status, 200)
        self.assertEqual(body["status"], "UPDATED")
        self.assertEqual([r["id"] for r in body["memory"]], ["old", "keep", "new"])
        self.assertEqual(body["memory"][0]["last_seen"], 10.0)
        self.assertEqual(body["memory"][0]["data"], {"z": True})
        self.assertEqual(body["memory"][1]["data"], {})
        self.assertEqual(body["seen"], ["old", "new"])
        self.assertEqual(body["forgotten"], [])

    def test_forgotten_ids_reported_in_old_memory_order(self):
        status, body = self.post(
            "/v1/perception/memory",
            json.dumps(
                {
                    "now": 10.0,
                    "retention": 5.0,
                    "memory": [
                        {"id": "g1", "kind": "k", "last_seen": 1.0, "confidence": 1,
                         "position": {"x": 0, "y": 0}},
                        {"id": "stay", "kind": "k", "last_seen": 8.0, "confidence": 1,
                         "position": {"x": 0, "y": 0}},
                        {"id": "g2", "kind": "k", "last_seen": 2.0, "confidence": 1,
                         "position": {"x": 0, "y": 0}},
                    ],
                    "observations": [],
                }
            ),
        )
        self.assertEqual(status, 200)
        self.assertEqual([r["id"] for r in body["memory"]], ["stay"])
        self.assertEqual(body["forgotten"], ["g1", "g2"])

    def test_invalid_request_returns_422_invalid_perception(self):
        status, body = self.post(
            "/v1/perception/memory",
            json.dumps(
                {
                    "now": 10.0,
                    "retention": 5.0,
                    "memory": [
                        {"id": "a", "kind": "k", "last_seen": 1.0, "confidence": 2.0,
                         "position": {"x": 0, "y": 0}},
                    ],
                }
            ),
        )
        self.assertEqual(status, 422)
        self.assertEqual(body["error"]["code"], "invalid_perception")

    def test_duplicate_ids_returns_422(self):
        status, body = self.post(
            "/v1/perception/memory",
            json.dumps(
                {
                    "now": 10.0,
                    "retention": 5.0,
                    "observations": [
                        {"id": "a", "kind": "k", "confidence": 1, "position": {"x": 0, "y": 0}},
                        {"id": "a", "kind": "k", "confidence": 1, "position": {"x": 0, "y": 0}},
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

    def test_unknown_route_still_404(self):
        status, body = self.post("/v1/perception/unknown", "{}")
        self.assertEqual(status, 404)
        self.assertEqual(body["error"]["code"], "not_found")


if __name__ == "__main__":
    unittest.main()
