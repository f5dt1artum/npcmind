import http.client
import json
import threading
import unittest
from http.server import ThreadingHTTPServer

from npcmind.server import Handler


class AttentionHttpTest(unittest.TestCase):
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

    def base_payload(self, **overrides):
        payload = {
            "observer": {
                "position": {"x": 0.0, "y": 0.0},
                "forward": {"x": 1.0, "y": 0.0},
                "max_distance": 10.0,
                "field_of_view_degrees": 180.0,
            },
            "now": 10.0,
            "memory_horizon": 5.0,
            "memory": [
                {
                    "id": "guard",
                    "kind": "npc",
                    "last_seen": 10.0,
                    "confidence": 1.0,
                    "position": {"x": 4.0, "y": 0.0},
                    "data": {"threat": 0.5},
                }
            ],
        }
        payload.update(overrides)
        return payload

    def test_selected_round_trip(self):
        status, body = self.post("/v1/attention/select", json.dumps(self.base_payload()))
        self.assertEqual(status, 200)
        self.assertEqual(body["status"], "SELECTED")
        self.assertEqual(body["selected"], "guard")
        self.assertEqual(body["score"], 0.3)
        (evaluation,) = body["evaluations"]
        self.assertEqual(evaluation["id"], "guard")
        self.assertTrue(evaluation["visible"])
        self.assertEqual(evaluation["distance"], 4.0)
        self.assertEqual(evaluation["threat"], 0.5)
        self.assertEqual(evaluation["proximity"], 0.6)
        self.assertEqual(evaluation["freshness"], 1.0)
        self.assertEqual(evaluation["score"], 0.3)

    def test_no_target_round_trip(self):
        status, body = self.post(
            "/v1/attention/select", json.dumps(self.base_payload(memory=[]))
        )
        self.assertEqual(status, 200)
        self.assertEqual(body["status"], "NO_TARGET")
        self.assertIsNone(body["selected"])
        self.assertIsNone(body["score"])
        self.assertEqual(body["evaluations"], [])

    def test_invalid_request_returns_422_invalid_attention(self):
        for bad in (
            {"not": "an attention request"},
            self.base_payload(observer={"position": {"x": 0, "y": 0}}),
            self.base_payload(now=-1.0),
            self.base_payload(memory_horizon=0),
            self.base_payload(
                memory=[
                    {
                        "id": "guard",
                        "kind": "npc",
                        "last_seen": 10.0,
                        "confidence": 1.0,
                        "position": {"x": 4.0, "y": 0.0},
                        "data": {"threat": 2.0},
                    }
                ]
            ),
        ):
            status, body = self.post("/v1/attention/select", json.dumps(bad))
            self.assertEqual(status, 422, bad)
            self.assertEqual(body["error"]["code"], "invalid_attention")

    def test_duplicate_memory_ids_return_422(self):
        record = self.base_payload()["memory"][0]
        status, body = self.post(
            "/v1/attention/select",
            json.dumps(self.base_payload(memory=[record, dict(record)])),
        )
        self.assertEqual(status, 422)
        self.assertEqual(body["error"]["code"], "invalid_attention")

    def test_unparseable_json_returns_400_invalid_json(self):
        status, body = self.post("/v1/attention/select", "{not valid json")
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
