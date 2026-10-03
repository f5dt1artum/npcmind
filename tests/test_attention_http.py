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
                "field_of_view_degrees": 90.0,
            },
            "now": 10.0,
            "memory_horizon": 20.0,
            "memory": [
                {
                    "id": "a",
                    "kind": "npc",
                    "last_seen": 10.0,
                    "confidence": 1.0,
                    "position": {"x": 2.0, "y": 0.0},
                    "data": {"threat": 0.5},
                },
                {
                    "id": "b",
                    "kind": "npc",
                    "last_seen": 5.0,
                    "confidence": 1.0,
                    "position": {"x": -2.0, "y": 0.0},
                    "data": {"threat": 1.0},
                },
            ],
        }
        payload.update(overrides)
        return payload

    def test_selected_round_trip(self):
        status, body = self.post("/v1/attention/select", json.dumps(self.base_payload()))
        self.assertEqual(status, 200)
        self.assertEqual(body["status"], "SELECTED")
        self.assertEqual(body["selected"], "a")
        self.assertAlmostEqual(body["score"], 0.4)
        evaluations = body["evaluations"]
        self.assertEqual([e["id"] for e in evaluations], ["a", "b"])
        self.assertTrue(evaluations[0]["visible"])
        self.assertFalse(evaluations[1]["visible"])
        self.assertEqual(evaluations[1]["score"], 0.0)
        self.assertEqual(evaluations[0]["threat"], 0.5)
        self.assertEqual(evaluations[0]["freshness"], 1.0)
        self.assertEqual(evaluations[1]["freshness"], 0.75)

    def test_no_target_round_trip(self):
        payload = self.base_payload(
            memory=[
                {
                    "id": "b",
                    "kind": "npc",
                    "last_seen": 5.0,
                    "confidence": 1.0,
                    "position": {"x": -2.0, "y": 0.0},
                    "data": {"threat": 1.0},
                }
            ]
        )
        status, body = self.post("/v1/attention/select", json.dumps(payload))
        self.assertEqual(status, 200)
        self.assertEqual(body["status"], "NO_TARGET")
        self.assertIsNone(body["selected"])
        self.assertIsNone(body["score"])
        self.assertEqual(len(body["evaluations"]), 1)

    def test_empty_memory_round_trip(self):
        status, body = self.post(
            "/v1/attention/select", json.dumps(self.base_payload(memory=[]))
        )
        self.assertEqual(status, 200)
        self.assertEqual(body["status"], "NO_TARGET")
        self.assertEqual(body["evaluations"], [])

    def test_invalid_request_returns_422_invalid_attention(self):
        for bad in (
            {"not": "an attention request"},
            self.base_payload(now=-1),
            self.base_payload(memory_horizon=0),
            self.base_payload(
                observer={
                    "position": {"x": 0.0, "y": 0.0},
                    "forward": {"x": 0.0, "y": 0.0},
                    "max_distance": 10.0,
                    "field_of_view_degrees": 90.0,
                }
            ),
            self.base_payload(
                observer={
                    "position": {"x": 0.0, "y": 0.0},
                    "forward": {"x": 1.0, "y": 0.0},
                    "max_distance": 0,
                    "field_of_view_degrees": 90.0,
                }
            ),
            self.base_payload(
                observer={
                    "position": {"x": 0.0, "y": 0.0},
                    "forward": {"x": 1.0, "y": 0.0},
                    "max_distance": 10.0,
                    "field_of_view_degrees": 400.0,
                }
            ),
            self.base_payload(
                memory=[
                    {
                        "id": "x",
                        "kind": "npc",
                        "last_seen": 11.0,
                        "confidence": 1.0,
                        "position": {"x": 1.0, "y": 0.0},
                        "data": {"threat": 1.0},
                    }
                ]
            ),
            self.base_payload(
                memory=[
                    {
                        "id": "x",
                        "kind": "npc",
                        "last_seen": 1.0,
                        "confidence": 1.0,
                        "position": {"x": 1.0, "y": 0.0},
                        "data": {"threat": 2.0},
                    }
                ]
            ),
        ):
            status, body = self.post("/v1/attention/select", json.dumps(bad))
            self.assertEqual(status, 422, bad)
            self.assertEqual(body["error"]["code"], "invalid_attention")

    def test_duplicate_memory_ids_return_422(self):
        payload = self.base_payload(
            memory=[
                {
                    "id": "dup",
                    "kind": "npc",
                    "last_seen": 1.0,
                    "confidence": 1.0,
                    "position": {"x": 1.0, "y": 0.0},
                    "data": {"threat": 1.0},
                },
                {
                    "id": "dup",
                    "kind": "npc",
                    "last_seen": 2.0,
                    "confidence": 1.0,
                    "position": {"x": 2.0, "y": 0.0},
                },
            ]
        )
        status, body = self.post("/v1/attention/select", json.dumps(payload))
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

    def test_existing_route_still_served(self):
        payload = {
            "position": {"x": 0.0, "y": 0.0},
            "radius": 0.5,
            "max_speed": 10.0,
            "desired_velocity": {"x": 1.0, "y": 0.0},
            "time_horizon": 1.0,
            "candidates": [
                {"id": "go", "velocity": {"x": 1.0, "y": 0.0}}
            ],
        }
        status, body = self.post("/v1/steering/avoid", json.dumps(payload))
        self.assertEqual(status, 200)
        self.assertEqual(body["status"], "SELECTED")
        self.assertEqual(body["selected"], "go")


if __name__ == "__main__":
    unittest.main()
