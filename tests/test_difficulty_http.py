import http.client
import json
import threading
import unittest
from http.server import ThreadingHTTPServer

from npcmind.server import Handler


class DifficultyHttpTest(unittest.TestCase):
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
            "current_difficulty": 0.5,
            "target": {"min": 0.4, "max": 0.6},
            "max_step": 0.1,
            "now": 100.0,
            "cooldown": 10.0,
            "signals": [
                {"id": "win_rate", "value": 0.9, "weight": 2.0},
                {"id": "deaths", "value": 0.3, "weight": 1.0},
            ],
        }
        payload.update(overrides)
        return payload

    def test_increased_round_trip(self):
        status, body = self.post("/v1/difficulty/adjust", json.dumps(self.base_payload()))
        self.assertEqual(status, 200)
        self.assertEqual(body["status"], "INCREASED")
        self.assertEqual(body["previous_difficulty"], 0.5)
        self.assertAlmostEqual(body["score"], 0.7)
        self.assertAlmostEqual(body["difficulty"], 0.6)
        self.assertAlmostEqual(body["adjustment"], 0.1)

    def test_decreased_round_trip(self):
        payload = self.base_payload(
            signals=[{"id": "win_rate", "value": 0.1, "weight": 1.0}]
        )
        status, body = self.post("/v1/difficulty/adjust", json.dumps(payload))
        self.assertEqual(status, 200)
        self.assertEqual(body["status"], "DECREASED")
        self.assertAlmostEqual(body["difficulty"], 0.4)
        self.assertAlmostEqual(body["adjustment"], -0.1)

    def test_unchanged_round_trip(self):
        payload = self.base_payload(
            signals=[{"id": "win_rate", "value": 0.5, "weight": 1.0}]
        )
        status, body = self.post("/v1/difficulty/adjust", json.dumps(payload))
        self.assertEqual(status, 200)
        self.assertEqual(body["status"], "UNCHANGED")
        self.assertEqual(body["difficulty"], 0.5)
        self.assertEqual(body["adjustment"], 0)

    def test_cooldown_round_trip(self):
        payload = self.base_payload(last_adjusted_at=95.0)
        status, body = self.post("/v1/difficulty/adjust", json.dumps(payload))
        self.assertEqual(status, 200)
        self.assertEqual(body["status"], "COOLDOWN")
        self.assertEqual(body["difficulty"], 0.5)
        self.assertEqual(body["adjustment"], 0)

    def test_invalid_request_returns_422_invalid_difficulty(self):
        for bad in (
            {"not": "a difficulty request"},
            {},
            self.base_payload(current_difficulty=True),
            self.base_payload(current_difficulty=1.5),
            self.base_payload(target={"min": 0.6, "max": 0.4}),
            self.base_payload(target={"min": 0.4}),
            self.base_payload(max_step=0),
            self.base_payload(max_step=2),
            self.base_payload(now=-1),
            self.base_payload(cooldown=-1),
            self.base_payload(signals=[]),
            self.base_payload(signals=[{"id": "s", "value": 0.5}]),
            self.base_payload(signals=[{"id": "s", "value": 0.5, "weight": 0}]),
            self.base_payload(signals=[{"id": "s", "value": True, "weight": 1}]),
            self.base_payload(
                signals=[
                    {"id": "s", "value": 0.5, "weight": 1},
                    {"id": "s", "value": 0.6, "weight": 1},
                ]
            ),
            self.base_payload(last_adjusted_at=-1),
            self.base_payload(last_adjusted_at=101.0),
        ):
            status, body = self.post("/v1/difficulty/adjust", json.dumps(bad))
            self.assertEqual(status, 422, bad)
            self.assertEqual(body["error"]["code"], "invalid_difficulty")

    def test_non_finite_constants_are_rejected(self):
        raw = json.dumps(self.base_payload()).replace("0.9", "NaN")
        status, body = self.post("/v1/difficulty/adjust", raw)
        self.assertEqual(status, 400)
        self.assertEqual(body["error"]["code"], "invalid_json")

    def test_unparseable_json_returns_400_invalid_json(self):
        status, body = self.post("/v1/difficulty/adjust", "{not valid json")
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
            "roles": [{"id": "tank", "capacity": 1}],
            "agents": [{"id": "a", "scores": {"tank": 0.9}}],
        }
        status, body = self.post("/v1/teams/assign", json.dumps(payload))
        self.assertEqual(status, 200)
        self.assertEqual(body["status"], "ASSIGNED")


if __name__ == "__main__":
    unittest.main()
