import http.client
import json
import threading
import unittest
from http.server import ThreadingHTTPServer

from npcmind.server import Handler


class ScheduleHttpTest(unittest.TestCase):
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
            "now": 600,
            "needs": {"hunger": 0.8},
            "activities": [
                {"id": "work", "start_minute": 540, "end_minute": 1020, "priority": 1},
                {
                    "id": "eat",
                    "start_minute": 1200,
                    "end_minute": 1260,
                    "priority": 3,
                    "need": "hunger",
                    "trigger": 0.7,
                    "relief": 0.5,
                },
            ],
        }
        payload.update(overrides)
        return payload

    def test_selected_round_trip(self):
        status, body = self.post("/v1/schedules/decide", json.dumps(self.base_payload()))
        self.assertEqual(status, 200)
        self.assertEqual(body["status"], "SELECTED")
        self.assertEqual(body["selected"], "eat")
        self.assertAlmostEqual(body["needs"]["hunger"], 0.3)
        evaluations = body["evaluations"]
        self.assertEqual([e["id"] for e in evaluations], ["work", "eat"])
        self.assertTrue(evaluations[0]["scheduled"])
        self.assertFalse(evaluations[0]["urgent"])
        self.assertIsNone(evaluations[0]["need_level"])
        self.assertFalse(evaluations[1]["scheduled"])
        self.assertTrue(evaluations[1]["urgent"])
        self.assertTrue(evaluations[1]["eligible"])
        self.assertEqual(evaluations[1]["need_level"], 0.8)

    def test_idle_round_trip(self):
        payload = self.base_payload(
            now=300,
            needs={"hunger": 0.4},
            activities=[
                {
                    "id": "eat",
                    "start_minute": 1200,
                    "end_minute": 1260,
                    "need": "hunger",
                    "trigger": 0.7,
                    "relief": 0.5,
                }
            ],
        )
        status, body = self.post("/v1/schedules/decide", json.dumps(payload))
        self.assertEqual(status, 200)
        self.assertEqual(body["status"], "IDLE")
        self.assertIsNone(body["selected"])
        self.assertEqual(body["needs"], {"hunger": 0.4})
        self.assertEqual(len(body["evaluations"]), 1)

    def test_invalid_request_returns_422_invalid_schedule(self):
        for bad in (
            {"not": "a schedule request"},
            self.base_payload(now=1440),
            self.base_payload(now=True),
            self.base_payload(needs={"hunger": 1.5}),
            self.base_payload(needs={"hunger": "high"}),
            self.base_payload(activities=[]),
            self.base_payload(
                activities=[
                    {"id": "a", "start_minute": 0, "end_minute": 1},
                    {"id": "a", "start_minute": 2, "end_minute": 3},
                ]
            ),
            self.base_payload(
                activities=[{"id": "a", "start_minute": 0, "end_minute": 1440}]
            ),
            self.base_payload(
                activities=[
                    {"id": "a", "start_minute": 0, "end_minute": 1, "priority": 0.5}
                ]
            ),
            self.base_payload(
                activities=[
                    {"id": "a", "start_minute": 0, "end_minute": 1, "need": "unknown",
                     "trigger": 0.5, "relief": 0.5}
                ]
            ),
            self.base_payload(
                activities=[
                    {"id": "a", "start_minute": 0, "end_minute": 1, "need": "hunger",
                     "trigger": 0.5}
                ]
            ),
            self.base_payload(
                activities=[
                    {"id": "a", "start_minute": 0, "end_minute": 1, "need": "hunger",
                     "trigger": 2, "relief": 0.5}
                ]
            ),
        ):
            status, body = self.post("/v1/schedules/decide", json.dumps(bad))
            self.assertEqual(status, 422, bad)
            self.assertEqual(body["error"]["code"], "invalid_schedule")

    def test_unparseable_json_returns_400_invalid_json(self):
        status, body = self.post("/v1/schedules/decide", "{not valid json")
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
            "utterance": "hello there",
            "intents": [{"id": "greet", "patterns": ["hello there"]}],
        }
        status, body = self.post("/v1/dialogue/intents/match", json.dumps(payload))
        self.assertEqual(status, 200)
        self.assertEqual(body["status"], "MATCHED")
        self.assertEqual(body["intent"], "greet")


if __name__ == "__main__":
    unittest.main()
