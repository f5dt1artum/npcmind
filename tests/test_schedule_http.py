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
            "needs": {"hunger": 0.2, "fun": 0.1},
            "activities": [
                {"id": "patrol", "start_minute": 540, "end_minute": 660, "priority": 1},
                {
                    "id": "eat",
                    "start_minute": 0,
                    "end_minute": 300,
                    "need": "hunger",
                    "trigger": 0.5,
                    "relief": 0.25,
                },
            ],
        }
        payload.update(overrides)
        return payload

    def test_selected_round_trip(self):
        status, body = self.post("/v1/schedules/decide", json.dumps(self.base_payload()))
        self.assertEqual(status, 200)
        self.assertEqual(body["status"], "SELECTED")
        self.assertEqual(body["selected"], "patrol")
        self.assertEqual(body["needs"], {"hunger": 0.2, "fun": 0.1})
        self.assertEqual(
            body["evaluations"],
            [
                {
                    "id": "patrol",
                    "scheduled": True,
                    "urgent": False,
                    "eligible": True,
                    "need_level": None,
                },
                {
                    "id": "eat",
                    "scheduled": False,
                    "urgent": False,
                    "eligible": False,
                    "need_level": 0.2,
                },
            ],
        )

    def test_urgent_round_trip(self):
        payload = self.base_payload(needs={"hunger": 0.9, "fun": 0.1})
        status, body = self.post("/v1/schedules/decide", json.dumps(payload))
        self.assertEqual(status, 200)
        self.assertEqual(body["status"], "SELECTED")
        self.assertEqual(body["selected"], "eat")
        self.assertEqual(body["needs"], {"hunger": 0.65, "fun": 0.1})
        eat = body["evaluations"][1]
        self.assertFalse(eat["scheduled"])
        self.assertTrue(eat["urgent"])
        self.assertTrue(eat["eligible"])

    def test_idle_round_trip(self):
        payload = self.base_payload(now=400, needs={"hunger": 0.1, "fun": 0.0})
        status, body = self.post("/v1/schedules/decide", json.dumps(payload))
        self.assertEqual(status, 200)
        self.assertEqual(body["status"], "IDLE")
        self.assertIsNone(body["selected"])
        self.assertEqual(body["needs"], {"hunger": 0.1, "fun": 0.0})
        self.assertEqual(len(body["evaluations"]), 2)

    def test_invalid_request_is_422(self):
        status, body = self.post(
            "/v1/schedules/decide",
            json.dumps(self.base_payload(now=1440)),
        )
        self.assertEqual(status, 422)
        self.assertEqual(body["error"]["code"], "invalid_schedule")

    def test_unknown_need_reference_is_422(self):
        payload = self.base_payload(
            activities=[
                {
                    "id": "a",
                    "start_minute": 0,
                    "end_minute": 1,
                    "need": "missing",
                    "trigger": 0.5,
                    "relief": 0.5,
                }
            ]
        )
        status, body = self.post("/v1/schedules/decide", json.dumps(payload))
        self.assertEqual(status, 422)
        self.assertEqual(body["error"]["code"], "invalid_schedule")

    def test_malformed_json_is_400(self):
        status, body = self.post("/v1/schedules/decide", "{not json")
        self.assertEqual(status, 400)
        self.assertEqual(body["error"]["code"], "invalid_json")

    def test_unknown_route_is_404(self):
        status, body = self.post("/v1/schedules/unknown", json.dumps(self.base_payload()))
        self.assertEqual(status, 404)
        self.assertEqual(body["error"]["code"], "not_found")

    def test_health_still_ok(self):
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=5)
        conn.request("GET", "/healthz")
        response = conn.getresponse()
        body = json.loads(response.read().decode("utf-8"))
        conn.close()
        self.assertEqual(response.status, 200)
        self.assertEqual(body["status"], "ok")


if __name__ == "__main__":
    unittest.main()
