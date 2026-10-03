import http.client
import json
import threading
import unittest
from http.server import ThreadingHTTPServer

from npcmind.server import Handler


class DialogueHttpTest(unittest.TestCase):
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
            "utterance": "attack the goblin",
            "intents": [
                {"id": "attack", "patterns": ["attack the {target}"], "priority": 1},
                {"id": "flee", "patterns": ["run away"]},
            ],
            "context": {"armed": True},
        }
        payload.update(overrides)
        return payload

    def test_matched_round_trip(self):
        status, body = self.post(
            "/v1/dialogue/intents/match", json.dumps(self.base_payload())
        )
        self.assertEqual(status, 200)
        self.assertEqual(body["status"], "MATCHED")
        self.assertEqual(body["intent"], "attack")
        self.assertEqual(body["slots"], {"target": "goblin"})
        self.assertEqual(
            body["candidates"],
            [
                {
                    "id": "attack",
                    "priority": 1,
                    "literal_count": 2,
                    "pattern_index": 0,
                    "slots": {"target": "goblin"},
                }
            ],
        )

    def test_no_match_round_trip(self):
        status, body = self.post(
            "/v1/dialogue/intents/match",
            json.dumps(self.base_payload(utterance="hello there")),
        )
        self.assertEqual(status, 200)
        self.assertEqual(body["status"], "NO_MATCH")
        self.assertIsNone(body["intent"])
        self.assertEqual(body["slots"], {})
        self.assertEqual(body["candidates"], [])

    def test_invalid_request_is_422(self):
        status, body = self.post(
            "/v1/dialogue/intents/match",
            json.dumps(self.base_payload(intents=[])),
        )
        self.assertEqual(status, 422)
        self.assertEqual(body["error"]["code"], "invalid_dialogue")

    def test_invalid_slot_name_is_422(self):
        status, body = self.post(
            "/v1/dialogue/intents/match",
            json.dumps(self.base_payload(intents=[{"id": "a", "patterns": ["{9bad}"]}])),
        )
        self.assertEqual(status, 422)
        self.assertEqual(body["error"]["code"], "invalid_dialogue")

    def test_malformed_json_is_400(self):
        status, body = self.post("/v1/dialogue/intents/match", "{not json")
        self.assertEqual(status, 400)
        self.assertEqual(body["error"]["code"], "invalid_json")

    def test_unknown_route_is_404(self):
        status, body = self.post("/v1/dialogue/unknown", json.dumps(self.base_payload()))
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
