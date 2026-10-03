import http.client
import json
import threading
import unittest
from http.server import ThreadingHTTPServer

from npcmind.server import Handler


class SteeringHttpTest(unittest.TestCase):
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

    def payload(self, **overrides):
        base = {
            "agent": {"position": {"x": 0.0, "y": 0.0}, "radius": 1.0},
            "max_speed": 10.0,
            "desired_velocity": {"x": 1.0, "y": 0.0},
            "time_horizon": 1.0,
            "candidates": [
                {"id": "straight", "velocity": {"x": 5.0, "y": 0.0}},
                {"id": "dodge", "velocity": {"x": 0.0, "y": 5.0}},
            ],
            "obstacles": [{"id": "wall", "position": {"x": 5.0, "y": 0.0}, "radius": 1.0}],
        }
        base.update(overrides)
        return base

    def test_selected_round_trip(self):
        status, body = self.post("/v1/steering/avoid", json.dumps(self.payload()))
        self.assertEqual(status, 200)
        self.assertEqual(body["status"], "SELECTED")
        self.assertEqual(body["selected"], "dodge")
        self.assertEqual(body["velocity"], {"x": 0.0, "y": 5.0})
        self.assertEqual([e["id"] for e in body["evaluations"]], ["straight", "dodge"])
        self.assertEqual(body["evaluations"][0]["collision_ids"], ["wall"])
        self.assertFalse(body["evaluations"][0]["admissible"])
        self.assertTrue(body["evaluations"][1]["admissible"])

    def test_blocked_round_trip(self):
        payload = self.payload(
            candidates=[{"id": "fast", "velocity": {"x": 5.0, "y": 0.0}}],
        )
        status, body = self.post("/v1/steering/avoid", json.dumps(payload))
        self.assertEqual(status, 200)
        self.assertEqual(body["status"], "BLOCKED")
        self.assertIsNone(body["selected"])
        self.assertEqual(body["velocity"], {"x": 0, "y": 0})
        self.assertEqual(len(body["evaluations"]), 1)

    def test_invalid_request_returns_422_invalid_steering(self):
        bad = self.payload()
        del bad["agent"]
        status, body = self.post("/v1/steering/avoid", json.dumps(bad))
        self.assertEqual(status, 422)
        self.assertEqual(body["error"]["code"], "invalid_steering")

    def test_duplicate_object_id_returns_422(self):
        bad = self.payload(
            neighbors=[
                {"id": "wall", "position": {"x": 9.0, "y": 0.0}, "radius": 1.0,
                 "velocity": {"x": 0.0, "y": 0.0}}
            ]
        )
        status, body = self.post("/v1/steering/avoid", json.dumps(bad))
        self.assertEqual(status, 422)
        self.assertEqual(body["error"]["code"], "invalid_steering")

    def test_unparseable_json_returns_400_invalid_json(self):
        status, body = self.post("/v1/steering/avoid", "{not valid json")
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
