import http.client
import json
import threading
import unittest
from http.server import ThreadingHTTPServer

from npcmind.server import Handler


class FlockingHttpTest(unittest.TestCase):
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
            "id": "self",
            "position": {"x": 0.0, "y": 0.0},
            "velocity": {"x": 1.0, "y": 0.0},
            "neighbors": [
                {"id": "n1", "position": {"x": 1.0, "y": 0.0},
                 "velocity": {"x": 0.0, "y": 1.0}},
                {"id": "n2", "position": {"x": -1.0, "y": 0.0},
                 "velocity": {"x": 0.0, "y": -1.0}},
            ],
            "perception_radius": 5.0,
            "separation_radius": 2.0,
            "max_speed": 10.0,
            "max_acceleration": 10.0,
            "delta_time": 0.5,
            "separation": 2.0,
            "alignment": 1.0,
            "cohesion": 1.0,
        }
        payload.update(overrides)
        return payload

    def test_steered_round_trip(self):
        status, body = self.post("/v1/steering/flock", json.dumps(self.base_payload()))
        self.assertEqual(status, 200)
        self.assertEqual(body["status"], "STEERED")
        self.assertEqual(body["neighbor_ids"], ["n1", "n2"])
        self.assertEqual(body["separation"], {"x": 0.0, "y": 0.0})
        self.assertEqual(body["alignment"], {"x": -1.0, "y": 0.0})
        self.assertEqual(body["cohesion"], {"x": 0.0, "y": 0.0})
        self.assertEqual(body["acceleration"], {"x": -1.0, "y": 0.0})
        self.assertEqual(body["velocity"], {"x": 0.5, "y": 0.0})

    def test_no_neighbors_round_trip(self):
        status, body = self.post(
            "/v1/steering/flock",
            json.dumps(self.base_payload(neighbors=[])),
        )
        self.assertEqual(status, 200)
        self.assertEqual(body["status"], "NO_NEIGHBORS")
        self.assertEqual(body["neighbor_ids"], [])
        self.assertEqual(body["acceleration"], {"x": 0, "y": 0})
        self.assertEqual(body["velocity"], {"x": 1.0, "y": 0.0})

    def test_invalid_request_returns_422_invalid_flocking(self):
        for bad in (
            {"not": "a flocking request"},
            self.base_payload(id=""),
            self.base_payload(perception_radius=0),
            self.base_payload(separation_radius=-1),
            self.base_payload(max_speed=True),
            self.base_payload(max_acceleration=0),
            self.base_payload(delta_time=0),
            self.base_payload(separation=-0.5),
            self.base_payload(alignment=None),
            self.base_payload(perception_radius=2.0, separation_radius=3.0),
            self.base_payload(position={"x": 0, "y": "north"}),
            self.base_payload(neighbors={}),
            self.base_payload(neighbors=[{"id": "self", "position": {"x": 1, "y": 0},
                                          "velocity": {"x": 0, "y": 0}}]),
        ):
            status, body = self.post("/v1/steering/flock", json.dumps(bad))
            self.assertEqual(status, 422, bad)
            self.assertEqual(body["error"]["code"], "invalid_flocking")

    def test_unparseable_json_returns_400_invalid_json(self):
        status, body = self.post("/v1/steering/flock", "{not valid json")
        self.assertEqual(status, 400)
        self.assertEqual(body["error"]["code"], "invalid_json")

    def test_non_finite_json_constant_returns_400_invalid_json(self):
        status, body = self.post(
            "/v1/steering/flock",
            json.dumps(self.base_payload(max_speed=1.0)).replace(
                '"max_speed": 1.0', '"max_speed": NaN'
            ),
        )
        self.assertEqual(status, 400)
        self.assertEqual(body["error"]["code"], "invalid_json")

    def test_avoid_route_is_unchanged(self):
        payload = {
            "position": {"x": 0.0, "y": 0.0},
            "radius": 1.0,
            "max_speed": 10.0,
            "desired_velocity": {"x": 1.0, "y": 0.0},
            "time_horizon": 1.0,
            "candidates": [{"id": "go", "velocity": {"x": 1.0, "y": 0.0}}],
        }
        status, body = self.post("/v1/steering/avoid", json.dumps(payload))
        self.assertEqual(status, 200)
        self.assertEqual(body["status"], "SELECTED")
        self.assertEqual(body["selected"], "go")

    def test_unknown_route_returns_404(self):
        status, body = self.post("/v1/unknown", "{}")
        self.assertEqual(status, 404)
        self.assertEqual(body["error"]["code"], "not_found")


if __name__ == "__main__":
    unittest.main()
