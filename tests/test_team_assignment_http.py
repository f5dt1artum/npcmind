import http.client
import json
import threading
import unittest
from http.server import ThreadingHTTPServer

from npcmind.server import Handler


class TeamAssignmentHttpTest(unittest.TestCase):
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
            "roles": [
                {"id": "tank", "capacity": 1},
                {"id": "healer", "capacity": 2},
            ],
            "agents": [
                {"id": "a", "scores": {"tank": 0.9, "healer": 0.5}},
                {"id": "b", "scores": {"tank": 0.8, "healer": 0.4}},
                {"id": "c", "scores": {"healer": 0.6}},
            ],
        }
        payload.update(overrides)
        return payload

    def test_assigned_round_trip(self):
        status, body = self.post("/v1/teams/assign", json.dumps(self.base_payload()))
        self.assertEqual(status, 200)
        self.assertEqual(body["status"], "ASSIGNED")
        # a->tank 0.9, c->healer 0.6, b->healer 0.4 = 1.9 (tank capacity 1).
        self.assertEqual(
            body["assignments"],
            [
                {"agent": "a", "role": "tank", "score": 0.9},
                {"agent": "b", "role": "healer", "score": 0.4},
                {"agent": "c", "role": "healer", "score": 0.6},
            ],
        )
        self.assertEqual(body["unassigned"], [])
        self.assertAlmostEqual(body["total_score"], 1.9)
        self.assertEqual(
            body["roles"],
            [
                {"id": "tank", "capacity": 1, "agents": ["a"], "score": 0.9},
                {"id": "healer", "capacity": 2, "agents": ["b", "c"], "score": 1.0},
            ],
        )

    def test_empty_assignment_round_trip(self):
        payload = self.base_payload(
            agents=[
                {"id": "a", "scores": {}},
                {"id": "b", "scores": {"tank": 0}},
            ]
        )
        status, body = self.post("/v1/teams/assign", json.dumps(payload))
        self.assertEqual(status, 200)
        self.assertEqual(body["status"], "ASSIGNED")
        self.assertEqual(body["assignments"], [])
        self.assertEqual(body["unassigned"], ["a", "b"])
        self.assertEqual(body["total_score"], 0)
        self.assertEqual(
            body["roles"],
            [
                {"id": "tank", "capacity": 1, "agents": [], "score": 0},
                {"id": "healer", "capacity": 2, "agents": [], "score": 0},
            ],
        )

    def test_invalid_request_returns_422_invalid_team_assignment(self):
        for bad in (
            {"not": "a team assignment request"},
            {},
            self.base_payload(roles=[]),
            self.base_payload(roles="nope"),
            self.base_payload(agents=[]),
            self.base_payload(agents="nope"),
            self.base_payload(roles=[{}]),
            self.base_payload(roles=[{"id": "tank", "capacity": 1},
                                     {"id": "tank", "capacity": 1}]),
            self.base_payload(roles=[{"id": "tank", "capacity": 0}]),
            self.base_payload(roles=[{"id": "tank", "capacity": True}]),
            self.base_payload(roles=[{"id": "tank", "capacity": 1.5}]),
            self.base_payload(agents=["nope"]),
            self.base_payload(agents=[{"id": "a", "scores": {}},
                                     {"id": "a", "scores": {}}]),
            self.base_payload(agents=[{"id": "a"}]),
            self.base_payload(agents=[{"id": "a", "scores": []}]),
            self.base_payload(agents=[{"id": "a", "scores": {"rogue": 0.5}}]),
            self.base_payload(agents=[{"id": "a", "scores": {"tank": True}}]),
            self.base_payload(agents=[{"id": "a", "scores": {"tank": 1.5}}]),
            self.base_payload(agents=[{"id": "a", "scores": {"tank": -0.1}}]),
        ):
            status, body = self.post("/v1/teams/assign", json.dumps(bad))
            self.assertEqual(status, 422, bad)
            self.assertEqual(body["error"]["code"], "invalid_team_assignment")

    def test_non_finite_constants_are_rejected(self):
        raw = json.dumps(self.base_payload()).replace('0.9', 'NaN')
        status, body = self.post("/v1/teams/assign", raw)
        self.assertEqual(status, 400)
        self.assertEqual(body["error"]["code"], "invalid_json")

    def test_unparseable_json_returns_400_invalid_json(self):
        status, body = self.post("/v1/teams/assign", "{not valid json")
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
        payload = {"now": 600, "needs": {}, "activities": [
            {"id": "idle", "start_minute": 0, "end_minute": 0}
        ]}
        status, body = self.post("/v1/schedules/decide", json.dumps(payload))
        self.assertEqual(status, 200)
        self.assertEqual(body["status"], "SELECTED")
        self.assertEqual(body["selected"], "idle")


if __name__ == "__main__":
    unittest.main()
