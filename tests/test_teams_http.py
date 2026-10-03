import http.client
import json
import threading
import unittest
from http.server import ThreadingHTTPServer

from npcmind.server import Handler


class TeamsHttpTest(unittest.TestCase):
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
            "agents": [
                {"id": "alice", "scores": {"tank": 0.9, "healer": 0.4}},
                {"id": "bob", "scores": {"tank": 0.7, "dps": 0.8}},
                {"id": "carol", "scores": {"healer": 0.6}},
            ],
            "roles": [
                {"id": "tank", "capacity": 1},
                {"id": "healer", "capacity": 1},
                {"id": "dps", "capacity": 2},
            ],
        }
        payload.update(overrides)
        return payload

    def test_assigned_round_trip(self):
        status, body = self.post("/v1/teams/assign", json.dumps(self.base_payload()))
        self.assertEqual(status, 200)
        self.assertEqual(body["status"], "ASSIGNED")
        self.assertEqual(
            body["assignments"],
            [
                {"agent": "alice", "role": "tank", "score": 0.9},
                {"agent": "bob", "role": "dps", "score": 0.8},
                {"agent": "carol", "role": "healer", "score": 0.6},
            ],
        )
        self.assertEqual(body["unassigned"], [])
        self.assertAlmostEqual(body["total_score"], 2.3)
        self.assertEqual(
            body["roles"],
            [
                {"id": "tank", "capacity": 1, "agents": ["alice"], "score": 0.9},
                {"id": "healer", "capacity": 1, "agents": ["carol"], "score": 0.6},
                {"id": "dps", "capacity": 2, "agents": ["bob"], "score": 0.8},
            ],
        )

    def test_empty_assignment_round_trip(self):
        payload = self.base_payload(
            agents=[{"id": "alice", "scores": {"tank": 0}}],
            roles=[{"id": "tank", "capacity": 1}],
        )
        status, body = self.post("/v1/teams/assign", json.dumps(payload))
        self.assertEqual(status, 200)
        self.assertEqual(body["status"], "ASSIGNED")
        self.assertEqual(body["assignments"], [])
        self.assertEqual(body["unassigned"], ["alice"])
        self.assertEqual(body["total_score"], 0)
        self.assertEqual(
            body["roles"], [{"id": "tank", "capacity": 1, "agents": [], "score": 0}]
        )

    def test_invalid_request_returns_422_invalid_team_assignment(self):
        for bad in (
            {"not": "a team request"},
            self.base_payload(agents=[]),
            self.base_payload(roles=[]),
            self.base_payload(agents=["not a dict"]),
            self.base_payload(roles=["not a dict"]),
            self.base_payload(agents=[{"id": "a", "scores": {}}, {"id": "a", "scores": {}}]),
            self.base_payload(
                roles=[{"id": "r", "capacity": 1}, {"id": "r", "capacity": 1}]
            ),
            self.base_payload(roles=[{"id": "r", "capacity": 0}]),
            self.base_payload(roles=[{"id": "r", "capacity": True}]),
            self.base_payload(roles=[{"id": "r", "capacity": 1.5}]),
            self.base_payload(agents=[{"id": "a"}]),
            self.base_payload(agents=[{"id": "a", "scores": "high"}]),
            self.base_payload(agents=[{"id": "a", "scores": {"unknown": 0.5}}]),
            self.base_payload(agents=[{"id": "a", "scores": {"tank": True}}]),
            self.base_payload(agents=[{"id": "a", "scores": {"tank": 1.5}}]),
            self.base_payload(agents=[{"id": "a", "scores": {"tank": -0.1}}]),
        ):
            status, body = self.post("/v1/teams/assign", json.dumps(bad))
            self.assertEqual(status, 422, bad)
            self.assertEqual(body["error"]["code"], "invalid_team_assignment")

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
        payload = {
            "now": 600,
            "needs": {},
            "activities": [{"id": "work", "start_minute": 540, "end_minute": 1020}],
        }
        status, body = self.post("/v1/schedules/decide", json.dumps(payload))
        self.assertEqual(status, 200)
        self.assertEqual(body["status"], "SELECTED")
        self.assertEqual(body["selected"], "work")


if __name__ == "__main__":
    unittest.main()
