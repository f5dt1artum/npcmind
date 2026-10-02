import http.client
import json
import threading
import unittest
from http.server import ThreadingHTTPServer

from npcmind.server import Handler


class NavigationHttpTest(unittest.TestCase):
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

    def test_success_round_trip(self):
        status, body = self.post(
            "/v1/navigation/path",
            json.dumps(
                {
                    "grid": [[1, 1, 1], [1, 1, 1]],
                    "start": {"x": 0, "y": 0},
                    "goal": {"x": 2, "y": 1},
                }
            ),
        )
        self.assertEqual(status, 200)
        self.assertEqual(body["status"], "SUCCESS")
        self.assertEqual(body["cost"], 3)
        self.assertEqual(body["path"][0], {"x": 0, "y": 0})
        self.assertEqual(body["path"][-1], {"x": 2, "y": 1})

    def test_unreachable_round_trip(self):
        status, body = self.post(
            "/v1/navigation/path",
            json.dumps(
                {
                    "grid": [[1, None], [None, 1]],
                    "start": {"x": 0, "y": 0},
                    "goal": {"x": 1, "y": 1},
                }
            ),
        )
        self.assertEqual(status, 200)
        self.assertEqual(body["status"], "UNREACHABLE")
        self.assertEqual(body["path"], [])
        self.assertIsNone(body["cost"])

    def test_invalid_request_returns_422_invalid_navigation(self):
        status, body = self.post(
            "/v1/navigation/path", json.dumps({"grid": [], "start": {}, "goal": {}})
        )
        self.assertEqual(status, 422)
        self.assertEqual(body["error"]["code"], "invalid_navigation")

    def test_unparseable_json_returns_400_invalid_json(self):
        status, body = self.post("/v1/navigation/path", "{not valid json")
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
