import http.client
import json
import threading
import unittest
from http.server import ThreadingHTTPServer

from npcmind.server import Handler


class UtilityHttpTest(unittest.TestCase):
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
            "/v1/utility/select",
            json.dumps(
                {
                    "context": {"hunger": 0.9},
                    "options": [
                        {
                            "id": "eat",
                            "base": 2,
                            "considerations": [
                                {"key": "hunger", "min": 0, "max": 1, "curve": "linear"}
                            ],
                        },
                        {"id": "idle"},
                    ],
                }
            ),
        )
        self.assertEqual(status, 200)
        self.assertEqual(body["status"], "SELECTED")
        self.assertEqual(body["selected"], "eat")
        self.assertAlmostEqual(body["score"], 1.8)
        self.assertEqual([c["id"] for c in body["candidates"]], ["eat", "idle"])

    def test_no_selection_round_trip(self):
        status, body = self.post(
            "/v1/utility/select",
            json.dumps(
                {
                    "context": {},
                    "options": [{"id": "a", "enabled": False}],
                }
            ),
        )
        self.assertEqual(status, 200)
        self.assertEqual(body["status"], "NO_SELECTION")
        self.assertIsNone(body["selected"])
        self.assertIsNone(body["score"])

    def test_invalid_request_returns_422_invalid_utility(self):
        status, body = self.post(
            "/v1/utility/select",
            json.dumps({"context": {}, "options": [{"id": "a"}, {"id": "a"}]}),
        )
        self.assertEqual(status, 422)
        self.assertEqual(body["error"]["code"], "invalid_utility")

    def test_unparseable_json_returns_400_invalid_json(self):
        status, body = self.post("/v1/utility/select", "{not valid json")
        self.assertEqual(status, 400)
        self.assertEqual(body["error"]["code"], "invalid_json")

    def test_unknown_route_returns_404(self):
        status, body = self.post("/v1/utility/unknown", "{}")
        self.assertEqual(status, 404)
        self.assertEqual(body["error"]["code"], "not_found")


if __name__ == "__main__":
    unittest.main()
