"""HTTP entry point for NpcMind."""

from __future__ import annotations

import argparse
import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from .service import Service


def _reject_constant(value: str) -> None:
    raise ValueError(f"invalid JSON constant {value!r}")


def env_address() -> tuple[str, int]:
    raw = os.environ.get("NPCMIND_ADDR", "127.0.0.1:8080")
    host, _, port = raw.rpartition(":")
    if not host or not port.isdigit():
        raise SystemExit(f"invalid NPCMIND_ADDR: {raw!r}")
    return host, int(port)


class Handler(BaseHTTPRequestHandler):
    service = Service()
    post_routes = {
        "/v1/behavior-trees/evaluate": ("evaluate_behavior", "invalid_tree"),
        "/v1/state-machines/step": ("step_state_machine", "invalid_state_machine"),
    }

    def send_json(self, status: int, payload: dict) -> None:
        body = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        if self.path == "/healthz":
            self.send_json(200, self.service.health())
            return
        self.send_json(404, {"error": {"code": "not_found", "message": f"no route for {self.path}"}})

    def do_POST(self) -> None:
        route = self.post_routes.get(self.path)
        if route is None:
            self.send_json(404, {"error": {"code": "not_found", "message": f"no route for {self.path}"}})
            return
        method_name, error_code = route
        try:
            length = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            length = 0
        raw = self.rfile.read(max(length, 0))
        try:
            request = json.loads(raw.decode("utf-8"), parse_constant=_reject_constant)
        except ValueError as exc:
            self.send_json(400, {"error": {"code": "invalid_json", "message": f"request body is not valid JSON: {exc}"}})
            return
        try:
            result = getattr(self.service, method_name)(request)
        except ValueError as exc:
            self.send_json(422, {"error": {"code": error_code, "message": str(exc)}})
            return
        self.send_json(200, result)

    def log_message(self, fmt: str, *args: object) -> None:
        """Silence per-request logging so recorded output stays stable."""


def main() -> int:
    parser = argparse.ArgumentParser(prog="npcmind.server", description="游戏 AI 与 NPC 智能行为引擎")
    host, port = env_address()
    parser.add_argument("--host", default=host)
    parser.add_argument("--port", type=int, default=port)
    args = parser.parse_args()
    httpd = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"NpcMind listening on http://{args.host}:{httpd.server_address[1]}", flush=True)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
