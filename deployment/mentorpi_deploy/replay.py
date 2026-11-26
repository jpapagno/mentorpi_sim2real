"""Serve the episode viewer and local recordings on loopback."""

import argparse
import json
from contextlib import suppress
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import unquote, urlsplit

from .paths import ROOT


class ReplayHandler(BaseHTTPRequestHandler):
    def respond(self, body, content_type):
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        route = unquote(urlsplit(self.path).path)
        diagnostics = (ROOT / "diagnostics").resolve()
        if route in ("/", "/replay.html"):
            self.respond((ROOT / "replay.html").read_bytes(), "text/html; charset=utf-8")
        elif route == "/episodes":
            files = sorted(
                (path for path in diagnostics.glob("policy-*.json") if path.is_file() and not path.is_symlink()),
                reverse=True,
            )
            self.respond(json.dumps([path.name for path in files]).encode(), "application/json")
        elif route.startswith("/episodes/"):
            name = route.removeprefix("/episodes/")
            path = diagnostics / name
            if (
                path.parent == diagnostics
                and name.startswith("policy-")
                and path.suffix == ".json"
                and path.is_file()
                and not path.is_symlink()
            ):
                self.respond(path.read_bytes(), "application/json")
                return
            self.send_error(404)
        else:
            self.send_error(404)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    with ThreadingHTTPServer(("127.0.0.1", args.port), ReplayHandler) as server:
        print(f"Episode replay at http://127.0.0.1:{server.server_port}/", flush=True)
        with suppress(KeyboardInterrupt):
            server.serve_forever()


if __name__ == "__main__":
    main()
