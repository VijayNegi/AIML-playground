"""Local web app on 127.0.0.1: static review/batch pages plus their JSON APIs."""

import argparse
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
from pathlib import Path
import secrets
from urllib.parse import urlsplit
import webbrowser

from inpaint_lab import batch_api, review_api, runs_api
from inpaint_lab.batches import log
from inpaint_lab.config import ROOT, RUNS_DIR, STATIC_DIR
from inpaint_lab.tracking import sync_saved_reviews


PAGES = {"/": "runs.html", "/run": "run.html", "/review": "review.html", "/batches": "batch.html", "/compare": "compare.html"}
STATIC_TYPES = {".js": "text/javascript; charset=utf-8", ".css": "text/css; charset=utf-8"}
ROUTES = (runs_api, review_api, batch_api)
MAX_JSON_BYTES = 131072


class ReviewHandler(BaseHTTPRequestHandler):
    def valid_host(self):
        if self.headers.get("Host") != f"127.0.0.1:{self.server.server_port}":
            self.send_error(403, "Use the printed loopback URL")
            return False
        return True

    def respond(self, body, content_type="text/html; charset=utf-8", status=200):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(body)

    def send_json(self, data, status=200):
        if status >= 400:
            self.error_detail = data.get("error")
        self.respond(json.dumps(data, allow_nan=False).encode(), "application/json; charset=utf-8", status)

    def send_error(self, code, message=None, explain=None):
        self.error_detail = message
        super().send_error(code, message, explain)

    def log_request(self, code="-", size="-"):
        # Successful GETs are page loads, polling and images: noise. Show actions and failures with their reason.
        code = int(code) if str(code).isdigit() else 0
        if (self.command == "GET" and code < 400) or self.path == "/favicon.ico":
            return
        detail = getattr(self, "error_detail", None)
        log(f"{self.command} {self.path} → {code}" + (f": {detail}" if detail else ""))

    def log_message(self, format, *args):
        pass  # Only send_error's duplicate "code 404" line and timeouts arrive here; log_request covers requests.

    def read_json(self):
        """CSRF-checked JSON object body. Sends 403 and returns None for a bad token."""
        if not secrets.compare_digest(self.headers.get("X-CSRF-Token", ""), self.server.token):
            self.send_json({"error": "Invalid form token"}, 403)
            return None
        length = int(self.headers.get("Content-Length", "0"))
        if not 0 < length <= MAX_JSON_BYTES or self.headers.get_content_type() != "application/json":
            raise ValueError("Expected a JSON request of at most 128KiB")
        data = json.loads(self.rfile.read(length))
        if not isinstance(data, dict):
            raise ValueError("Expected a JSON object")
        return data

    def do_GET(self):
        if not self.valid_host():
            return
        address = urlsplit(self.path)
        if address.path in PAGES:
            page = (STATIC_DIR / PAGES[address.path]).read_text().replace("__TOKEN__", self.server.token)
            self.respond(page.encode())
            return
        if address.path.startswith("/static/"):
            name = address.path.removeprefix("/static/")
            path = STATIC_DIR / name
            # Flat directory, fixed types: no traversal and no serving of templates or source.
            if "/" in name or "\\" in name or path.suffix not in STATIC_TYPES or not path.is_file():
                self.send_error(404)
                return
            self.respond(path.read_bytes(), STATIC_TYPES[path.suffix])
            return
        if not any(route.get_route(self, address) for route in ROUTES):
            self.send_error(404)

    def do_POST(self):
        if not self.valid_host():
            return
        if not any(route.post_route(self) for route in ROUTES):
            self.send_error(404)


class ReviewServer(HTTPServer):
    def server_close(self):
        batch_manager = getattr(self, "batch_manager", None)
        if batch_manager is not None:
            batch_manager.close()
        super().server_close()


def review_server(runs_dir, port=0):
    # shortcut: single-user loopback server; use a production server and authentication before network sharing.
    server = ReviewServer(("127.0.0.1", port), ReviewHandler)
    server.runs_dir = runs_dir
    server.token = secrets.token_urlsafe(32)
    server.project_root = ROOT
    return server


def main():
    parser = argparse.ArgumentParser(description="Review runs and manage batches in a local browser app.")
    parser.add_argument("--runs-dir", type=Path, default=RUNS_DIR)
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    if not args.runs_dir.is_dir():
        parser.error(f"Runs directory not found: {args.runs_dir}")
    sync_saved_reviews(args.runs_dir)
    with review_server(args.runs_dir, args.port) as server:
        url = f"http://127.0.0.1:{server.server_port}/"
        print(f"Inpaint Lab: {url}\nPress Ctrl+C to stop.")
        webbrowser.open(url)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            print("\nStopped. Saved reviews are preserved.")


if __name__ == "__main__":
    main()
