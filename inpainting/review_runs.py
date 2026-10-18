"""Review completed runs in a local browser and sync saved reviews to Aim."""

import argparse
from datetime import datetime, timezone
import json
from html import escape
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
import secrets
from urllib.parse import parse_qs, quote, urlsplit
import webbrowser


RATINGS = {
    "seam_quality": "Seam quality (1=obvious seam, 5=invisible seam)",
    "scene_coherence": "Scene coherence (1=implausible/artifacts, 5=believable)",
    "preservation": "Preservation (1=unwanted changes, 5=original content preserved)",
}


def pending_runs(runs_dir):
    """Find recorded runs with saved outputs and no existing visual review."""
    for params_path in sorted(runs_dir.glob("*/params.json")):
        run_dir = params_path.parent
        if (run_dir / "review.json").exists():
            continue
        try:
            params = json.loads(params_path.read_text())
            output = run_dir / Path(params["output"]).name
            comparison = output.with_name(f"{output.stem}_comparison.png")
            if output.is_file() and comparison.is_file():
                yield run_dir, params, comparison
        except (OSError, ValueError, KeyError, TypeError) as error:
            print(f"Skipping {run_dir}: {error}")


def sync_review(run_dir, review):
    """Update one existing Aim run; never create a replacement or duplicate run."""
    repo = None
    run = None
    try:
        from aim import Repo, Run, Text

        if any(type(review.get(name)) is not int or not 1 <= review[name] <= 5 for name in RATINGS):
            raise ValueError("Review ratings must be whole numbers from 1 to 5")
        repo = Repo(str(Path(__file__).resolve().parent))
        matches = []
        already_synced = False
        for candidate in repo.iter_runs():
            try:
                if candidate.get("run_dir") == str(run_dir.resolve()):
                    matches.append(candidate.hash)
                    already_synced = review.get("reviewed_at") is not None and candidate.get("review_synced_at") == review["reviewed_at"]
            finally:
                try:
                    candidate.close()
                except AttributeError as error:
                    # shortcut: Aim 3 read-only close lacks sequence_infos; remove when Aim fixes its cleanup.
                    if "sequence_infos" not in str(error):
                        raise
        if len(matches) != 1:
            raise ValueError(f"Expected one matching Aim run, found {len(matches)}")
        if already_synced:
            return
        run = Run(matches[0], repo=repo, system_tracking_interval=None,
                  log_system_params=False, capture_terminal_logs=False)
        for name in RATINGS:
            run.track(review[name], name=f"review.{name}", step=0)
        run.track(Text(review.get("notes", "")), name="review.notes", step=0)
        run["review_synced_at"] = review.get("reviewed_at")
        if run.get("review_synced_at") != review.get("reviewed_at"):
            raise RuntimeError("Aim did not save the review sync timestamp")
        if run.get("review") is not None:
            del run["review"]
        print(f"Synced review to Aim: {matches[0]}")
    except Exception as error:
        print(f"Warning: Aim review sync failed for {run_dir}: {error}. Local review is preserved; retry by running this script again.")
    finally:
        for resource in (run, repo):
            if resource is not None:
                try:
                    resource.close()
                except Exception as error:
                    print(f"Warning: Aim close failed: {error}")


def sync_saved_reviews(runs_dir):
    for review_path in sorted(runs_dir.glob("*/review.json")):
        try:
            sync_review(review_path.parent, json.loads(review_path.read_text()))
        except (OSError, ValueError) as error:
            print(f"Cannot load {review_path}: {error}")


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

    def do_GET(self):
        if not self.valid_host():
            return
        address = urlsplit(self.path)
        runs = list(pending_runs(self.server.runs_dir))
        name = parse_qs(address.query).get("run", [""])[0]
        selected = next((entry for entry in runs if entry[0].name == name), None)
        if address.path == "/image":
            if selected is None:
                self.send_error(404)
                return
            self.respond(selected[2].read_bytes(), "image/png")
            return
        if address.path != "/":
            self.send_error(404)
            return
        if name and selected is None:
            self.send_error(404, "Run is missing or already reviewed")
            return
        selected = selected or (runs[0] if runs else None)
        content = "<h1>All available runs reviewed</h1>"
        if selected:
            run_dir, params, _ = selected
            options = "".join(f'<option value="{escape(entry[0].name, quote=True)}"'
                              f'{" selected" if entry == selected else ""}>{escape(entry[0].name)}</option>'
                              for entry in runs)
            fields = "".join(f'<fieldset><legend>{escape(label)}</legend>' + "".join(
                f'<label><input type="radio" name="{key}" value="{value}" required> {value}</label> '
                for value in range(1, 6)) + '</fieldset>' for key, label in RATINGS.items())
            index = runs.index(selected)
            next_run = runs[(index + 1) % len(runs)][0].name
            content = f'''<h1>Visual review</h1><p>{len(runs)} unreviewed runs</p>
<form method="get"><label>Choose run <select name="run">{options}</select></label><button>Open</button></form>
<h2>{escape(run_dir.name)}</h2><details><summary>All settings</summary><pre>{escape(json.dumps(params, indent=2))}</pre></details>
<p>Top: before. Bottom: after. Click image for full resolution.</p>
<a href="/image?run={quote(run_dir.name)}" target="_blank" rel="noopener"><img alt="Before and after collage comparison" src="/image?run={quote(run_dir.name)}"></a>
<form method="post" action="/review"><input type="hidden" name="token" value="{self.server.token}">
<input type="hidden" name="run" value="{escape(run_dir.name, quote=True)}">{fields}
<p><label>Notes<br><textarea name="notes" rows="3" maxlength="10000"></textarea></label></p>
<button>Save review and sync to Aim</button> <a href="/?run={quote(next_run)}">Skip / next</a></form>'''
        message = parse_qs(address.query).get("message", [""])[0]
        page = f'''<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Inpainting review</title>
<style>body{{font:16px system-ui;max-width:1100px;margin:24px auto;padding:0 16px}}img{{width:100%;height:auto}}fieldset{{margin:16px 0}}button,select,textarea{{font:inherit}}textarea{{width:100%;box-sizing:border-box}}pre{{white-space:pre-wrap;overflow-wrap:anywhere}}label{{margin-right:12px}}</style>
</head><body><p role="status">{escape(message)}</p>{content}</body></html>'''
        self.respond(page.encode())

    def do_POST(self):
        if not self.valid_host():
            return
        if self.path != "/review":
            self.send_error(404)
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if not 0 < length <= 65536:
                raise ValueError("Invalid request size")
            fields = parse_qs(self.rfile.read(length).decode("utf-8"), keep_blank_values=True)
            if not secrets.compare_digest(fields.get("token", [""])[0], self.server.token):
                self.send_error(403, "Invalid form token")
                return
            name = fields.get("run", [""])[0]
            selected = next((entry for entry in pending_runs(self.server.runs_dir) if entry[0].name == name), None)
            if selected is None:
                self.send_error(409, "Run is missing or already reviewed")
                return
            review = {}
            for key in RATINGS:
                value = fields.get(key, [""])[0]
                if value not in ("1", "2", "3", "4", "5"):
                    raise ValueError("Each rating must be a whole number from 1 to 5")
                review[key] = int(value)
            review["notes"] = fields.get("notes", [""])[0].strip()
            if len(review["notes"]) > 10000:
                raise ValueError("Notes exceed 10000 characters")
            review["reviewed_at"] = datetime.now(timezone.utc).isoformat()
            run_dir = selected[0]
            with (run_dir / "review.json").open("x") as file:
                file.write(json.dumps(review, indent=2) + "\n")
        except (ValueError, OSError) as error:
            self.send_error(400, str(error))
            return
        from contextlib import redirect_stdout
        import io
        with redirect_stdout(io.StringIO()) as messages:
            sync_review(run_dir, review)
        message = f"Review saved for {name}. {messages.getvalue().strip()}"
        self.send_response(303)
        self.send_header("Location", "/?message=" + quote(message))
        self.end_headers()


def review_server(runs_dir, port=0):
    # shortcut: single-user loopback server; use a production server and authentication before network sharing.
    server = HTTPServer(("127.0.0.1", port), ReviewHandler)
    server.runs_dir = runs_dir
    server.token = secrets.token_urlsafe(32)
    return server


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-dir", type=Path, default=Path(__file__).resolve().parent / "runs")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    if not args.runs_dir.is_dir():
        parser.error(f"Runs directory not found: {args.runs_dir}")
    sync_saved_reviews(args.runs_dir)
    with review_server(args.runs_dir, args.port) as server:
        url = f"http://127.0.0.1:{server.server_port}/"
        print(f"Review in your browser: {url}\nPress Ctrl+C to stop.")
        webbrowser.open(url)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            print("\nStopped. Saved reviews are preserved.")


if __name__ == "__main__":
    main()
