"""Shared test helpers: fixture runs, a background-server HTTP client, and headless Chrome rendering."""

import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import tempfile
import threading
import time
from http.client import HTTPConnection

from PIL import Image

from inpaint_lab.server import review_server


CHROME_PATHS = ("/Applications/Google Chrome.app/Contents/MacOS/Google Chrome", "google-chrome", "chromium", "chrome")
CHROME_FLAGS = ("--headless=new", "--disable-gpu", "--no-first-run", "--no-default-browser-check",
                "--disable-background-networking", "--disable-component-update", "--disable-sync",
                "--virtual-time-budget=5000")
SAMPLE_METRICS = {
    "canvas_size": [500, 448], "quality_band_width": 8,
    "quality_gap_bbox": [200, 0, 280, 448], "quality_effective_mask_bbox": [190, 0, 290, 448],
    "quality_editable_bbox": [180, 0, 300, 448], "crop_bbox": [100, 0, 400, 448],
    "quality_gap_left_delta_e76": 2.5, "quality_gap_right_gradient_excess": 0.03,
    "quality_patch_left_gradient_excess": 0.1, "quality_preservation_changed_fraction": 0.125,
    "quality_maniqa": 0.6, "quality_maniqa_min": 0.4, "quality_maniqa_min_bbox": [128, 0, 352, 224],
    "quality_maniqa_tiles": [{"bbox": [128, 0, 352, 224], "score": 0.4},
                             {"bbox": [128, 224, 352, 448], "score": 0.8}],
}


def make_run(root, name, metrics=None, params=None):
    run_dir = root / name
    run_dir.mkdir()
    (run_dir / "params.json").write_text(json.dumps(params or {"output": f"runs/{name}/output.png", "seed": 42}))
    Image.new("RGB", (8, 8)).save(run_dir / "output.png")
    Image.new("RGB", (8, 16)).save(run_dir / "output_comparison.png")
    if metrics is not None:
        (run_dir / "metrics.json").write_text(json.dumps(metrics))
    return run_dir


class Client:
    """Minimal HTTP client against a background server."""
    def __init__(self, server):
        self.server = server
        self.thread = threading.Thread(target=server.serve_forever)
        self.thread.start()

    def request(self, method, path, data=None, token=None, raw=None):
        connection = HTTPConnection("127.0.0.1", self.server.server_port, timeout=10)
        try:
            headers = {"Content-Type": "application/json", "X-CSRF-Token": token or "bad"}
            body = raw if raw is not None else (json.dumps(data) if data is not None else None)
            connection.request(method, path, body=body, headers=headers)
            response = connection.getresponse()
            return response.status, response.read()
        finally:
            connection.close()

    def close(self):
        self.server.shutdown()
        self.thread.join()
        self.server.server_close()


def chrome():
    for candidate in CHROME_PATHS:
        if os.path.isfile(candidate) or shutil.which(candidate):
            return candidate
    return None


def render(runs_dir, path="/", project_root=None):
    """Load a page in headless Chrome, run its JavaScript, and return the resulting DOM."""
    server = review_server(runs_dir)
    if project_root is not None:
        server.project_root = project_root
    client = Client(server)
    try:
        with tempfile.TemporaryDirectory() as profile:
            output = Path(profile) / "dom.html"
            with output.open("w") as stdout:
                process = subprocess.Popen(
                    [chrome(), *CHROME_FLAGS, f"--user-data-dir={profile}/data", "--dump-dom",
                     f"http://127.0.0.1:{server.server_port}{path}"],
                    stdout=stdout, stderr=subprocess.DEVNULL, start_new_session=True)
            # Chrome on macOS can print the DOM and then linger, so stop once the document is complete.
            deadline = time.monotonic() + 60
            try:
                while "</html>" not in output.read_text() and process.poll() is None and time.monotonic() < deadline:
                    time.sleep(0.2)
            finally:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except (ProcessLookupError, PermissionError):  # macOS: EPERM once the group has exited
                    pass
                process.wait()
            dom = output.read_text()
        if "</html>" not in dom:
            raise AssertionError("Chrome did not render the page in time")
        return dom
    finally:
        client.close()
