"""Review API, region geometry, Aim sync, and page rendering checks without model inference."""

from contextlib import redirect_stdout
from datetime import datetime
import io
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import tempfile
import threading
import time
import types
import unittest
from http.client import HTTPConnection
from unittest.mock import MagicMock, patch

from PIL import Image

from inpaint_lab.review_api import metric_regions, run_payload
from inpaint_lab.runs import pending_runs
from inpaint_lab.server import review_server
from inpaint_lab.tracking import sync_review, sync_saved_reviews


CHROME_PATHS = ("/Applications/Google Chrome.app/Contents/MacOS/Google Chrome", "google-chrome", "chromium", "chrome")
CHROME_FLAGS = ("--headless=new", "--disable-gpu", "--no-first-run", "--no-default-browser-check",
                "--disable-background-networking", "--disable-component-update", "--disable-sync",
                "--virtual-time-budget=5000", "--dump-dom")
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


def make_run(root, name, metrics=None):
    run_dir = root / name
    run_dir.mkdir()
    (run_dir / "params.json").write_text(json.dumps({"output": f"runs/{name}/output.png", "seed": 42}))
    Image.new("RGB", (8, 8)).save(run_dir / "output.png")
    Image.new("RGB", (8, 16)).save(run_dir / "output_comparison.png")
    if metrics is not None:
        (run_dir / "metrics.json").write_text(json.dumps(metrics))
    return run_dir


class Client:
    """Minimal HTTP client against a background review server."""
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


class ReviewTests(unittest.TestCase):
    def test_metric_regions_match_recorded_measurement_geometry(self):
        metrics = {"canvas_size": [100, 60], "quality_gap_bbox": [40, 0, 60, 60],
                   "quality_editable_bbox": [32, 0, 68, 60], "crop_bbox": [10, 4, 90, 56],
                   "quality_band_width": 8, "quality_maniqa_crop_bbox": [20, 0, 80, 60]}
        self.assertEqual(metric_regions("quality_maniqa", metrics), [[20, 0, 80, 60]])
        self.assertEqual(metric_regions("quality_preservation_mae", metrics), [[0, 0, 32, 60], [68, 0, 100, 60]])
        self.assertEqual(metric_regions("quality_gap_left_gradient", metrics), [[39, 0, 41, 60]])
        self.assertEqual(metric_regions("quality_gap_left_delta_e76", metrics), [[32, 0, 40, 60], [40, 0, 48, 60]])
        self.assertEqual(metric_regions("quality_patch_top_gradient_excess", metrics),
                         [[10, 3, 90, 5], [10, 0, 90, 4], [10, 4, 90, 9]])
        self.assertEqual(metric_regions("quality_editable_top_gradient", metrics), [])
        self.assertEqual(metric_regions("quality_maniqa", {}), [])
        metrics["quality_maniqa_crop_bbox"] = [-1, 0, 80, 60]
        self.assertEqual(metric_regions("quality_maniqa", metrics), [])

    def test_run_payload_regions_tiles_and_invalid_metrics(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_dir = make_run(root, "scored", SAMPLE_METRICS)
            payload = run_payload(run_dir, {"seed": 42})
            self.assertIsNone(payload["metrics_error"])
            regions = payload["regions"]
            self.assertEqual(len(regions["quality_maniqa"]), 2)
            self.assertEqual(regions["quality_maniqa_tile_1"], [[128, 224, 352, 448]])
            self.assertEqual(regions["quality_maniqa_min"], [[128, 0, 352, 224]])
            self.assertIn("quality_gap_left_delta_e76", regions)
            self.assertNotIn("quality_maniqa_tile_2", regions)
            (run_dir / "metrics.json").write_text('{"quality_maniqa": NaN, "quality_maniqa_tiles": "broken"}')
            payload = run_payload(run_dir, {"seed": float("nan")})
            self.assertIsNone(payload["metrics"]["quality_maniqa"])
            self.assertIsNone(payload["params"]["seed"])
            json.dumps(payload, allow_nan=False)
            (run_dir / "metrics.json").write_text("{broken")
            payload = run_payload(run_dir, {})
            self.assertIsNone(payload["metrics"])
            self.assertIn("missing or invalid", payload["metrics_error"])
            self.assertEqual(payload["regions"], {})

    def test_sync_updates_existing_run_and_skips_identical_review(self):
        directory = Path("runs/example")
        review = {"seam_quality": 4, "scene_coherence": 3, "preservation": 5, "notes": "Good", "reviewed_at": "2026-10-12T00:00:00+00:00"}
        candidate = MagicMock()
        candidate.hash = "existing-hash"
        candidate.close.side_effect = AttributeError("'RunTracker' object has no attribute 'sequence_infos'")
        candidate.get.side_effect = lambda key: str(directory.resolve()) if key == "run_dir" else None
        repo = MagicMock()
        repo.iter_runs.return_value = [candidate]
        run = MagicMock()
        run.get.side_effect = lambda key: review["reviewed_at"] if key == "review_synced_at" else review
        aim = types.ModuleType("aim")
        aim.Repo = MagicMock(return_value=repo)
        aim.Run = MagicMock(return_value=run)
        aim.Text = MagicMock(side_effect=lambda text: text)
        with patch.dict("sys.modules", {"aim": aim}), redirect_stdout(io.StringIO()):
            sync_review(directory, review)
        self.assertEqual(aim.Run.call_args.args, ("existing-hash",))
        run.__setitem__.assert_called_once_with("review_synced_at", review["reviewed_at"])
        run.track.assert_any_call(4, name="review.seam_quality", step=0)
        run.track.assert_any_call("Good", name="review.notes", step=0)
        run.__delitem__.assert_called_once_with("review")
        run.close.assert_called_once()
        candidate.close.assert_called_once()
        repo.close.assert_called_once()
        aim.Run.reset_mock()
        candidate.get.side_effect = lambda key: str(directory.resolve()) if key == "run_dir" else review["reviewed_at"]
        candidate.__getitem__.return_value = {"review_synced_at": review["reviewed_at"]}
        with patch.dict("sys.modules", {"aim": aim}), redirect_stdout(io.StringIO()):
            sync_review(directory, review)
        aim.Run.assert_not_called()

    def test_missing_or_ambiguous_aim_run_never_creates_run(self):
        for count in (0, 2):
            repo = MagicMock()
            candidates = [MagicMock() for _ in range(count)]
            for candidate in candidates:
                candidate.get.return_value = str(Path("runs/example").resolve())
            repo.iter_runs.return_value = candidates
            aim = types.ModuleType("aim")
            aim.Repo = MagicMock(return_value=repo)
            aim.Run = MagicMock()
            aim.Text = MagicMock()
            with patch.dict("sys.modules", {"aim": aim}), redirect_stdout(io.StringIO()) as output:
                sync_review(Path("runs/example"), {"seam_quality": 4, "scene_coherence": 3, "preservation": 5})
            aim.Run.assert_not_called()
            self.assertIn("Local review is preserved", output.getvalue())

    def test_only_completed_unreviewed_runs_are_selected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pending = make_run(root, "pending")
            reviewed = make_run(root, "reviewed")
            (reviewed / "review.json").write_text("{}")
            incomplete = make_run(root, "incomplete")
            (incomplete / "output_comparison.png").unlink()
            self.assertEqual([entry[0] for entry in pending_runs(root)], [pending])

    def test_pages_static_files_and_traversal(self):
        with tempfile.TemporaryDirectory() as directory:
            client = Client(review_server(Path(directory)))
            try:
                status, page = client.request("GET", "/")
                self.assertEqual(status, 200)
                self.assertIn(client.server.token.encode(), page)
                self.assertIn(b'src="/static/review.js"', page)
                for name in ("review.js", "batch.js", "common.js", "app.css"):
                    self.assertEqual(client.request("GET", f"/static/{name}")[0], 200, name)
                for path in ("/static/review.html", "/static/../server.py", "/static/..%2Fserver.py",
                             "/static/missing.js", "/review", "/review.js", "/batch.js"):
                    self.assertEqual(client.request("GET", path)[0], 404, path)
            finally:
                client.close()

    def test_review_api_validation_save_and_sync(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_dir = make_run(root, "example", SAMPLE_METRICS)
            make_run(root, "other")
            client = Client(review_server(root))
            try:
                with patch("inpaint_lab.tracking.sync_review") as sync:
                    status, body = client.request("GET", "/api/review/runs")
                    self.assertEqual(status, 200)
                    data = json.loads(body)
                    self.assertEqual(data["runs"], ["example", "other"])
                    self.assertIn("seam_quality", data["ratings"])
                    status, body = client.request("GET", "/api/review/run?run=example")
                    self.assertEqual(status, 200)
                    self.assertEqual(json.loads(body)["regions"]["quality_maniqa_tile_0"], [[128, 0, 352, 224]])
                    self.assertEqual(client.request("GET", "/api/review/run?run=../secret")[0], 404)
                    status, output_bytes = client.request("GET", "/image?run=example&view=output")
                    self.assertEqual(status, 200)
                    with Image.open(io.BytesIO(output_bytes)) as output_image:
                        self.assertEqual(output_image.size, (8, 8))
                    status, input_bytes = client.request("GET", "/image?run=example&view=input")
                    with Image.open(io.BytesIO(input_bytes)) as input_image:
                        self.assertEqual(input_image.size, (8, 8))
                    self.assertEqual(client.request("GET", "/image?run=example&view=unknown")[0], 400)
                    self.assertEqual(client.request("GET", "/image?run=../secret")[0], 404)

                    review = {"run": "example", "seam_quality": 4, "scene_coherence": 3, "preservation": 5, "notes": " Good seam "}
                    token = client.server.token
                    self.assertEqual(client.request("POST", "/api/review/save", review)[0], 403)
                    for invalid in ({"seam_quality": 6}, {"seam_quality": "4"}, {"seam_quality": 2.5},
                                    {"seam_quality": True}, {"notes": 5}, {"notes": "x" * 10001}):
                        status, _ = client.request("POST", "/api/review/save", dict(review, **invalid), token)
                        self.assertEqual(status, 400, invalid)
                    self.assertEqual(client.request("POST", "/api/review/save", raw="[1]", token=token)[0], 400)
                    self.assertEqual(client.request("POST", "/api/review/save", raw="{broken", token=token)[0], 400)
                    self.assertFalse((run_dir / "review.json").exists())
                    sync.assert_not_called()

                    status, body = client.request("POST", "/api/review/save", review, token)
                    self.assertEqual(status, 200)
                    self.assertIn("Review saved for example", json.loads(body)["message"])
                    saved = json.loads((run_dir / "review.json").read_text())
                    self.assertEqual(saved["seam_quality"], 4)
                    self.assertEqual(saved["notes"], "Good seam")
                    self.assertEqual(datetime.fromisoformat(saved["reviewed_at"]).utcoffset(),
                                     datetime.now().astimezone().utcoffset())
                    sync.assert_called_once_with(run_dir, saved)
                    self.assertEqual(client.request("POST", "/api/review/save", review, token)[0], 409)
                    self.assertEqual(json.loads(client.request("GET", "/api/review/runs")[1])["runs"], ["other"])
            finally:
                client.close()
            with patch("inpaint_lab.tracking.sync_review") as retry:
                sync_saved_reviews(root)
            retry.assert_called_once_with(run_dir, saved)


def chrome():
    for candidate in CHROME_PATHS:
        if os.path.isfile(candidate) or shutil.which(candidate):
            return candidate
    return None


@unittest.skipUnless(chrome(), "Chrome/Chromium not installed")
class RenderedPageTests(unittest.TestCase):
    """Executes the real JavaScript in headless Chrome and inspects the rendered DOM."""
    def render(self, root, path="/"):
        client = Client(review_server(root))
        try:
            with tempfile.TemporaryDirectory() as profile:
                output = Path(profile) / "dom.html"
                with output.open("w") as stdout:
                    process = subprocess.Popen(
                        [chrome(), *CHROME_FLAGS[:-1], f"--user-data-dir={profile}/data", CHROME_FLAGS[-1],
                         f"http://127.0.0.1:{client.server.server_port}{path}"],
                        stdout=stdout, stderr=subprocess.DEVNULL, start_new_session=True)
                # Chrome on macOS can print the DOM and then linger, so stop once the document is complete.
                deadline = time.monotonic() + 60
                try:
                    while "</html>" not in output.read_text() and process.poll() is None and time.monotonic() < deadline:
                        time.sleep(0.2)
                finally:
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    process.wait()
                dom = output.read_text()
            self.assertIn("</html>", dom, "Chrome did not render the page in time")
            return dom
        finally:
            client.close()

    def test_review_page_renders_scores_rulers_tiles_and_form(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            metrics = dict(SAMPLE_METRICS, quality_maniqa_error="<script>bad</script>")
            make_run(root, "example", metrics)
            dom = self.render(root)
        for text in ("1. Gap quality", "2. Seams", "3. Patch", "Visual quality · MANIQA: 0.600",
                     "Tile 1: 0.400", "Tile 2: 0.800", "Lowest tile: 0.400", "L color 2.5", "R extra edge 0.03",
                     "Original pixels changed: 12.50%", "Left edge of pasted crop", "Model mask", "Outer feather",
                     "Pasted crop", "1: 0.400", "2: 0.800", "Save review and sync to Aim", "1 unreviewed runs",
                     "left:40%;width:16%", "&lt;script&gt;bad&lt;/script&gt;"):
            self.assertIn(text, dom)
        self.assertLess(dom.index("1. Gap quality"), dom.index("2. Seams"))
        self.assertNotIn("<script>bad</script>", dom)
        self.assertEqual(dom.count('type="radio"'), 15)

    def test_missing_metrics_and_all_reviewed_states(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            make_run(root, "legacy")
            dom = self.render(root, "/?run=missing")
            self.assertIn("Run is missing or already reviewed", dom)
            self.assertIn("Visual quality · MANIQA: Not available", dom)
            self.assertIn("missing or invalid", dom)
            self.assertIn("No recorded spatial metrics.", dom)
            (root / "legacy" / "review.json").write_text("{}")
            dom = self.render(root)
            self.assertRegex(dom, r'<h1 id="all-done">All available runs reviewed</h1>')

    def test_batch_page_renders_with_shared_modules(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "samples").mkdir()
            Image.new("RGB", (256, 256)).save(root / "samples" / "left.png")
            with patch("inpaint_lab.server.ROOT", root):
                dom = self.render(root / "samples", "/batches")
        self.assertIn("Batch history", dom)
        self.assertIn("samples/left.png", dom)
        self.assertIn('aria-label="Gap (px) baseline"', dom)


if __name__ == "__main__":
    unittest.main()
