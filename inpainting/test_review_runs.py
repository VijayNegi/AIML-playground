"""Visual review checks without opening an image viewer."""

from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import types
import threading
import unittest
from urllib.parse import urlencode
from http.client import HTTPConnection
from unittest.mock import MagicMock, patch

from PIL import Image

from review_runs import automatic_scores_html, metric_regions, pending_runs, review_server, surrounding_metrics_html, sync_saved_reviews, sync_review


class ReviewTests(unittest.TestCase):
    def test_spatial_ranges_and_tile_values(self):
        metrics = {"canvas_size": [500, 448], "quality_gap_bbox": [200, 0, 280, 448],
                   "quality_effective_mask_bbox": [190, 0, 290, 448],
                   "quality_editable_bbox": [180, 0, 300, 448], "crop_bbox": [100, 0, 400, 448],
                   "quality_band_width": 8, "quality_gap_left_delta_e76": 2.5,
                   "quality_gap_right_gradient_excess": 0.03,
                   "quality_maniqa_tiles": [{"bbox": [128, 0, 352, 224], "score": 0.7},
                                             {"bbox": [128, 224, 352, 448], "score": 0.8}]}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.assertEqual(surrounding_metrics_html(root)[:2], ("", ""))
            (root / "metrics.json").write_text(json.dumps(metrics))
            top, bottom, side = surrounding_metrics_html(root)
        self.assertIn("left:40.000000%;width:16.000000%", top)
        self.assertIn("L color 2.5", top)
        self.assertIn("R extra edge 0.03", top)
        self.assertIn("Model mask", top)
        self.assertIn("Outer feather", bottom)
        self.assertIn("Pasted crop", bottom)
        self.assertIn("1: 0.700", side)
        self.assertIn("2: 0.800", side)
        self.assertIn('data-metric="quality_maniqa_tile_1"', side)
        self.assertIn('y1="112.0"', side)

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

    def test_tiled_maniqa_scores_and_highlights(self):
        metrics = {"canvas_size": [500, 448], "quality_maniqa": 0.6, "quality_maniqa_min": 0.4,
                   "quality_maniqa_min_bbox": [100, 0, 324, 224],
                   "quality_maniqa_tiles": [{"bbox": [100, 0, 324, 224], "score": 0.4},
                                             {"bbox": [100, 224, 324, 448], "score": 0.8}]}
        self.assertEqual(len(metric_regions("quality_maniqa", metrics)), 2)
        self.assertEqual(metric_regions("quality_maniqa_tile_1", metrics), [[100, 224, 324, 448]])
        self.assertEqual(metric_regions("quality_maniqa_min", metrics), [[100, 0, 324, 224]])
        self.assertEqual(metric_regions("quality_maniqa_tile_99", metrics), [])
        with tempfile.TemporaryDirectory() as directory:
            (Path(directory) / "metrics.json").write_text(json.dumps(metrics))
            page = automatic_scores_html(Path(directory))
        self.assertIn("Tile 1: 0.400", page)
        self.assertIn("Tile 2: 0.800", page)
        self.assertIn("Summary is the median", page)

    def test_maniqa_always_visible_with_failure_explanation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            page = automatic_scores_html(root)
            self.assertIn("Visual quality · MANIQA", page)
            self.assertIn("Not available", page)
            (root / "metrics.json").write_text(json.dumps({
                "quality_maniqa_error": "No module named 'pyiqa'",
                "quality_preservation_changed_fraction": 0.125,
                "quality_patch_left_gradient_excess": 0.1}))
            page = automatic_scores_html(root)
            self.assertIn("12.50%", page)
            self.assertIn("Scoring failed", page)
            self.assertIn("install requirements.txt", page)
            self.assertIn("Left edge of pasted crop: Extra edge strength", page)
            self.assertLess(page.index("1. Gap quality"), page.index("2. Seams"))
            self.assertLess(page.index("2. Seams"), page.index("3. Patch"))
            self.assertNotIn("<details>", page)

    def test_automatic_scores_scales_missing_values_and_escaped_errors(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.assertIn("missing or invalid", automatic_scores_html(root))
            (root / "metrics.json").write_text(json.dumps({
                "quality_maniqa": 0.7, "quality_preservation_mae": 2.5,
                "quality_preservation_changed_fraction": None,
                "quality_gap_left_gradient_excess": 0.1,
                "quality_gap_left_delta_e76": 3, "quality_maniqa_error": "<script>bad</script>",
                "inference_time": 15}))
            page = automatic_scores_html(root)
            self.assertIn("quality_maniqa", page)
            self.assertIn("0.7", page)
            self.assertIn("0–255", page)
            self.assertIn("Unavailable", page)
            self.assertIn("&lt;script&gt;", page)
            self.assertNotIn("<script>", page)
            self.assertNotIn("inference_time", page)

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

    def make_run(self, root, name):
        run_dir = root / name
        run_dir.mkdir()
        (run_dir / "params.json").write_text(json.dumps({"output": f"runs/{name}/output.png", "seed": 42}))
        for filename in ("output.png", "output_comparison.png"):
            Image.new("RGB", (8, 16 if filename == "output_comparison.png" else 8)).save(run_dir / filename)
        return run_dir

    def test_only_completed_unreviewed_runs_are_selected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pending = self.make_run(root, "pending")
            reviewed = self.make_run(root, "reviewed")
            (reviewed / "review.json").write_text("{}")
            incomplete = self.make_run(root, "incomplete")
            (incomplete / "output_comparison.png").unlink()
            self.assertEqual([entry[0] for entry in pending_runs(root)], [pending])

    def test_browser_review_validation_save_and_sync(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_dir = self.make_run(root, "example")
            with review_server(root) as server, patch("review_runs.sync_review") as sync:
                thread = threading.Thread(target=server.serve_forever)
                thread.start()
                connection = HTTPConnection("127.0.0.1", server.server_port, timeout=5)
                def request(method, path, fields=None):
                    connection.request(method, path, body=urlencode(fields) if fields else None,
                                       headers={"Content-Type": "application/x-www-form-urlencoded"})
                    response = connection.getresponse()
                    return response.status, response.read()
                try:
                    status, page = request("GET", "/")
                    self.assertEqual(status, 200)
                    self.assertIn(b'Save review and sync to Aim', page)
                    self.assertIn(b'id="metric-overlay"', page)
                    self.assertIn(b'id="flip-image"', page)
                    self.assertIn(b'class="metric-workspace"', page)
                    self.assertIn(b'MANIQA tiles', page)
                    self.assertIn(b'alt="Inpainted output"', page)
                    self.assertEqual(request("GET", "/review.js")[0], 200)
                    self.assertEqual(request("GET", "/image?run=example")[0], 200)
                    status, output_bytes = request("GET", "/image?run=example&view=output")
                    self.assertEqual(status, 200)
                    with Image.open(io.BytesIO(output_bytes)) as output_image:
                        self.assertEqual(output_image.size, (8, 8))
                    status, input_bytes = request("GET", "/image?run=example&view=input")
                    self.assertEqual(status, 200)
                    with Image.open(io.BytesIO(input_bytes)) as input_image:
                        self.assertEqual(input_image.size, (8, 8))
                    self.assertEqual(request("GET", "/image?run=example&view=unknown")[0], 400)
                    self.assertEqual(request("GET", "/image?run=../secret")[0], 404)
                    fields = {"run": "example", "token": "wrong", "seam_quality": "4",
                              "scene_coherence": "3", "preservation": "5", "notes": "Good seam"}
                    self.assertEqual(request("POST", "/review", fields)[0], 403)
                    fields["token"] = server.token
                    fields["seam_quality"] = "6"
                    self.assertEqual(request("POST", "/review", fields)[0], 400)
                    self.assertFalse((run_dir / "review.json").exists())
                    fields["seam_quality"] = "4"
                    self.assertEqual(request("POST", "/review", fields)[0], 303)
                    review = json.loads((run_dir / "review.json").read_text())
                    self.assertEqual(review["seam_quality"], 4)
                    self.assertEqual(review["notes"], "Good seam")
                    sync.assert_called_once_with(run_dir, review)
                    self.assertEqual(request("POST", "/review", fields)[0], 409)
                    self.assertIn(b'All available runs reviewed', request("GET", "/")[1])
                finally:
                    connection.close()
                    server.shutdown()
                    thread.join()
            with patch("review_runs.sync_review") as retry:
                sync_saved_reviews(root)
            retry.assert_called_once_with(run_dir, review)


if __name__ == "__main__":
    unittest.main()
