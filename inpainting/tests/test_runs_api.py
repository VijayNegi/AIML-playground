"""Run history API, single runs through the batch worker, rerun settings, and page rendering."""

from copy import deepcopy
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image

from inpaint_lab.batches import preview_batch
from inpaint_lab.runs import image_pair_id
from inpaint_lab.server import review_server

from helpers import SAMPLE_METRICS, Client, chrome, make_run, render


def project(directory):
    """Temporary project root with samples/ and runs/."""
    root = Path(directory)
    (root / "samples").mkdir()
    (root / "runs").mkdir()
    Image.new("RGB", (256, 256), "green").save(root / "samples" / "left.png")
    Image.new("RGB", (256, 256), "blue").save(root / "samples" / "right.png")
    return root


def server(root):
    instance = review_server(root / "runs")
    instance.project_root = root
    return Client(instance)


def fake_run(args, resources=None):
    folder = Path(args.output).parent
    folder.mkdir(parents=True)
    for name in ("output.png", "output_comparison.png", "debug_crop_mask.png"):
        Image.new("RGB", (8, 8)).save(folder / name)
    (folder / "params.json").write_text(json.dumps(vars(args)))
    metrics = {"quality_maniqa": 0.5, "inference_time": 3.0}
    (folder / "metrics.json").write_text(json.dumps(metrics))
    return folder, metrics


class RunsApiTests(unittest.TestCase):
    def test_list_orders_newest_first_and_reports_status(self):
        with tempfile.TemporaryDirectory() as directory:
            root = project(directory)
            runs = root / "runs"
            reviewed = make_run(runs, "reviewed", {"quality_maniqa": 0.7})
            (reviewed / "review.json").write_text("{}")
            pending = make_run(runs, "pending")
            incomplete = make_run(runs, "incomplete")
            (incomplete / "output.png").unlink()
            failed = make_run(runs, "failed")
            (runs / "legacy-folder-without-params").mkdir()
            for age, folder in enumerate((failed, incomplete, pending, reviewed)):
                os.utime(folder / "params.json", (1_700_000_000 + age, 1_700_000_000 + age))
            (root / "batches.json").write_text(json.dumps({"version": 1, "batches": [{
                "id": "b1", "name": "Sweep", "kind": "batch", "status": "failed", "jobs": [{
                    "status": "failed", "attempts": [{"run_dir": str(failed), "status": "failed", "error": "CUDA out of memory"}]}]}]}))
            client = server(root)
            try:
                status, body = client.request("GET", "/api/runs")
            finally:
                client.close()
        self.assertEqual(status, 200)
        listed = json.loads(body)["runs"]
        self.assertEqual([run["name"] for run in listed], ["reviewed", "pending", "incomplete", "failed"])
        self.assertEqual([run["status"] for run in listed], ["reviewed", "pending review", "incomplete", "failed"])
        self.assertEqual(listed[0]["maniqa"], 0.7)
        self.assertEqual(listed[3]["batch"]["error"], "CUDA out of memory")
        self.assertEqual(listed[3]["batch"]["batch_name"], "Sweep")

    def test_detail_images_metrics_and_file_serving(self):
        with tempfile.TemporaryDirectory() as directory:
            root = project(directory)
            folder = make_run(root / "runs", "example", SAMPLE_METRICS)
            for name in ("debug_crop_result.png", "collage_mask.png", "zz_extra.png"):
                Image.new("RGB", (4, 4)).save(folder / name)
            (folder / "notes.txt").write_text("not an image")
            (folder / "intermediate_steps").mkdir()
            Image.new("RGB", (4, 4)).save(folder / "intermediate_steps" / "step_0000.png")
            (root / "runs" / "secret.png").write_bytes(b"outside the run folder")
            client = server(root)
            try:
                status, body = client.request("GET", "/api/runs/detail?run=example")
                self.assertEqual(status, 200)
                detail = json.loads(body)
                self.assertEqual(detail["images"], ["output.png", "output_comparison.png", "collage_mask.png",
                                                    "debug_crop_result.png", "zz_extra.png"])
                self.assertEqual(detail["step_images"], 1)
                self.assertEqual(detail["metrics"]["quality_maniqa"], 0.6)
                self.assertIsNone(detail["review"])
                status, body = client.request("GET", "/api/runs/file?run=example&name=debug_crop_result.png")
                self.assertEqual(status, 200)
                self.assertTrue(body.startswith(b"\x89PNG"))
                for path in ("/api/runs/file?run=example&name=notes.txt", "/api/runs/file?run=example&name=../secret.png",
                             "/api/runs/file?run=..&name=secret.png", "/api/runs/file?run=example%2F..&name=secret.png",
                             "/api/runs/file?run=example&name=intermediate_steps/step_0000.png",
                             "/api/runs/file?run=missing&name=output.png"):
                    self.assertEqual(client.request("GET", path)[0], 404, path)
                for name in ("..", "missing", "example/..", ""):
                    self.assertEqual(client.request("GET", f"/api/runs/detail?run={name}")[0], 404, name)
            finally:
                client.close()

    def test_rerun_settings_map_inputs_and_warn(self):
        with tempfile.TemporaryDirectory() as directory:
            root = project(directory)
            with Image.open(root / "samples/left.png") as left, Image.open(root / "samples/right.png") as right:
                pair = image_pair_id([left, right])
            base = {"output": "runs/x/output.png", "gap": 30, "feather": 15, "steps": 20, "guidance_scale": 5.0,
                    "seed": 7, "model": "sd-xl", "prompt": "p", "negative_prompt": "n", "height": None}
            cases = {
                "exact": (dict(base, images=[str(root / "samples/left.png"), "samples/right.png"], input_pair_id=pair), []),
                "legacy": (dict(base, images=["./left.png", "./right.png"]), ["matched by filename"]),
                "changed": (dict(base, images=["samples/right.png", "samples/left.png"], input_pair_id=pair), ["differ"]),
                "missing": (dict(base, images=["samples/gone.png", "samples/right.png"], input_pair_id=pair), ["gone.png"]),
            }
            for name, (params, _) in cases.items():
                make_run(root / "runs", name, params=params)
            client = server(root)
            try:
                details = {name: json.loads(client.request("GET", f"/api/runs/detail?run={name}")[1]) for name in cases}
            finally:
                client.close()
        exact = details["exact"]
        self.assertEqual(exact["rerun"]["images"], ["samples/left.png", "samples/right.png"])
        self.assertEqual(exact["rerun_warnings"], [])
        self.assertEqual({key: exact["rerun"][key] for key in ("gap", "feather", "steps", "guidance_scale", "seed")},
                         {"gap": 30, "feather": 15, "steps": 20, "guidance_scale": 5.0, "seed": 7})
        self.assertNotIn("output", exact["rerun"])
        self.assertEqual(details["legacy"]["rerun"]["images"], ["samples/left.png", "samples/right.png"])
        self.assertEqual(details["missing"]["rerun"]["images"], [None, "samples/right.png"])
        for name, (_, fragments) in cases.items():
            for fragment in fragments:
                self.assertTrue(any(fragment in warning for warning in details[name]["rerun_warnings"]), (name, fragment))

    def test_single_run_through_batch_worker_appears_in_history(self):
        with tempfile.TemporaryDirectory() as directory:
            root = project(directory)
            definition = {"name": "Single run", "kind": "single", "mode": "one_at_a_time", "sweeps": {},
                          "images": ["samples/left.png", "samples/right.png"],
                          "baseline": {"gap": 16, "save_intermediate": True}}
            client = server(root)
            try:
                token = client.server.token
                with patch("inpaint_lab.runner.hardware_id", return_value="test"):
                    status, body = client.request("POST", "/api/batches/preview", definition, token)
                self.assertEqual(status, 200)
                checked = json.loads(body)
                self.assertEqual(len(checked["jobs"]), 1)
                self.assertTrue(checked["jobs"][0]["params"]["save_intermediate"])
                status, body = client.request("POST", "/api/batches/create",
                                              {"definition": definition, "expected": checked["expected"]}, token)
                self.assertEqual(status, 200)
                batch = json.loads(body)
                self.assertEqual(batch["kind"], "single")
                client.server.batch_manager.runner = fake_run
                self.assertEqual(client.request("POST", "/api/batches/action", {"id": batch["id"], "action": "resume"}, token)[0], 200)
                client.server.batch_manager.thread.join(5)
                listed = json.loads(client.request("GET", "/api/runs")[1])["runs"]
            finally:
                client.close()
        self.assertEqual(len(listed), 1)
        self.assertEqual(listed[0]["status"], "pending review")
        self.assertEqual(listed[0]["batch"]["batch_kind"], "single")
        self.assertEqual(listed[0]["params"]["gap"], 16)

    def test_single_kind_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = project(directory)
            base = {"images": ["samples/left.png", "samples/right.png"], "baseline": {"gap": 16}}
            for invalid in ({"kind": "single", "sweeps": {"gap": [8, 16]}}, {"kind": "other"},
                            {"baseline": {"gap": 16, "save_intermediate": "yes"}}):
                with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                    preview_batch(root, dict(deepcopy(base), **invalid))
            self.assertEqual(preview_batch(root, dict(base, kind="single"))["kind"], "single")
            self.assertEqual(preview_batch(root, deepcopy(base))["kind"], "batch")


@unittest.skipUnless(chrome(), "Chrome/Chromium not installed")
class RenderedRunsPageTests(unittest.TestCase):
    def test_runs_page_lists_and_shows_detail(self):
        with tempfile.TemporaryDirectory() as directory:
            root = project(directory)
            params = {"output": "runs/example/output.png", "gap": 30, "feather": 15, "steps": 20,
                      "guidance_scale": 5.0, "seed": 7, "prompt": "<b>bold</b>"}
            folder = make_run(root / "runs", "example", dict(SAMPLE_METRICS, inference_time=12.34), params=params)
            Image.new("RGB", (8, 8)).save(folder / "debug_crop_mask.png")
            dom = render(root / "runs", "/?run=example", project_root=root)
        for text in ("1 of 1 runs", "pending review", "gap 30 · feather 15 · steps 20 · guidance 5 · seed 7",
                     "MANIQA 0.600", "Rerun with these settings", 'href="/run?from=example"', 'href="/review?run=example"',
                     "Crop mask", "Before / after", "12.3 s", "12.50%", "Guidance scale", "quality_maniqa_tiles",
                     "&lt;b&gt;bold&lt;/b&gt;", 'href="/" aria-current="page"'):
            self.assertIn(text, dom)
        self.assertNotIn("<b>bold</b>", dom)

    def test_run_page_loads_settings_from_previous_run(self):
        with tempfile.TemporaryDirectory() as directory:
            root = project(directory)
            make_run(root / "runs", "previous", params={"output": "x", "gap": 30, "images": ["./left.png", "./right.png"]})
            dom = render(root / "runs", "/run?from=previous", project_root=root)
        for text in ("Settings loaded from", 'href="/?run=previous"', "matched by filename",
                     'value="samples/left.png"', "Choose an image…", 'name="save_intermediate"',
                     'href="/run" aria-current="page"'):
            self.assertIn(text, dom)


if __name__ == "__main__":
    unittest.main()
