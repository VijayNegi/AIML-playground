"""Batch previews, recovery, timing, and HTTP controls without model inference."""

import argparse
from contextlib import redirect_stdout
from copy import deepcopy
from http.client import HTTPConnection
import io
import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch
from urllib.parse import urlencode

from PIL import Image

from inpaint_lab.batches import BatchManager, estimate_jobs, preview_batch
from inpaint_lab.server import review_server
from inpaint_lab.quality import QUALITY_VERSION


def inputs(root):
    (root / "samples").mkdir()
    for name in ("left.png", "right.png"):
        Image.new("RGB", (256, 256), "green").save(root / "samples" / name)
    return {"name": "Test batch", "images": ["samples/left.png", "samples/right.png"],
            "baseline": {"gap": 10}, "mode": "one_at_a_time", "sweeps": {}}


def fake_run(args, resources=None):
    resources.setdefault("model", object())
    folder = Path(args.output).parent
    folder.mkdir(parents=True)
    for name in ("output.png", "output_comparison.png"):
        Image.new("RGB", (8, 8)).save(folder / name)
    (folder / "params.json").write_text(json.dumps(vars(args)))
    metrics = {"aim_status": "logged"}
    (folder / "metrics.json").write_text(json.dumps(metrics))
    return folder, metrics


class BatchTests(unittest.TestCase):
    def test_one_parameter_expansion_and_grid(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            definition = inputs(root)
            definition["sweeps"] = {"gap": [10, 30, 60, 100], "feather": [0, 15, 30, 60],
                                    "steps": [20, 30, 50], "guidance_scale": [3, 5, 7.5, 10]}
            batch = preview_batch(root, deepcopy(definition))
            self.assertEqual(len(batch["jobs"]), 12)
            baseline = batch["jobs"][0]["params"]
            for job in batch["jobs"]:
                self.assertLessEqual(sum(job["params"][key] != baseline[key] for key in definition["sweeps"]), 1)
            definition["mode"] = "grid"
            self.assertEqual(len(preview_batch(root, definition)["jobs"]), 192)

    def test_reject_invalid_inputs_parameters_and_overlarge_grid(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            baseline = inputs(root)
            invalid = [dict(baseline, images=["../outside.png", "samples/right.png"]),
                       dict(baseline, baseline={"gap": -1}), dict(baseline, baseline={"steps": 2.5}),
                       dict(baseline, baseline={"guidance_scale": float("nan")}),
                       dict(baseline, baseline={"height": 2048}), dict(baseline, baseline={"output": "output.png"}),
                       dict(baseline, sweeps={"bad": [1]}), dict(baseline, sweeps={"steps": []}),
                       dict(baseline, mode="grid", sweeps={"seed": list(range(1000)), "steps": [20, 30]})]
            for definition in invalid:
                with self.subTest(definition=definition), self.assertRaises(ValueError):
                    preview_batch(root, definition)
            # This input pair pads a requested 224px gap to 232px, outside tiled scoring support.
            Image.new("RGB", (257, 256)).save(root / "samples/left.png")
            with self.assertRaises(ValueError):
                preview_batch(root, dict(baseline, baseline={"gap": 224}))

    def test_pause_resume_persist_and_prevent_concurrent_worker(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            definition = inputs(root)
            definition["sweeps"] = {"steps": [20, 30]}
            entered, release = threading.Event(), threading.Event()
            calls = []
            def runner(args, resources=None):
                calls.append(args)
                if len(calls) == 1:
                    entered.set()
                    if not release.wait(5):
                        raise RuntimeError("Test timed out")
                return fake_run(args, resources)
            manager = BatchManager(root, runner=runner)
            try:
                batch = manager.create(definition)
                manager.action(batch["id"], "resume")
                self.assertTrue(entered.wait(5))
                with self.assertRaises(ValueError):
                    manager.action(batch["id"], "resume")
                manager.action(batch["id"], "pause")
                self.assertEqual(manager.snapshot()["batches"][0]["status"], "pause_requested")
                release.set()
                manager.thread.join(5)
                paused = manager.snapshot()["batches"][0]
                self.assertEqual(paused["status"], "paused")
                self.assertEqual(len(calls), 1)
                manager.action(batch["id"], "resume")
                manager.thread.join(5)
                finished = manager.snapshot()["batches"][0]
                self.assertEqual(finished["status"], "completed")
                self.assertEqual(len(calls), 3)
                self.assertTrue(all(len(job["attempts"]) == 1 for job in finished["jobs"]))
                self.assertTrue((root / "batches.json").is_file())
                self.assertFalse((root / "batches.json.tmp").exists())
                # A second process/controller cannot own the same manifest.
                with self.assertRaises(OSError):
                    BatchManager(root)
            finally:
                release.set()
                manager.close()

    def test_failure_retry_and_input_changes_preserve_attempts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            definition = inputs(root)
            calls = []
            def runner(args, resources=None):
                calls.append(args.output)
                if len(calls) == 1:
                    folder = Path(args.output).parent
                    folder.mkdir(parents=True)
                    (folder / "partial.txt").write_text("preserve me")
                    raise RuntimeError("Synthetic inference failure")
                return fake_run(args, resources)
            manager = BatchManager(root, runner=runner)
            try:
                batch = manager.create(definition)
                manager.action(batch["id"], "resume")
                manager.thread.join(5)
                self.assertEqual(manager.snapshot()["batches"][0]["status"], "failed")
                manager.action(batch["id"], "retry")
                manager.thread.join(5)
                job = manager.snapshot()["batches"][0]["jobs"][0]
                self.assertEqual(job["status"], "completed")
                self.assertEqual(len(job["attempts"]), 2)
                self.assertNotEqual(calls[0], calls[1])
                self.assertEqual((Path(calls[0]).parent / "partial.txt").read_text(), "preserve me")
                (Path(calls[1]).parent / "metrics.json").unlink()
                with self.assertRaises(ValueError):
                    manager.action(batch["id"], "resume")
                self.assertEqual(manager.snapshot()["batches"][0]["jobs"][0]["status"], "interrupted")
                Image.new("RGB", (256, 256), "blue").save(root / "samples/left.png")
                with self.assertRaises(ValueError):
                    manager.action(batch["id"], "retry")
            finally:
                manager.close()

    def test_restart_marks_interrupted_and_corrupt_manifest_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            definition = inputs(root)
            manager = BatchManager(root)
            batch = manager.create(definition)
            saved = manager.data["batches"][0]
            saved["status"] = "running"
            saved["jobs"][0].update(status="running", attempts=[{"status": "running", "run_dir": "old"}])
            manager.save()
            manager.close()
            manager = BatchManager(root)
            try:
                self.assertEqual(manager.snapshot()["batches"][0]["status"], "interrupted")
                with self.assertRaises(ValueError):
                    manager.action(batch["id"], "resume")
            finally:
                manager.close()
            (root / "batches.json").write_text("{broken")
            with self.assertRaises(ValueError):
                BatchManager(root)
            self.assertEqual((root / "batches.json").read_text(), "{broken")

    def test_estimate_matches_hardware_geometry_and_protocol(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            batch = preview_batch(root, inputs(root))
            job = batch["jobs"][0]
            self.assertIsNone(estimate_jobs(root, [job], "this-machine")["seconds"])
            folder = root / "runs/history"
            folder.mkdir(parents=True)
            metrics = {"hardware_id": "this-machine", "crop_size": job["geometry"]["crop_size"],
                       "inference_time": 10, "model_load_time": 2, "wall_time": 18,
                       "quality_version": QUALITY_VERSION, "quality_maniqa_tile_count": 2,
                       "quality_assessment_time": 5, "quality_model_load_time": 1}
            (folder / "params.json").write_text(json.dumps(job["params"]))
            (folder / "metrics.json").write_text(json.dumps(metrics))
            estimate = estimate_jobs(root, [job], "this-machine")
            self.assertEqual(estimate["seconds"], 18)
            self.assertEqual(estimate["minimum_samples"], 1)
            self.assertIsNone(estimate_jobs(root, [job], "other-machine")["seconds"])
            metrics.pop("hardware_id")
            (folder / "metrics.json").write_text(json.dumps(metrics))
            self.assertIsNone(estimate_jobs(root, [job], "this-machine")["seconds"])
            self.assertEqual(estimate_jobs(root, [job], "this-machine", allow_legacy=True)["seconds"], 18)

    def test_http_preview_save_and_csrf_do_not_start_inference(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            definition = inputs(root)
            (root / "runs").mkdir()
            definition["sweeps"] = {"steps": [20]}
            with review_server(root / "runs") as server:
                server.project_root = root
                thread = threading.Thread(target=server.serve_forever)
                thread.start()
                connection = HTTPConnection("127.0.0.1", server.server_port, timeout=10)
                def request(path, data=None, token=None):
                    headers = {"Content-Type": "application/json", "X-CSRF-Token": token or "bad"}
                    connection.request("POST" if data is not None else "GET", path,
                                       body=json.dumps(data) if data is not None else None, headers=headers)
                    response = connection.getresponse()
                    return response.status, response.read()
                try:
                    self.assertEqual(request("/batches")[0], 200)
                    self.assertEqual(request("/static/batch.js")[0], 200)
                    self.assertEqual(request("/api/batches/catalog")[0], 200)
                    self.assertEqual(request("/api/batches/sample?path=../outside")[0], 400)
                    self.assertEqual(request("/api/batches/preview", definition)[0], 403)
                    with patch("inpaint_lab.runner.hardware_id", return_value="test"):
                        status, body = request("/api/batches/preview", definition, server.token)
                    self.assertEqual(status, 200)
                    preview = json.loads(body)
                    self.assertEqual(request("/api/batches/create", {"definition": definition}, server.token)[0], 400)
                    status, body = request("/api/batches/create", {"definition": definition, "expected": preview["expected"]}, server.token)
                    self.assertEqual(status, 200)
                    created = json.loads(body)
                    self.assertEqual(created["status"], "paused")
                    self.assertIsNone(server.batch_manager.thread)
                    self.assertEqual(request("/api/batches")[0], 200)
                    definition["baseline"]["gap"] = 20
                    self.assertEqual(request("/api/batches/create", {"definition": definition, "expected": preview["expected"]}, server.token)[0], 400)
                    server.batch_manager.runner = fake_run
                    self.assertEqual(request("/api/batches/action", {"id": created["id"], "action": "resume"}, server.token)[0], 200)
                    server.batch_manager.thread.join(5)
                    self.assertEqual(server.batch_manager.snapshot()["batches"][0]["status"], "completed")
                    job_id = created["jobs"][0]["id"]
                    query = urlencode({"batch": created["id"], "job": job_id, "attempt": 0})
                    self.assertEqual(request("/api/batches/output?" + query)[0], 200)
                finally:
                    connection.close()
                    server.shutdown()
                    thread.join()


if __name__ == "__main__":
    unittest.main()
