"""Run logging checks without model downloads or GPU inference."""

import argparse
from datetime import datetime, timedelta, timezone
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import MagicMock, patch

from PIL import Image

import collage
from inpainting_utils import inpaint_crop_and_patch


class RunLoggingTests(unittest.TestCase):
    def test_two_runs_reuse_batch_pipeline(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.png"
            Image.new("RGB", (256, 256), "green").save(source)
            resources = {}
            perf = {"preprocessing_time": 0.1, "inference_time": 0.2, "total_time": 0.3}
            with patch.object(collage, "load_pipeline", return_value=(object(), 0.4)) as loader, \
                    patch.object(collage, "inpaint_crop_and_patch", side_effect=lambda **kwargs: (Image.new("RGB", (520, 256)), dict(perf))), \
                    patch.object(collage, "track_aim_run", return_value="test-hash"), \
                    patch.object(collage, "assess_quality", return_value={}) as assessment, \
                    redirect_stdout(io.StringIO()):
                for index in range(2):
                    args = collage.parse_collage_args(["--images", str(source), str(source), "--gap", "8",
                                                      "--output", str(root / f"run-{index}" / "output.png")])
                    collage.run_collage(args, resources=resources)
            loader.assert_called_once()
            self.assertIs(assessment.call_args.kwargs["model_cache"], resources)
            first = json.loads((root / "run-0/metrics.json").read_text())
            second = json.loads((root / "run-1/metrics.json").read_text())
            self.assertEqual(first["model_load_time"], 0.4)
            self.assertEqual(second["model_load_time"], 0)
            self.assertIn("hardware_id", second)
            self.assertGreater(second["wall_time"], 0)

    def test_run_folder_uses_local_time_and_offset(self):
        local = datetime(2026, 10, 19, 14, 30, 5, 123456, tzinfo=timezone(timedelta(hours=5, minutes=30)))
        with tempfile.TemporaryDirectory() as directory, patch.object(collage, "datetime") as clock:
            clock.now.return_value.astimezone.return_value = local
            with patch.object(collage, "Path", side_effect=lambda path: Path(directory) / path):
                run_dir, _ = collage.prepare_run(argparse.Namespace(output=None, steps_dir=None, seed=42, model="sd-xl"))
            self.assertEqual(run_dir.name, "2026-10-19_14-30-05.123456_+0530")

    def test_input_pair_id_is_stable_ordered_and_content_based(self):
        left = Image.new("RGB", (8, 8), "green")
        right = Image.new("RGB", (8, 8), "blue")
        pair_id = collage.image_pair_id([left, right])
        self.assertEqual(pair_id, collage.image_pair_id([left.copy(), right.copy()]))
        self.assertNotEqual(pair_id, collage.image_pair_id([right, left]))
        right.putpixel((0, 0), (255, 0, 0))
        self.assertNotEqual(pair_id, collage.image_pair_id([left, right]))

    def test_aim_tracks_params_metrics_images_and_closes(self):
        run = MagicMock()
        aim = types.ModuleType("aim")
        aim.Run = MagicMock(return_value=run)
        aim.Image = MagicMock(side_effect=lambda image: image)
        args = argparse.Namespace(seed=42)
        image = Image.new("RGB", (8, 8))
        with patch.dict("sys.modules", {"aim": aim}), redirect_stdout(io.StringIO()):
            collage.track_aim_run(args, Path("runs/example"),
                                  {"inference_time": 1.2, "crop_bbox": [0, 0, 8, 8]}, image, image)
        self.assertEqual(run.name, "example")
        run.__setitem__.assert_any_call("params", vars(args))
        run.__setitem__.assert_any_call("metadata", {"crop_bbox": [0, 0, 8, 8]})
        self.assertNotIn("metrics", [call.args[0] for call in run.__setitem__.call_args_list])
        run.track.assert_any_call(1.2, name="inference_time", step=0)
        run.track.assert_any_call(image, name="output", step=0)
        run.track.assert_any_call(image, name="comparison", step=0)
        self.assertEqual(run.track.call_count, 3)
        run.close.assert_called_once()

    def test_aim_failure_warns_and_closes(self):
        run = MagicMock()
        run.track.side_effect = RuntimeError("storage unavailable")
        aim = types.ModuleType("aim")
        aim.Run = MagicMock(return_value=run)
        aim.Image = MagicMock()
        stderr = io.StringIO()
        with patch.dict("sys.modules", {"aim": aim}), patch("sys.stderr", stderr):
            collage.track_aim_run(argparse.Namespace(), Path("runs/example"), {"inference_time": 1.2}, None, None)
        self.assertIn("Aim tracking failed", stderr.getvalue())
        run.close.assert_called_once()

    def test_missing_aim_warns_without_failing(self):
        stderr = io.StringIO()
        with patch.dict("sys.modules", {"aim": None}), patch("sys.stderr", stderr):
            collage.track_aim_run(argparse.Namespace(), Path("runs/example"), {}, None, None)
        self.assertIn("Local results are saved", stderr.getvalue())

    def test_default_runs_are_unique(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(collage, "Path", side_effect=lambda path: Path(directory) / path):
                first = argparse.Namespace(output=None, steps_dir=None, seed=42, model="sd-xl")
                second = argparse.Namespace(output=None, steps_dir=None, seed=42, model="sd-xl")
                first_dir, _ = collage.prepare_run(first)
                second_dir, _ = collage.prepare_run(second)
            self.assertNotEqual(first_dir, second_dir)
            self.assertRegex(first_dir.name, r"^\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}\.\d{6}_[+-]\d{4}$")
            params = json.loads((first_dir / "params.json").read_text())
            self.assertEqual(params, vars(first))
            self.assertEqual(first.steps_dir, str(first_dir / "intermediate_steps"))

    def test_existing_output_is_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "output.png"
            output.write_bytes(b"previous result")
            args = argparse.Namespace(output=str(output), steps_dir=None)
            with self.assertRaises(FileExistsError):
                collage.prepare_run(args)
            self.assertEqual(output.read_bytes(), b"previous result")

    def test_main_logs_settings_metrics_and_images(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_path = root / "input.png"
            Image.new("RGB", (32, 32), "green").save(input_path)
            output = root / "run" / "result.jpg"
            command = ["collage.py", "--images", str(input_path), str(input_path),
                       "--output", str(output), "--gap", "8", "--seed", "123",
                       "--save-intermediate"]
            perf = {"preprocessing_time": 0.1, "inference_time": 0.2, "total_time": 0.3}
            with patch("sys.argv", command), patch.object(collage, "load_pipeline", return_value=(object(), 0.4)), \
                    patch.object(collage, "inpaint_crop_and_patch", return_value=(Image.new("RGB", (72, 32)), perf)) as inference, \
                    patch.object(collage, "track_aim_run") as tracking, \
                    patch.object(collage, "assess_quality", return_value={"quality_maniqa": 0.5}) as assessment, \
                    redirect_stdout(io.StringIO()):
                collage.main()
            tracking.assert_called_once()
            assessment.assert_called_once()
            params = json.loads((output.parent / "params.json").read_text())
            metrics = json.loads((output.parent / "metrics.json").read_text())
            self.assertEqual(params["seed"], 123)
            self.assertEqual(len(params["input_pair_id"]), 64)
            for name, value in {"feather": 30, "steps": 50, "guidance_scale": 7.5,
                                "strength": 0.99, "max_crop_size": 1024,
                                "model": "sd-xl", "save_steps": False, "save_every": 1,
                                "height": None}.items():
                self.assertEqual(params[name], value, name)
            self.assertEqual(params["model_id"], collage.AVAILABLE_MODELS["sd-xl"])
            self.assertEqual(inference.call_args.kwargs["strength"], params["strength"])
            self.assertEqual(inference.call_args.kwargs["max_crop_size"], params["max_crop_size"])
            self.assertEqual(metrics["model_load_time"], 0.4)
            self.assertEqual(metrics["adjusted_gap"], 8)
            self.assertEqual(metrics["inference_time"], 0.2)
            self.assertEqual(metrics["quality_maniqa"], 0.5)
            self.assertEqual(inference.call_args.kwargs["debug_dir"], output.parent)
            for name in ("result.jpg", "result_comparison.png", "collage_canvas.png", "collage_mask.png"):
                self.assertTrue((output.parent / name).is_file(), name)
            with self.assertRaises(FileExistsError):
                collage.prepare_run(argparse.Namespace(**params))

    def test_crop_debug_images_use_run_directory(self):
        class FakePipeline:
            def __call__(self, **kwargs):
                return argparse.Namespace(images=[kwargs["image"]])

        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            image = Image.new("RGB", (1632, 1024), "green")
            mask = collage.create_blend_mask(image.size, 776, 80)
            result, perf = inpaint_crop_and_patch(FakePipeline(), image, mask, "landscape", debug_dir=directory)
            self.assertEqual(result.size, image.size)
            self.assertTrue(all(type(value) is int for value in perf["crop_bbox"]))
            self.assertEqual(json.loads(json.dumps(perf))["crop_bbox"], list(perf["crop_bbox"]))
            for name in ("debug_crop_image.png", "debug_crop_mask.png", "debug_crop_result.png"):
                self.assertTrue((Path(directory) / name).is_file(), name)


if __name__ == "__main__":
    unittest.main()
