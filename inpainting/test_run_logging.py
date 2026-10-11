"""Run logging checks without model downloads or GPU inference."""

import argparse
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image

import collage
from inpainting_utils import inpaint_crop_and_patch


class RunLoggingTests(unittest.TestCase):
    def test_default_runs_are_unique(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(collage, "Path", side_effect=lambda path: Path(directory) / path):
                first = argparse.Namespace(output=None, steps_dir=None, seed=42, model="sd-xl")
                second = argparse.Namespace(output=None, steps_dir=None, seed=42, model="sd-xl")
                first_dir, _ = collage.prepare_run(first)
                second_dir, _ = collage.prepare_run(second)
            self.assertNotEqual(first_dir, second_dir)
            self.assertRegex(first_dir.name, r"^\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}\.\d{6}_UTC$")
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
                    redirect_stdout(io.StringIO()):
                collage.main()
            params = json.loads((output.parent / "params.json").read_text())
            metrics = json.loads((output.parent / "metrics.json").read_text())
            self.assertEqual(params["seed"], 123)
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
