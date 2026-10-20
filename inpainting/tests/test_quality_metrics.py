"""Synthetic checks for no-reference scoring without model downloads."""

from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
from PIL import Image
import torch

from inpaint_lab.compare import correlations, spearman
from inpaint_lab.quality import assess_quality, diagnostics, maniqa_score, maniqa_tiles
from inpaint_lab.canvas import create_blend_mask
from inpaint_lab.review_api import metric_regions


class QualityTests(unittest.TestCase):
    def test_batch_maniqa_model_is_loaded_once(self):
        metric = MagicMock(return_value=torch.tensor(0.7))
        metric.lower_better = False
        pyiqa = types.ModuleType("pyiqa")
        pyiqa.create_metric = MagicMock(return_value=metric)
        cache = {}
        with patch.dict("sys.modules", {"pyiqa": pyiqa}), patch("inpaint_lab.quality.version", return_value="test"):
            for _ in range(2):
                scores = maniqa_score(Image.new("RGB", (224, 224)), 80, 64, Path("unused"), model_cache=cache)
        pyiqa.create_metric.assert_called_once()
        self.assertEqual(scores["quality_model_load_time"], 0)
    def test_feather_edges_follow_actual_mask_and_model_threshold(self):
        canvas = Image.new("RGB", (100, 32), "gray")
        for feather in (0, 10, 50):
            mask = create_blend_mask(canvas.size, 40, 20, feather)
            scores = diagnostics(canvas, canvas, 40, 20, feather, (8, 0, 92, 32), mask, True)
            alpha = np.asarray(mask.convert("L"))
            support = np.where(alpha[0] > 0)[0]
            active = np.where(alpha[0] >= 128)[0]
            self.assertEqual(scores["quality_editable_bbox"], [int(support.min()), 0, int(support.max()) + 1, 32])
            self.assertEqual(scores["quality_effective_mask_bbox"], [int(active.min()), 0, int(active.max()) + 1, 32])
            scores["canvas_size"] = list(canvas.size)
            edge = int(active.min())
            self.assertEqual(metric_regions("quality_mask_left_gradient", scores), [[edge - 1, 0, edge + 1, 32]])
            self.assertEqual(scores["quality_mask_left_gradient"], 0)
        mask = create_blend_mask(canvas.size, 40, 20, 10)
        changed = canvas.copy()
        changed.putpixel((30, 0), (0, 0, 0))
        scores = diagnostics(canvas, changed, 40, 20, 10, (8, 0, 92, 32), mask, True)
        self.assertGreater(scores["quality_preservation_mae"], 0)

    def test_creativity_inside_editable_area_is_not_preservation_error(self):
        canvas = Image.new("RGB", (64, 32), "gray")
        result = canvas.copy()
        result.paste("green", (24, 0, 40, 32))
        metrics = diagnostics(canvas, result, 28, 8, 4, (8, 0, 56, 32))
        self.assertEqual(metrics["quality_preservation_mae"], 0)
        self.assertEqual(metrics["quality_preservation_changed_fraction"], 0)
        self.assertGreater(metrics["quality_editable_left_delta_e76"], 0)
        self.assertGreater(metrics["quality_editable_left_gradient_excess"], 0)
        result.putpixel((0, 0), (255, 255, 255))
        self.assertGreater(diagnostics(canvas, result, 28, 8, 4, None)["quality_preservation_mae"], 0)
        json.dumps(metrics, allow_nan=False)

    def test_patch_edges_and_no_unchanged_area(self):
        canvas = Image.new("RGB", (64, 32), "black")
        result = canvas.copy()
        result.paste("white", (8, 4, 56, 28))
        scores = diagnostics(canvas, result, 28, 8, 4, (8, 4, 56, 28))
        for name in ("left", "right", "top", "bottom"):
            self.assertEqual(scores[f"quality_patch_{name}_gradient"], 1)
        self.assertIsNone(diagnostics(canvas, result, 0, 64, 0, None)["quality_preservation_mae"])
        with self.assertRaises(ValueError):
            diagnostics(canvas, Image.new("RGB", (8, 8)), 28, 8, 4, None)

    def test_maniqa_crop_and_scalar(self):
        metric = MagicMock()
        metric.side_effect = [torch.tensor(value) for value in (0.9, 0.1, 0.7, 0.2, 0.8, 0.3, 0.6, 0.4, 0.5)]
        metric.lower_better = False
        pyiqa = types.ModuleType("pyiqa")
        pyiqa.create_metric = MagicMock(return_value=metric)
        with tempfile.TemporaryDirectory() as directory, patch.dict("sys.modules", {"pyiqa": pyiqa}), \
                patch("inpaint_lab.quality.version", return_value="test"):
            scores = maniqa_score(Image.new("RGB", (1632, 1024)), 776, 80, Path(directory))
            self.assertEqual(scores["quality_maniqa_tiles"][0]["bbox"], [704, 0, 928, 224])
            self.assertEqual(scores["quality_maniqa_tiles"][-1]["bbox"], [704, 800, 928, 1024])
            self.assertEqual(scores["quality_maniqa_tile_count"], 9)
            self.assertAlmostEqual(scores["quality_maniqa"], 0.5)
            self.assertAlmostEqual(scores["quality_maniqa_min"], 0.1)
            self.assertEqual(scores["quality_maniqa_min_bbox"], [704, 112, 928, 336])
            self.assertEqual(tuple(metric.call_args.args[0].shape), (1, 3, 224, 224))
            self.assertEqual(metric.call_count, 9)
            json.dumps(scores, allow_nan=False)
            pyiqa.create_metric.assert_called_once_with("maniqa-pipal", device="cpu", test_sample=1)

    def test_maniqa_tiles_cover_gap_without_resize_or_duplicates(self):
        for image_size, start, gap in (((500, 224), 100, 80), ((500, 225), 0, 224),
                                      ((500, 1000), 276, 224), ((224, 448), 0, 224)):
            boxes = maniqa_tiles(image_size, start, gap)
            self.assertEqual(boxes[0][1], 0)
            self.assertEqual(boxes[-1][3], image_size[1])
            self.assertEqual(len(boxes), len(set(map(tuple, boxes))))
            for x1, y1, x2, y2 in boxes:
                self.assertEqual((x2 - x1, y2 - y1), (224, 224))
                self.assertTrue(0 <= x1 <= start and start + gap <= x2 <= image_size[0])
            for previous, current in zip(boxes, boxes[1:]):
                self.assertLessEqual(current[1], previous[3])
        for size, start, gap in (((500, 300), 100, 225), ((500, 223), 100, 80), ((223, 500), 50, 80)):
            with self.assertRaises(ValueError):
                maniqa_tiles(size, start, gap)

    def test_model_failure_keeps_diagnostics(self):
        image = Image.new("RGB", (64, 32))
        with patch("inpaint_lab.quality.maniqa_score", side_effect=RuntimeError("offline")), redirect_stdout(io.StringIO()):
            scores = assess_quality(image, image, 28, 8, 4, None, Path("unused"))
        self.assertEqual(scores["quality_maniqa_error"], "offline")
        self.assertEqual(scores["quality_preservation_mae"], 0)

    def test_spearman_ties_and_constants(self):
        self.assertAlmostEqual(spearman([1, 1, 3], [2, 2, 5]), 1)
        self.assertAlmostEqual(spearman([1, 2, 3], [3, 2, 1]), -1)
        self.assertIsNone(spearman([1, 1, 1], [1, 2, 3]))

    def test_correlations_do_not_pool_image_pairs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for pair, count in (("a", 3), ("b", 2)):
                for i in range(count):
                    folder = root / f"{pair}-{i}"
                    folder.mkdir()
                    for name, content in {"params": {"input_pair_id": pair},
                                          "metrics": {"quality_version": 1, "quality_maniqa": i},
                                          "review": {"seam_quality": i + 1}}.items():
                        (folder / f"{name}.json").write_text(json.dumps(content))
            rows = list(correlations(root))
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["input_pair_id"], "a")
            self.assertEqual(rows[0]["runs"], 3)
            self.assertAlmostEqual(rows[0]["spearman"], 1)


if __name__ == "__main__":
    unittest.main()
