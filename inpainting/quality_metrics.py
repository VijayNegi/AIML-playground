"""No-reference collage diagnostics and seam-context MANIQA assessment."""

from importlib.metadata import version
import math
import time

import cv2
import numpy as np
from PIL import Image


QUALITY_VERSION = 3
BAND_WIDTH = 8
TILE_SIZE = 224
TILE_STRIDE = 112


def boundary_metrics(rgb, axis, position, start, end):
    """Measure a boundary against nearby parallel gradients in the same span."""
    pixels = rgb[start:end] if axis == "x" else rgb[:, start:end].transpose(1, 0, 2)
    width = pixels.shape[1]
    if not 0 < position < width or not pixels.size:
        return {}
    band = min(BAND_WIDTH, position, width - position)
    edges = np.abs(np.diff(pixels, axis=1)).mean(axis=(0, 2))
    seam = float(edges[position - 1])
    nearby = np.concatenate((edges[max(0, position - band - 1):position - 1],
                             edges[position:min(width - 1, position + band)]))
    baseline = float(nearby.mean()) if nearby.size else None
    lab = cv2.cvtColor(np.ascontiguousarray(pixels[:, position - band:position + band]), cv2.COLOR_RGB2LAB)
    left = lab[:, :band].mean(axis=1)
    right = lab[:, band:].mean(axis=1)
    return {"gradient": seam, "nearby_gradient": baseline,
            "gradient_excess": seam - baseline if baseline is not None else None,
            "delta_e76": float(np.linalg.norm(left - right, axis=1).mean())}


def diagnostics(canvas, result, gap_start, gap_width, feather, crop_bbox, mask=None, mask_binarized=False):
    if result.size != canvas.size:
        raise ValueError("Quality assessment requires matching canvas and output sizes")
    before = np.asarray(canvas.convert("RGB"), dtype=np.float32)
    after = np.asarray(result.convert("RGB"), dtype=np.float32)
    width, height = canvas.size
    if gap_width <= 0 or feather < 0 or not 0 <= gap_start < gap_start + gap_width <= width:
        raise ValueError("Invalid gap or feather geometry")
    editable_start = max(0, gap_start - feather)
    editable_end = min(width, gap_start + gap_width + feather)
    unchanged = np.ones((height, width), dtype=bool)
    unchanged[:, editable_start:editable_end] = False
    effective_bbox = None
    if mask is not None:
        if mask.size != canvas.size:
            raise ValueError("Mask and canvas sizes must match for assessment")
        alpha = np.asarray(mask.convert("L"))
        if not np.all(alpha == alpha[0]):
            raise ValueError("Mask-edge diagnostics currently require a vertical strip mask")
        unchanged = alpha == 0
        support = mask.convert("L").getbbox()
        if support is not None:
            editable_start, _, editable_end, _ = support
        effective = alpha >= 128 if mask_binarized else alpha > 0
        if crop_bbox is not None:
            x1, y1, x2, y2 = crop_bbox
            covered = np.zeros_like(effective)
            covered[y1:y2, x1:x2] = True
            effective &= covered
        effective_bbox = Image.fromarray(effective).getbbox()
    difference = np.abs(after - before)[unchanged]
    scores = {
        "quality_version": QUALITY_VERSION,
        "quality_band_width": BAND_WIDTH,
        "quality_editable_bbox": [editable_start, 0, editable_end, height],
        "quality_gap_bbox": [gap_start, 0, gap_start + gap_width, height],
        "quality_effective_mask_bbox": list(effective_bbox) if effective_bbox else None,
        "quality_mask_binarized": mask_binarized,
        "quality_preservation_mae": float(difference.mean()) if difference.size else None,
        "quality_preservation_changed_fraction": float(np.any(difference != 0, axis=1).mean()) if difference.size else None,
    }
    rgb = after / 255.0
    boundaries = [("gap_left", "x", gap_start, 0, height),
                  ("gap_right", "x", gap_start + gap_width, 0, height),
                  ("editable_left", "x", editable_start, 0, height),
                  ("editable_right", "x", editable_end, 0, height)]
    if effective_bbox is not None:
        x1, y1, x2, y2 = effective_bbox
        boundaries.extend((("mask_left", "x", x1, y1, y2), ("mask_right", "x", x2, y1, y2)))
    if crop_bbox is not None:
        x1, y1, x2, y2 = crop_bbox
        boundaries.extend((("patch_left", "x", x1, y1, y2), ("patch_right", "x", x2, y1, y2),
                           ("patch_top", "y", y1, x1, x2), ("patch_bottom", "y", y2, x1, x2)))
    for name, axis, position, start, end in boundaries:
        scores.update({f"quality_{name}_{key}": value for key, value in
                       boundary_metrics(rgb, axis, position, start, end).items()})
    return scores


def maniqa_tiles(image_size, gap_start, gap_width):
    """Cover a narrow vertical gap with native-resolution, overlapping tiles."""
    # shortcut: one tile column supports gaps up to 224px; add horizontal tiling if wider gaps are needed.
    width, height = image_size
    if not 0 < gap_width <= TILE_SIZE or not 0 <= gap_start < gap_start + gap_width <= width:
        raise ValueError("Tiled MANIQA requires a gap between 1 and 224 pixels inside the image")
    if min(width, height) < TILE_SIZE:
        raise ValueError("Tiled MANIQA requires images at least 224×224; no resizing or padding is applied")
    x = max(0, min(width - TILE_SIZE, gap_start + gap_width // 2 - TILE_SIZE // 2))
    positions = list(range(0, height - TILE_SIZE + 1, TILE_STRIDE))
    if positions[-1] != height - TILE_SIZE:
        positions.append(height - TILE_SIZE)
    return [[x, y, x + TILE_SIZE, y + TILE_SIZE] for y in positions]


def maniqa_score(result, gap_start, gap_width, run_dir):
    """Score native-resolution gap tiles, retaining every score and its location."""
    boxes = maniqa_tiles(result.size, gap_start, gap_width)
    import pyiqa
    import torch

    metric = pyiqa.create_metric("maniqa-pipal", device="cpu", test_sample=1)
    metric.eval()
    tiles = []
    with torch.inference_mode():
        for bbox in boxes:
            crop = result.crop(bbox).convert("RGB")
            tensor = torch.from_numpy(np.asarray(crop).copy()).permute(2, 0, 1).float().unsqueeze(0) / 255
            score = float(metric(tensor).item())
            if not math.isfinite(score):
                raise ValueError("MANIQA produced a non-finite tile score")
            tiles.append({"bbox": bbox, "score": score})
    lowest = min(tiles, key=lambda tile: tile["score"])
    return {"quality_maniqa": float(np.median([tile["score"] for tile in tiles])),
            "quality_maniqa_min": lowest["score"], "quality_maniqa_min_bbox": lowest["bbox"],
            "quality_maniqa_tiles": tiles, "quality_maniqa_tile_count": len(tiles),
            "quality_maniqa_stride": TILE_STRIDE, "quality_maniqa_aggregation": "median",
            "quality_maniqa_model": "maniqa-pipal",
            "quality_maniqa_input_size": [224, 224], "quality_maniqa_device": "cpu",
            "quality_maniqa_test_sample": 1, "quality_pyiqa_version": version("pyiqa"),
            "quality_maniqa_lower_better": bool(metric.lower_better)}


def assess_quality(canvas, result, gap_start, gap_width, feather, crop_bbox, run_dir, mask=None, mask_binarized=False):
    start = time.perf_counter()
    scores = diagnostics(canvas, result, gap_start, gap_width, feather, crop_bbox, mask, mask_binarized)
    try:
        scores.update(maniqa_score(result, gap_start, gap_width, run_dir))
    except Exception as error:
        scores["quality_maniqa_error"] = str(error)
        print(f"Warning: MANIQA assessment failed: {error}. Diagnostics and output are preserved.")
    scores["quality_assessment_time"] = time.perf_counter() - start
    return scores
