"""Review JSON routes: pending runs, run details with highlight regions, image views, saving."""

from contextlib import redirect_stdout
from datetime import datetime
import io
import json
import math
from pathlib import Path
from urllib.parse import parse_qs

from PIL import Image

from inpaint_lab import tracking
from inpaint_lab.runs import RATINGS, pending_runs


def metric_regions(name, metrics):
    """Return measured rectangles in output pixels, not a defect heatmap."""
    size = metrics.get("canvas_size")
    if not isinstance(size, list) or len(size) != 2 or any(type(v) is not int or v <= 0 for v in size):
        return []
    width, height = size
    def valid_box(value):
        if (isinstance(value, list) and len(value) == 4 and all(type(v) is int for v in value)
                and 0 <= value[0] < value[2] <= width and 0 <= value[1] < value[3] <= height):
            return value
        return None
    def box(key):
        return valid_box(metrics.get(key))
    if name == "quality_maniqa":
        if "quality_maniqa_tiles" in metrics:
            return [region for tile in metrics["quality_maniqa_tiles"]
                    if (region := valid_box(tile.get("bbox"))) is not None]
        region = box("quality_maniqa_crop_bbox")
        return [region] if region else []
    if name == "quality_maniqa_min":
        region = box("quality_maniqa_min_bbox")
        return [region] if region else []
    if name.startswith("quality_maniqa_tile_"):
        index = name.removeprefix("quality_maniqa_tile_")
        tiles = metrics.get("quality_maniqa_tiles", [])
        if index.isdigit() and int(index) < len(tiles):
            region = valid_box(tiles[int(index)].get("bbox"))
            return [region] if region else []
        return []
    if name.startswith("quality_preservation_"):
        region = box("quality_editable_bbox")
        if not region:
            return []
        left, _, right, _ = region
        return [r for r in ([0, 0, left, height], [right, 0, width, height]) if r[0] < r[2]]
    for group, key in (("gap", "quality_gap_bbox"), ("mask", "quality_effective_mask_bbox"),
                       ("editable", "quality_editable_bbox"), ("patch", "crop_bbox")):
        region = box(key)
        if not region:
            continue
        for edge in ("left", "right", "top", "bottom"):
            prefix = f"quality_{group}_{edge}_"
            if not name.startswith(prefix):
                continue
            vertical = edge in ("left", "right")
            position = region[{"left": 0, "top": 1, "right": 2, "bottom": 3}[edge]]
            extent = width if vertical else height
            if not 0 < position < extent:
                return []
            band_width = metrics.get("quality_band_width")
            if type(band_width) is not int or band_width <= 0:
                return []
            band = min(band_width, position, extent - position)
            def strip(start, end):
                return [start, region[1], end, region[3]] if vertical else [region[0], start, region[2], end]
            suffix = name[len(prefix):]
            if suffix == "delta_e76":
                return [strip(position - band, position), strip(position, position + band)]
            boundary = strip(position - 1, position + 1)
            neighbors = [strip(max(0, position - band - 1), position),
                         strip(position, min(extent, position + band + 1))]
            if suffix == "nearby_gradient":
                return neighbors
            if suffix == "gradient_excess":
                return [boundary] + neighbors
            if suffix == "gradient":
                return [boundary]
    return []


def finite(value):
    """Replace NaN/inf (accepted by json.loads) so responses stay valid JSON."""
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: finite(item) for key, item in value.items()}
    if isinstance(value, list):
        return [finite(item) for item in value]
    return value


def run_payload(run_dir, params):
    """Recorded params and metrics plus server-computed regions; nothing is recalculated."""
    try:
        metrics = finite(json.loads((run_dir / "metrics.json").read_text()))
        if not isinstance(metrics, dict):
            raise ValueError("Metrics must be an object")
        error = None
    except (OSError, ValueError):
        metrics, error = None, "Automatic scores unavailable: metrics.json is missing or invalid."
    regions = {}
    if metrics:
        tiles = metrics.get("quality_maniqa_tiles")
        names = list(metrics) + [f"quality_maniqa_tile_{index}" for index in range(len(tiles) if isinstance(tiles, list) else 0)]
        for name in names:
            try:
                if found := metric_regions(name, metrics):
                    regions[name] = found
            except (TypeError, AttributeError, KeyError):
                continue  # Malformed legacy geometry: show the value without a highlight.
    return {"name": run_dir.name, "params": finite(params), "metrics": metrics,
            "metrics_error": error, "regions": regions}


def find_pending(handler, name):
    return next((entry for entry in pending_runs(handler.server.runs_dir) if entry[0].name == name), None)


def get_route(handler, address):
    query = parse_qs(address.query)
    if address.path == "/api/review/runs":
        handler.send_json({"runs": [entry[0].name for entry in pending_runs(handler.server.runs_dir)],
                           "ratings": RATINGS})
    elif address.path == "/api/review/run":
        selected = find_pending(handler, query.get("run", [""])[0])
        if selected is None:
            handler.send_json({"error": "Run is missing or already reviewed"}, 404)
        else:
            handler.send_json(run_payload(selected[0], selected[1]))
    elif address.path == "/image":
        selected = find_pending(handler, query.get("run", [""])[0])
        if selected is None:
            handler.send_error(404)
            return True
        view = query.get("view", ["output"])[0]
        if view not in ("input", "output"):
            handler.send_error(400, "Unknown image view")
            return True
        try:
            image_path = selected[2] if view == "input" else selected[0] / Path(selected[1]["output"]).name
            with Image.open(image_path) as image:
                if view == "input":
                    image = image.crop((0, 0, image.width, image.height // 2))
                buffer = io.BytesIO()
                image.save(buffer, format="PNG")
            handler.respond(buffer.getvalue(), "image/png")
        except OSError as error:
            handler.send_error(404, str(error))
    else:
        return False
    return True


def post_route(handler):
    if handler.path != "/api/review/save":
        return False
    try:
        data = handler.read_json()
        if data is None:
            return True
        name = data.get("run")
        selected = find_pending(handler, name) if isinstance(name, str) else None
        if selected is None:
            handler.send_json({"error": "Run is missing or already reviewed"}, 409)
            return True
        review = {}
        for key in RATINGS:
            if type(data.get(key)) is not int or not 1 <= data[key] <= 5:
                raise ValueError("Each rating must be a whole number from 1 to 5")
            review[key] = data[key]
        notes = data.get("notes", "")
        if not isinstance(notes, str) or len(notes.strip()) > 10000:
            raise ValueError("Notes must be text of at most 10000 characters")
        review["notes"] = notes.strip()
        review["reviewed_at"] = datetime.now().astimezone().isoformat()
        run_dir = selected[0]
        with (run_dir / "review.json").open("x") as file:
            file.write(json.dumps(review, indent=2) + "\n")
    except FileExistsError:
        handler.send_json({"error": "Run is missing or already reviewed"}, 409)
        return True
    except (ValueError, OSError) as error:
        handler.send_json({"error": str(error)}, 400)
        return True
    with redirect_stdout(io.StringIO()) as messages:
        tracking.sync_review(run_dir, review)
    handler.send_json({"message": f"Review saved for {name}. {messages.getvalue().strip()}".strip()})
    return True
