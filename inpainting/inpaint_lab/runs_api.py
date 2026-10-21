"""Run history JSON routes: list all runs, run details with debug images, and rerun settings."""

from datetime import datetime
import json
from pathlib import Path
from urllib.parse import parse_qs

from inpaint_lab.review_api import finite
from inpaint_lab.runs import image_pair_id

IMAGE_TYPES = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg"}
# Known files first, in pipeline order; anything else follows alphabetically.
IMAGE_ORDER = ("output.png", "output_comparison.png", "collage_canvas.png", "collage_mask.png",
               "debug_crop_image.png", "debug_crop_mask.png", "debug_crop_result.png", "quality_context.png")
RERUN_KEYS = ("model", "prompt", "negative_prompt", "height", "gap", "feather", "steps",
              "guidance_scale", "seed", "save_intermediate")


def read_json(path):
    try:
        data = finite(json.loads(path.read_text()))
        return data if isinstance(data, dict) else None
    except (OSError, ValueError):
        return None


def run_dir(handler, name):
    """A direct child of runs_dir only; rejects traversal and arbitrary paths."""
    if not isinstance(name, str) or not name or "/" in name or "\\" in name or name in (".", ".."):
        return None
    folder = handler.server.runs_dir / name
    return folder if folder.is_dir() and (folder / "params.json").is_file() else None


def attempts(handler):
    """Map run folder name to its batch attempt; empty if the manifest is unavailable."""
    from inpaint_lab.batch_api import manager
    try:
        data = manager(handler.server).snapshot()
    except Exception:  # Another process owns the manifest, or it is unreadable: history still works.
        return {}
    found = {}
    for batch in data["batches"]:
        for job in batch["jobs"]:
            for attempt in job["attempts"]:
                found[Path(attempt["run_dir"]).name] = {
                    "batch_id": batch["id"], "batch_name": batch["name"], "batch_kind": batch.get("kind", "batch"),
                    "status": attempt["status"], "error": attempt.get("error")}
    return found


def status(folder, params, attempt):
    if (folder / "review.json").is_file():
        return "reviewed"
    if attempt and attempt["status"] in ("running", "failed", "interrupted"):
        return attempt["status"]
    output = folder / Path(str(params.get("output", "output.png"))).name
    if output.is_file() and output.with_name(f"{output.stem}_comparison.png").is_file():
        return "pending review"
    return "incomplete"


def summary(folder, attempt):
    params = read_json(folder / "params.json") or {}
    metrics = read_json(folder / "metrics.json") or {}
    created = datetime.fromtimestamp((folder / "params.json").stat().st_mtime).astimezone()
    return {"name": folder.name, "created": created.isoformat(), "status": status(folder, params, attempt),
            "params": {key: params.get(key) for key in ("model", "gap", "feather", "steps", "guidance_scale", "seed")},
            "maniqa": metrics.get("quality_maniqa"), "inference_time": metrics.get("inference_time"),
            "batch": attempt}


def rerun_settings(root, params):
    """Form values from a run; inputs must still be in samples/ and match the recorded pixels."""
    settings = {key: params.get(key) for key in RERUN_KEYS if key in params}
    warnings, images = [], []
    samples = (root / "samples").resolve()
    for value in params.get("images") or []:
        path = Path(str(value))
        candidates = [path if path.is_absolute() else root / path, samples / path.name]
        found = next((c.resolve() for c in candidates if c.resolve().is_relative_to(samples) and c.is_file()), None)
        if found is None:
            warnings.append(f"Input {value} is not available in samples/; choose it manually.")
        images.append(str(found.relative_to(root.resolve())) if found else None)
    if len(images) == 2 and all(images) and params.get("input_pair_id"):
        from PIL import Image
        loaded = []
        for image in images:
            with Image.open(root / image) as opened:
                loaded.append(opened.convert("RGB"))
        if image_pair_id(loaded) != params["input_pair_id"]:
            warnings.append("Selected inputs differ from the pixels used by this run; check both images.")
    elif all(images) and images:
        warnings.append("This run has no input fingerprint, so the images were matched by filename only; confirm them.")
    settings["images"] = images
    return settings, warnings


def get_route(handler, address):
    query = parse_qs(address.query)
    if address.path == "/api/runs":
        found = attempts(handler)
        folders = [path.parent for path in handler.server.runs_dir.glob("*/params.json")]
        folders.sort(key=lambda folder: (folder / "params.json").stat().st_mtime, reverse=True)
        handler.send_json({"runs": [summary(folder, found.get(folder.name)) for folder in folders]})
    elif address.path == "/api/runs/detail":
        folder = run_dir(handler, query.get("run", [""])[0])
        if folder is None:
            handler.send_json({"error": "Run not found"}, 404)
            return True
        attempt = attempts(handler).get(folder.name)
        params = read_json(folder / "params.json") or {}
        images = [path.name for path in folder.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_TYPES]
        images.sort(key=lambda name: (IMAGE_ORDER.index(name) if name in IMAGE_ORDER else len(IMAGE_ORDER), name))
        steps = folder / "intermediate_steps"
        settings, warnings = rerun_settings(handler.server.project_root, params)
        handler.send_json(dict(summary(folder, attempt), params=params, metrics=read_json(folder / "metrics.json"),
                               review=read_json(folder / "review.json"), images=images,
                               step_images=len(list(steps.glob("*.png"))) if steps.is_dir() else 0,
                               rerun=settings, rerun_warnings=warnings))
    elif address.path == "/api/runs/file":
        folder = run_dir(handler, query.get("run", [""])[0])
        name = query.get("name", [""])[0]
        path = folder / name if folder and name and "/" not in name and "\\" not in name else None
        if path is None or path.suffix.lower() not in IMAGE_TYPES or not path.is_file():
            handler.send_error(404)
        else:
            handler.respond(path.read_bytes(), IMAGE_TYPES[path.suffix.lower()])
    else:
        return False
    return True


def post_route(handler):
    return False
