"""Batch builder JSON routes: catalog, previews, history, controls, and images."""

from copy import deepcopy
import io
from pathlib import Path
from urllib.parse import parse_qs

from PIL import Image

from inpaint_lab.batches import BatchManager, estimate_jobs, preview_batch


GET_PATHS = ("/api/batches", "/api/batches/catalog", "/api/batches/sample", "/api/batches/output")
POST_PATHS = ("/api/batches/preview", "/api/batches/create", "/api/batches/action")


def manager(server):
    if getattr(server, "batch_manager", None) is None:
        server.batch_manager = BatchManager(server.project_root)
    return server.batch_manager


def get_route(handler, address):
    if address.path not in GET_PATHS:
        return False
    try:
        root = handler.server.project_root
        if address.path == "/api/batches/catalog":
            from inpaint_lab.models import AVAILABLE_MODELS
            from inpaint_lab.runner import parse_collage_args
            files = [str(path.relative_to(root)) for path in sorted((root / "samples").rglob("*"))
                     if path.is_file() and path.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp")
                     and path.resolve().is_relative_to((root / "samples").resolve())]
            defaults = vars(parse_collage_args(["--images", "left", "right"]))
            handler.send_json({"images": files, "models": list(AVAILABLE_MODELS), "defaults": defaults})
        elif address.path == "/api/batches/output":
            query = parse_qs(address.query)
            batch = next(item for item in manager(handler.server).snapshot()["batches"] if item["id"] == query.get("batch", [""])[0])
            job = next(item for item in batch["jobs"] if item["id"] == query.get("job", [""])[0])
            index = int(query.get("attempt", ["0"])[0])
            if not 0 <= index < len(job["attempts"]):
                raise ValueError("Attempt not found")
            output = (Path(job["attempts"][index]["run_dir"]) / "output.png").resolve()
            if not output.is_relative_to((root / "runs").resolve()):
                raise ValueError("Output is outside runs/")
            handler.respond(output.read_bytes(), "image/png")
        elif address.path == "/api/batches/sample":
            path = (root / parse_qs(address.query).get("path", [""])[0]).resolve()
            if not path.is_relative_to((root / "samples").resolve()) or not path.is_file():
                raise ValueError("Sample not found")
            with Image.open(path) as image:
                image.thumbnail((320, 240))
                buffer = io.BytesIO()
                image.convert("RGB").save(buffer, "PNG")
            handler.respond(buffer.getvalue(), "image/png")
        else:
            from inpaint_lab.runner import hardware_id
            data = manager(handler.server).snapshot()
            selected = parse_qs(address.query).get("batch", [""])[0]
            batch = next((item for item in data["batches"] if item["id"] == selected), None)
            if batch:
                jobs = [job for job in batch["jobs"] if job["status"] in ("pending", "running")]
                estimate = estimate_jobs(root, jobs, hardware_id(), batch["definition"].get("allow_legacy_timings", False),
                                         include_loads=batch["status"] not in ("running", "pause_requested"))
                elapsed = batch["running_elapsed_seconds"]
                for key in ("seconds", "known_seconds", "observed_low_seconds", "observed_high_seconds"):
                    if estimate[key] is not None:
                        estimate[key] = max(0, estimate[key] - elapsed)
                data["estimate"] = estimate
            handler.send_json(data)
    except (ValueError, OSError, TypeError, KeyError, StopIteration) as error:
        handler.send_json({"error": str(error)}, 400)
    return True


def post_route(handler):
    if handler.path not in POST_PATHS:
        return False
    try:
        data = handler.read_json()
        if data is None:
            return True
        root = handler.server.project_root
        if handler.path == "/api/batches/preview":
            from inpaint_lab.runner import hardware_id
            batch = preview_batch(root, deepcopy(data))
            batch["estimate"] = estimate_jobs(root, batch["jobs"], hardware_id(), data.get("allow_legacy_timings", False))
            batch["expected"] = {"jobs": [job["id"] for job in batch["jobs"]], "input_hashes": batch["input_hashes"]}
            handler.send_json(batch)
        elif handler.path == "/api/batches/create":
            if not isinstance(data.get("expected"), dict):
                raise ValueError("Preview the batch before saving or starting")
            batch = manager(handler.server).create(data["definition"], expected=data["expected"])
            handler.send_json(batch)
        else:
            manager(handler.server).action(data["id"], data["action"])
            handler.send_json({"ok": True})
    except (ValueError, OSError, TypeError, KeyError) as error:
        handler.send_json({"error": str(error)}, 400)
    return True
