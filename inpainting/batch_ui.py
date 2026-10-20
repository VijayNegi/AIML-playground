"""HTTP routes for the batch builder, hosted by the existing review server."""

from copy import deepcopy
import io
import json
from pathlib import Path
import secrets
from urllib.parse import parse_qs

from PIL import Image

from batch_runs import BatchManager, estimate_jobs, preview_batch


def manager(server):
    if getattr(server, "batch_manager", None) is None:
        server.batch_manager = BatchManager(server.project_root)
    return server.batch_manager


def json_response(handler, data, status=200):
    handler.respond(json.dumps(data, allow_nan=False).encode(), "application/json; charset=utf-8", status)


def get_route(handler, address):
    if address.path not in ("/batches", "/batch.js", "/api/batches", "/api/batches/catalog", "/api/batches/sample", "/api/batches/output"):
        return False
    try:
        root = handler.server.project_root
        if address.path == "/batches":
            html = Path(__file__).with_name("batch.html").read_text().replace("__TOKEN__", handler.server.token)
            handler.respond(html.encode())
        elif address.path == "/batch.js":
            handler.respond(Path(__file__).with_name("batch.js").read_bytes(), "text/javascript; charset=utf-8")
        elif address.path == "/api/batches/catalog":
            from collage import AVAILABLE_MODELS, parse_collage_args
            files = [str(path.relative_to(root)) for path in sorted((root / "samples").rglob("*"))
                     if path.is_file() and path.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp")
                     and path.resolve().is_relative_to((root / "samples").resolve())]
            defaults = vars(parse_collage_args(["--images", "left", "right"]))
            json_response(handler, {"images": files, "models": list(AVAILABLE_MODELS), "defaults": defaults})
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
            from collage import hardware_id
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
            json_response(handler, data)
    except (ValueError, OSError, TypeError, KeyError, StopIteration) as error:
        json_response(handler, {"error": str(error)}, 400)
    return True


def post_route(handler):
    if handler.path not in ("/api/batches/preview", "/api/batches/create", "/api/batches/action"):
        return False
    try:
        if not secrets.compare_digest(handler.headers.get("X-CSRF-Token", ""), handler.server.token):
            json_response(handler, {"error": "Invalid form token"}, 403)
            return True
        length = int(handler.headers.get("Content-Length", "0"))
        if not 0 < length <= 131072 or handler.headers.get_content_type() != "application/json":
            raise ValueError("Expected a JSON request of at most 128KiB")
        data = json.loads(handler.rfile.read(length))
        if not isinstance(data, dict):
            raise ValueError("Expected a JSON object")
        root = handler.server.project_root
        if handler.path == "/api/batches/preview":
            from collage import hardware_id
            batch = preview_batch(root, deepcopy(data))
            batch["estimate"] = estimate_jobs(root, batch["jobs"], hardware_id(), data.get("allow_legacy_timings", False))
            batch["expected"] = {"jobs": [job["id"] for job in batch["jobs"]], "input_hashes": batch["input_hashes"]}
            json_response(handler, batch)
        elif handler.path == "/api/batches/create":
            if not isinstance(data.get("expected"), dict):
                raise ValueError("Preview the batch before saving or starting")
            batch = manager(handler.server).create(data["definition"], expected=data["expected"])
            json_response(handler, batch)
        else:
            manager(handler.server).action(data["id"], data["action"])
            json_response(handler, {"ok": True})
    except (ValueError, OSError, TypeError, KeyError) as error:
        json_response(handler, {"error": str(error)}, 400)
    return True
