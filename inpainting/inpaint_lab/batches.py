"""Local, single-worker batches with durable history and conservative timing estimates."""

import argparse
from contextlib import redirect_stdout
from copy import deepcopy
from datetime import datetime
import hashlib
import io
from itertools import product
import json
import math
import os
from pathlib import Path
import statistics
import sys
import threading
import time
import traceback
import uuid

from PIL import Image


PARAMETERS = {"gap": (int, 1, 224), "feather": (int, 0, 1024), "steps": (int, 2, 1000),
              "guidance_scale": (float, 0, 50), "seed": (int, 0, 2**32 - 1)}
MAX_JOBS = 1000
# Modules whose changes alter generated images or scores; resuming across them would mix protocols.
INFERENCE_MODULES = ("runner.py", "models.py", "canvas.py", "inpaint.py", "quality.py", "runs.py")


def software_signature():
    package = Path(__file__).resolve().parent
    return hashlib.sha256(b"".join((package / name).read_bytes() for name in INFERENCE_MODULES)).hexdigest()


def log(text):
    """One console line with local time; stderr so it is never captured into a run log."""
    print(f"{datetime.now():%H:%M:%S}  {text}", file=sys.stderr, flush=True)


def now():
    return datetime.now().astimezone().isoformat()


def atomic_json(path, data):
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w") as file:
        file.write(json.dumps(data, indent=2, allow_nan=False) + "\n")
        file.flush()
        os.fsync(file.fileno())
    temporary.replace(path)


def file_hash(path):
    with Path(path).open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def number(name, value):
    kind, minimum, maximum = PARAMETERS[name]
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    if kind is int and type(value) is not int:
        raise ValueError(f"{name} must be a whole number")
    if not minimum <= value <= maximum:
        raise ValueError(f"{name} must be between {minimum} and {maximum}")
    return kind(value)


def preview_batch(root, definition):
    from inpaint_lab.models import AVAILABLE_MODELS
    from inpaint_lab.runner import parse_collage_args
    from inpaint_lab.runs import image_pair_id
    from inpaint_lab.quality import maniqa_tiles

    if not isinstance(definition, dict) or set(definition) - {"name", "images", "baseline", "sweeps", "mode", "allow_legacy_timings", "kind"}:
        raise ValueError("Invalid batch definition or unsupported fields")
    if type(definition.get("allow_legacy_timings", False)) is not bool:
        raise ValueError("Legacy timing opt-in must be true or false")
    kind = definition.get("kind", "batch")
    if kind not in ("batch", "single"):
        raise ValueError("Kind must be batch or single")
    if kind == "single" and definition.get("sweeps"):
        raise ValueError("A single run cannot have sweeps")
    name = definition.get("name", "Batch")
    if not isinstance(name, str) or not 1 <= len(name.strip()) <= 100:
        raise ValueError("Batch name must be 1–100 characters")
    paths = definition.get("images")
    if not isinstance(paths, list) or len(paths) != 2 or any(not isinstance(path, str) for path in paths):
        raise ValueError("Choose two sample images")
    paths = [(root / path).resolve() for path in paths]
    if any(not path.is_relative_to((root / "samples").resolve()) or not path.is_file() for path in paths):
        raise ValueError("Images must be existing files inside samples/")
    images = []
    for path in paths:
        with Image.open(path) as image:
            images.append(image.convert("RGB"))
    baseline = definition.get("baseline", {})
    if not isinstance(baseline, dict) or set(baseline) - (set(PARAMETERS) | {"model", "prompt", "negative_prompt", "height", "save_intermediate"}):
        raise ValueError("Unsupported baseline parameters")
    if type(baseline.get("save_intermediate", False)) is not bool:
        raise ValueError("save_intermediate must be true or false")
    defaults = vars(parse_collage_args(["--images", *map(str, paths)]))
    defaults.update(baseline)
    for key in PARAMETERS:
        defaults[key] = number(key, defaults[key])
    if defaults["model"] not in AVAILABLE_MODELS:
        raise ValueError("Unknown model")
    for key in ("prompt", "negative_prompt"):
        if not isinstance(defaults[key], str) or len(defaults[key]) > 10000:
            raise ValueError(f"{key} must be text of at most 10000 characters")
    height = defaults["height"]
    if height is not None and (type(height) is not int or not 224 <= height <= 1024):
        raise ValueError("Height must be automatic or a whole number between 224 and 1024")
    height = height or min(image.height for image in images)
    if not 224 <= height <= 1024:
        raise ValueError("Batch images must normalize to 224–1024px high; use the height setting")
    widths = [int(height * (image.width / image.height)) for image in images]
    if min(widths) < 1:
        raise ValueError("Normalized image width is too small")
    pair = image_pair_id(images)
    hashes = [file_hash(path) for path in paths]
    sweeps = definition.get("sweeps", {})
    if not isinstance(sweeps, dict) or set(sweeps) - set(PARAMETERS):
        raise ValueError("Sweep only gap, feather, steps, guidance_scale, or seed")
    for key, values in sweeps.items():
        if not isinstance(values, list) or not values or len(values) > MAX_JOBS:
            raise ValueError(f"{key} needs a nonempty list of at most {MAX_JOBS} values")
        sweeps[key] = list(dict.fromkeys(number(key, value) for value in values))
    mode = definition.get("mode", "one_at_a_time")
    if mode not in ("one_at_a_time", "grid"):
        raise ValueError("Choose one_at_a_time or grid")
    count = math.prod(map(len, sweeps.values())) if mode == "grid" else 1 + sum(map(len, sweeps.values()))
    if count > MAX_JOBS:
        raise ValueError(f"Batch exceeds the {MAX_JOBS}-job preview limit")
    overrides = [({}, "baseline")]
    if mode == "grid" and sweeps:
        overrides = [(dict(zip(sweeps, values)), "grid") for values in product(*sweeps.values())]
    elif mode == "one_at_a_time":
        overrides += [({key: value}, key) for key, values in sweeps.items() for value in values]
    jobs, seen = [], set()
    for override, label in overrides:
        params = dict(defaults, **override, input_pair_id=pair)
        canonical = json.dumps(params, sort_keys=True, allow_nan=False)
        identity = hashlib.sha256((canonical + json.dumps(hashes)).encode()).hexdigest()
        if identity in seen:
            continue
        seen.add(identity)
        raw_width = sum(widths) + params["gap"]
        canvas_width = (raw_width + 7) // 8 * 8
        canvas_height = (height + 7) // 8 * 8
        adjusted_gap = params["gap"] + canvas_width - raw_width
        tiles = maniqa_tiles((canvas_width, canvas_height), widths[0], adjusted_gap)
        jobs.append({"id": identity[:16], "sweep": label, "params": params,
                     "geometry": {"crop_size": f"{min(1024, canvas_width)}x{min(1024, canvas_height)}",
                                  "adjusted_gap": adjusted_gap, "tiles": len(tiles)},
                     "status": "pending", "attempts": []})
    return {"name": name.strip(), "kind": kind, "definition": deepcopy(definition), "input_hashes": hashes,
            "jobs": jobs, "status": "paused", "created_at": now(), "elapsed_seconds": 0.0,
            "software_signature": software_signature()}


def estimate_jobs(root, jobs, hardware, allow_legacy=False, include_loads=True):
    """Use matched observed timings; unknown components remain explicitly unknown."""
    from inpaint_lab.quality import QUALITY_VERSION

    history = []
    for path in (root / "runs").glob("*/metrics.json"):
        try:
            metrics = json.loads(path.read_text())
            params = json.loads((path.parent / "params.json").read_text())
            if (not isinstance(metrics, dict) or not isinstance(params, dict)
                    or type(params.get("guidance_scale", 7.5)) not in (int, float)):
                continue
            if metrics.get("hardware_id") == hardware or (allow_legacy and not metrics.get("hardware_id")):
                history.append((params, metrics))
        except (OSError, ValueError, TypeError):
            continue
    warnings = set()
    if allow_legacy:
        warnings.add("Legacy records without hardware identity are assumed to be from this machine.")
    total, low, high = 0.0, 0.0, 0.0
    unknown = False
    samples = set()
    for index, job in enumerate(jobs):
        params, geometry = job["params"], job["geometry"]
        matching = [(p, m) for p, m in history if p.get("model") == params["model"]
                    and m.get("crop_size") == geometry["crop_size"]
                    and (p.get("guidance_scale", 7.5) > 1) == (params["guidance_scale"] > 1)]
        def valid(value):
            return type(value) in (int, float) and math.isfinite(value) and value >= 0
        rates, quality, overhead, loads, quality_loads = [], [], [], [], []
        for p, m in matching:
            if valid(m.get("inference_time")) and valid(p.get("steps")) and p["steps"] > 0:
                # shortcut: step-linear timing assumes matching crop/CFG; calibrate richer models if measured estimates drift.
                rates.append(m["inference_time"] / p["steps"] * params["steps"])
            if valid(m.get("model_load_time")) and m["model_load_time"] > 0:
                loads.append(m["model_load_time"])
            if (m.get("quality_version") == QUALITY_VERSION and valid(m.get("quality_assessment_time"))
                    and valid(m.get("quality_maniqa_tile_count")) and m["quality_maniqa_tile_count"] > 0
                    and not m.get("quality_maniqa_error")):
                load = m.get("quality_model_load_time", 0)
                if valid(load):
                    quality.append(max(0, m["quality_assessment_time"] - load) / m["quality_maniqa_tile_count"] * geometry["tiles"])
                    if load > 0:
                        quality_loads.append(load)
            if all(valid(m.get(key)) for key in ("wall_time", "model_load_time", "inference_time", "quality_assessment_time")):
                overhead.append(max(0, m["wall_time"] - m["model_load_time"] - m["inference_time"] - m["quality_assessment_time"]))
        components = [("inference", rates), ("quality assessment", quality), ("saving/tracking overhead", overhead)]
        if index == 0 and include_loads:
            components += [("diffusion model loading", loads), ("MANIQA model loading", quality_loads)]
        for name, values in components:
            if not values:
                warnings.add(f"Insufficient matching history for {name}.")
                unknown = True
            else:
                samples.add(len(values))
                total += statistics.median(values)
                low += min(values)
                high += max(values)
    return {"seconds": None if unknown else total, "known_seconds": total, "observed_low_seconds": low,
            "observed_high_seconds": high, "minimum_samples": min(samples, default=0), "warnings": sorted(warnings),
            "confidence": "insufficient history" if unknown else "historical approximation"}


class BatchManager:
    """One process owns the manifest; one worker runs sequential jobs."""
    def __init__(self, root, runner=None):
        self.root = Path(root).resolve()
        self.path = self.root / "batches.json"
        self.lock = threading.RLock()
        self.thread = None
        self.closing = False
        self.runner = runner
        self.current_started = None
        self.process_lock = (self.root / ".batches.lock").open("a+")
        try:
            if os.name == "nt":
                import msvcrt
                self.process_lock.write("0")
                self.process_lock.flush()
                self.process_lock.seek(0)
                msvcrt.locking(self.process_lock.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(self.process_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            self.data = json.loads(self.path.read_text()) if self.path.exists() else {"version": 1, "batches": []}
            if self.data.get("version") != 1 or not isinstance(self.data.get("batches"), list):
                raise ValueError("Unsupported batches.json; refusing to overwrite it")
            recovered = False
            for batch in self.data["batches"]:
                if batch["status"] in ("running", "pause_requested"):
                    batch["status"] = "interrupted"
                    recovered = True
                    for job in batch["jobs"]:
                        if job["status"] == "running":
                            job["status"] = "interrupted"
                            job["attempts"][-1].update(status="interrupted", error="Worker stopped before completion", ended_at=now())
            if recovered:
                self.save()
        except BaseException:
            self.process_lock.close()
            raise

    def save(self):
        atomic_json(self.path, self.data)

    def snapshot(self):
        with self.lock:
            data = deepcopy(self.data)
            for batch in data["batches"]:
                batch["running_elapsed_seconds"] = (time.perf_counter() - self.current_started
                    if self.current_started is not None and batch["status"] in ("running", "pause_requested") else 0)
            return data

    def create(self, definition, expected=None):
        batch = preview_batch(self.root, deepcopy(definition))
        if expected is not None and expected != {"jobs": [job["id"] for job in batch["jobs"]], "input_hashes": batch["input_hashes"]}:
            raise ValueError("Preview is out of date or input files changed; preview again")
        batch["id"] = datetime.now().astimezone().strftime("batch-%Y-%m-%d_%H-%M-%S_%z-") + uuid.uuid4().hex[:6]
        with self.lock:
            self.data["batches"].append(batch)
            self.save()
        return deepcopy(batch)

    def action(self, batch_id, action):
        with self.lock:
            batch = next((b for b in self.data["batches"] if b["id"] == batch_id), None)
            if batch is None:
                raise ValueError("Batch not found")
            if action == "pause":
                if batch["status"] == "running":
                    batch["status"] = "pause_requested"
                    self.save()
                return
            if action not in ("resume", "retry"):
                raise ValueError("Unknown batch action")
            if self.closing or (self.thread and self.thread.is_alive()):
                raise ValueError("A batch worker is already active; pause it first")
            if software_signature() != batch["software_signature"]:
                raise ValueError("Inference/scoring code changed; create a new batch instead of mixing protocols")
            paths = batch["jobs"][0]["params"]["images"]
            if [file_hash(path) for path in paths] != batch["input_hashes"]:
                raise ValueError("Input files changed; create a new batch")
            for job in batch["jobs"]:
                if job["status"] == "completed":
                    folder = Path(job["attempts"][-1]["run_dir"])
                    if not all((folder / name).is_file() for name in ("output.png", "output_comparison.png", "params.json", "metrics.json")):
                        job["status"] = "interrupted"
                        batch["status"] = "interrupted"
                        self.save()
                if action == "retry" and job["status"] in ("failed", "interrupted"):
                    job["status"] = "pending"
            if action == "resume" and any(job["status"] == "interrupted" for job in batch["jobs"]):
                raise ValueError("Interrupted or missing-output jobs require Retry failed/interrupted")
            if not any(job["status"] == "pending" for job in batch["jobs"]):
                raise ValueError("No pending jobs; use Retry failed/interrupted if needed")
            batch["status"] = "running"
            batch.setdefault("started_at", now())
            self.save()
            self.thread = threading.Thread(target=self.work, args=(batch,), daemon=True)
            self.thread.start()

    def work(self, batch):
        resources = {}
        try:
            from inpaint_lab.runner import run_collage
            runner = self.runner or run_collage
            while True:
                with self.lock:
                    job = next((j for j in batch["jobs"] if j["status"] == "pending"), None)
                    if job is None:
                        batch.update(status="failed" if any(j["status"] == "failed" for j in batch["jobs"]) else "completed", ended_at=now())
                        self.save()
                        failed = sum(j["status"] == "failed" for j in batch["jobs"])
                        log(f"■ {batch['name']}: {batch['status']} ({len(batch['jobs'])} jobs, {failed} failed)")
                        return
                    if self.closing or batch["status"] == "pause_requested":
                        batch.update(status="paused", paused_at=now())
                        self.save()
                        log(f"⏸ {batch['name']}: paused")
                        return
                    folder = self.root / "runs" / f"{batch['id']}_{job['id']}_attempt-{len(job['attempts']) + 1}"
                    attempt = {"run_dir": str(folder), "status": "running", "started_at": now()}
                    job["attempts"].append(attempt)
                    job["status"] = "running"
                    self.current_started = time.perf_counter()
                    self.save()
                started = time.perf_counter()
                p = job["params"]
                label = f"{batch['name']} · job {batch['jobs'].index(job) + 1}/{len(batch['jobs'])}"
                log(f"▶ {label}: gap {p.get('gap')}, feather {p.get('feather')}, steps {p.get('steps')}, "
                    f"cfg {p.get('guidance_scale')}, seed {p.get('seed')}")
                output = io.StringIO()
                try:
                    if [file_hash(path) for path in job["params"]["images"]] != batch["input_hashes"]:
                        raise ValueError("Input files changed during the batch")
                    args = argparse.Namespace(**dict(job["params"], output=str(folder / "output.png"),
                                                    batch_id=batch["id"], configuration_id=job["id"], sweep=job["sweep"]))
                    # shortcut: redirect_stdout is process-wide, so server-thread prints during a job also land in its
                    # run.log; switch the pipeline to the logging module if that matters.
                    with redirect_stdout(output):
                        _, metrics = runner(args, resources=resources)
                    result = {"status": "completed", "warnings": {key: metrics[key] for key in
                              ("quality_error", "quality_maniqa_error") if key in metrics}}
                    if metrics.get("aim_status") == "failed":
                        result["warnings"]["aim_status"] = "failed"
                except Exception as error:
                    output.write(traceback.format_exc())
                    result = {"status": "failed", "error": str(error)}
                    metrics = {}
                elapsed = time.perf_counter() - started
                if folder.is_dir():
                    (folder / "run.log").write_text(output.getvalue())
                if result["status"] == "completed":
                    maniqa = metrics.get("quality_maniqa")
                    score = f", MANIQA {maniqa:.3f}" if isinstance(maniqa, (int, float)) else ""
                    warned = f", warnings: {', '.join(result['warnings'])}" if result["warnings"] else ""
                    log(f"✓ {label}: done in {elapsed:.1f}s{score}{warned} → runs/{folder.name}")
                else:
                    log(f"✗ {label}: failed after {elapsed:.1f}s: {result['error']}"
                        + (f" (details: runs/{folder.name}/run.log)" if folder.is_dir() else ""))
                with self.lock:
                    attempt.update(result, ended_at=now(), elapsed_seconds=elapsed)
                    job["status"] = result["status"]
                    batch["elapsed_seconds"] += attempt["elapsed_seconds"]
                    self.current_started = None
                    self.save()
        except Exception as error:
            with self.lock:
                batch.update(status="interrupted", error=str(error))
                log(f"✗ {batch['name']}: interrupted: {error}")
                for job in batch["jobs"]:
                    if job["status"] == "running":
                        job["status"] = "interrupted"
                        job["attempts"][-1].update(status="interrupted", error=str(error), ended_at=now())
                self.save()
        finally:
            resources.clear()

    def close(self):
        with self.lock:
            self.closing = True
        if self.thread and self.thread.is_alive():
            log("Waiting for the current batch job to finish before shutdown...")
            self.thread.join()
        self.process_lock.close()
