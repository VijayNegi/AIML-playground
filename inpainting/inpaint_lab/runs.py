"""Run folders: identity, reserved metadata, and pending-review discovery."""

from datetime import datetime
import hashlib
import json
from pathlib import Path

from inpaint_lab.config import RUNS_DIR
from inpaint_lab.models import AVAILABLE_MODELS


def image_pair_id(images):
    """Identify ordered input pixels independently of filenames or run settings."""
    digest = hashlib.sha256()
    for image in images:
        image = image.convert("RGB")
        digest.update(f"{image.width}x{image.height}:".encode("ascii"))
        digest.update(image.tobytes())
    return digest.hexdigest()


def prepare_run(args):
    """Create the output directory and reserve metadata without overwriting a run."""
    if args.output is None:
        run_dir = RUNS_DIR / datetime.now().astimezone().strftime("%Y-%m-%d_%H-%M-%S.%f_%z")
        run_dir.mkdir(parents=True, exist_ok=False)
        args.output = str(run_dir / "output.png")
    else:
        run_dir = Path(args.output).parent
        run_dir.mkdir(parents=True, exist_ok=True)
    output = Path(args.output)
    comparison_path = output.with_name(f"{output.stem}_comparison.png")
    if any(path.exists() for path in (output, comparison_path, run_dir / "params.json", run_dir / "metrics.json")):
        raise FileExistsError(f"Run output or metadata already exists in {run_dir}; choose a fresh directory")
    if args.steps_dir is None:
        args.steps_dir = str(run_dir / "intermediate_steps")
    args.strength = 0.99
    args.max_crop_size = 1024
    args.model_id = AVAILABLE_MODELS[args.model]
    with (run_dir / "params.json").open("x") as file:
        json.dump(vars(args), file, indent=2)
        file.write("\n")
    return run_dir, comparison_path


RATINGS = {
    "seam_quality": "Seam quality (1=obvious seam, 5=invisible seam)",
    "scene_coherence": "Scene coherence (1=implausible/artifacts, 5=believable)",
    "preservation": "Preservation (1=unwanted changes, 5=original content preserved)",
}


def pending_runs(runs_dir):
    """Find recorded runs with saved outputs and no existing visual review."""
    for params_path in sorted(runs_dir.glob("*/params.json")):
        run_dir = params_path.parent
        if (run_dir / "review.json").exists():
            continue
        try:
            params = json.loads(params_path.read_text())
            output = run_dir / Path(params["output"]).name
            comparison = output.with_name(f"{output.stem}_comparison.png")
            if output.is_file() and comparison.is_file():
                yield run_dir, params, comparison
        except (OSError, ValueError, KeyError, TypeError) as error:
            print(f"Skipping {run_dir}: {error}")
