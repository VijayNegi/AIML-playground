# Seamless Photo Collage

The v1 MVP blends two images horizontally using AI inpainting. Optimization is
ongoing; multi-image and vertical/grid layouts are planned, not implemented.

## Folder layout

- Python scripts remain in this directory so existing imports still work.
- `docs/`: performance and PyTorch guides, v2 design, and an example command.
- `samples/`: reusable source images, split inputs, and an existing input mask.
- `runs/`: existing experiment outputs, sweep folders, debug images, and diffusion
  steps. Files retain their original names; missing settings are not inferred.

`runs/` is ignored by Git, not deleted. Back it up separately. Samples and docs
are not ignored, but this reorganization does not stage or commit them.

## Install dependencies

From this directory, activate the repository's virtual environment first. Use
`python` and `python -m pip` only after activation, rather than a direct path to
the virtual environment's Python executable.

```bash
source ../.venv/bin/activate
python -m pip install -r requirements.txt
```

For a CUDA machine, install the appropriate PyTorch build using the
[official installation selector](https://pytorch.org/get-started/locally/) first.
Optional CUDA acceleration packages such as xformers are not required here.

## Run from this directory

Each invocation creates a unique UTC-timestamped directory under `runs/`:

Example: `runs/2026-10-12_14-30-05.123456_UTC/`. Fractional seconds keep rapid
runs distinct. Existing run folders retain their names.

```bash
source ../.venv/bin/activate
python collage.py \
  --images samples/split_main_1024_left.png samples/split_main_1024_right.png \
  --prompt "a green landscape" \
  --gap 80
```

Each run contains `params.json` (all CLI settings, including defaults, plus
inference strength, maximum crop size, and full model ID), `metrics.json`
(inference/preprocessing/patch timings, model load time, actual gap, and canvas
size), `output.png`, `output_comparison.png`, and crop debug images. Settings
are saved before model loading, so failed runs retain their configuration.
Metrics are saved after inference; a failed inference has no `metrics.json`.

Add `--save-intermediate` for the canvas and mask, or `--save-steps` for denoising
steps in the run's `intermediate_steps/` directory.

To name a run, pass `--output runs/my-experiment/output.png`; its parent directory
is created automatically. Existing outputs or run metadata are refused rather
than overwritten. Use one directory per run. `--steps-dir` can still override
the denoising step directory; choose a fresh directory to avoid overwriting steps.

Other scripts retain their existing output paths. JSON records settings, not
copies of inputs or a locked software environment; preserve samples separately.

## Aim tracking

Every completed collage run is tracked in Aim. Activate the virtual environment
first and install dependencies:

```bash
source ../.venv/bin/activate
python -m pip install -r requirements.txt
python collage.py \
  --images samples/split_main_1024_left.png samples/split_main_1024_right.png \
  --prompt "a green landscape" --gap 80
aim up --repo . --host 127.0.0.1
```

Open the local URL printed by Aim. The repository is stored in `inpainting/.aim/`,
regardless of the working directory used to launch the Python script. Run the
`aim up` command from `inpainting/`.

Aim mirrors settings, numeric metrics, output, and comparison after local files
are saved. Its run name matches the folder name, and `run_dir` records the
absolute output location. Aim failure prints a warning without discarding local
results. No tracking flag is required.

Only completed runs are sent to Aim; failed inference remains recorded locally
through `params.json`. System/environment and terminal-log capture are disabled.
Aim stores image copies, so back up both `runs/` and `.aim/`; neither is tracked
by Git. Existing runs are not automatically imported.

### Compare settings for the same inputs

Every new run records `params.input_pair_id`, a SHA-256 identifier of the ordered
left/right RGB pixels and dimensions before height normalization. Renaming a file
does not change the ID; swapping left/right or changing pixels does. Different
settings, including output height, share an ID when their original inputs match.

In Aim, filter runs with `run.params.input_pair_id == "<ID from params.json>"`.
Compare their `output` or `comparison` image sequences and display gap, feather,
steps, guidance scale, seed, and inference time. Keep seed fixed when isolating
another setting; test several seeds before judging robustness.

Numeric metrics are tracked once, not duplicated as Aim parameters. Nonnumeric
details such as crop coordinates are stored under `metadata`. Existing Aim runs
retain their old fields and do not acquire an input pair ID automatically.

### Assess quality

Review completed runs that do not yet have `review.json`:

```bash
source ../.venv/bin/activate
python review_runs.py
```

The script opens `http://127.0.0.1:8765/` in your browser. The page shows each
before/after comparison, all settings, three 1–5 rating groups, and optional notes.
Click the image for full resolution. Choose a run from the dropdown or use
Skip / next; click Save review and sync to Aim to save and advance.
It saves `review.json` with a UTC timestamp and never overwrites completed reviews.
Aim sync success or failure appears on the next page; local reviews remain safe.
Press Ctrl+C in the terminal to stop the server. Unsaved form entries are not kept.
Use `--runs-dir PATH` for another collection or `--port 8766` if the port is busy.
The server listens only on loopback, with form-token protection; it is not intended
for shared network access. No additional UI dependencies are required.

Only runs with `params.json`, output, and comparison images are selected. Legacy
images without metadata and incomplete runs are skipped. After saving, the script
syncs the review into the existing Aim run matched by its absolute `run_dir`.
Ratings appear in Aim's Metrics column group as `review.seam_quality`,
`review.scene_coherence`, and `review.preservation`, not as run parameters.
Notes are tracked text named `review.notes` and can be viewed in Aim's text
explorer. `review_synced_at` is a run parameter used only to avoid repeated sync.
Previously synced review dictionaries are migrated on the next script startup.
It does not overwrite the Aim description or create new runs.

On startup, existing local reviews are synced too, allowing recovery after an Aim
failure without entering ratings again. Identical reviews are not rewritten.
Missing or ambiguous matches print a warning and preserve the local review.
Moving run folders breaks absolute-path matching; sync before relocating them.

Timing is not a quality score. For unrelated photographs, there is no single
correct gap: inspect seam visibility, color/lighting continuity, scene coherence,
and changes to the original images. Record consistent ratings with the browser
review form to support comparison.

For inputs split from one original photo, the missing strip provides a reference.
PSNR/SSIM can measure reconstruction inside that strip when dimensions and pixel
alignment match; perceptual metrics such as LPIPS provide another signal. These
scores can penalize plausible alternative content, so retain visual review.
Do not score the whole canvas: unchanged pixels can hide a poor seam.
Automatic quality scoring is not implemented.

## Checks

Check run logging without GPU inference:

```bash
source ../.venv/bin/activate
python -m unittest test_run_logging test_review_runs
```

## Documentation

- [Performance guide](docs/PERFORMANCE_GUIDE.md)
- [PyTorch upgrade guide](docs/PYTORCH_UPGRADE.md)
- [v2 design](docs/inpainting_temp.txt)
- [Example command](docs/inpaintint_cmd.txt)

The existing performance guide references `benchmark_performance.py`, which is
not present in this repository.
