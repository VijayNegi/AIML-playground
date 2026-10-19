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

Each invocation creates a unique local-time directory under `runs/`:

Example in UTC+05:30: `runs/2026-10-12_14-30-05.123456_+0530/`. The timestamp uses
the machine's local timezone; the offset distinguishes daylight-saving changes.
Fractional seconds keep rapid runs distinct. Existing run folders retain their names.

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
output by default, all settings, three 1–5 rating groups, and optional notes.
Use **Show input / Show output** to flip between the assembled input canvas and
the inpainted result. The input is read from the existing comparison image.
Spatial metrics surround the image: gap and model-mask ranges above, outer-feather
and pasted-crop ranges below, and MANIQA tile values on the right. Range bars align
with image x coordinates; L/R buttons show color difference and extra edge strength.
Connector lines link each tile score to its vertical center, with evenly spaced
labels to avoid crowding near overlapping tiles. Hover/focus or select these values
for the same region overlays, without automatically scrolling away from the image.
The image scales with viewport height to keep nearby metrics in view; click for
full resolution. Detailed metric tables remain below. Small screens or very tall
images with many tiles may still require scrolling or a larger window.
The compact **Automatic assessment** panel beneath the image keeps scores visible
in priority order: **1. Gap quality** (MANIQA), **2. Seams** (gap and blend edges),
then **3. Patch** (preservation and pasted-crop edges). Boundary checks use one row
per edge, with color mismatch, extra edge strength, edge strength, and nearby-edge
values in columns. No metric groups are collapsed. Short explanations give scales
and interpretation; MANIQA shows a reason when unavailable. Technical metric keys
are available on value tooltips. No red/green quality thresholds are invented.
Hover or keyboard-focus a metric value or **Show area** button to preview
its area on the output. Click to pin the highlight and
scroll the image into view; click again or use **Clear highlight** to remove it.
Orange outlines the first region; blue/purple distinguish additional bands.
Gradient checks show the adjacent pixel pair, color checks show both color bands,
and gradient excess also shows nearby context. Preservation highlights all pixels
eligible for checking outside the editable region, not just changed pixels.
MANIQA highlights its gap tiles; individual tile scores and the lowest tile can
also be selected. Older overview runs still highlight their recorded context crop.
These are measurement regions, not defect maps.
Highlights hide while viewing input and return when switching back to output.
Runs missing recorded geometry show an explanation instead of guessed highlights.
Review the image before reading scores to reduce anchoring. Missing scores are not
computed by the page or replaced with zeros; older runs can show no automatic scores.
Click the image for full resolution. Choose a run from the dropdown or use
Skip / next; click Save review and sync to Aim to save and advance.
It saves `review.json` with a local ISO timestamp including the timezone offset,
and never overwrites completed reviews. Review sync preserves that same timestamp.
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

### Automatic no-reference assessment

New collage runs automatically compute separate diagnostics and a MANIQA score
after saving the output. No original/reference photo is required, and creative
content within the gap is not treated as reconstruction error.

Activate the environment and install the updated requirements before running:

```bash
source ../.venv/bin/activate
python -m pip install -r requirements.txt
```

Results are saved in `metrics.json` and sent to Aim as `quality_*` metrics:

- `quality_preservation_mae`: mean RGB change outside the gap and feather zone,
  in 0–255 units. Lower is better; zero means no change.
- `quality_preservation_changed_fraction`: fraction of outside pixels changed
  in any channel, with no tolerance threshold. Not a perceptual severity score.
- `quality_{boundary}_gradient`: mean adjacent-pixel RGB difference, in 0–1 units.
  `nearby_gradient` measures parallel edges within eight pixels, excluding the
  boundary itself; `gradient_excess` subtracts that baseline. Lower excess is a
  diagnostic preference, not proof of better quality.
- `quality_{boundary}_delta_e76`: mean row-wise Lab color distance between bands
  up to eight pixels wide on each side. Lower suggests a smoother color transition.
  Natural edges, shadows, and textures can legitimately produce high values.
- Boundaries cover gap edges, outer editable/feather edges, and crop-patch edges.
  Edges on the canvas border are omitted because they have no outside neighbor.
  Missing baselines or preservation regions are JSON `null`, not fabricated zeros.
- Assessment version 2 uses the actual grayscale mask: preservation checks pixels
  with mask value zero, and outer feather bounds enclose nonzero mask pixels.
  This avoids assuming the first nominal feather pixel is editable when its value
  is actually zero. Model-mask checks separately measure the boundary after mask
  thresholding and cropping. SDXL binarizes at 0.5, so feather values 128–255 are
  masked and 0–127 are unmasked; passing a feathered mask does not by itself produce
  soft alpha compositing. The current pipeline still pastes the entire generated
  crop, and this change does not alter inference or compositing.
  Diagnostics currently support the v1 vertical-strip mask, not arbitrary shapes.
  Old scores and their recorded overlays are not silently recalculated.
- Edge strength measures the pixel pair across the named edge. Nearby edges are
  deliberately measured on either side, excluding that crossing. Their highlights
  therefore sit beside the edge, not directly on it. Extra edge strength compares
  the two measurements. Feathering creates an interval, not one universal seam:
  gap edges, nonzero feather bounds, and thresholded model-mask edges differ.
- `quality_maniqa`: PyIQA's `maniqa-pipal` model; higher is better. This is a raw
  model score, not a calibrated 1–5 rating. No hand-weighted overall score is made.

Assessment version 3 replaces the resized overview with native-resolution
224×224 gap-centered tiles, stepping vertically by 112 pixels (50% overlap).
The final tile is anchored to the bottom edge, covering the full gap height.
Each tile contains the full gap width and some neighboring context where possible.
The actual adjusted gap must be 1–224 pixels, and the image must be at least
224×224. Unsupported sizes skip MANIQA with an error rather than resizing/padding
or evaluating only part of the gap. A 224-pixel-wide gap leaves no side context.

`quality_maniqa` is the median tile score; `quality_maniqa_min` is the lowest tile
score, a possible weak spot rather than an automatic failure. Both use higher-is-
better scores. `quality_maniqa_tiles` retains every score and bounding box, with
model/library versions, stride, aggregation, and input size also recorded. Review
mode displays each tile score and can highlight all tiles or a single selected tile.
No new overview image is saved; tiles are taken directly from the saved output.
The model loads once per assessment and processes tiles sequentially on CPU to
avoid competing with diffusion for GPU memory. More tiles add scoring time.
The first assessment downloads pretrained weights and can take time. CPU scoring
adds latency; `quality_assessment_time` is separate from inference timing.

If MANIQA fails (missing dependency, offline weights, incompatible packages), the
run retains diagnostics, images, and a `quality_maniqa_error`. No score is invented.
Existing runs are not backfilled. The current pipeline pastes the entire crop, so
preservation diagnostics can expose changes outside the intended editable region.

### Validate automatic scores against your reviews

Run several settings/seeds for the same input pair, review them in the browser,
then print a CSV correlation report:

```bash
source ../.venv/bin/activate
python compare_quality.py
```

The report computes Spearman rank correlations with tie handling, separately for
each input pair, assessment version, model/library version, and crop/tile layout.
At least three scored and reviewed runs are required per comparison; that minimum
is exploratory, not enough to establish reliability. Constant scores/ratings
produce a blank correlation. Positive correlation is expected for MANIQA and
negative for error/discontinuity diagnostics when they agree with human ratings.

Inspect seam diagnostics against seam ratings and preservation against preservation
ratings first. Do not pool unrelated scenes or turn these correlations into weights.
Test promising metrics on held-out image pairs and several seeds before using
them for automatic selection. Browser reviews remain the human check; no learned
ranking, combined score, or automatic winner selection is implemented.

## Checks

Check run logging without GPU inference:

```bash
source ../.venv/bin/activate
python -m unittest test_run_logging test_review_runs test_quality_metrics
```

## Documentation

- [Performance guide](docs/PERFORMANCE_GUIDE.md)
- [PyTorch upgrade guide](docs/PYTORCH_UPGRADE.md)
- [v2 design](docs/inpainting_temp.txt)
- [Example command](docs/inpaintint_cmd.txt)

The existing performance guide references `benchmark_performance.py`, which is
not present in this repository.
