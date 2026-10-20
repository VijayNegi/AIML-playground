# Inpainting Batch Sweeps

## Goal

Understand how inference settings affect natural-looking collage transitions using
controlled experiments. Preserve existing per-run outputs, JSON metadata, automatic
quality metrics, human reviews, and Aim tracking.

The initial local browser batch builder and sequential worker are implemented.
They support one-at-a-time/grid previews, model reuse, pause/resume, root-level
`batches.json` history, explicit retries, and historical runtime estimates.
Standalone JSON-file import, explicit arbitrary configuration lists, and the real
two-run inference smoke test remain follow-up work. The stages below describe the
experiment strategy; they do not authorize launching expensive inference automatically.

## Scope

- Sequential, single-machine batch execution.
- One-parameter sweeps and explicit lists of configurations for targeted grids.
- JSON experiment definitions using Python's standard library.
- Reuse the diffusion pipeline and MANIQA model within a batch.
- Resume successful configurations without overwriting runs.
- Local-time timestamps with timezone offsets, consistent with the existing CLI.
- No new tracking service, parallel GPU jobs, automatic optimizer, reference-image
  scoring, or automatic selection based solely on MANIQA.

## Baseline

Use the latest recorded run's numeric settings as the initial baseline, with the
CLI's generic default prompt shared across input pairs:

| Setting | Value |
| --- | --- |
| Left input | `samples/split_main_1024_left.png` |
| Right input | `samples/split_main_1024_right.png` |
| Model | `sd-xl` |
| Prompt | `seamless natural transition, matching surroundings, continuous scene` |
| Gap | 10 pixels |
| Feather | 30 pixels per side |
| Height | Automatic: minimum input height |
| Steps | 50 |
| Guidance scale | 7.5 |
| Seed | 42 |
| Strength | 0.99 |
| Maximum crop size | 1024 pixels |

Retain the CLI's existing negative prompt. Capture its resolved value and all other
defaults in each run rather than relying on defaults remaining unchanged.
Omitting `--prompt` already selects this generic prompt. Keep it fixed across the
initial sweeps rather than using the scene-specific `a green landscape` prompt.

## Experiment Stages

### Stage 0: Smoke Test

Run the baseline and one changed configuration before launching a larger batch.
Confirm outputs, quality metrics, metadata, Aim entries, and batch resumption work.

### Stage 1: One-Parameter Sweeps

Keep inputs, prompt, model, height, and seed fixed. For each sweep, vary only the
named setting and retain baseline values for the others.

| Parameter | Values |
| --- | --- |
| Guidance scale | 3, 5, 7.5, 10 |
| Feather | 0, 15, 30, 60 |
| Steps | 20, 30, 50 |
| Gap | 10, 30, 60, 100 |

Deduplicate the shared baseline, yielding 12 unique configurations. Reuse baseline
results within this batch; do not automatically reuse historical runs without
checking settings, input identity, and assessment protocol.

Gap sweeps change canvas geometry, image placement, and the width of generated
content. Analyze them separately from tuning inference settings for a fixed task.
Record both requested gap and actual padded gap.

### Stage 2: Targeted Interactions

After inspecting the first sweep, run a small grid as an explicit configuration
list. Initial candidate:

- Guidance scale: 3, 5, 7.5.
- Feather: 15, 30, 60.

This produces nine configurations before deduplication. Prefer settings supported
by the first-stage observations rather than expanding every parameter combination.
The full first-stage Cartesian grid would contain 192 configurations per seed and
is deliberately out of scope for the first batch.

### Stage 3: Reliability Across Seeds and Inputs

- Select three promising configurations and retain the baseline as a control.
- Use the same three fixed seeds for every configuration.
- Repeat on several input pairs, including landscapes, architecture, and scenes
  with contrasting lighting or content.
- Count reused baseline configurations only once.
- Validate conclusions on held-out input pairs before trusting automatic rankings.

## Batch Runner Design

### Configuration

Add a JSON experiment definition containing baseline settings and either:

1. A mapping of parameter names to values for one-at-a-time sweeps.
2. An explicit list of setting overrides for interaction experiments.

Do not silently interpret sweep lists as a Cartesian product. Validate supported
keys, value types, numeric ranges, inputs, and configuration structure before model
loading. Resolve input paths consistently relative to the experiment definition.
Do not allow per-configuration output paths to overwrite other configurations.

Provide a dry-run preview listing resolved unique configurations and their count.
Use a stable configuration identifier derived from canonical resolved settings and
input identity, not timestamps or output paths.

### Execution and Model Reuse

Inspect the current callers before extracting the smallest shared single-run
execution function from `collage.py`. Both the existing CLI and batch runner should
use it so generation, logging, quality assessment, and error handling stay aligned.

- Process configurations sequentially.
- Reuse a diffusion pipeline for configurations using the same model.
- Initialize MANIQA once per batch and score tiles sequentially on CPU.
- Preserve normal single-run CLI behavior and invocation.
- Keep lazy loading and clear error reporting if MANIQA or weights are unavailable.
- Do not repeat an expensive image generation merely because Aim or MANIQA failed.

### Output and Metadata

Retain the current per-run layout under `runs/`, including `params.json`,
`metrics.json`, output/comparison images, debug images, and optional `review.json`.

Add batch ID, configuration ID, and sweep label to run metadata and Aim parameters.
Maintain a batch manifest containing:

- Resolved experiment definition and input identities.
- Configuration identifiers, settings, and output locations.
- Status for each configuration: pending, running, completed, or failed.
- Error messages for failed configurations.
- Local-time start/end timestamps with explicit timezone offsets.

Write manifest updates atomically. Do not modify or delete existing run outputs.
Use the manifest to find results instead of relying on chronological sorting of
folder names, especially when older UTC names coexist with local-time names.

### Resume and Failure Handling

- Resume an explicitly selected batch manifest.
- Refuse silent resumption if the experiment settings or input identity changed.
- Verify required artifacts exist before skipping a completed configuration.
- Treat interrupted running entries as incomplete, not successful.
- Retry failed/incomplete generation only when explicitly requested, in a fresh
  run folder that preserves the earlier attempt.
- Record quality-assessment and Aim-sync failures separately from generation
  failures. Successful saved outputs must remain usable.
- Record failures, continue with remaining configurations, and exit nonzero when
  any generation failed; Ctrl+C should stop promptly and preserve progress.

## Evaluation

Compare within the same ordered input pair, seed, and compatible assessment
protocol wherever possible. Record settings that alter geometry explicitly.

Use existing measurements without combining them into an arbitrary overall score:

- Tiled MANIQA median and lowest tile score.
- Mask/gap/feather seam color and gradient diagnostics.
- Preservation error and changed-pixel fraction.
- Crop-patch edge diagnostics.
- Diffusion inference time and quality-assessment time separately.
- Human seam quality, scene coherence, preservation ratings, and notes.

The tiled scorer currently requires an actual adjusted gap of at most 224 pixels
and images at least 224×224. Unsupported assessment geometry must produce a clear
assessment error rather than a misleading score.

Filter runs by batch ID and input-pair ID in Aim. Existing browser reviews remain
the human check. Use `compare_quality.py` for exploratory within-pair correlations;
do not pool incompatible score versions or learn weights from the same image pairs
used to judge the resulting ranking.

## Tests and Acceptance Criteria

- Expansion gives 12 unique first-stage configurations and changes only the intended
  setting in each one-parameter configuration.
- Explicit configuration lists work without implicit Cartesian expansion.
- Input and parameter validation happens before model loading.
- Existing single-run tests continue to pass.
- Mocked batches load each needed model once and record every configuration.
- Resume skips verified completed runs and never overwrites previous artifacts.
- Interrupted, failed, and missing-artifact cases are handled explicitly.
- Aim/MANIQA failures do not discard successful generated outputs.
- Batch/configuration identifiers appear in local metadata and Aim logging calls.
- A real two-run smoke test verifies the full workflow before the 12-run sweep.

## Suggested Implementation Order

1. Inspect current CLI, scoring, logging, and tests; identify shared execution boundaries.
2. Add JSON validation, configuration expansion/deduplication, and dry-run tests.
3. Share single-run execution and support batch-level model reuse.
4. Add sequential execution, manifest persistence, and explicit resume/retry behavior.
5. Wire batch metadata into existing JSON and Aim records; update usage notes.
6. Run automated tests and the real two-run smoke test.
7. Launch the 12-configuration sweep, review outputs, then choose targeted follow-ups.

## Environment and Constraints

Activate the virtual environment before running Python commands. From `inpainting/`:

```bash
source ../.venv/bin/activate
```

Use `python` and `python -m pip` after activation, not a direct virtual-environment
Python path. No implementation or benchmark execution is authorized by writing
this plan. Runtime, memory usage, and automatic-score reliability remain unmeasured
for the proposed batches.
