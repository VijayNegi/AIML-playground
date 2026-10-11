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

## Run from this directory

Create a fresh output directory for each experiment, then point the CLI at it:

```bash
mkdir runs/my-experiment
python collage.py \
  --images samples/split_main_1024_left.png samples/split_main_1024_right.png \
  --prompt "a green landscape" \
  --gap 80 \
  --output runs/my-experiment/output.png \
  --steps-dir runs/my-experiment/intermediate_steps
```

Add `--save-steps` to save denoising steps. The comparison image is saved next to
the output. Existing scripts still write some debug files into the working
directory; root-level PNGs are ignored. Move any results worth keeping into
`runs/` before another run overwrites them.

Run logging, JSON metadata, and Aim integration are deferred to a separate change.

## Documentation

- [Performance guide](docs/PERFORMANCE_GUIDE.md)
- [PyTorch upgrade guide](docs/PYTORCH_UPGRADE.md)
- [v2 design](docs/inpainting_temp.txt)
- [Example command](docs/inpaintint_cmd.txt)

The existing performance guide references `benchmark_performance.py`, which is
not present in this repository.
