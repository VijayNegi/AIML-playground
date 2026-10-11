"""
Seamless Photo Collage
======================
Create seamless collages by blending multiple images with AI-powered inpainting.
Instead of hard borders, the AI generates natural transitions between images.

This is the MVP implementation supporting 2 images in a horizontal layout.

Requirements:
    pip install diffusers transformers accelerate torch pillow opencv-python

Usage:
    # Basic usage - blend two images with generic prompt (no scene analysis needed)
    python collage.py --images photo1.jpg photo2.jpg
    
    # With custom scene-specific prompt (optional)
    python collage.py --images img1.jpg img2.jpg --prompt "forest landscape"
    
    # With custom options
    python collage.py --images img1.jpg img2.jpg \\
        --gap 80 \\
        --feather 40 \\
        --model sd-xl \\
        --output seamless_collage.png

How it works:
    1. Load both images and normalize to same height
    2. Create a canvas with a gap between images
    3. Generate a feathered blend mask for the gap region
    4. Use AI inpainting to fill the gap with seamless content
    5. Save the final blended image

Key Concepts (for learning):
    - Diffusion Models: Generate images by gradually denoising random noise
    - Inpainting: Fill masked regions while keeping unmasked areas intact
    - Feathered Masks: Gradient edges for smooth blending (no hard seams)
    - Guidance Scale: Higher = follows prompt strictly, Lower = more creative
"""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
from PIL import Image

from inpainting_utils import (
    AVAILABLE_MODELS,
    load_pipeline,
    
    inpaint_crop_and_patch,
    normalize_image_heights,
    create_blend_mask,
)


def parse_collage_args():
    """Parse command line arguments for the collage tool."""
    parser = argparse.ArgumentParser(
        description="Create seamless photo collages with AI-powered blending",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic usage - uses generic prompt that matches any scene
    python collage.py --images photo1.jpg photo2.jpg
    
    # With optional custom prompt if you want more control
    python collage.py --images photo1.jpg photo2.jpg --prompt "beach sunset"
    
    # Custom gap and feathering
    python collage.py --images a.jpg b.jpg --gap 100 --feather 50
    
    # Lower guidance scale (good for better context matching)
    python collage.py --images a.jpg b.jpg --guidance-scale 5.0
        """
    )
    
    # Required arguments
    parser.add_argument(
        "--images", "-i",
        type=str,
        nargs=2,
        required=True,
        help="Two input images to blend (e.g., --images left.jpg right.jpg)"
    )
    
    parser.add_argument(
        "--prompt", "-p",
        type=str,
        default="seamless natural transition, matching surroundings, continuous scene",
        help="Text prompt for the gap region. Default is a generic prompt that works "
             "universally by telling the model to match surrounding context. "
             "For object removal, try: 'clean background' or 'natural continuation'"
    )
    
    # Optional arguments
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Output file path (default: runs/<UTC timestamp>/output.png)"
    )
    
    parser.add_argument(
        "--gap",
        type=int,
        default=80,
        help="Width of the gap between images in pixels (default: 80)"
    )
    
    parser.add_argument(
        "--feather",
        type=int,
        default=30,
        help="Feather/gradient size at mask edges in pixels (default: 30)"
    )
    
    parser.add_argument(
        "--height",
        type=int,
        default=None,
        help="Target height for output (default: uses minimum of input heights)"
    )
    
    parser.add_argument(
        "--model",
        type=str,
        default="sd-xl",
        choices=list(AVAILABLE_MODELS.keys()),
        help=f"Inpainting model to use (default: sd-xl). Available: {', '.join(AVAILABLE_MODELS.keys())}"
    )
    
    parser.add_argument(
        "--negative-prompt",
        type=str,
        default="blurry, low quality, distorted, seams, visible borders, artifacts, "
                "harsh edges, discontinuity, mismatched colors, abrupt transition",
        help="What to avoid in generation (default includes seam-related terms)"
    )
    
    parser.add_argument(
        "--steps",
        type=int,
        default=50,
        help="Number of inference steps (default: 50, higher = better quality)"
    )
    
    parser.add_argument(
        "--guidance-scale",
        type=float,
        default=7.5,
        help="Guidance scale - how closely to follow the prompt (default: 7.5)"
    )
    
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)"
    )
    
    parser.add_argument(
        "--save-intermediate",
        action="store_true",
        help="Save intermediate images (canvas, mask) for debugging"
    )
    
    parser.add_argument(
        "--save-steps",
        action="store_true",
        help="Save image after each denoising step (visualize diffusion process)"
    )
    
    parser.add_argument(
        "--steps-dir",
        type=str,
        default=None,
        help="Directory to save step images (default: <run directory>/intermediate_steps)"
    )
    
    parser.add_argument(
        "--save-every",
        type=int,
        default=1,
        help="Save every N steps (default: 1 = every step, use 5 for every 5th)"
    )

    
    return parser.parse_args()


def create_collage_canvas(
    images: list,
    gap: int,
    target_height: int = None,
) -> tuple:
    """
    Create a canvas with two images placed side by side with a gap.
    
    Args:
        images: List of two PIL Images
        gap: Width of gap between images in pixels
        target_height: Optional target height (uses min height if None)
        
    Returns:
        Tuple of (canvas, gap_x_start, normalized_images)
        - canvas: PIL Image with both images placed
        - gap_x_start: X coordinate where the gap begins
        - normalized_images: The height-normalized input images
    """
    if len(images) != 2:
        raise ValueError(f"Expected 2 images, got {len(images)}")
    
    # Normalize heights first
    print(f"Normalizing image heights...")
    normalized = normalize_image_heights(images, target_height)
    img1, img2 = normalized
    
    print(f"  Image 1 (after height norm): {img1.size[0]}x{img1.size[1]}")
    print(f"  Image 2 (after height norm): {img2.size[0]}x{img2.size[1]}")
    
    # Calculate raw canvas dimensions
    raw_width = img1.width + gap + img2.width
    raw_height = img1.height  # All images now have same height
    
    # Ensure dimensions are divisible by 8 (required by Stable Diffusion)
    # Round UP to avoid cutting off images
    canvas_width = ((raw_width + 7) // 8) * 8
    canvas_height = ((raw_height + 7) // 8) * 8
    
    # Adjust gap to account for any width increase (distribute padding evenly)
    width_padding = canvas_width - raw_width
    adjusted_gap = gap + width_padding
    
    print(f"Canvas size: {canvas_width}x{canvas_height} (padded from {raw_width}x{raw_height})")
    print(f"Gap adjusted: {gap} -> {adjusted_gap} (added {width_padding}px padding)")
    
    # Create canvas (black background - will be visible in the gap)
    canvas = Image.new("RGB", (canvas_width, canvas_height), (0, 0, 0))
    
    # Place images
    # Image 1 on the left
    canvas.paste(img1, (0, 0))
    
    # Image 2 on the right (after the adjusted gap)
    gap_x_start = img1.width
    img2_x_start = img1.width + adjusted_gap
    canvas.paste(img2, (img2_x_start, 0))
    
    print(f"Gap region: x={gap_x_start} to x={img2_x_start} (width={adjusted_gap})")
    
    # Return adjusted gap for mask creation
    return canvas, gap_x_start, normalized, adjusted_gap


def prepare_run(args):
    """Create the output directory and reserve metadata without overwriting a run."""
    if args.output is None:
        run_dir = Path("runs") / datetime.now(timezone.utc).strftime("%Y-%m-%d_%H-%M-%S.%f_UTC")
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


def track_aim_run(args, run_dir, perf, result, comparison):
    """Mirror a completed local run into Aim without risking saved results."""
    run = None
    try:
        from aim import Run, Image as AimImage

        run = Run(repo=str(Path(__file__).resolve().parent), experiment="collage",
                  system_tracking_interval=None, log_system_params=False,
                  capture_terminal_logs=False)
        run.name = run_dir.name
        run["params"] = vars(args)
        run["run_id"] = run_dir.name
        run["run_dir"] = str(run_dir.resolve())
        run["metrics"] = perf
        for name, value in perf.items():
            if isinstance(value, (int, float)):
                run.track(value, name=name, step=0)
        run.track(AimImage(result), name="output", step=0)
        run.track(AimImage(comparison), name="comparison", step=0)
        print(f"Aim run: {run.hash}")
    except Exception as error:
        print(f"Warning: Aim tracking failed: {error}. Local results are saved in {run_dir}", file=sys.stderr)
    finally:
        if run is not None:
            try:
                run.close()
            except Exception as error:
                print(f"Warning: Aim close failed: {error}", file=sys.stderr)


def main():
    """Main function for the seamless collage tool."""
    args = parse_collage_args()
    
    print("=" * 60)
    print("Seamless Photo Collage")
    print("=" * 60)
    
    # Validate input images exist
    for img_path in args.images:
        if not os.path.exists(img_path):
            print(f"Error: Image not found: {img_path}")
            sys.exit(1)
    
    # Load input images
    print(f"\nLoading images...")
    images = []
    for img_path in args.images:
        img = Image.open(img_path).convert("RGB")
        images.append(img)
        print(f"  Loaded: {img_path} ({img.size[0]}x{img.size[1]})")
    
    # Create the canvas with gap
    print(f"\nCreating canvas with {args.gap}px gap...")
    canvas, gap_x_start, normalized_images, adjusted_gap = create_collage_canvas(
        images=images,
        gap=args.gap,
        target_height=args.height,
    )
    
    # Generate blend mask using the adjusted gap (accounts for padding)
    print(f"\nGenerating blend mask (feather={args.feather}px)...")
    mask = create_blend_mask(
        canvas_size=canvas.size,
        gap_x_start=gap_x_start,
        gap_width=adjusted_gap,
        feather=args.feather,
    )

    run_dir, comparison_path = prepare_run(args)
    print(f"Run directory: {run_dir}")
    
    # Save intermediate files if requested
    if args.save_intermediate:
        canvas.save(run_dir / "collage_canvas.png")
        mask.save(run_dir / "collage_mask.png")
        print(f"  Saved canvas and mask in: {run_dir}")
    
    # Load the inpainting model
    print(f"\nLoading inpainting model: {args.model}")
    print("-" * 40)
    model_id = args.model_id
    pipe, load_time = load_pipeline(model_id)
    print(f"Model loaded in {load_time:.2f}s")
    
    # Perform inpainting using crop-and-patch (no resizing, max 1024x1024 crop)
    print(f"\nInpainting the gap region (crop-and-patch mode)...")
    print(f"  Prompt: '{args.prompt}'")
    print(f"  Negative: '{args.negative_prompt}'")
    print(f"  Steps: {args.steps}, Guidance: {args.guidance_scale}, Seed: {args.seed}")
    print("-" * 40)
    
    result, perf = inpaint_crop_and_patch(
        pipe=pipe,
        image=canvas,
        mask=mask,
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance_scale,
        seed=args.seed,
        max_crop_size=args.max_crop_size,  # SDXL optimal size
        strength=args.strength,
        save_intermediate=args.save_steps,
        intermediate_dir=args.steps_dir,
        save_every=args.save_every,
        debug_dir=run_dir,
    )

    perf['model_load_time'] = load_time
    perf['adjusted_gap'] = adjusted_gap
    perf['canvas_size'] = list(canvas.size)
    # Save the result
    result.save(args.output)
    print(f"\n✓ Seamless collage saved to: {args.output}")

    metrics_json = json.dumps(perf, indent=2)
    with (run_dir / "metrics.json").open("x") as file:
        file.write(metrics_json + "\n")
    
    # Performance summary
    print(f"\nPerformance:")
    print(f"  Preprocessing: {perf['preprocessing_time']:.3f}s")
    print(f"  Inference: {perf['inference_time']:.2f}s")
    if 'patch_time' in perf:
        print(f"  Patch back: {perf['patch_time']:.3f}s")
    print(f"  Total: {perf['total_time']:.2f}s")
    print(f"  Original canvas: {perf.get('original_size', 'N/A')}")
    print(f"  Crop sent to model: {perf.get('crop_size', 'N/A')}")
    
    # Create comparison image
    comparison_width = canvas.size[0]
    comparison_height = canvas.size[1] * 2
    comparison = Image.new("RGB", (comparison_width, comparison_height))
    comparison.paste(canvas, (0, 0))
    comparison.paste(result, (0, canvas.size[1]))
    
    comparison.save(comparison_path)
    print(f"  Comparison saved to: {comparison_path}")

    track_aim_run(args, run_dir, perf, result, comparison)
    
    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)


if __name__ == "__main__":
    main()
