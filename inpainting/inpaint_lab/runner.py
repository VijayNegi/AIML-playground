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
    python -m inpaint_lab run --images photo1.jpg photo2.jpg
    
    # With custom scene-specific prompt (optional)
    python -m inpaint_lab run --images img1.jpg img2.jpg --prompt "forest landscape"
    
    # With custom options
    python -m inpaint_lab run --images img1.jpg img2.jpg \\
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
import json
import hashlib
import os
import time
import platform
from PIL import Image

from inpaint_lab.canvas import create_blend_mask, create_collage_canvas
from inpaint_lab.inpaint import inpaint_crop_and_patch
from inpaint_lab.models import AVAILABLE_MODELS, load_pipeline
from inpaint_lab.quality import assess_quality
from inpaint_lab.runs import image_pair_id, prepare_run
from inpaint_lab.tracking import track_aim_run


def parse_collage_args(argv=None):
    """Parse command line arguments for the collage tool."""
    parser = argparse.ArgumentParser(
        description="Create seamless photo collages with AI-powered blending",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic usage - uses generic prompt that matches any scene
    python -m inpaint_lab run --images photo1.jpg photo2.jpg
    
    # With optional custom prompt if you want more control
    python -m inpaint_lab run --images photo1.jpg photo2.jpg --prompt "beach sunset"
    
    # Custom gap and feathering
    python -m inpaint_lab run --images a.jpg b.jpg --gap 100 --feather 50
    
    # Lower guidance scale (good for better context matching)
    python -m inpaint_lab run --images a.jpg b.jpg --guidance-scale 5.0
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
        help="Output file path (default: runs/<local timestamp with timezone offset>/output.png)"
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

    
    return parser.parse_args(argv)


def hardware_id():
    """Keep timing history separate across machines and accelerator/software changes."""
    import torch
    device = torch.cuda.get_device_name(0) if torch.cuda.is_available() else (
        "mps" if torch.backends.mps.is_available() else "cpu")
    return hashlib.sha256(f"{platform.node()}|{platform.machine()}|{device}|{torch.__version__}".encode()).hexdigest()


def run_collage(args, resources=None):
    """Execute one run; an optional batch-owned cache retains loaded models."""
    wall_start = time.perf_counter()
    
    print("=" * 60)
    print("Seamless Photo Collage")
    print("=" * 60)
    
    # Validate input images exist
    for img_path in args.images:
        if not os.path.exists(img_path):
            print(f"Error: Image not found: {img_path}")
            raise FileNotFoundError(img_path)
    
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

    args.input_pair_id = image_pair_id(images)
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
    if resources is not None and resources.get("model_id") == model_id:
        pipe, load_time = resources["pipeline"], 0.0
    else:
        pipe, load_time = load_pipeline(model_id)
        if resources is not None:
            resources.update(model_id=model_id, pipeline=pipe)
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
    perf['hardware_id'] = hardware_id()
    # Save the result
    result.save(args.output)
    print(f"\n✓ Seamless collage saved to: {args.output}")

    try:
        perf.update(assess_quality(canvas, result, gap_x_start, adjusted_gap,
                                   args.feather, perf.get("crop_bbox"), run_dir,
                                   mask=mask, mask_binarized=perf.get("mask_binarized", False),
                                   model_cache=resources))
    except Exception as error:
        perf["quality_error"] = str(error)
        print(f"Warning: Quality diagnostics failed: {error}. Output is saved.")

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

    aim_hash = track_aim_run(args, run_dir, perf, result, comparison)
    perf["aim_status"] = "logged" if aim_hash else "failed"
    perf["wall_time"] = time.perf_counter() - wall_start
    temporary = run_dir / "metrics.json.tmp"
    temporary.write_text(json.dumps(perf, indent=2) + "\n")
    temporary.replace(run_dir / "metrics.json")
    
    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)
    return run_dir, perf


def main():
    """Main function for the seamless collage tool."""
    run_collage(parse_collage_args())


if __name__ == "__main__":
    main()
