"""
Seamless Photo Collage
======================
Create seamless collages by blending multiple images with AI-powered inpainting.
Instead of hard borders, the AI generates natural transitions between images.

This is the MVP implementation supporting 2 images in a horizontal layout.

Requirements:
    pip install diffusers transformers accelerate torch pillow opencv-python

Usage:
    # Basic usage - blend two images horizontally
    python collage.py --images photo1.jpg photo2.jpg --prompt "sunny beach scene"
    
    # With custom options
    python collage.py --images img1.jpg img2.jpg \\
        --prompt "forest landscape" \\
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
import os
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
    # Basic usage
    python collage.py --images photo1.jpg photo2.jpg --prompt "beach sunset"
    
    # Custom gap and feathering
    python collage.py --images a.jpg b.jpg --prompt "forest" --gap 100 --feather 50
    
    # Use a specific model
    python collage.py --images a.jpg b.jpg --prompt "city" --model sd-xl
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
        required=True,
        help="Text prompt describing the scene (guides the AI for the gap region)"
    )
    
    # Optional arguments
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="collage_output.png",
        help="Output file path (default: collage_output.png)"
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
        default="blurry, low quality, distorted, seams, visible borders, artifacts",
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
    
    # Save intermediate files if requested
    if args.save_intermediate:
        canvas.save("collage_canvas.png")
        mask.save("collage_mask.png")
        print("  Saved: collage_canvas.png, collage_mask.png")
    
    # Load the inpainting model
    print(f"\nLoading inpainting model: {args.model}")
    print("-" * 40)
    model_id = AVAILABLE_MODELS[args.model]
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
        max_crop_size=1024,  # SDXL optimal size
    )
    
    # Save the result
    result.save(args.output)
    print(f"\n✓ Seamless collage saved to: {args.output}")
    
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
    
    comparison_path = args.output.replace(".png", "_comparison.png")
    comparison.save(comparison_path)
    print(f"  Comparison saved to: {comparison_path}")
    
    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)


if __name__ == "__main__":
    main()
