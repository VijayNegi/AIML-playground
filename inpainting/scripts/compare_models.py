"""
Inpainting Example using Stable Diffusion
==========================================
This script demonstrates how to use Stable Diffusion Inpainting to replace
parts of an image with AI-generated content based on a text prompt.

Requirements:
    pip install diffusers transformers accelerate torch pillow opencv-python

Usage:
    # Interactive mask drawing with single model:
    python -m scripts.compare_models --input image.jpg --interactive --prompt "your prompt"

    # Use existing mask with multiple models:
    python -m scripts.compare_models --input image.jpg --mask mask.png --models sd-1.5 sd-2 --prompt "your prompt"

    # Use all available models:
    python -m scripts.compare_models --input image.jpg --mask mask.png --models all --prompt "your prompt"
"""

import argparse
import os

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from inpaint_lab.inpaint import inpaint
from inpaint_lab.models import AVAILABLE_MODELS, load_pipelines


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Stable Diffusion Inpainting",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--input", "-i",
        type=str,
        default="input.png",
        help="Path to input image (default: input.png)"
    )
    parser.add_argument(
        "--mask", "-m",
        type=str,
        default="mask.png",
        help="Path to mask image, white=inpaint, black=keep (default: mask.png)"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="output.png",
        help="Path to save output image (default: output.png)"
    )
    parser.add_argument(
        "--prompt", "-p",
        type=str,
        default="a beautiful garden with colorful flowers",
        help="Text prompt describing what to generate"
    )
    parser.add_argument(
        "--negative-prompt",
        type=str,
        default="blurry, low quality, distorted",
        help="Negative prompt (what to avoid)"
    )
    parser.add_argument(
        "--seed", "-s",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)"
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=50,
        help="Number of inference steps (default: 50)"
    )
    parser.add_argument(
        "--guidance-scale",
        type=float,
        default=7.5,
        help="Guidance scale (default: 7.5)"
    )
    parser.add_argument(
        "--no-comparison",
        action="store_true",
        help="Skip creating comparison image"
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Use interactive mask drawing with OpenCV"
    )
    parser.add_argument(
        "--brush-size",
        type=int,
        default=20,
        help="Brush size for interactive mask drawing (default: 20)"
    )
    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        default=["sd-1.5"],
        choices=list(AVAILABLE_MODELS.keys()) + ["all"],
        help=f"Model(s) to use for inpainting. Available: {', '.join(AVAILABLE_MODELS.keys())}, or 'all' for all models (default: sd-1.5)"
    )
    return parser.parse_args()


def create_mask_interactive(image_path: str, brush_size: int = 20) -> np.ndarray:
    """
    Create a mask interactively using OpenCV.

    Controls:
        - Left Mouse Button: Draw mask (white area = inpaint)
        - Right Mouse Button: Erase mask (black area = keep original)
        - Mouse Wheel / +/-: Adjust brush size
        - 'c': Clear mask
        - 'r': Reset to original
        - 's': Save and continue
        - 'ESC' or 'q': Cancel and exit

    Args:
        image_path: Path to the input image
        brush_size: Initial brush size in pixels

    Returns:
        Mask as numpy array (white = inpaint, black = keep)
    """
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image from {image_path}")

    # Create mask (all black initially)
    mask = np.zeros(image.shape[:2], dtype=np.uint8)

    # State variables
    drawing = False
    erasing = False
    current_brush_size = brush_size

    def draw_circle(event, x, y, flags, param):
        nonlocal drawing, erasing, current_brush_size, mask

        if event == cv2.EVENT_LBUTTONDOWN:
            drawing = True
            cv2.circle(mask, (x, y), current_brush_size, 255, -1)
        elif event == cv2.EVENT_LBUTTONUP:
            drawing = False
        elif event == cv2.EVENT_RBUTTONDOWN:
            erasing = True
            cv2.circle(mask, (x, y), current_brush_size, 0, -1)
        elif event == cv2.EVENT_RBUTTONUP:
            erasing = False
        elif event == cv2.EVENT_MOUSEMOVE:
            if drawing:
                cv2.circle(mask, (x, y), current_brush_size, 255, -1)
            elif erasing:
                cv2.circle(mask, (x, y), current_brush_size, 0, -1)
        elif event == cv2.EVENT_MOUSEWHEEL:
            # Adjust brush size with mouse wheel
            if flags > 0:
                current_brush_size = min(current_brush_size + 2, 100)
            else:
                current_brush_size = max(current_brush_size - 2, 1)

    # Create window and set mouse callback
    window_name = "Draw Mask - Left:Draw | Right:Erase | +/-:Size | c:Clear | s:Save | ESC:Cancel"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window_name, draw_circle)

    print("\nInteractive Mask Drawing")
    print("=" * 50)
    print("Controls:")
    print("  Left Mouse: Draw mask (areas to inpaint)")
    print("  Right Mouse: Erase mask")
    print("  Mouse Wheel / +/-: Adjust brush size")
    print("  'c': Clear mask")
    print("  'r': Reset view")
    print("  's': Save and continue")
    print("  'ESC' or 'q': Cancel")
    print("=" * 50)

    while True:
        # Create display image with mask overlay
        display = image.copy()

        # Create colored mask overlay (green for areas to inpaint)
        mask_colored = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        mask_colored[:, :, 0] = 0  # Remove blue channel
        mask_colored[:, :, 2] = 0  # Remove red channel

        # Blend image with mask
        overlay = cv2.addWeighted(display, 0.7, mask_colored, 0.3, 0)

        # Show brush size indicator in corner
        cv2.putText(overlay, f"Brush Size: {current_brush_size}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Display
        cv2.imshow(window_name, overlay)

        # Handle keyboard input
        key = cv2.waitKey(1) & 0xFF

        if key == ord('s'):
            # Save and exit
            print("\nMask saved!")
            break
        elif key == ord('c'):
            # Clear mask
            mask = np.zeros(image.shape[:2], dtype=np.uint8)
            print("Mask cleared")
        elif key == ord('r'):
            # Reset view (just redraw)
            pass
        elif key == ord('+') or key == ord('='):
            current_brush_size = min(current_brush_size + 5, 100)
            print(f"Brush size: {current_brush_size}")
        elif key == ord('-') or key == ord('_'):
            current_brush_size = max(current_brush_size - 5, 1)
            print(f"Brush size: {current_brush_size}")
        elif key == 27 or key == ord('q'):  # ESC or 'q'
            print("\nCancelled")
            cv2.destroyAllWindows()
            return None

    cv2.destroyAllWindows()
    return mask


def main():
    """Main function demonstrating the inpainting workflow."""
    args = parse_args()

    # Check if input image exists, if not create a sample
    if not os.path.exists(args.input):
        print(f"No input image found at '{args.input}'")
        print("Creating a sample gradient image for demonstration...")

        # Create a sample 512x512 gradient image
        sample = Image.new("RGB", (512, 512))
        for x in range(512):
            for y in range(512):
                r = int(255 * x / 512)
                g = int(255 * y / 512)
                b = 128
                sample.putpixel((x, y), (r, g, b))
        sample.save(args.input)
        print(f"Sample image saved to '{args.input}'")

    # Load input image
    image = Image.open(args.input)
    print(f"Loaded image: {image.size}")

    # Load or create mask
    mask = None

    if args.interactive:
        # Interactive mask creation with OpenCV
        print("\nStarting interactive mask creation...")
        mask_np = create_mask_interactive(args.input, brush_size=args.brush_size)

        if mask_np is None:
            print("Mask creation cancelled. Exiting.")
            return

        # Convert numpy array to PIL Image
        mask = Image.fromarray(mask_np).convert("RGB")
        mask.save(args.mask)
        print(f"Mask saved to '{args.mask}'")

    elif os.path.exists(args.mask):
        # Load existing mask
        mask = Image.open(args.mask)
        print(f"Loaded mask: {mask.size}")

    else:
        # No mask provided
        print(f"Error: No mask found at '{args.mask}'")
        print("Please either:")
        print("  - Use --interactive flag to draw a mask interactively")
        print("  - Provide an existing mask with --mask <path>")
        return

    # Load the inpainting pipelines
    print(f"\nLoading models: {', '.join(args.models)}")
    print("=" * 50)
    pipelines, load_times = load_pipelines(args.models)

    if not pipelines:
        print("Error: No models were successfully loaded!")
        return

    print(f"\n✓ Loaded {len(pipelines)} model(s)")

    # Display load times
    print("\nModel Load Times:")
    for model_name, load_time in load_times.items():
        print(f"  {model_name}: {load_time:.2f}s")
    print("=" * 50)

    # Perform inpainting with each model
    results = {}
    performance_logs = {}
    print(f"\nInpainting with prompt: '{args.prompt}'")
    print("=" * 50)

    for model_name, pipe in pipelines.items():
        print(f"\nRunning {model_name}...")
        try:
            result, perf = inpaint(
                pipe=pipe,
                image=image,
                mask=mask,
                prompt=args.prompt,
                negative_prompt=args.negative_prompt,
                num_inference_steps=args.steps,
                guidance_scale=args.guidance_scale,
                seed=args.seed,
            )
            results[model_name] = result
            performance_logs[model_name] = perf

            # Display performance metrics
            print(f"  Preprocessing: {perf['preprocessing_time']:.3f}s")
            print(f"  Inference: {perf['inference_time']:.2f}s")
            print(f"  Total: {perf['total_time']:.2f}s")
            print(f"  Image size: {perf['image_size']}")

            # Save individual result
            output_path = args.output
            if len(pipelines) > 1:
                # Add model name to filename if multiple models
                base, ext = os.path.splitext(args.output)
                output_path = f"{base}_{model_name}{ext}"

            result.save(output_path)
            print(f"✓ Result saved to '{output_path}'")

        except Exception as e:
            print(f"✗ Failed: {e}")

    if not results:
        print("\nError: All models failed to generate results!")
        return

    # Optional: Create comparison images
    if not args.no_comparison:
        # Create individual comparison for each model
        for model_name, result in results.items():
            comparison = Image.new("RGB", (image.width * 3, image.height))
            comparison.paste(image, (0, 0))
            comparison.paste(mask, (image.width, 0))
            comparison.paste(result, (image.width * 2, 0))

            comparison_path = f"comparison_{model_name}.png"
            comparison.save(comparison_path)
            print(f"Comparison saved to '{comparison_path}'")

        # Create grid comparison if multiple models
        if len(results) > 1:
            # Grid: Original | Mask | Model1 | Model2 | ...
            grid_width = (2 + len(results)) * image.width
            grid_height = image.height
            grid = Image.new("RGB", (grid_width, grid_height))

            # Paste original and mask
            grid.paste(image, (0, 0))
            grid.paste(mask, (image.width, 0))

            # Paste results
            x_offset = 2 * image.width
            for model_name, result in results.items():
                grid.paste(result, (x_offset, 0))

                # Add label
                draw = ImageDraw.Draw(grid)
                try:
                    font = ImageFont.truetype("Arial.ttf", 20)
                except Exception:
                    font = ImageFont.load_default()
                draw.text((x_offset + 10, 10), model_name, fill="white", font=font)

                x_offset += image.width

            grid.save("comparison_grid.png")
            print("Grid comparison saved to 'comparison_grid.png'")

    # Display performance summary
    if performance_logs:
        print(f"\n{'=' * 50}")
        print("Performance Summary")
        print("=" * 50)
        for model_name, perf in performance_logs.items():
            load_time = load_times.get(model_name, 0)
            steps_per_sec = perf["num_steps"] / perf["inference_time"] if perf["inference_time"] > 0 else 0

            print(f"\n{model_name}:")
            print(f"  Load time: {load_time:.2f}s")
            print(f"  Inference time: {perf['inference_time']:.2f}s")
            print(f"  Steps/sec: {steps_per_sec:.2f}")


if __name__ == "__main__":
    main()
