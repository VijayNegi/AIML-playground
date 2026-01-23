"""
Inpainting Utilities
====================
Core functions for AI-powered inpainting using various models.
Supports Stable Diffusion, Kandinsky, and other inpainting models.

Requirements:
    pip install diffusers transformers accelerate torch pillow opencv-python
"""

import argparse
import cv2
import numpy as np
import torch
from diffusers import AutoPipelineForInpainting
from PIL import Image, ImageDraw


# Available inpainting models
# AutoPipelineForInpainting automatically detects and loads the correct pipeline class
AVAILABLE_MODELS = {
    "sd-1.5": "runwayml/stable-diffusion-inpainting",
    "sd-2": "sd2-community/stable-diffusion-2-inpainting",
    "sd-xl": "diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
    "kandinsky-2.2": "kandinsky-community/kandinsky-2-2-decoder-inpaint",
}


def get_available_models():
    """Return dictionary of available models."""
    return AVAILABLE_MODELS


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


def load_pipeline(model_id: str = "runwayml/stable-diffusion-inpainting"):
    """
    Load an inpainting pipeline with auto-detection of the correct pipeline class.
    
    Args:
        model_id: HuggingFace model identifier
        
    Returns:
        Configured inpainting pipeline
    """
    # Determine device and dtype
    if torch.cuda.is_available():
        device = "cuda"
        dtype = torch.float16
    elif torch.backends.mps.is_available():
        device = "mps"
        dtype = torch.float16
    else:
        device = "cpu"
        dtype = torch.float32
    
    print(f"Loading model '{model_id}' on {device}...")
    
    # Use AutoPipelineForInpainting to automatically detect the correct pipeline class
    pipe = AutoPipelineForInpainting.from_pretrained(
        model_id,
        torch_dtype=dtype,
    )
    pipe = pipe.to(device)
    
    # Enable memory efficient attention if available
    if device == "cuda":
        try:
            pipe.enable_xformers_memory_efficient_attention()
        except Exception:
            pass  # xformers not available
    
    return pipe


def load_pipelines(model_names: list):
    """
    Load multiple inpainting pipelines.
    
    Args:
        model_names: List of model names (keys from AVAILABLE_MODELS) or ["all"]
        
    Returns:
        Dictionary mapping model names to their pipelines
    """
    pipelines = {}
    
    # Handle "all" keyword
    if "all" in model_names:
        model_names = list(AVAILABLE_MODELS.keys())
    
    for model_name in model_names:
        if model_name not in AVAILABLE_MODELS:
            print(f"Warning: Unknown model '{model_name}', skipping...")
            continue
        
        model_id = AVAILABLE_MODELS[model_name]
        try:
            pipelines[model_name] = load_pipeline(model_id)
            print(f"✓ Loaded {model_name}")
        except Exception as e:
            print(f"✗ Failed to load {model_name}: {e}")
    
    return pipelines


def create_sample_mask(image: Image.Image, mask_region: tuple) -> Image.Image:
    """
    Create a simple rectangular mask for demonstration.
    
    Args:
        image: The input image
        mask_region: Tuple of (x1, y1, x2, y2) defining the mask area
        
    Returns:
        PIL Image mask (white = inpaint, black = keep)
    """
    mask = Image.new("RGB", image.size, "black")
    draw = ImageDraw.Draw(mask)
    draw.rectangle(mask_region, fill="white")
    return mask


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


def inpaint(
    pipe,
    image: Image.Image,
    mask: Image.Image,
    prompt: str,
    negative_prompt: str = "blurry, low quality, distorted",
    num_inference_steps: int = 50,
    guidance_scale: float = 7.5,
    seed: int = None,
) -> Image.Image:
    """
    Perform inpainting on an image.
    
    Args:
        pipe: The inpainting pipeline
        image: Input image to inpaint
        mask: Mask image (white = areas to replace, black = keep)
        prompt: Text description of what to generate
        negative_prompt: What to avoid in generation
        num_inference_steps: Number of denoising steps (higher = better quality)
        guidance_scale: How closely to follow the prompt (7-12 recommended)
        seed: Random seed for reproducibility
        
    Returns:
        Inpainted image
    """
    # Set seed for reproducibility
    generator = None
    if seed is not None:
        generator = torch.Generator(device=pipe.device).manual_seed(seed)
    
    # Ensure images are in RGB mode and correct size
    image = image.convert("RGB")
    mask = mask.convert("RGB")
    
    # Resize to be divisible by 8 (required by SD)
    width, height = image.size
    new_width = (width // 8) * 8
    new_height = (height // 8) * 8
    
    if (new_width, new_height) != (width, height):
        image = image.resize((new_width, new_height), Image.LANCZOS)
        mask = mask.resize((new_width, new_height), Image.NEAREST)
    
    # Run inpainting
    result = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        image=image,
        mask_image=mask,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        generator=generator,
    ).images[0]
    
    return result
