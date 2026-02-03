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
import time
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
        Tuple of (pipeline, load_time_seconds)
    """
    start_time = time.time()
    
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
        safety_checker=None,  # Disable safety checker to avoid CPU fallback
        use_safetensors=True,  # Use safetensors to bypass PyTorch 2.6 requirement
    )
    pipe = pipe.to(device)
    
    # Ensure all components are on the correct device
    if device == "cuda":
        # Explicitly move all submodules to GPU
        if hasattr(pipe, 'vae'):
            pipe.vae = pipe.vae.to(device)
        if hasattr(pipe, 'text_encoder'):
            pipe.text_encoder = pipe.text_encoder.to(device)
        if hasattr(pipe, 'unet'):
            pipe.unet = pipe.unet.to(device)
        
        # Enable memory efficient attention if available
        try:
            pipe.enable_xformers_memory_efficient_attention()
            print("  ✓ xformers memory efficient attention enabled")
        except Exception as e:
            # xformers not available or incompatible version - this is fine
            # The model will use standard attention instead (10-20% slower but works)
            print(f"  ℹ xformers not available, using standard attention (slightly slower)")
        
        # Enable attention slicing for lower memory usage
        try:
            pipe.enable_attention_slicing(1)
            print("  ✓ Attention slicing enabled")
        except Exception:
            pass
    
    load_time = time.time() - start_time
    print(f"✓ Loaded in {load_time:.2f}s")
    
    return pipe, load_time


def load_pipelines(model_names: list):
    """
    Load multiple inpainting pipelines.
    
    Args:
        model_names: List of model names (keys from AVAILABLE_MODELS) or ["all"]
        
    Returns:
        Tuple of (pipelines_dict, load_times_dict)
    """
    pipelines = {}
    load_times = {}
    
    # Handle "all" keyword
    if "all" in model_names:
        model_names = list(AVAILABLE_MODELS.keys())
    
    for model_name in model_names:
        if model_name not in AVAILABLE_MODELS:
            print(f"Warning: Unknown model '{model_name}', skipping...")
            continue
        
        model_id = AVAILABLE_MODELS[model_name]
        try:
            pipe, load_time = load_pipeline(model_id)
            pipelines[model_name] = pipe
            load_times[model_name] = load_time
            print(f"✓ Loaded {model_name}")
        except Exception as e:
            print(f"✗ Failed to load {model_name}: {e}")
    
    return pipelines, load_times


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


def normalize_image_heights(
    images: list,
    target_height: int = None,
) -> list:
    """
    Resize multiple images to have the same height while preserving aspect ratios.
    
    This is essential for horizontal collages where images need to align vertically.
    Each image is scaled proportionally so heights match, widths vary based on
    original aspect ratios.
    
    Args:
        images: List of PIL Image objects
        target_height: Desired height in pixels. If None, uses the minimum height
                      among all images to avoid upscaling.
                      
    Returns:
        List of resized PIL Images, all with the same height
        
    Example:
        >>> img1 = Image.open("photo1.jpg")  # 800x600
        >>> img2 = Image.open("photo2.jpg")  # 1200x900
        >>> normalized = normalize_image_heights([img1, img2], target_height=400)
        >>> # img1 becomes 533x400, img2 becomes 533x400
    """
    if not images:
        return []
    
    # Determine target height if not specified
    if target_height is None:
        target_height = min(img.height for img in images)
    
    resized_images = []
    for img in images:
        # Calculate new width to preserve aspect ratio
        aspect_ratio = img.width / img.height
        new_width = int(target_height * aspect_ratio)
        
        # Resize using high-quality LANCZOS resampling
        resized = img.resize((new_width, target_height), Image.LANCZOS)
        resized_images.append(resized)
    
    return resized_images


def create_blend_mask(
    canvas_size: tuple,
    gap_x_start: int,
    gap_width: int,
    feather: int = 30,
) -> Image.Image:
    """
    Create a gradient blend mask for the gap region between two images.
    
    The mask uses feathering (soft edges) to create smooth transitions.
    This helps the AI inpainting blend naturally with surrounding pixels
    rather than creating hard seams.
    
    Mask values:
        - White (255): Areas to be inpainted (the gap)
        - Black (0): Areas to keep unchanged
        - Gray gradients: Feathered transition zones
    
    Args:
        canvas_size: Tuple of (width, height) for the full canvas
        gap_x_start: X coordinate where the gap begins
        gap_width: Width of the gap in pixels
        feather: Size of the gradient feather on each edge (default: 30px)
        
    Returns:
        PIL Image mask in RGB mode (white=inpaint, black=keep)
        
    Visual representation:
        |  Image 1  |<-feather->|  GAP  |<-feather->|  Image 2  |
        |  BLACK    |  GRADIENT | WHITE |  GRADIENT |  BLACK    |
    """
    width, height = canvas_size
    
    # Create a grayscale mask
    mask = Image.new("L", canvas_size, 0)  # Start with all black (keep)
    
    # Convert to numpy for easier gradient creation
    mask_np = np.array(mask, dtype=np.float32)
    
    # Define the gap region boundaries
    gap_x_end = gap_x_start + gap_width
    
    # Left feather zone: gradient from black to white
    feather_left_start = max(0, gap_x_start - feather)
    feather_left_end = gap_x_start
    
    # Right feather zone: gradient from white to black  
    feather_right_start = gap_x_end
    feather_right_end = min(width, gap_x_end + feather)
    
    # Fill the solid white gap region
    mask_np[:, gap_x_start:gap_x_end] = 255
    
    # Create left feather gradient (0 -> 255)
    if feather_left_end > feather_left_start:
        for x in range(feather_left_start, feather_left_end):
            # Linear interpolation from 0 to 255
            t = (x - feather_left_start) / (feather_left_end - feather_left_start)
            mask_np[:, x] = int(255 * t)
    
    # Create right feather gradient (255 -> 0)
    if feather_right_end > feather_right_start:
        for x in range(feather_right_start, feather_right_end):
            # Linear interpolation from 255 to 0
            t = (x - feather_right_start) / (feather_right_end - feather_right_start)
            mask_np[:, x] = int(255 * (1 - t))
    
    # Convert back to PIL Image and then to RGB (required by inpainting pipeline)
    mask = Image.fromarray(mask_np.astype(np.uint8), mode="L")
    mask_rgb = mask.convert("RGB")
    
    return mask_rgb


def get_crop_region_around_mask(
    mask: Image.Image,
    max_size: int = 1024,
) -> tuple:
    """
    Find a crop region around the mask, limited to max_size.
    
    The crop is centered on the mask and expanded to include context,
    but never exceeds max_size in either dimension.
    
    Args:
        mask: Mask image (white = areas to inpaint)
        max_size: Maximum width/height of the crop (default 1024 for SDXL)
        
    Returns:
        Tuple of (x1, y1, x2, y2) crop coordinates, or None if no mask found
    """
    # Convert to grayscale numpy array
    mask_gray = mask.convert("L")
    mask_np = np.array(mask_gray)
    
    # Find non-zero (white) pixels
    white_pixels = np.where(mask_np > 127)
    
    if len(white_pixels[0]) == 0:
        return None
    
    # Get bounding box of the mask
    y_min, y_max = white_pixels[0].min(), white_pixels[0].max()
    x_min, x_max = white_pixels[1].min(), white_pixels[1].max()
    
    mask_width = x_max - x_min
    mask_height = y_max - y_min
    mask_center_x = (x_min + x_max) // 2
    mask_center_y = (y_min + y_max) // 2
    
    img_width, img_height = mask.size
    
    # Determine crop size (max_size or image size, whichever is smaller)
    crop_width = min(max_size, img_width)
    crop_height = min(max_size, img_height)
    
    # If mask is larger than max_size, we have a problem - warn but proceed
    if mask_width > crop_width or mask_height > crop_height:
        print(f"  Warning: Mask region ({mask_width}x{mask_height}) exceeds max crop size ({crop_width}x{crop_height})")
    
    # Center the crop on the mask center
    x1 = mask_center_x - crop_width // 2
    y1 = mask_center_y - crop_height // 2
    x2 = x1 + crop_width
    y2 = y1 + crop_height
    
    # Clamp to image boundaries
    if x1 < 0:
        x1 = 0
        x2 = crop_width
    if y1 < 0:
        y1 = 0
        y2 = crop_height
    if x2 > img_width:
        x2 = img_width
        x1 = max(0, img_width - crop_width)
    if y2 > img_height:
        y2 = img_height
        y1 = max(0, img_height - crop_height)
    
    return (x1, y1, x2, y2)


def inpaint(
    pipe,
    image: Image.Image,
    mask: Image.Image,
    prompt: str,
    negative_prompt: str = "blurry, low quality, distorted",
    num_inference_steps: int = 50,
    guidance_scale: float = 7.5,
    seed: int = None,
    strength: float = 0.99,
):
    """
    Perform inpainting on an image with performance logging.
    
    Args:
        pipe: The inpainting pipeline
        image: Input image to inpaint
        mask: Mask image (white = areas to replace, black = keep)
        prompt: Text description of what to generate
        negative_prompt: What to avoid in generation
        num_inference_steps: Number of denoising steps (higher = better quality)
        guidance_scale: How closely to follow the prompt (7-12 recommended)
        seed: Random seed for reproducibility
        strength: How much to transform the masked region (0.0-1.0).
                  Use <1.0 for SDXL to avoid quality issues (default 0.99)
        
    Returns:
        Tuple of (inpainted_image, performance_dict)
    """
    perf = {}
    total_start = time.time()
    
    # Set seed for reproducibility
    generator = None
    if seed is not None:
        generator = torch.Generator(device=pipe.device).manual_seed(seed)
    
    # Ensure images are in RGB mode and correct size
    # Preprocessing
    prep_start = time.time()
    image = image.convert("RGB")
    mask = mask.convert("RGB")
    
    # Resize to be divisible by 8 (required by SD)
    # Round UP to preserve image content (avoid cutting off edges)
    width, height = image.size
    new_width = ((width + 7) // 8) * 8
    new_height = ((height + 7) // 8) * 8
    
    if (new_width, new_height) != (width, height):
        image = image.resize((new_width, new_height), Image.LANCZOS)
        mask = mask.resize((new_width, new_height), Image.NEAREST)
    
    perf['preprocessing_time'] = time.time() - prep_start
    
    # Run inpainting
    inference_start = time.time()
    result = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        image=image,
        mask_image=mask,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        strength=strength,  # Important for SDXL: use <1.0 to avoid quality issues
        generator=generator,
    ).images[0]
    perf['inference_time'] = time.time() - inference_start
    
    perf['total_time'] = time.time() - total_start
    perf['image_size'] = f"{new_width}x{new_height}"
    perf['num_steps'] = num_inference_steps
    perf['guidance_scale'] = guidance_scale
    perf['strength'] = strength
    
    return result, perf


def inpaint_crop_and_patch(
    pipe,
    image: Image.Image,
    mask: Image.Image,
    prompt: str,
    negative_prompt: str = "blurry, low quality, distorted",
    num_inference_steps: int = 50,
    guidance_scale: float = 7.5,
    seed: int = None,
    max_crop_size: int = 1024,
    strength: float = 0.99,
):
    """
    Crop the region around the mask (max 1024x1024), inpaint it, and paste back.
    
    No resizing is performed - the crop is taken at original resolution and
    pasted back directly. This is ideal for SDXL which works best at 1024x1024.
    
    Workflow:
    1. Find the mask region and crop a max 1024x1024 area centered on it
    2. Run inpainting on this crop (at original resolution, no scaling)
    3. Paste the result back into the original image
    
    Args:
        pipe: The inpainting pipeline
        image: Input image (can be any size)
        mask: Mask image (white = areas to replace, black = keep)
        prompt: Text description of what to generate
        negative_prompt: What to avoid in generation
        num_inference_steps: Number of denoising steps
        guidance_scale: How closely to follow the prompt
        seed: Random seed for reproducibility
        max_crop_size: Maximum crop dimension (default 1024 for SDXL)
        strength: How much to transform the masked region (0.0-1.0). 
                  Use <1.0 for SDXL to avoid quality issues (default 0.99)
        
    Returns:
        Tuple of (inpainted_image, performance_dict)
    """
    perf = {}
    total_start = time.time()
    
    # Set seed for reproducibility
    generator = None
    if seed is not None:
        generator = torch.Generator(device=pipe.device).manual_seed(seed)
    
    # Ensure images are in RGB mode
    prep_start = time.time()
    image = image.convert("RGB")
    #mask = mask.convert("RGB")
    original_size = image.size
    
    # Find crop region around the mask
    bbox = get_crop_region_around_mask(mask, max_size=max_crop_size)
    
    if bbox is None:
        print("  Warning: No mask region found, returning original image")
        perf['total_time'] = time.time() - total_start
        perf['error'] = "No mask region found"
        return image, perf
    
    x1, y1, x2, y2 = bbox
    crop_width = x2 - x1
    crop_height = y2 - y1
    
    print(f"  Original image: {original_size[0]}x{original_size[1]}")
    print(f"  Crop region: ({x1}, {y1}) to ({x2}, {y2}) = {crop_width}x{crop_height}")
    
    # Crop the image and mask (no resizing!)
    crop_image = image.crop(bbox)
    crop_mask = mask.crop(bbox)
    
    # Make divisible by 8 for SD (minimal adjustment)
    adj_width = ((crop_width + 7) // 8) * 8
    adj_height = ((crop_height + 7) // 8) * 8
    
    if (adj_width, adj_height) != (crop_width, crop_height):
        # Expand crop slightly to be divisible by 8
        # Prefer expanding over shrinking to keep all mask content
        new_x2 = min(original_size[0], x1 + adj_width)
        new_y2 = min(original_size[1], y1 + adj_height)
        new_x1 = max(0, new_x2 - adj_width)
        new_y1 = max(0, new_y2 - adj_height)
        
        bbox = (new_x1, new_y1, new_x2, new_y2)
        crop_image = image.crop(bbox)
        crop_mask = mask.crop(bbox)
        crop_width = new_x2 - new_x1
        crop_height = new_y2 - new_y1
        print(f"  Adjusted for SD: {crop_width}x{crop_height}")
    
    perf['preprocessing_time'] = time.time() - prep_start
    perf['crop_bbox'] = bbox
    perf['crop_size'] = f"{crop_width}x{crop_height}"
    
    # Debug: Save crops for inspection
    crop_image.save("debug_crop_image.png")
    crop_mask.save("debug_crop_mask.png")
    print(f"  Debug: Saved debug_crop_image.png and debug_crop_mask.png")
    
    # Run inpainting on the crop
    inference_start = time.time()
    result = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        image=crop_image,
        mask_image=crop_mask,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        strength=strength,  # Important for SDXL: use <1.0 to avoid quality issues
        generator=generator,
    ).images[0]
    perf['inference_time'] = time.time() - inference_start
    perf['strength'] = strength
    
    # Debug: Save model output
    result.save("debug_crop_result.png")
    print(f"  Debug: Saved debug_crop_result.png")
    patch_start = time.time()
    final_result = image.copy()
    final_result.paste(result, (bbox[0], bbox[1]))
    perf['patch_time'] = time.time() - patch_start
    
    perf['total_time'] = time.time() - total_start
    perf['original_size'] = f"{original_size[0]}x{original_size[1]}"
    perf['num_steps'] = num_inference_steps
    perf['guidance_scale'] = guidance_scale
    
    print(f"  Inpainting complete: {perf['inference_time']:.2f}s inference")
    
    return final_result, perf
