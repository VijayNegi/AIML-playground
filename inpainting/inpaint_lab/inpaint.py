"""
Inpainting
==========
Crop-and-patch inference around a mask, with optional denoising-step capture.
Model registry and loading live in models.py; canvas and masks in canvas.py.
"""

import numpy as np
from pathlib import Path
import time
import torch
from PIL import Image


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
    y_min, y_max = int(white_pixels[0].min()), int(white_pixels[0].max())
    x_min, x_max = int(white_pixels[1].min()), int(white_pixels[1].max())
    
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


def create_step_callback(output_dir: str = "intermediate_steps", save_every: int = 1):
    """
    Create a callback function to save intermediate images during diffusion.
    
    The callback decodes the latents at each step and saves them as images,
    allowing you to visualize the denoising process.
    
    Args:
        output_dir: Directory to save intermediate images
        save_every: Save every N steps (1 = every step, 5 = every 5th step)
        
    Returns:
        Callback function compatible with diffusers pipelines
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    def callback_fn(pipe, step: int, timestep: int, callback_kwargs: dict):
        """
        Callback called at each denoising step.
        
        Args:
            pipe: The pipeline instance
            step: Current step number (0-indexed)
            timestep: Current timestep value from the scheduler
            callback_kwargs: Dict containing 'latents' and other state
            
        Returns:
            callback_kwargs (required by diffusers)
        """
        if step % save_every != 0:
            return callback_kwargs
        
        latents = callback_kwargs["latents"]
        
        # Decode latents to image
        # Scale latents (required for VAE decoding)
        with torch.no_grad():
            # Get the VAE scaling factor
            scaling_factor = pipe.vae.config.scaling_factor
            scaled_latents = latents / scaling_factor
            
            # Decode to image
            decoded = pipe.vae.decode(scaled_latents, return_dict=False)[0]
            
            # Convert to PIL Image
            # Output is in range [-1, 1], convert to [0, 255]
            decoded = (decoded / 2 + 0.5).clamp(0, 1)
            decoded = decoded.cpu().permute(0, 2, 3, 1).numpy()
            decoded = (decoded * 255).round().astype(np.uint8)
            
            # Save the image
            for i, img_array in enumerate(decoded):
                img = Image.fromarray(img_array)
                img.save(os.path.join(output_dir, f"step_{step:04d}_t{timestep}.png"))
                print(f"  Saved intermediate: step_{step:04d}_t{timestep}.png")
        
        return callback_kwargs
    
    return callback_fn


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
    save_intermediate: bool = False,
    intermediate_dir: str = "intermediate_steps",
    save_every: int = 1,
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
        save_intermediate: If True, save images at each denoising step
        intermediate_dir: Directory to save intermediate images
        save_every: Save every N steps (1 = every step, 5 = every 5th step)
        
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
    
    # Setup callback for intermediate images
    callback = None
    if save_intermediate:
        callback = create_step_callback(intermediate_dir, save_every)
        print(f"  Saving intermediate images to: {intermediate_dir}/")
    
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
        callback_on_step_end=callback,
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
    save_intermediate: bool = False,
    intermediate_dir: str = "intermediate_steps",
    save_every: int = 1,
    debug_dir=".",
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
        save_intermediate: If True, save images at each denoising step
        intermediate_dir: Directory to save intermediate images
        save_every: Save every N steps (1 = every step, 5 = every 5th step)
        debug_dir: Directory for crop input, mask, and result debug images
        
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
    perf['mask_binarized'] = bool(getattr(getattr(getattr(pipe, 'mask_processor', None), 'config', None), 'do_binarize', False))
    perf['crop_size'] = f"{crop_width}x{crop_height}"
    
    # Debug: Save crops for inspection
    debug_dir = Path(debug_dir)
    debug_dir.mkdir(parents=True, exist_ok=True)
    crop_image.save(debug_dir / "debug_crop_image.png")
    crop_mask.save(debug_dir / "debug_crop_mask.png")
    print(f"  Debug: Saved crop image and mask in: {debug_dir}")
    
    # Setup callback for intermediate images
    callback = None
    if save_intermediate:
        callback = create_step_callback(intermediate_dir, save_every)
        print(f"  Saving intermediate images to: {intermediate_dir}/")
    
    # Run inpainting on the crop
    # IMPORTANT: Specify width/height to prevent SDXL from resizing to 1024x1024
    inference_start = time.time()
    result = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        image=crop_image,
        mask_image=crop_mask,
        width=crop_width,
        height=crop_height,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        strength=strength,  # Important for SDXL: use <1.0 to avoid quality issues
        generator=generator,
        callback_on_step_end=callback,
    ).images[0]
    perf['inference_time'] = time.time() - inference_start
    perf['strength'] = strength
    
    # Debug: Save model output
    result.save(debug_dir / "debug_crop_result.png")
    print(f"  Debug: Saved {debug_dir / 'debug_crop_result.png'} (size: {result.size})")
    
    # Ensure result matches crop size (safety check)
    if result.size != (crop_width, crop_height):
        print(f"  Warning: Result size {result.size} != crop size ({crop_width}, {crop_height}), resizing...")
        result = result.resize((crop_width, crop_height), Image.LANCZOS)
    
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
