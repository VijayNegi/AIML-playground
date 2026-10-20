"""Collage canvas layout and feathered blend masks."""

import numpy as np
from PIL import Image


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
