"""
Image Splitter Utility
======================
Resize and split images for testing the seamless collage tool.

This utility helps prepare test images by:
1. Resizing an image to a target width or height
2. Splitting it horizontally or vertically into two parts
3. Skipping a configurable number of pixels in the middle (to test inpainting quality)

Usage:
    # Split horizontally (left/right) with default 80px skip
    python image_splitter.py --input photo.jpg --split horizontal
    
    # Split vertically (top/bottom) with custom skip
    python image_splitter.py --input photo.jpg --split vertical --skip 100
    
    # Resize to specific width before splitting
    python image_splitter.py --input photo.jpg --width 1024 --split horizontal
    
    # Resize to specific height before splitting
    python image_splitter.py --input photo.jpg --height 512 --split vertical

Output:
    Creates two files: {input_name}_left.png, {input_name}_right.png (horizontal)
    Or: {input_name}_top.png, {input_name}_bottom.png (vertical)

Testing workflow:
    1. Take a good photo
    2. Split it with this tool (skipping some pixels in the middle)
    3. Run collage.py on the split parts
    4. Compare the AI-generated gap with the original skipped region
"""

import argparse
import os
from PIL import Image


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Resize and split images for collage testing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic horizontal split
    python image_splitter.py --input beach.jpg --split horizontal
    
    # Split with larger gap to skip
    python image_splitter.py --input landscape.jpg --split horizontal --skip 150
    
    # Resize to width first, then split
    python image_splitter.py --input large_photo.jpg --width 1920 --split horizontal
    
    # Vertical split for testing top/bottom blending
    python image_splitter.py --input tall_image.jpg --split vertical --skip 100
        """
    )
    
    parser.add_argument(
        "--input", "-i",
        type=str,
        required=True,
        help="Path to input image"
    )
    
    parser.add_argument(
        "--split", "-s",
        type=str,
        choices=["horizontal", "vertical", "h", "v"],
        default="horizontal",
        help="Split direction: horizontal (left/right) or vertical (top/bottom). Default: horizontal"
    )
    
    parser.add_argument(
        "--skip",
        type=int,
        default=80,
        help="Number of pixels to skip in the middle (default: 80). This creates the gap that will be inpainted."
    )
    
    parser.add_argument(
        "--width", "-w",
        type=int,
        default=None,
        help="Resize image to this width before splitting (maintains aspect ratio)"
    )
    
    parser.add_argument(
        "--height", "-H",
        type=int,
        default=None,
        help="Resize image to this height before splitting (maintains aspect ratio)"
    )
    
    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        default=None,
        help="Output directory for split images (default: same as input)"
    )
    
    parser.add_argument(
        "--prefix",
        type=str,
        default=None,
        help="Custom prefix for output files (default: input filename without extension)"
    )
    
    return parser.parse_args()


def resize_image(image: Image.Image, width: int = None, height: int = None) -> Image.Image:
    """
    Resize image to target width or height while preserving aspect ratio.
    
    Args:
        image: PIL Image to resize
        width: Target width (if specified, height is calculated)
        height: Target height (if specified, width is calculated)
        
    Returns:
        Resized PIL Image
    """
    if width is None and height is None:
        return image
    
    original_width, original_height = image.size
    aspect_ratio = original_width / original_height
    
    if width is not None and height is not None:
        # Both specified - use width and ignore height
        new_width = width
        new_height = int(width / aspect_ratio)
    elif width is not None:
        new_width = width
        new_height = int(width / aspect_ratio)
    else:  # height is not None
        new_height = height
        new_width = int(height * aspect_ratio)
    
    resized = image.resize((new_width, new_height), Image.LANCZOS)
    return resized


def split_horizontal(image: Image.Image, skip: int) -> tuple:
    """
    Split image horizontally (left and right parts) with a gap in the middle.
    
    Args:
        image: PIL Image to split
        skip: Number of pixels to skip in the middle
        
    Returns:
        Tuple of (left_image, right_image)
        
    Visual:
        |<--- left --->|<- skip ->|<--- right --->|
    """
    width, height = image.size
    
    # Calculate split point (middle of image)
    mid_x = width // 2
    half_skip = skip // 2
    
    # Left part: from 0 to (mid - half_skip)
    left_end = mid_x - half_skip
    left = image.crop((0, 0, left_end, height))
    
    # Right part: from (mid + half_skip) to end
    right_start = mid_x + half_skip + (skip % 2)  # Handle odd skip values
    right = image.crop((right_start, 0, width, height))
    
    return left, right


def split_vertical(image: Image.Image, skip: int) -> tuple:
    """
    Split image vertically (top and bottom parts) with a gap in the middle.
    
    Args:
        image: PIL Image to split
        skip: Number of pixels to skip in the middle
        
    Returns:
        Tuple of (top_image, bottom_image)
        
    Visual:
        ┌─────────┐
        │   top   │
        ├─────────┤
        │  skip   │
        ├─────────┤
        │ bottom  │
        └─────────┘
    """
    width, height = image.size
    
    # Calculate split point (middle of image)
    mid_y = height // 2
    half_skip = skip // 2
    
    # Top part: from 0 to (mid - half_skip)
    top_end = mid_y - half_skip
    top = image.crop((0, 0, width, top_end))
    
    # Bottom part: from (mid + half_skip) to end
    bottom_start = mid_y + half_skip + (skip % 2)  # Handle odd skip values
    bottom = image.crop((0, bottom_start, width, height))
    
    return top, bottom


def main():
    """Main function for the image splitter tool."""
    args = parse_args()
    
    # Validate input
    if not os.path.exists(args.input):
        print(f"Error: Input file not found: {args.input}")
        return 1
    
    print("=" * 50)
    print("Image Splitter Utility")
    print("=" * 50)
    
    # Load image
    print(f"\nLoading: {args.input}")
    image = Image.open(args.input).convert("RGB")
    print(f"  Original size: {image.size[0]}x{image.size[1]}")
    
    # Resize if requested
    if args.width or args.height:
        image = resize_image(image, args.width, args.height)
        print(f"  Resized to: {image.size[0]}x{image.size[1]}")
    
    # Validate skip size
    is_horizontal = args.split in ["horizontal", "h"]
    dimension = image.size[0] if is_horizontal else image.size[1]
    
    if args.skip >= dimension:
        print(f"Error: Skip value ({args.skip}) must be less than image {'width' if is_horizontal else 'height'} ({dimension})")
        return 1
    
    # Split the image
    print(f"\nSplitting {'horizontally' if is_horizontal else 'vertically'}...")
    print(f"  Skipping {args.skip} pixels in the middle")
    
    if is_horizontal:
        part1, part2 = split_horizontal(image, args.skip)
        suffix1, suffix2 = "_left", "_right"
    else:
        part1, part2 = split_vertical(image, args.skip)
        suffix1, suffix2 = "_top", "_bottom"
    
    print(f"  Part 1: {part1.size[0]}x{part1.size[1]}")
    print(f"  Part 2: {part2.size[0]}x{part2.size[1]}")
    
    # Determine output paths
    input_dir = os.path.dirname(args.input) or "."
    input_name = os.path.splitext(os.path.basename(args.input))[0]
    
    output_dir = args.output_dir or input_dir
    prefix = args.prefix or input_name
    
    # Create output directory if needed
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Save split images
    output1 = os.path.join(output_dir, f"{prefix}{suffix1}.png")
    output2 = os.path.join(output_dir, f"{prefix}{suffix2}.png")
    
    part1.save(output1)
    part2.save(output2)
    
    print(f"\nSaved:")
    print(f"  {output1}")
    print(f"  {output2}")
    
    # Print usage hint
    print(f"\n{'=' * 50}")
    print("Next step - test with collage.py:")
    print(f"{'=' * 50}")
    
    if is_horizontal:
        print(f"""
python collage.py \\
    --images "{output1}" "{output2}" \\
    --prompt "your scene description" \\
    --gap {args.skip}
""")
    else:
        print(f"""
Note: Current collage.py only supports horizontal blending.
For vertical blending, you'll need the v2 multi-layout feature.

To test horizontally, run:
python image_splitter.py --input {args.input} --split horizontal
""")
    
    return 0


if __name__ == "__main__":
    exit(main())
