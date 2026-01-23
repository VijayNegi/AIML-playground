"""
Inpainting Example using Stable Diffusion
==========================================
This script demonstrates how to use Stable Diffusion Inpainting to replace
parts of an image with AI-generated content based on a text prompt.

Requirements:
    pip install diffusers transformers accelerate torch pillow opencv-python

Usage:
    # Interactive mask drawing with single model:
    python test.py --input image.jpg --interactive --prompt "your prompt"
    
    # Use existing mask with multiple models:
    python test.py --input image.jpg --mask mask.png --models sd-1.5 sd-2 --prompt "your prompt"
    
    # Use all available models:
    python test.py --input image.jpg --mask mask.png --models all --prompt "your prompt"
"""

from PIL import Image, ImageDraw, ImageFont
import os

from inpainting_utils import (
    load_pipelines, 
    create_mask_interactive,
    inpaint, 
    parse_args
)


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
    pipelines = load_pipelines(args.models)
    
    if not pipelines:
        print("Error: No models were successfully loaded!")
        return
    
    print(f"\n✓ Loaded {len(pipelines)} model(s)")
    print("=" * 50)
    
    # Perform inpainting with each model
    results = {}
    print(f"\nInpainting with prompt: '{args.prompt}'")
    print("=" * 50)
    
    for model_name, pipe in pipelines.items():
        print(f"\nRunning {model_name}...")
        try:
            result = inpaint(
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


if __name__ == "__main__":
    main()
