"""
Quick GPU Test for Inpainting
==============================
A minimal test to verify GPU is being used properly.

Usage:
    python quick_gpu_test.py
"""

import torch
from PIL import Image
import time

def quick_test():
    """Quick test to verify GPU usage."""
    
    print("\n" + "=" * 60)
    print("QUICK GPU TEST")
    print("=" * 60)
    
    # Check CUDA
    if not torch.cuda.is_available():
        print("❌ CUDA not available!")
        print("\nInstall PyTorch with CUDA:")
        print("pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121")
        return False
    
    print(f"✓ GPU: {torch.cuda.get_device_name(0)}")
    print(f"✓ CUDA: {torch.version.cuda}")
    print(f"✓ PyTorch: {torch.__version__}")
    
    # Load model
    print("\nLoading model (this may take a minute)...")
    from diffusers import AutoPipelineForInpainting
    
    start = time.time()
    pipe = AutoPipelineForInpainting.from_pretrained(
        "runwayml/stable-diffusion-inpainting",
        torch_dtype=torch.float16,
        safety_checker=None,
    )
    
    # Move to GPU
    pipe = pipe.to('cuda')
    
    # Verify components are on GPU
    print(f"\nChecking components:")
    print(f"  VAE: {pipe.vae.device}")
    print(f"  UNet: {pipe.unet.device}")
    print(f"  Text Encoder: {pipe.text_encoder.device}")
    
    load_time = time.time() - start
    print(f"\n✓ Model loaded in {load_time:.1f}s")
    
    # Check memory
    mem_gb = torch.cuda.memory_allocated(0) / 1024**3
    print(f"✓ GPU Memory: {mem_gb:.2f} GB")
    
    # Create test images
    print("\nCreating test images...")
    image = Image.new("RGB", (512, 512), color='white')
    mask = Image.new("RGB", (512, 512), color='black')
    
    # Draw a circle in the mask
    from PIL import ImageDraw
    draw = ImageDraw.Draw(mask)
    draw.ellipse([200, 200, 312, 312], fill='white')
    
    # Run inference
    print("\nRunning inference (5 steps for quick test)...")
    print("👀 NOW CHECK nvidia-smi IN ANOTHER TERMINAL!")
    print("   You should see GPU-Util at 80-100%")
    
    start = time.time()
    with torch.inference_mode():
        result = pipe(
            prompt="a red ball",
            image=image,
            mask_image=mask,
            num_inference_steps=5,
            guidance_scale=7.5,
        ).images[0]
    
    inference_time = time.time() - start
    print(f"\n✓ Inference completed in {inference_time:.2f}s")
    
    # Save result
    result.save("quick_test_output.png")
    print(f"✓ Result saved to 'quick_test_output.png'")
    
    print("\n" + "=" * 60)
    print("TEST PASSED!")
    print("=" * 60)
    print("\nIf GPU-Util was 0% during inference, check:")
    print("1. Run the full diagnostic: python gpu_diagnostic.py")
    print("2. Reinstall PyTorch with CUDA support")
    print("3. Check GPU_TROUBLESHOOTING.md for detailed guide")
    
    return True


if __name__ == "__main__":
    try:
        quick_test()
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
