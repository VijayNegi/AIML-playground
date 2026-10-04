"""
GPU Diagnostic Script
=====================
This script helps diagnose why GPU usage is zero while memory is allocated.
Run this to identify the issue before running the main inpainting script.

Usage:
    python gpu_diagnostic.py
"""

import torch
import time
import sys

def check_basic_cuda():
    """Check basic CUDA availability."""
    print("\n" + "=" * 70)
    print("STEP 1: Basic CUDA Check")
    print("=" * 70)
    
    if not torch.cuda.is_available():
        print("❌ CUDA is NOT available!")
        print("\nPossible causes:")
        print("1. PyTorch CPU-only version installed")
        print("2. NVIDIA drivers not installed")
        print("3. CUDA toolkit mismatch")
        print("\nSolution (works with CUDA 12.x and 13.x drivers):")
        print("pip uninstall torch torchvision torchaudio")
        print("pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")
        print("\nNote: PyTorch CUDA 12.1 works with CUDA driver 12.x, 13.x, and newer")
        return False
    
    print(f"✓ CUDA available: {torch.cuda.is_available()}")
    print(f"✓ CUDA version: {torch.version.cuda}")
    print(f"✓ PyTorch version: {torch.__version__}")
    print(f"✓ cuDNN version: {torch.backends.cudnn.version()}")
    print(f"✓ Device count: {torch.cuda.device_count()}")
    print(f"✓ Current device: {torch.cuda.current_device()}")
    print(f"✓ Device name: {torch.cuda.get_device_name(0)}")
    print(f"✓ Device capability: {torch.cuda.get_device_capability(0)}")
    
    return True


def check_memory():
    """Check GPU memory status."""
    print("\n" + "=" * 70)
    print("STEP 2: GPU Memory Status")
    print("=" * 70)
    
    allocated = torch.cuda.memory_allocated(0) / 1024**3
    reserved = torch.cuda.memory_reserved(0) / 1024**3
    total = torch.cuda.get_device_properties(0).total_memory / 1024**3
    
    print(f"Total memory: {total:.2f} GB")
    print(f"Allocated: {allocated:.2f} GB ({allocated/total*100:.1f}%)")
    print(f"Reserved: {reserved:.2f} GB ({reserved/total*100:.1f}%)")
    print(f"Free: {total - reserved:.2f} GB")


def test_simple_computation():
    """Test a simple GPU computation to verify it's working."""
    print("\n" + "=" * 70)
    print("STEP 3: Simple GPU Computation Test")
    print("=" * 70)
    
    try:
        # Create tensors on GPU
        print("Creating tensors on GPU...")
        x = torch.randn(1000, 1000, device='cuda')
        y = torch.randn(1000, 1000, device='cuda')
        
        # Warmup
        _ = torch.matmul(x, y)
        torch.cuda.synchronize()
        
        # Timed computation
        print("Running matrix multiplication...")
        start = time.time()
        for _ in range(10):
            z = torch.matmul(x, y)
        torch.cuda.synchronize()
        elapsed = time.time() - start
        
        print(f"✓ Completed 10 matrix multiplications in {elapsed:.3f}s")
        print(f"✓ Average time per operation: {elapsed/10*1000:.1f}ms")
        
        # Check memory again
        allocated = torch.cuda.memory_allocated(0) / 1024**3
        print(f"✓ GPU memory used: {allocated:.3f} GB")
        
        return True
        
    except Exception as e:
        print(f"❌ GPU computation failed: {e}")
        return False


def test_model_loading():
    """Test loading a small model to GPU."""
    print("\n" + "=" * 70)
    print("STEP 4: Model Loading Test")
    print("=" * 70)
    
    try:
        # Create a simple neural network
        print("Creating a simple neural network...")
        model = torch.nn.Sequential(
            torch.nn.Linear(512, 1024),
            torch.nn.ReLU(),
            torch.nn.Linear(1024, 512)
        )
        
        # Move to GPU
        print("Moving model to GPU...")
        model = model.to('cuda')
        
        # Check if model is on GPU
        is_on_gpu = next(model.parameters()).is_cuda
        print(f"✓ Model on GPU: {is_on_gpu}")
        
        # Test inference
        print("Running inference...")
        x = torch.randn(32, 512, device='cuda')
        
        start = time.time()
        with torch.no_grad():
            for _ in range(100):
                y = model(x)
        torch.cuda.synchronize()
        elapsed = time.time() - start
        
        print(f"✓ Completed 100 forward passes in {elapsed:.3f}s")
        print(f"✓ Average time per pass: {elapsed/100*1000:.1f}ms")
        
        # Memory check
        allocated = torch.cuda.memory_allocated(0) / 1024**3
        print(f"✓ GPU memory used: {allocated:.3f} GB")
        
        return True
        
    except Exception as e:
        print(f"❌ Model test failed: {e}")
        return False


def check_diffusers():
    """Check if diffusers can use GPU properly."""
    print("\n" + "=" * 70)
    print("STEP 5: Diffusers GPU Test (This may take a while...)")
    print("=" * 70)
    
    try:
        from diffusers import AutoPipelineForInpainting
        print("✓ Diffusers imported successfully")
        
        # Try to load a small model
        print("Loading a small inpainting model...")
        print("(This will download ~1.7GB if not cached)")
        
        model_id = "runwayml/stable-diffusion-inpainting"
        
        start = time.time()
        pipe = AutoPipelineForInpainting.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            safety_checker=None,  # Disable safety checker
        )
        load_time = time.time() - start
        print(f"✓ Model loaded in {load_time:.1f}s")
        
        # Move to GPU
        print("Moving pipeline to GPU...")
        start = time.time()
        pipe = pipe.to('cuda')
        move_time = time.time() - start
        print(f"✓ Moved to GPU in {move_time:.1f}s")
        
        # Check each component
        print("\nChecking pipeline components:")
        components = ['vae', 'text_encoder', 'unet']
        for comp_name in components:
            if hasattr(pipe, comp_name):
                comp = getattr(pipe, comp_name)
                if hasattr(comp, 'device'):
                    device = comp.device
                    print(f"  ✓ {comp_name}: {device}")
                else:
                    # Check parameters
                    try:
                        device = next(comp.parameters()).device
                        print(f"  ✓ {comp_name}: {device}")
                    except:
                        print(f"  ? {comp_name}: Unknown device")
        
        # Memory check
        allocated = torch.cuda.memory_allocated(0) / 1024**3
        reserved = torch.cuda.memory_reserved(0) / 1024**3
        print(f"\n✓ GPU memory allocated: {allocated:.2f} GB")
        print(f"✓ GPU memory reserved: {reserved:.2f} GB")
        
        # Try a simple inference
        print("\nRunning a quick inference test...")
        from PIL import Image
        
        # Create a small test image
        image = Image.new("RGB", (512, 512), color='blue')
        mask = Image.new("RGB", (512, 512), color='black')
        
        start = time.time()
        with torch.inference_mode():
            result = pipe(
                prompt="red circle",
                image=image,
                mask_image=mask,
                num_inference_steps=5,  # Very few steps for quick test
                guidance_scale=7.5,
            ).images[0]
        inference_time = time.time() - start
        
        print(f"✓ Inference completed in {inference_time:.2f}s")
        print(f"✓ Result size: {result.size}")
        
        return True
        
    except ImportError:
        print("❌ Diffusers not installed!")
        print("Install with: pip install diffusers transformers accelerate")
        return False
    except Exception as e:
        print(f"❌ Diffusers test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all diagnostic checks."""
    print("\n" + "=" * 70)
    print("GPU DIAGNOSTIC TOOL FOR STABLE DIFFUSION INPAINTING")
    print("=" * 70)
    print("This tool will help identify why GPU usage is zero.")
    print("Please wait while we run the diagnostics...\n")
    
    results = {}
    
    # Run checks
    results['cuda'] = check_basic_cuda()
    if not results['cuda']:
        print("\n" + "=" * 70)
        print("DIAGNOSIS: PyTorch is not configured for CUDA")
        print("=" * 70)
        print("You need to reinstall PyTorch with CUDA support.")
        print("\nRun these commands (works with CUDA driver 12.x, 13.x, and newer):")
        print("  pip uninstall torch torchvision torchaudio")
        print("  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")
        print("  pip install diffusers transformers accelerate pillow opencv-python xformers")
        return
    
    check_memory()
    results['computation'] = test_simple_computation()
    results['model'] = test_model_loading()
    results['diffusers'] = check_diffusers()
    
    # Summary
    print("\n" + "=" * 70)
    print("DIAGNOSTIC SUMMARY")
    print("=" * 70)
    
    for test_name, passed in results.items():
        status = "✓ PASSED" if passed else "❌ FAILED"
        print(f"{test_name.upper()}: {status}")
    
    if all(results.values()):
        print("\n" + "=" * 70)
        print("✓ ALL TESTS PASSED!")
        print("=" * 70)
        print("\nYour GPU is working correctly. If you're still seeing zero")
        print("GPU usage during actual inpainting:")
        print("\n1. Check nvidia-smi while the model is running")
        print("2. Try disabling safety_checker (already done in the updated code)")
        print("3. Check if any components are being moved back to CPU")
        print("4. Monitor GPU usage: watch -n 0.5 nvidia-smi")
    else:
        print("\n" + "=" * 70)
        print("❌ SOME TESTS FAILED")
        print("=" * 70)
        print("\nPlease check the errors above and fix them before running")
        print("the inpainting script.")


if __name__ == "__main__":
    main()
