"""Inpainting model registry and pipeline loading."""

import time

import torch
from diffusers import AutoPipelineForInpainting


# Available inpainting models


# AutoPipelineForInpainting automatically detects and loads the correct pipeline class


AVAILABLE_MODELS = {
    "sd-1.5": "runwayml/stable-diffusion-inpainting",
    "sd-2": "sd2-community/stable-diffusion-2-inpainting",
    "sd-xl": "diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
    "kandinsky-2.2": "kandinsky-community/kandinsky-2-2-decoder-inpaint",
}


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
