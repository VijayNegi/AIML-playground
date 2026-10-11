# PyTorch 2.6+ Installation Guide

## The Problem
You're getting this error:
```
ValueError: Due to a serious vulnerability issue in torch.load, even with weights_only=True,
we now require users to upgrade torch to at least v2.6
```

This means PyTorch 2.5.1 is too old due to security vulnerability CVE-2025-32434.

## Solution Options

### Option 1: Install Latest PyTorch (Recommended)

Try installing the latest stable version:

```bash
pip install --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### Option 2: Install Nightly Build (If 2.6+ Not Available in Stable)

If stable channel doesn't have 2.6 yet, use nightly:

```bash
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu121
```

### Option 3: Use CPU Version Temporarily (To Check Available Versions)

```bash
pip install --upgrade torch torchvision torchaudio
```

Then check version:
```bash
python -c "import torch; print(torch.__version__)"
```

## Complete Reinstallation (If Something Broke)

If PyTorch got uninstalled or corrupted, do a clean reinstall:

```bash
# 1. Completely remove PyTorch
pip uninstall torch torchvision torchaudio xformers -y

# 2. Install latest stable with CUDA
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 3. Check version
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA:', torch.cuda.is_available())"

# 4. If version is still < 2.6, try nightly
pip uninstall torch torchvision torchaudio -y
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu121

# 5. Install xformers after PyTorch is installed
pip install xformers --no-cache-dir

# 6. Install other dependencies
pip install diffusers transformers accelerate pillow opencv-python
```

## Verification Commands

After installation, verify everything:

```bash
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"
```

## Alternative: Use Safetensors (Workaround)

If you can't get PyTorch 2.6+ right now, the error message mentions that the restriction doesn't apply when using safetensors. You can try:

```python
# In your code, when loading models
pipe = AutoPipelineForInpainting.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    safety_checker=None,
    use_safetensors=True,  # Add this line
)
```

But this is a workaround - upgrading PyTorch is the proper solution.

## Check What's Available

To see what versions are available:

```bash
# Check stable channel
pip index versions torch --index-url https://download.pytorch.org/whl/cu121

# Or check PyPI
pip index versions torch
```

## Troubleshooting

### If pip install doesn't work:
1. Make sure you're using the correct Python environment
2. Try with `python -m pip` instead of `pip`
3. Check if you're behind a proxy or firewall

### If CUDA support breaks after upgrade:
```bash
pip uninstall torch torchvision torchaudio -y
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### If nothing works:
Use conda instead:
```bash
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
```

## Expected Results

After successful installation:
- PyTorch version should be 2.6.0 or higher
- CUDA should still be available (True)
- GPU should be detected (NVIDIA GeForce RTX 4080)

## Next Steps After Installation

1. Verify installation
2. Reinstall xformers
3. Run your code
4. Run benchmark if needed

---

Let me know which option worked for you!
