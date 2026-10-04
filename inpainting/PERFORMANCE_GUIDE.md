# Performance Optimization Guide - RTX 4080

## Good News: 100% GPU Usage ✅

Your GPU usage is at 100%, which means the zero GPU usage issue is **completely fixed**! Your GPU is working properly.

## Understanding Performance

### Expected Performance (RTX 4080 Laptop)

The RTX 4080 Laptop has varying performance depending on several factors:

#### With xformers (Optimal):
| Resolution | Steps | Expected Time | it/s |
|------------|-------|---------------|------|
| 512x512 | 20 | 2-4s | 5-10 |
| 512x512 | 50 | 5-10s | 5-10 |
| 768x768 | 50 | 12-20s | 2.5-4 |
| 1024x1024 (SDXL) | 50 | 20-40s | 1-2.5 |

#### Without xformers:
Add 30-50% more time to the above.

### Why Performance Varies

RTX 4080 **Laptop** performance depends on:

1. **Thermal Throttling** - Laptop GPUs throttle when hot
2. **Power Mode** - Battery vs Plugged in, Performance mode
3. **TGP (Total Graphics Power)** - Laptop models vary (80W-175W)
4. **xformers** - 30-50% speed difference
5. **Background processes** - Other apps using GPU
6. **VRAM speed** - Memory clock throttling

## Quick Performance Test

Run the benchmark:
```bash
python benchmark_performance.py
```

This will show your actual performance and compare it to expected values.

## Common Performance Issues

### Issue 1: xformers Not Working
**Symptom:** 30-50% slower than expected

**Check:**
```python
python -c "import xformers; print(xformers.__version__)"
```

**Fix:**
```bash
pip uninstall xformers -y
pip install xformers --no-cache-dir
```

### Issue 2: Thermal Throttling
**Symptom:** Starts fast, then slows down. GPU temp > 80°C

**Check:** Look at temperature in nvidia-smi
```powershell
nvidia-smi
```

**Fix:**
- Use on flat, hard surface (not bed/couch)
- Clean laptop vents
- Use cooling pad
- Elevate back of laptop for airflow
- Lower room temperature

### Issue 3: Power Throttling
**Symptom:** Inconsistent performance, especially on battery

**Check:** Windows Power Plan

**Fix:**
- Plug in laptop (crucial!)
- Windows Settings > System > Power & Battery > Power Mode > Best Performance
- NVIDIA Control Panel > Manage 3D Settings > Power Management > Prefer Maximum Performance

### Issue 4: Wrong Resolution
**Symptom:** Very slow inference

**Check:** What resolution are you using?

**Reality Check:**
- 512x512: Fast (base resolution)
- 768x768: 2.25x slower than 512x512
- 1024x1024: 4x slower than 512x512
- 1536x1536: 9x slower than 512x512

### Issue 5: Too Many Steps
**Symptom:** Long wait times

**Reality:**
- 20 steps: Good quality, fast
- 50 steps: Better quality, 2.5x slower
- 100 steps: Marginal improvement, 5x slower

**Recommendation:** Use 20-30 steps for most cases

### Issue 6: SDXL vs SD 1.5
**Symptom:** Extremely slow

**Check:** Which model are you using?

**Performance:**
- SD 1.5 (512x512, 50 steps): 5-10s ✅ Fast
- SDXL (1024x1024, 50 steps): 20-40s ⚠️ Much slower

SDXL is 4-8x slower but higher quality.

## Optimization Checklist

### Software Optimizations:
- [x] GPU usage at 100% ✅ (Already fixed!)
- [ ] xformers installed and working
- [ ] PyTorch with CUDA 12.1
- [ ] Attention slicing enabled (already in code)
- [ ] Safety checker disabled (already in code)
- [ ] Latest NVIDIA drivers

### Hardware Optimizations:
- [ ] Laptop plugged in (not on battery)
- [ ] Windows Power Mode: Best Performance
- [ ] Good cooling/ventilation
- [ ] No other GPU applications running
- [ ] GPU temperature < 80°C

### Usage Optimizations:
- [ ] Use 512x512 for speed
- [ ] Use 20-30 steps (not 50-100)
- [ ] Use SD 1.5 (not SDXL) for speed
- [ ] Close other applications

## Benchmark Your System

Run this to see your actual performance:
```bash
python benchmark_performance.py
```

This will:
- Test different step counts
- Measure actual it/s (iterations per second)
- Compare against expected performance
- Show memory usage
- Verify xformers status

## Real-World Expectations

### What "Fast" Means:

**SD 1.5 @ 512x512:**
- RTX 4080 Laptop (with xformers): 5-10 it/s → **50 steps in 5-10 seconds** ✅
- RTX 4080 Laptop (no xformers): 3-7 it/s → **50 steps in 7-17 seconds**
- RTX 3080 Laptop: 3-5 it/s → **50 steps in 10-17 seconds**
- RTX 4090 Desktop: 12-15 it/s → **50 steps in 3-4 seconds** (much faster!)

### Your RTX 4080 Laptop:

**Expected real-world performance:**
- **Best case** (everything optimized): 8-10 it/s
- **Typical case** (good setup): 5-7 it/s
- **Worst case** (thermal throttling, no xformers): 2-4 it/s

## Current Status

You said:
- ✅ GPU usage is 100%
- ❓ Speed not matching expectations

**Next Steps:**
1. Tell me your current performance:
   - How many seconds for 50 steps at 512x512?
   - What it/s are you seeing during inference?

2. Run benchmark:
   ```bash
   python benchmark_performance.py
   ```

3. Check nvidia-smi during inference:
   ```powershell
   nvidia-smi
   ```
   Look for:
   - GPU temp
   - Power usage (should be near 80W for laptop)
   - GPU clock speeds

## Performance Reference Table

| GPU | Resolution | Steps | Expected Time | it/s |
|-----|------------|-------|---------------|------|
| RTX 4080 Laptop (optimal) | 512x512 | 50 | 5-10s | 5-10 |
| RTX 4080 Laptop (throttled) | 512x512 | 50 | 10-17s | 3-5 |
| RTX 4080 Desktop | 512x512 | 50 | 3-6s | 8-15 |
| RTX 4090 Desktop | 512x512 | 50 | 3-4s | 12-17 |
| RTX 3080 Laptop | 512x512 | 50 | 10-17s | 3-5 |

## Most Common Culprit: Laptop vs Desktop

**Important:** RTX 4080 **Laptop** is significantly slower than desktop:
- Desktop RTX 4080: 320W TGP, ~12-15 it/s
- Laptop RTX 4080: 80-175W TGP, ~5-10 it/s

This is normal! Laptop GPUs are power-limited.

## What to Share

To help you optimize further, please share:

1. **Current performance:**
   ```
   How many seconds for 50 steps at 512x512?
   What does the progress bar show? (e.g., "5.23it/s")
   ```

2. **Run benchmark:**
   ```bash
   python benchmark_performance.py
   ```
   Share the output

3. **nvidia-smi output:**
   During inference, what's the:
   - Temperature
   - Power (W)
   - GPU clock speed

4. **Settings:**
   - Image resolution
   - Number of steps
   - Which model (SD 1.5, SD 2, SDXL)
   - xformers status
