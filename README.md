# ComfyUI-INT8-Fast-ROCM — rocblas DP4a backend fork

(Vibecoded) Fork of [patientx/ComfyUI-INT8-Fast-ROCM](https://github.com/patientx/ComfyUI-INT8-Fast-ROCM)
that swaps the pack's Triton INT8 GEMMs for a **rocBLAS-backed native INT8
(DP4a) backend on AMD RDNA2**. Everything else — model loading, ComfyKitchen
integration, ConvRot rotation, LoRA baking, per-row/per-channel quant — is
unchanged and comes from the original pack. Very WIP, expect jank. All courtesy of Deepseek-V4-Flash-0731.

(Set `ROCM_INT8_ROCBLAS=0` to revert to the original Triton
kernels at runtime.)

## Why

On RDNA2 (RX 6600 / gfx1032) the pack's Triton `tl.dot` INT8 kernels compile to
FMA emulation — no native INT8 path. rocBLAS ships real DP4a kernels for
gfx1032, giving a genuine ~2× instruction rate on INT8 GEMMs.

**Important:** the loader's `weight_dtype` must be set to **`fp16`** (not
default/bf16) — bf16 runs at FP32 rate on RDNA2 and it is what the original
pack's default uses.

## Results (Anima, 1024×1024, 10 steps, CFG 5 — KSampler s/it)

| Variant | s/it |
|---|---|
| fp8 (bf16 compute) | 13.4 |
| INT8 Triton (pack), bf16 compute | 7.52 |
| INT8 Triton (pack), fp16 compute | 5.73 |
| **INT8 rocblas (this fork), fp16 compute** | **5.51** |

That is **~2.4× faster than fp8** and **~1.04× faster than the pack's Triton
kernels at the same fp16 compute settings** (the gap widens to ~1.36× when both
use the bf16 default, which forces the Triton path's fp32-upcast quantize).

## Which models does it work on?

The backend is model-agnostic: it replaces the GEMM behind whatever the pack
already handles (W8A8/ConvRot DiTs — Flux2, Anima, Chroma, Z-Image, etc.), and
supports both per-channel and per-row weight scales, fp16 and bf16 compute
outputs. **It has only been tested on Anima so far** —
expect it to work on the others but verify before relying on it.

## Caveats / notes

- The original pack's Triton kernels beat rocBLAS on some small/odd GEMM shapes;
  the workflow-level win comes from the steady state at real model shapes.
- The compiled extension lives as C++/HIP source inside `rocblas_int8.py`:
  on first import it builds via torch's `load_inline` when no prebuilt `.pyd`
  is found (needs MSVC once, plus the ROCm SDK include/lib paths — the `DEVEL`
  constant at the top is hardcoded to the dev machine, adjust it if building
  elsewhere), and is then loaded directly at runtime — no toolchain needed
  after the first build.
- Only tested on Windows + ROCm 7.14 (ComfyUI 0.33), RX 6600.

## Fused DP4a kernel (B1, optional)

An optional hand-written DP4a GEMM with the dequant epilogue fused in-kernel,
achieving ~45% of DP4a peak (vs rocBLAS ~35–38%) while remaining
**byte-identical** to the 3-launch path (same seed, same PNG).

Enable: `ROCM_INT8_B1=1` before launching. Default OFF — no change to shipped
behaviour without this flag.

| | 3-launch | B1 fused |
|---|---|---|
| s/it (Anima 1024², fp16) | 5.51 | **5.31** |
| PNG | `5573d639…` | `5573d639…` (byte-identical) |
| launches per linear | 3 | 2 (quantize + fused GEMM+dequant) |

DP4a true peak on RX 6600 = 17.85 T-MAC/s (1792 × 4 MAC/cyc × 2.49 GHz);
int8 is exactly 2× fp16 on this hardware (32-bit datapath: 4×8b vs 2×16b).
Attention (SDPA math) is a bandwidth wall (~1.9 s/step, no flash on gfx1032)
and is the remaining ~36% of step time that cannot be kernel-optimized away.

---

<details>
<summary><b>Original README (patientx/ComfyUI-INT8-Fast-ROCM)</b></summary>

# 🎉 INT8 is now officially supported in ComfyUI 🎉
https://github.com/Comfy-Org/ComfyUI/commit/1a510f04234e5a213d3985a1a54f65652623f4bc

No, I did not help at all with this and had no involvement. My **existing quants are likely to not work** due to a quant naming missmatch, but [silveroxides](https://huggingface.co/silveroxides) are likely to work as they were quite involved in the process of making this happen.

Existing INT8 fast quants can be converted to the proper native format via this script https://github.com/BobJohnson24/ComfyUI-INT8-Fast/blob/main/convert_to_comfy.py

```
python convert_comfy_quant.py I8Fast.safetensors I8Comfy.safetensors
or
python convert_comfy_quant.py I8Fast.safetensors --inplace
```

I am glad to retire with a Piña colada in my hands, on the beach. Might slim this node down to an exclusively pre-lora focused node in the future, if that does not become a default comfy feature.

# Comfy INT8 Acceleration

This node speeds up Flux2, Ideogram4, Chroma, Z-Image, Ernie Image in ComfyUI by using INT8 quantization, delivering between 1.5~2x faster inference on my 3090 depending on the model. It should work on any NVIDIA GPU with enough INT8 TOPS. It appears to be faster than FP8 on 40-Series and above as well. 
Works with lora, torch compile.

Further Reading:

[Quality Metrics comparing against MXFP8, FP8, GGUF, etc.](Metrics.md)

[Speed](Speed.md)

[List of Prequantized Checkpoints](Models.md)

---

Updates:

2026-06-06

Fixes for 20-series GPUs

Ensuring proper handling of static weights when dynamic is deactivated

2026-24-05:

RAM usage for lora loading is fixed and on par with base comfy.

RAM usage for model loading is fixed.

Only thing that remains is on the fly quantization will create an extra int8 copy in memory, but it is too much of a hassle to work around. Please rely on swap or pre converted models if this is an issue.

Fixed an issue with loading loras on models that include .bias layers (WAN, LTX2.X) which would cause a OOM error.

2026-15-05:

Bringing back stochastic lora. Some loras appear to need it, others don't, try it if your lora is not working and you don't like pre-lora. TLDR is "sometimes it really helps, sometimes its a little worse". See our measurements [here](https://github.com/BobJohnson24/ComfyUI-INT8-Fast/blob/RAMExp/Metrics.md#some-loras-require-stochastic-lora-to-work).

Attempt at reducing RAM usage

Fixed an issue with Pre-Lora crashing on windows

2026-10-05:

Overhauled the entire lora system. Normal lora loader node works now, no need for specialized lora loaders.

Converted QuaRot to ConvRot, which is a small but free quality gain.

Added Pre-Lora node, which you can connect to the INT8 Model loader to merge loras before utilizing on the fly quantization. 

For more info on quality of convrot, lora approaches see the [Metrics](Metrics.md)

---

# Common GPU related issues:

RTX 20-Series will require you to either use Triton-Windows on windows, triton==3.2.0 or compile triton yourself with SM75 support which was dropped in 3.3.0.

A100 has no possible INT8 Speed-up https://github.com/BobJohnson24/ComfyUI-INT8-Fast/issues/71


## FAQ:

Q: How do I quantize myself?

A: It is not recommended to quantize the human existence. If you would like to quantize a model, see example_workflows/int8_save_convrot_model.json

Q: What is ConvRot?

A: ConvRot is a variant of QuaRot. It basically rotates model weights and activations to eliminate outliers before quantization. This has some inference overhead, but is generally a large quality boost.

Q: What is Pre-Lora?

A: Pre-Lora is a way to merge the lora weights to a BF16 checkpoint within ComfyUI before you quantize the model. This requires an unquantized base model, and enabling on-the-fly quantization. It is generally a higher quality way to apply a lora.

Q: Torch compile takes forever and I hate it

A: Use the torch compile node from [KJ Nodes](https://github.com/kijai/ComfyUI-KJNodes) and ensure you set the disable dynamic VRAM toggle.


# Requirements:
Working ComfyKitchen (needs latest comfy and possibly pytorch with cu130)

Triton

Windows untested, but I hear triton-windows exists.

# Credits:

## dxqb for the *entirety* of the INT8 code during the very early versions of this node, it would have been impossible without them:
https://github.com/Nerogar/OneTrainer/pull/1034

If you have a 30-Series GPU, OneTrainer is also the fastest current lora trainer thanks to this. Please go check them out!!

## newgrit1004 for the base ConvRot code we modified into proper ConvRot 
https://github.com/newgrit1004/ComfyUI-ZImage-Triton

## silveroxides for providing a base to hack the INT8 conversion code onto.
https://github.com/silveroxides/convert_to_quant

## Also silveroxides for showing how to properly register new data types to comfy
https://github.com/silveroxides/ComfyUI-QuantOps

## The unholy trinity of AI slopsters I used to glue all this together over the course of multiple months now
</details>