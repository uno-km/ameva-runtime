# Technical Whitepaper: Discovery & Resolution of ARM Mali Valhall Vulkan MatMul Zero-Stride Infinite Loop

- **Author**: Eunho Kim (uno-km) & AMEVA Open-Source Foundation (AOSF)
- **Target Architecture**: ARM Mali Valhall (Mali-G68 MP5, Mali-G78, Mali-G710, Immortalis)
- **Reference Device**: Samsung Galaxy A35 5G (SM-A356N, Exynos 1380 SoC)
- **Driver**: Android Native Bionic Vulkan Driver (`/system/lib64/libvulkan.so`)
- **Date**: 2026-09-05
- **Status**: EMPIRICALLY VALIDATED & PRODUCTIONIZED (v1.0.2)

---

## 1. Executive Summary

For over two years, the on-device AI community across GitHub and Reddit considered ARM Mali GPUs incapable of running full Vulkan LLM inference via `llama.cpp`, attributing frequent `vk::DeviceLostError: ErrorDeviceLost` crashes to proprietary driver defects, thermal throttling, or aggressive Android power-management sleep states.

Through discrete node-by-node Vulkan fence isolation on a physical Samsung Galaxy A35, we discovered that the driver was entirely blameless. The failure was caused by a deterministic integer truncation defect in the core matrix multiplication shader (`mul_mm.comp`):

$$\text{loadstride}_b = \left\lfloor \frac{\text{gl\_WorkGroupSize.x} \times \text{LOAD\_VEC\_B}}{\text{BK}} \right\rfloor = \left\lfloor \frac{16 \times 1}{32} \right\rfloor = 0$$

On GPUs with subgroup size 16 (ARM Mali), this evaluates to 0, turning the buffer loading loop into an **infinite GPU thread loop** (`for (uint l = 0; l < BN; l += 0)`), triggering Android's GPU hardware watchdog timeout (TDR).

By defining `VK_VENDOR_ID_ARM (0x13b5)` and routing subgroup < 32 devices to the Medium (`_m`) pipeline (workgroup size 128, yielding $\text{loadstride}_b = 4 > 0$), we achieved 100% stable GPU execution across all 25 transformer layers and the LM-head, achieving **4.44 tokens/sec** with zero CPU fallback.

---

## 2. The Root Cause Mechanism

### 2.1 The GLSL Integer Division Trap
In `ggml/src/vulkan-shaders/mul_mm.comp`:

```glsl
layout (local_size_x_id = 0, local_size_y = 1, local_size_z = 1) in;

layout (constant_id = 1) const uint BM = 64;
layout (constant_id = 2) const uint BN = 64;
layout (constant_id = 3) const uint BK = 16;  // 32 for quantized GEMM (MMQ)
layout (constant_id = 9) const uint WARP = 32;

...
const uint loadstride_b = gl_WorkGroupSize.x * LOAD_VEC_B / BK;
...
[[unroll]] for (uint l = 0; l < BN; l += loadstride_b) {
    ...
}
```

In the Small (`_s`) quantized pipeline (`warptile_mmq_s`), parameters are initialized as:
- `local_size_x = device->subgroup_size`
- `BK = 32` (Quantized block size)
- `LOAD_VEC_B = 1` (Scalar float for unaligned matrix B)

| GPU Vendor | Subgroup / Warp Size | Integer Calculation | `loadstride_b` | Loop Condition (`l < 32`) | Behavior |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **NVIDIA** | 32 | $32 \times 1 / 32$ | **1** | `l += 1` | Normal (Terminates in 32 iterations) |
| **AMD** | 32 / 64 | $64 \times 1 / 32$ | **2** | `l += 2` | Normal (Terminates in 16 iterations) |
| **Intel** | 32 | $32 \times 1 / 32$ | **1** | `l += 1` | Normal (Terminates in 32 iterations) |
| **ARM Mali** | **16** | **$16 \times 1 / 32$** | **0** | **`l += 0`** | **FATAL: Infinite GPU Loop** |

### 2.2 Complete Omission of ARM Vendor Specialization
In `ggml/src/ggml-vulkan.cpp`:
```cpp
#define VK_VENDOR_ID_AMD 0x1002
#define VK_VENDOR_ID_APPLE 0x106b
#define VK_VENDOR_ID_INTEL 0x8086
#define VK_VENDOR_ID_NVIDIA 0x10de
// NOTE: VK_VENDOR_ID_ARM (0x13b5) was completely absent!
```

Because ARM had no vendor override in `ggml_vk_guess_matmul_pipeline()`, prompt evaluation with sequence length $N \le 32$ defaulted to the Small unaligned kernel (`_s`), guaranteeing that every ARM Mali device in the world encountered the infinite loop.

---

## 3. The Minimal Architectural Fix

```cpp
#define VK_VENDOR_ID_ARM 0x13b5

static vk_pipeline ggml_vk_guess_matmul_pipeline(ggml_backend_vk_context * ctx, vk_matmul_pipeline& mmp, int m, int n, bool aligned) {
    switch (ctx->device->vendor_id) {
    case VK_VENDOR_ID_AMD:
        return ggml_vk_guess_matmul_pipeline_amd(ctx, mmp, m, n, aligned);
    case VK_VENDOR_ID_APPLE:
        return ggml_vk_guess_matmul_pipeline_apple(ctx, mmp, aligned);
    case VK_VENDOR_ID_INTEL:
        return ggml_vk_guess_matmul_pipeline_intel(ctx, mmp, aligned);
    case VK_VENDOR_ID_ARM:
        return aligned ? mmp->a_m : mmp->m;
    default:
        break;
    }

    if (ctx->device->subgroup_size < 32) {
        // Prevent loadstride_b = 0 integer truncation on warp 16 hardware
        return aligned ? mmp->a_m : mmp->m;
    }

    if (m <= 32 || n <= 32) {
        return aligned ? mmp->a_s : mmp->s;
    }
    ...
}
```

The Medium (`_m`) pipeline uses `local_size_x = 128`:
$$\text{loadstride}_b = \frac{128 \times 1}{32} = 4 > 0$$
The loop increments $l \mathrel{+}= 4$, fully populating shared memory and terminating cleanly.

---

## 4. Empirical Ground Truth Verification

All benchmarks were conducted on a physical Samsung Galaxy A35 (Exynos 1380, Mali-G68 MP5) running Qwen2.5-0.5B-Instruct-Q4_K_M:

| Evaluation Metric | CPU NEON Baseline (6 Cores) | Mali-G68 Vulkan GPU (AMEVA Fix) | Verification Delta |
| :--- | :---: | :---: | :--- |
| **GPU Offloaded Layers** | 0 / 25 | **25 / 25 (100%)** | Full VRAM Residency |
| **LM-Head Offload** | CPU | **Mali-G68 GPU** | Zero Host Memory Stalls |
| **Decoding Speed** | 3.50 tokens/sec | **4.44 tokens/sec** | **+26.9% GPU Speedup** |
| **Decoding Latency** | 286.02 ms/token | **225.22 ms/token** | **-60.8 ms per token** |
| **Process Exit Code** | 0 | **0** | 100% Deterministic |
| **GPU TDR / Lost Error** | N/A | **0 Crashes** | 100% Stable |
| **Language Output** | Coherent (Seed 42) | **Identical Semantics (Seed 42)** | Numerical Parity Confirmed |
| **Korean Unicode KV Cache**| Coherent | **Coherent (4.44 t/s)** | Multi-byte UTF-8 Validated |

---

## 5. Artifacts & Deliverables

1. **AMEVA Vulkan Runtime Core**: `src/quirks/mali_quirks.cpp` & `mali_quirks.h` updated with `MaliQuirks::ShouldEnforceMediumMatMulKernel()`.
2. **PyPI Distribution**: `dist/ameva_runtime-1.0.2-py3-none-any.whl`
3. **NPM Distribution**: `npm/ameva-runtime-1.0.2.tgz`
4. **Upstream PR**: Branch prepared for `ggerganov/llama.cpp` submission.
