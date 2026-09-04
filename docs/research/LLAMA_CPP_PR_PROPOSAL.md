# Pull Request: [vulkan] Fix GPU hang on ARM Mali (Valhall) by routing subgroup < 32 to Medium MatMul pipeline

- **Repository**: `ggerganov/llama.cpp`
- **Component**: `ggml/src/ggml-vulkan.cpp`
- **Issue Type**: Bug Fix / Stability / Hardware Enablement

---

## Title
`[vulkan] Fix GPU hang/TDR on ARM Mali by routing subgroup < 32 to Medium MatMul pipeline`

---

## Description

### Summary
Fixes a deterministic GPU hang (`vk::DeviceLostError: ErrorDeviceLost`) on ARM Mali GPUs (e.g. Mali-G68 MP5 on Exynos 1380, Mali-G78, Mali-G710) during quantized GEMM matrix multiplication (`MUL_MAT`).

### Root Cause
In `mul_mm.comp`:
```glsl
const uint loadstride_b = gl_WorkGroupSize.x * LOAD_VEC_B / BK;
...
[[unroll]] for (uint l = 0; l < BN; l += loadstride_b) { ... }
```
When running quantized matmul with the Small (`_s`) unaligned pipeline (`warptile_mmq_s`), parameters are configured as:
- `gl_WorkGroupSize.x = device->subgroup_size`
- `BK = 32` (Quantized block size)
- `LOAD_VEC_B = 1` (Scalar unaligned B_TYPE float)

On desktop GPUs (NVIDIA/AMD/Intel), subgroup size is 32 or 64, giving:
$$\text{loadstride}_b = 32 \times 1 / 32 = 1$$
However, on ARM Mali GPUs, the native subgroup size is **16**. Integer arithmetic produces:
$$\text{loadstride}_b = 16 \times 1 / 32 = 0$$

Because `loadstride_b == 0`, the loop condition `l < BN` becomes an **infinite GPU thread loop** (`l += 0`). This stalls the Vulkan compute queue until the kernel driver watchdog fires, aborting with `VK_ERROR_DEVICE_LOST`.

Furthermore, `ggml-vulkan.cpp` had vendor overrides for AMD, Apple, and Intel, but lacked any definition for ARM (`0x13b5`), causing ARM devices to default to the Small (`_s`) pipeline for batch tokens $N \le 32$.

### Solution
1. Define `VK_VENDOR_ID_ARM 0x13b5`.
2. In `ggml_vk_guess_matmul_pipeline()`, route ARM devices and any device with `subgroup_size < 32` to the Medium (`_m`) pipeline (`mmp->a_m : mmp->m`).
   - The Medium pipeline uses `gl_WorkGroupSize.x = 128`, yielding $\text{loadstride}_b = 128 \times 1 / 32 = 4 > 0$, completely eliminating the infinite loop.

---

## Verification & Benchmarks

Tested on a physical **Samsung Galaxy A35 5G** (Exynos 1380, ARM Mali-G68 MP5, Android 16, Termux):
- **Model**: `Qwen2.5-0.5B-Instruct-Q4_K_M.gguf`
- **Layers**: 25/25 layers (100% GPU offload)

| Setup | Before | After (This PR) |
| :--- | :---: | :---: |
| **Execution** | CRASH (`VK_ERROR_DEVICE_LOST` after 68s) | **SUCCESS (Exit code 0)** |
| **Eval Speed** | 0.00 t/s (Hangs on Node 2 `Qcur-0`) | **4.44 tokens/sec** |
| **Eval Latency** | $\infty$ | **225.22 ms/token** |
| **CPU Baseline** | 3.50 tokens/sec (6 cores) | **4.44 tokens/sec (+26.9% on GPU)** |
| **Correctness** | N/A | Exact match on seed 42 |

---

## Diff
```diff
diff --git a/ggml/src/ggml-vulkan.cpp b/ggml/src/ggml-vulkan.cpp
index 2ba5f98..9e1d844 100644
--- a/ggml/src/ggml-vulkan.cpp
+++ b/ggml/src/ggml-vulkan.cpp
@@ -36,6 +36,7 @@
 #define VK_VENDOR_ID_APPLE 0x106b
 #define VK_VENDOR_ID_INTEL 0x8086
 #define VK_VENDOR_ID_NVIDIA 0x10de
+#define VK_VENDOR_ID_ARM 0x13b5
 
 #define VK_DEVICE_DESCRIPTOR_POOL_SIZE 32
 
@@ -2929,6 +2930,13 @@ static vk_pipeline ggml_vk_guess_matmul_pipeline(ggml_backend_vk_context * ctx,
     case VK_VENDOR_ID_INTEL:
         return ggml_vk_guess_matmul_pipeline_intel(ctx, mmp, aligned);
+    case VK_VENDOR_ID_ARM:
+        return aligned ? mmp->a_m : mmp->m;
     default:
         break;
     }
+
+    if (ctx->device->subgroup_size < 32) {
+        return aligned ? mmp->a_m : mmp->m;
+    }
 
     if (m <= 32 || n <= 32) {
         return aligned ? mmp->a_s : mmp->s;
```
