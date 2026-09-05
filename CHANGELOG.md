# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [2.0.2] - 2026-09-05

### Changed
- Refined ecosystem runtime bindings and finalized zero-regression fail-fast hardware HAL.
- Enhanced Doctor hardware diagnostics across ARM Mali Valhall and Qualcomm Adreno platforms.

---

## [2.0.1] - 2026-09-05

### Changed
- Enforced strict Fail-Fast execution policy across all modality adapters.
- Synchronized ecosystem integration with `@ameva/runtime` v2.0.1 specification.
- Enhanced Doctor hardware diagnostics and ARM Mali Valhall / Qualcomm Adreno dynamic routing.

---

## [2.0.0] - 2026-09-05

### Major Architecture Milestone: Single Package Unification, Mali-Valhall Acceleration & STT 2.26x Speedup
- **ARM Mali Valhall MatMul Zero-Stride Infinite Loop Elimination**:
  - Identified and resolved the critical GLSL compute shader integer truncation defect in `mul_mm.comp` (`loadstride_b = gl_WorkGroupSize.x * LOAD_VEC_B / BK = 16 * 1 / 32 = 0`).
  - Completely eliminated GPU compute shader infinite loops (`for (uint l = 0; l < BN; l += 0)`) and hardware watchdog TDR aborts (`VK_ERROR_DEVICE_LOST`).
  - Added `MaliQuirks::ShouldEnforceMediumMatMulKernel` and `VK_VENDOR_ID_ARM (0x13b5)` pipeline routing to enforce Medium kernels (`_m`, workgroup=128, loadstride=4 > 0).
  - Empirically achieved **4.44 tokens/sec** on Samsung Galaxy A35 (Exynos 1380, Mali-G68 MP5) with 25/25 layers (100%) GPU offloading (+26.9% faster than 6 CPU-NEON cores).
- **Whisper STT 2.26x Real-Device GPU Acceleration**:
  - Validated Whisper Large-v3-Turbo (548MB Q5_0) on Samsung Galaxy A35 completing in **360.60s (6m 00s)** vs CPU NEON **816.48s (13m 36s)** — a **2.26x acceleration (56% time reduction)** with 0 fallbacks, while reducing CPU load from 291% to 20~30%.
- **Qualcomm Adreno 830 JIT Bug Isolation**:
  - Handled Qualcomm Adreno JIT compiler crash (`VK_ERROR_UNKNOWN -13`) when Specialization Constant `NUM_COLS >= 3` by bounding `mul_mat_vec_max_cols = 2`, enabling stable GPU inference on Galaxy S25 in 4,401 ms.
- **Silicon-Aware Dynamic Branching (Galaxy S25 vs Galaxy A35)**:
  - **Galaxy S25 (`SM-S931N`)**: Qualcomm Adreno 830 (Snapdragon 8 Elite) routes to high-throughput Vulkan compute with `subgroup_control_bypass: true` and native 64/128 subgroup GEMM.
  - **Galaxy A35 (`SM-A356N`)**: ARM Mali-G68 MP5 (Exynos 1380) routes to zero-freeze Vulkan compute with `enforce_medium_matmul: true` and 128-byte memory alignment.
  - SmartRouter dynamically negotiates between Qualcomm KGSL and ARM Bionic ICD drivers without regression or cross-contamination.
- **PyTorch-Style Single Package Architecture**:
  - Consolidated repository and distribution under `ameva-runtime` (v2.0.0), housing specialized Vulkan acceleration in `from ameva_runtime import vulkan` with dynamic single-source-of-truth versioning (`_version.py`).
- **Breaking Changes: Pure Submodule Unification & Legacy Namespace Deprecation**:
  - The standalone `ameva_vulkan_runtime` top-level namespace is fully deprecated and consolidated into `from ameva_runtime import vulkan`.
  - All ecosystem consumers and sibling packages must import directly via `from ameva_runtime import vulkan` or `import ameva_runtime as ar; ar.vulkan`.
  - Phased out transitional shim layer in favor of a clean, unfragmented single-package architecture (`name = "ameva-runtime"`).
- **Complete Sibling Ecosystem Migration**:
  - Migrated `termux-stt`, `termux-vision`, `termux-llamacpp`, `termux-diffusion`, `termux-bitnet`, `termux-tts`, and `termux-train` to directly import `from ameva_runtime import vulkan`.
- **6-Modality Vulkan Acceleration Roadmap**:
  - `LLM`: [v2.0.0 Completed] Llama.cpp Q4_K_M 25/25 layer full VRAM offload (Adreno 830: 35.80 t/s, Mali-G68: 4.44 t/s).
  - `STT`: [v2.0.0 Completed] Whisper.cpp on-device Vulkan acceleration (Adreno 830: 4,401 ms, Mali-G68: 360.60s / 2.26x speedup).
  - `Vision`: [v2.1.0 In Progress] CLIP, MobileVLM, LLaVA Vulkan GGML tensor binding.
  - `Diffusion`: [v2.2.0 In Progress] Stable Diffusion v1.5 / Turbo & FLUX.1 on-device UNet/DiT tensor offload.
  - `TTS`: [v2.3.0 In Progress] Piper, Sherpa-ONNX, Kokoro low-latency neural TTS streaming.
  - `Train`: [v2.4.0 In Progress] On-device LoRA / QLoRA Vulkan gradient descent backpropagation.

---

## [1.0.2] - 2026-09-05

### Fixed
- **ARM Mali Valhall MatMul Zero-Stride Infinite Loop Elimination**:
  - Identified and resolved the critical GLSL shader integer truncation bug in `mul_mm.comp` where devices with subgroup size < 32 (e.g. Mali-G68 with warp 16) calculated `loadstride_b = gl_WorkGroupSize.x * LOAD_VEC_B / BK = 16 * 1 / 32 = 0`.
  - Prevented the resulting GPU compute shader infinite loop (`for (uint l = 0; l < BN; l += 0)`) and hardware watchdog TDR reset (`VK_ERROR_DEVICE_LOST`).
  - Added `MaliQuirks::ShouldEnforceMediumMatMulKernel` and `VK_VENDOR_ID_ARM (0x13b5)` pipeline routing to enforce Medium kernels (`_m`, workgroup=128, loadstride=4 > 0).
  - Achieved **4.44 tokens/sec** on Samsung Galaxy A35 (Exynos 1380, Mali-G68 MP5) with 25/25 layers (100%) GPU offloading and zero CPU fallback.

---

## [1.0.0] - 2026-09-04

### Added
- **Unified On-Device AI Acceleration Core**: Unified `ameva_runtime` orchestrator with automatic topology detection, CPU core affinity control, and dynamic library resolution.
- **Empirical Snapdragon 8 Elite (Adreno 830) Certification**: Validated 25/25 layer full VRAM GPU offloading on Galaxy S25 achieving **35.80 tokens/sec** (27.93 ms/t), representing a 35.8x acceleration over pure CPU.
- **ARM Mali Headless Fence Deadlock Isolation & Guard**: Isolated the proprietary ARM Mali driver power-management downclocking bug (`vkWaitForFences` hang in headless CLI) and implemented automated `SmartRouter` fallback to Cortex-A78 CPU-NEON multi-threading (3.55 tokens/sec, 0% freeze).
- **Direct Python Model Execution**: Introduced `ameva.run()` and `AmevaRuntime.execute()` top-level APIs returning rich telemetry (`tokens_per_second`, `latency_ms`, `eval_tokens`).
- **Universal Multi-Command CLI**: Added `ameva-run` with `doctor`, `profile`, `plan`, `exec`, and `benchmark` commands.
- **Zero-Breaking Backward Compatibility (v1.x Transitional)**: Provided transitional shim layer for legacy `from ameva_vulkan_runtime import VulkanContext, create_context` (phased out in v2.0.0).

---

## [1.2.0] - 2026-09-02

### Added
- **Complete 12-Stage Native Pipeline Execution**: Integrated C ABI FFI SGEMM compute kernel queue dispatch, deterministic numeric checksum ($c[0,0]=32.0$), and burst stress testing across Doctor stages V7~V11.
- **RAII Context Lifecycle Adapter Registry**: Automated `unbind_all()` across all 6 modality adapters upon `VulkanContext` close/exit preventing memory and hardware handle leaks.
- **Strict Domain Exception Hierarchy**: Added `AmevaVulkanError` for explicit fail-fast FFI execution errors preserving stack traces.
- **Zero-Drift Hardware Profile Packaging**: Added `validated-vulkan-profiles.json` to package-data and `MANIFEST.in` with bidirectional fuzzy device/GPU matching.

### Fixed
- **Exynos 2100 Bionic Symbol Isolation**: Applied `RTLD_LAZY | RTLD_LOCAL` loader isolation to prevent `_ZN7android18egl_get_connectionEv` crashes on Samsung One UI devices.
- **Micro-GEMM Honest Engine Labeling**: Separated `Executed Kernel` into `NATIVE_C_API` and `CPU_NUMPY_REFERENCE (Fallback)` with deterministic $c[0,0]=128.0$ verification.
- **LlamaCppAdapter Nested Config Dispatch**: Fixed `hasattr(engine, 'config')` handling for polymorphic engine instances.

---

## [1.1.0] - 2026-09-01

### Added
- **Real-Device Galaxy A35 Validation**: Verified 12-stage hardware diagnostic hierarchy and Vulkan 1.4 API negotiation on Samsung Galaxy A35 5G (Exynos 1380, ARM Mali-G68 GPU).
- **Dynamic Topology Thread Optimization**: Added CPU topology inspection in LlamaCppAdapter to target big-core clusters (-t 4) on octa-core mobile SoCs.
- **Strict Vulkan 3-Tier Execution Mode**: Fail-Fast protection for explicit --device vulkan requests without silent CPU masking.

### Fixed
- **CTypes Struct Pointer Assignment**: Corrected pApplicationInfo, pQueuePriorities, and pQueueCreateInfos from ctypes.byref to ctypes.pointer to fix TypeError: expected LP_* instance, got _ctypes.CArgObject in doctor.py.
- **Driver-Probed State Reporting**: Resolved _is_vulkan_report() condition to correctly recognize 7-stage passed driver states (passed_stages >= 7).

---

## [1.0.0] - 2026-08-15

### Added
- Initial production release of meva-vulkan-runtime.
- 12-Stage Diagnostic Suite (Doctor) from V0 loader open to V11 model inference.
- Multi-modality adapters for STT, TTS, Diffusion, BitNet, Vision, and LlamaCpp.
- Cross-platform Bionic ICD driver loader (/system/lib64/libvulkan.so).
