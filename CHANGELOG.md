# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
- **Zero-Breaking Backward Compatibility**: 100% transparent shim layer for legacy `from ameva_vulkan_runtime import VulkanContext, create_context`.

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
