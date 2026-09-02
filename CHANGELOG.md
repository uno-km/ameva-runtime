# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
