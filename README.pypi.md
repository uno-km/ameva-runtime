# AMEVA-Vulkan-Runtime (Python)

[![PyPI](https://img.shields.io/pypi/v/ameva-vulkan-runtime.svg?style=flat-square&color=0369a1)](https://pypi.org/project/ameva-vulkan-runtime/)
[![Python](https://img.shields.io/pypi/pyversions/ameva-vulkan-runtime.svg?style=flat-square)](https://pypi.org/project/ameva-vulkan-runtime/)
[![License](https://img.shields.io/badge/License-Apache_2.0-004499.svg?style=flat-square)](https://github.com/uno-km/ameva-vulkan-runtime)

> **안드로이드 모바일 환경을 위한 통합 크로스 모달 Vulkan GPU 가속 런타임 및 하드웨어 추상화 계층(HAL)**  
> *Unified Cross-Modal Vulkan GPU Acceleration Runtime & HAL for Mobile Android*

## Installation

```bash
pip install ameva-vulkan-runtime
```

## Quickstart

```python
import ameva_vulkan_runtime as avr

doctor = avr.Doctor()
report = doctor.run_self_test(verbose=True)
print(f"Backend: {report.recommended_backend}, Device: {report.device_name}")

ctx = avr.create_context(device="auto", memory_limit_mb=1024)
print(f"Context active: {ctx.is_active}, Backend: {ctx.backend_type}")
```

## Description
Provides a single, zero-hardcoded C++20 Vulkan Hardware Abstraction Layer (HAL) and universal runtime for STT, Vision, LLM, Diffusion, and Training with a granular 12-stage validation hierarchy (V0-V11) and zero-data-loss auto-recovery.

## Documentation
- [Official Documentation & API Reference](https://uno-km.vercel.app/lib/ameva-vulkan-runtime/)
- [GitHub Repository](https://github.com/uno-km/ameva-vulkan-runtime)

## Release Notes (v1.2.0)
- **Complete 12-Stage Native Pipeline Execution**: Integrated C ABI FFI SGEMM compute kernel queue dispatch, deterministic numeric checksum ($c[0,0]=32.0$), and burst stress testing across Doctor stages V7~V11.
- **RAII Context Lifecycle Adapter Registry**: Automated `unbind_all()` across all 6 modality adapters upon `VulkanContext` close/exit preventing memory and hardware handle leaks.
- **Strict Domain Exception Hierarchy**: Added `AmevaVulkanError` for explicit fail-fast FFI execution errors preserving stack traces.
- **Exynos 2100 Bionic Symbol Isolation**: Applied `RTLD_LAZY | RTLD_LOCAL` loader isolation to prevent `_ZN7android18egl_get_connectionEv` crashes on Samsung One UI devices.
- **Zero-Drift Hardware Profile Packaging**: Added `validated-vulkan-profiles.json` to package-data and `MANIFEST.in` with bidirectional fuzzy device/GPU matching.

## License
Apache-2.0 License. Copyright (c) 2026 Eunho Kim (@uno-km).
