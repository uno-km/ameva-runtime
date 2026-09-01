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

## Release Notes (v1.1.0)
- **Real-Device Galaxy A35 Validation**: Certified 12-stage hardware diagnostic hierarchy and Vulkan 1.4 API negotiation on Samsung Galaxy A35 5G (Exynos 1380, ARM Mali-G68 GPU).
- **CTypes Struct Pointer Patch**: Corrected struct pointer allocations to `ctypes.pointer` to fix `TypeError: expected LP_* instance, got _ctypes.CArgObject` in `doctor.py`.
- **Dynamic Topology Thread Optimization**: Added CPU topology inspection in `LlamaCppAdapter` targeting big-core clusters (`-t 4`) on mobile octa-core SoCs.
- **Strict Vulkan 3-Tier Execution Mode**: Fail-Fast protection for explicit `--device vulkan` requests without silent CPU masking.

## License
Apache-2.0 License. Copyright (c) 2026 Eunho Kim (@uno-km).
