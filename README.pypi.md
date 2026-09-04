# AMEVA-Vulkan-Runtime (Python)

[![PyPI](https://img.shields.io/pypi/v/ameva-vulkan-runtime.svg?style=flat-square&color=0369a1)](https://pypi.org/project/ameva-vulkan-runtime/)
[![Python](https://img.shields.io/pypi/pyversions/ameva-vulkan-runtime.svg?style=flat-square)](https://pypi.org/project/ameva-vulkan-runtime/)
[![License](https://img.shields.io/badge/License-Apache_2.0-004499.svg?style=flat-square)](https://github.com/uno-km/ameva-vulkan-runtime)

> **안드로이드 Termux 환경에서 디바이스 리소스를 활용하는 SoC 감지 및 적응형 추상화 런타임**  
> *SoC-Aware Adaptive Abstraction Runtime Utilizing Device Resources for Android Termux*

## Installation

```bash
pip install ameva-vulkan-runtime
```

## Quickstart

```python
import ameva_vulkan_runtime as avr

# 1. Probe & Validate Hardware (V0-V11)
doctor = avr.Doctor()
report = doctor.run_self_test()
print(f"GPU Backend Status: {report.overall_success}, Device: {report.device_name}")

# 2. Acquire High-Performance Vulkan Context for STT / LLM / Diffusion
ctx = avr.create_context(device="auto", memory_limit_mb=1024)
print(f"Context initialized via: {ctx.loader_path} (API {ctx.driver_version})")
```

## Description
Provides a C++20 Hardware Abstraction Layer (HAL) with Zero-Guesswork SoC auto-detection, utilizing hardware paths on Qualcomm Adreno and stable ARM NEON 4-Thread FP16 CPU execution on Exynos Mali.

## Documentation
- [Official Documentation & API Reference](https://uno-km.vercel.app/lib/ameva-vulkan-runtime/)
- [GitHub Repository](https://github.com/uno-km/ameva-vulkan-runtime)

## License
Apache-2.0 License. Copyright (c) 2026 Eunho Kim (@uno-km).
