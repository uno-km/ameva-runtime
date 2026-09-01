# ameva-vulkan-runtime (Python SDK)

[![PyPI](https://img.shields.io/pypi/v/ameva-vulkan-runtime.svg?style=flat-square&color=0369a1)](https://pypi.org/project/ameva-vulkan-runtime/)
[![Python](https://img.shields.io/pypi/pyversions/ameva-vulkan-runtime.svg?style=flat-square)](https://pypi.org/project/ameva-vulkan-runtime/)
[![License](https://img.shields.io/badge/License-Apache_2.0-004499.svg?style=flat-square)](https://github.com/uno-km/ameva-vulkan-runtime)

> **안드로이드 모바일 환경을 위한 통합 크로스 모달 Vulkan GPU 가속 런타임 및 하드웨어 추상화 계층 (HAL)**  
> *Unified Cross-Modal Vulkan GPU Acceleration Runtime & HAL for Mobile Android*

---

## Installation

```bash
pip install ameva-vulkan-runtime

# Modality extras:
pip install ameva-vulkan-runtime[stt]
pip install ameva-vulkan-runtime[diffusion]
pip install ameva-vulkan-runtime[llamacpp]
pip install ameva-vulkan-runtime[all]
```

## Quickstart

```python
import ameva_vulkan_runtime as avr

# 1. Probe & Validate Hardware (V0-V11)
doctor = avr.Doctor()
report = doctor.run_self_test()
print(f"GPU Backend Status: {report.status}, Device: {report.device_name}")

# 2. Acquire High-Performance Vulkan Context for STT / LLM / Diffusion
ctx = avr.create_context(device="auto", memory_limit_mb=1024)
print(f"Context initialized via: {ctx.loader_path} (API {ctx.vulkan_version})")
```

## Description
단일 C++20 Vulkan 하드웨어 추상화 계층(HAL)과 통합 런타임을 통해 STT, 비전, LLM, 디퓨전, 트레이닝 전 모달리티를 가속하며, 12단계 정밀 자체 검증(V0~V11)과 무손실 자동 복구 체계를 제공합니다.

## Documentation
- [Official Documentation & API Reference](https://uno-km.vercel.app/lib/vulkan/)
- [GitHub Repository](https://github.com/uno-km/ameva-vulkan-runtime)

## License
Apache-2.0 License. Copyright (c) 2026 Eunho Kim (@uno-km).
