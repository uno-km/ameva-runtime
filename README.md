# AMEVA-Vulkan-Runtime

[![PyPI](https://img.shields.io/pypi/v/ameva-vulkan-runtime.svg?style=flat-square&color=0369a1)](https://pypi.org/project/ameva-vulkan-runtime/)
[![Python](https://img.shields.io/pypi/pyversions/ameva-vulkan-runtime.svg?style=flat-square)](https://pypi.org/project/ameva-vulkan-runtime/)
[![npm](https://img.shields.io/npm/v/ameva-vulkan-runtime.svg?style=flat-square&color=b91c1c)](https://www.npmjs.com/package/ameva-vulkan-runtime)
[![npm downloads](https://img.shields.io/npm/dm/ameva-vulkan-runtime.svg?style=flat-square&color=b91c1c)](https://www.npmjs.com/package/ameva-vulkan-runtime)
[![License](https://img.shields.io/badge/License-Apache_2.0-004499.svg?style=flat-square)](https://github.com/uno-km/ameva-vulkan-runtime)

> **안드로이드 Termux 환경에서 디바이스 리소스를 활용하는 SoC 감지 및 적응형 추상화 런타임**  
> *SoC-Aware Adaptive Abstraction Runtime Utilizing Device Resources for Android Termux*

---

## 📌 Architecture & Overview

Zero-Guesswork SoC 자동 감지 기능을 갖춘 C++20 하드웨어 추상화 계층(HAL)을 통해 Qualcomm Adreno 환경에서는 하드웨어 경로를, Exynos Mali 환경에서는 안정적인 ARM NEON 4-스레드 FP16 CPU 연산을 지원합니다.

Provides a C++20 Hardware Abstraction Layer (HAL) with Zero-Guesswork SoC auto-detection, utilizing hardware paths on Qualcomm Adreno and stable ARM NEON 4-Thread FP16 CPU execution on Exynos Mali.

---

## 🚀 Installation & Quickstart

### Python (PyPI)
```bash
pip install ameva-vulkan-runtime
```
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

### Node.js / TypeScript (npm)
```bash
npm install ameva-vulkan-runtime
```
```typescript
import { Doctor, createContext } from "ameva-vulkan-runtime";

const doctor = new Doctor();
const report = await doctor.runSelfTest();
console.log(`GPU Status: ${report.overallSuccess}, GPU: ${report.deviceName}`);

const ctx = await createContext({ device: "auto" });
console.log(`Vulkan Context Ready on ${ctx.deviceName}`);
```

---

## 📖 Official Documentation & Benchmarks
- [Official Architecture & API Reference](https://uno-km.vercel.app/lib/ameva-vulkan-runtime/)
- [Ecosystem Metrics & Registry Stats](https://uno-km.vercel.app/foundation/metrics)
- [AMEVA Open-Source Foundation Portal](https://uno-km.vercel.app/foundation/index.html)

---

## 📄 License
Licensed under the Apache-2.0 License. Copyright (c) 2026 Eunho Kim ([@uno-km](https://github.com/uno-km)).
