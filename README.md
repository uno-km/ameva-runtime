# AMEVA-Vulkan-Runtime

[![PyPI](https://img.shields.io/pypi/v/ameva-vulkan-runtime.svg?style=flat-square&color=0369a1)](https://pypi.org/project/ameva-vulkan-runtime/)
[![Python](https://img.shields.io/pypi/pyversions/ameva-vulkan-runtime.svg?style=flat-square)](https://pypi.org/project/ameva-vulkan-runtime/)
[![npm](https://img.shields.io/npm/v/ameva-vulkan-runtime.svg?style=flat-square&color=b91c1c)](https://www.npmjs.com/package/ameva-vulkan-runtime)
[![npm downloads](https://img.shields.io/npm/dm/ameva-vulkan-runtime.svg?style=flat-square&color=b91c1c)](https://www.npmjs.com/package/ameva-vulkan-runtime)
[![License](https://img.shields.io/badge/License-Apache_2.0-004499.svg?style=flat-square)](https://github.com/uno-km/ameva-vulkan-runtime)

> **안드로이드 모바일 환경을 위한 통합 크로스 모달 Vulkan GPU 가속 런타임 및 하드웨어 추상화 계층(HAL)**  
> *Unified Cross-Modal Vulkan GPU Acceleration Runtime & HAL for Mobile Android*

---

## 📌 Architecture & Overview

단일 C++20 Vulkan 하드웨어 추상화 계층(HAL)과 통합 런타임을 통해 STT, 비전, LLM, 디퓨전, 트레이닝 전 모달리티를 가속하며, 12단계 정밀 자체 검증(V0~V11)과 무손실 자동 복구 체계를 제공합니다.

Provides a single, zero-hardcoded C++20 Vulkan Hardware Abstraction Layer (HAL) and universal runtime for STT, Vision, LLM, Diffusion, and Training with a granular 12-stage validation hierarchy (V0-V11) and zero-data-loss auto-recovery.

---

## 🚀 Installation & Quickstart

### Python (PyPI)
```bash
pip install ameva-vulkan-runtime
```
```python
import ameva_vulkan_runtime as avr

doctor = avr.Doctor()
report = doctor.run_self_test(verbose=True)
print(f"Backend: {report.recommended_backend}, Device: {report.device_name}")

ctx = avr.create_context(device="auto", memory_limit_mb=1024)
print(f"Context active: {ctx.is_active}, Backend: {ctx.backend_type}")
```

### Node.js / TypeScript (npm)
```bash
npm install ameva-vulkan-runtime
```
```typescript
import { Doctor, createContext } from "ameva-vulkan-runtime";

const doctor = new Doctor();
const report = await doctor.runSelfTest(true);
console.log(`Backend: ${report.recommendedBackend}, Device: ${report.deviceName}`);

const ctx = createContext({ device: "auto" });
console.log(`Vulkan Context Active: ${ctx.isActive} (${ctx.backendType})`);
```

---

## 📖 Official Documentation & Benchmarks
- [Official Architecture & API Reference](https://uno-km.vercel.app/lib/ameva-vulkan-runtime/)
- [Ecosystem Metrics & Registry Stats](https://uno-km.vercel.app/foundation/metrics)
- [AMEVA Open-Source Foundation Portal](https://uno-km.vercel.app/foundation/index.html)

---

## 📝 Release Notes

### [v1.1.0] — 2026-09-01
- **Real-Device Galaxy A35 Validation**: Certified 12-stage hardware diagnostic hierarchy and Vulkan 1.4 API negotiation on Samsung Galaxy A35 5G (Exynos 1380, ARM Mali-G68 GPU).
- **CTypes Struct Pointer Patch**: Corrected `pApplicationInfo`, `pQueuePriorities`, and `pQueueCreateInfos` to `ctypes.pointer` to fix `TypeError: expected LP_* instance, got _ctypes.CArgObject` in `doctor.py`.
- **Dynamic Topology Thread Optimization**: Added CPU topology inspection in `LlamaCppAdapter` targeting big-core clusters (`-t 4`) on mobile octa-core SoCs.
- **Strict Vulkan 3-Tier Execution Mode**: Fail-Fast protection for explicit `--device vulkan` requests without silent CPU masking.
- **Multi-Registry Availability**: Published synchronously to PyPI (`pip install ameva-vulkan-runtime`) and npm (`npm install ameva-vulkan-runtime`).

---

## 📄 License
Licensed under the Apache-2.0 License. Copyright (c) 2026 Eunho Kim ([@uno-km](https://github.com/uno-km)).
