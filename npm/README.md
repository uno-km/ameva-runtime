# ameva-vulkan-runtime (Node.js & TypeScript)

[![npm](https://img.shields.io/npm/v/ameva-vulkan-runtime.svg?style=flat-square&color=b91c1c)](https://www.npmjs.com/package/ameva-vulkan-runtime)
[![npm downloads](https://img.shields.io/npm/dm/ameva-vulkan-runtime.svg?style=flat-square&color=b91c1c)](https://www.npmjs.com/package/ameva-vulkan-runtime)
[![License](https://img.shields.io/badge/License-Apache_2.0-004499.svg?style=flat-square)](https://github.com/uno-km/ameva-vulkan-runtime)

> **안드로이드 모바일 환경을 위한 통합 크로스 모달 Vulkan GPU 가속 런타임 및 하드웨어 추상화 계층 (HAL)**  
> *Unified Cross-Modal Vulkan GPU Acceleration Runtime & HAL for Mobile Android*

---

## Installation

```bash
npm install ameva-vulkan-runtime
```

## Quickstart

```typescript
import { Doctor, createContext } from "ameva-vulkan-runtime";

const doctor = new Doctor();
const report = await doctor.runSelfTest();
console.log(`GPU Status: ${report.status}, GPU: ${report.deviceName}`);

const ctx = await createContext({ device: "auto" });
console.log(`Vulkan Context Ready on ${ctx.deviceName}`);
```

## Description
Provides a single, zero-hardcoded C++20 Vulkan Hardware Abstraction Layer (HAL) and universal runtime for STT, Vision, LLM, Diffusion, and Training with a granular 12-stage validation hierarchy (V0-V11) and zero-data-loss auto-recovery.

## Documentation
- [Official Documentation & API Reference](https://uno-km.vercel.app/lib/vulkan/)
- [GitHub Repository](https://github.com/uno-km/ameva-vulkan-runtime)

## License
Apache-2.0 License. Copyright (c) 2026 Eunho Kim (@uno-km).
