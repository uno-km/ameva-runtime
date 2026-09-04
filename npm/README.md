# AMEVA-Vulkan-Runtime (Node.js & TypeScript)

[![npm](https://img.shields.io/npm/v/ameva-vulkan-runtime.svg?style=flat-square&color=b91c1c)](https://www.npmjs.com/package/ameva-vulkan-runtime)
[![npm downloads](https://img.shields.io/npm/dm/ameva-vulkan-runtime.svg?style=flat-square&color=b91c1c)](https://www.npmjs.com/package/ameva-vulkan-runtime)
[![License](https://img.shields.io/badge/License-Apache_2.0-004499.svg?style=flat-square)](https://github.com/uno-km/ameva-vulkan-runtime)

> **안드로이드 Termux 환경에서 디바이스 리소스를 활용하는 SoC 감지 및 적응형 추상화 런타임**  
> *SoC-Aware Adaptive Abstraction Runtime Utilizing Device Resources for Android Termux*

## Installation

```bash
npm install ameva-vulkan-runtime
```

## Quickstart

```typescript
import { Doctor, createContext } from "ameva-vulkan-runtime";

const doctor = new Doctor();
const report = await doctor.runSelfTest();
console.log(`GPU Status: ${report.overallSuccess}, GPU: ${report.deviceName}`);

const ctx = await createContext({ device: "auto" });
console.log(`Vulkan Context Ready on ${ctx.deviceName}`);
```

## Description
Provides a C++20 Hardware Abstraction Layer (HAL) with Zero-Guesswork SoC auto-detection, utilizing hardware paths on Qualcomm Adreno and stable ARM NEON 4-Thread FP16 CPU execution on Exynos Mali.

## Documentation
- [Official Documentation & API Reference](https://uno-km.vercel.app/lib/ameva-vulkan-runtime/)
- [GitHub Repository](https://github.com/uno-km/ameva-vulkan-runtime)

## License
Apache-2.0 License. Copyright (c) 2026 Eunho Kim (@uno-km).
