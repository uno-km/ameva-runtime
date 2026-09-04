# AMEVA-Runtime (Node.js & TypeScript)

[![npm](https://img.shields.io/npm/v/%40ameva%2Fruntime.svg?style=flat-square&color=b91c1c)](https://www.npmjs.com/package/@ameva/runtime)
[![npm downloads](https://img.shields.io/npm/dm/%40ameva%2Fruntime.svg?style=flat-square&color=b91c1c)](https://www.npmjs.com/package/@ameva/runtime)
[![License](https://img.shields.io/badge/License-Apache_2.0-004499.svg?style=flat-square)](https://github.com/uno-km/ameva-runtime)

> **모바일 및 엣지 환경을 위한 차세대 통합 온디바이스 하드웨어 오케스트레이션 및 멀티모달 가속 런타임**  
> *Next-Gen Unified On-Device Hardware Orchestration & Multi-Modal Acceleration Runtime for Mobile & Edge*

## Installation

```bash
npm install @ameva/runtime
```

## Quickstart

```typescript
import { createContext, Doctor } from "@ameva/runtime";

const doctor = new Doctor();
const report = await doctor.runSelfTest();
console.log(`Topology Status: ${report.overallSuccess}, Hardware: ${report.deviceName}`);

const ctx = await createContext({ device: "auto" });
console.log(`Runtime Context Initialized on ${ctx.deviceName}`);

```

## Description
AMEVA Runtime provides a unified, zero-silent-fallback hardware orchestration engine for generative AI on mobile and edge devices. It automatically resolves vendor divergence, dispatching to native Vulkan hardware offload on Qualcomm Adreno while isolating driver deadlocks on ARM Mali through deterministic Cortex-A78 CPU-NEON multi-threading.

### Empirical Hardware Benchmarks (Qwen2.5-0.5B-Instruct)

| Device & SoC | GPU & Driver Architecture | Backend Route | Layer Offload | Generation Speed | Prompt Eval Speed | UI Freeze | Speedup |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Galaxy S25** (Snapdragon 8 Elite) | Qualcomm Adreno 830 (Vulkan 1.3) | **VULKAN** | **25/25 (100% VRAM)** | **35.80 t/s** (27.93 ms/t) | **4.53 t/s** | **0%** | **35.8x** |
| **Galaxy A35** (Exynos 1380) | ARM Mali-G68 MP5 (Vulkan 1.3) | **CPU-NEON** | **0/25 (Safe Guard)** | **3.55 t/s** (295 ms/t) | **8.05 t/s** | **0%** | Baseline |
| **Galaxy A35** (Forced Vulkan Test) | ARM Mali-G68 MP5 (vulkan.mali.so) | VULKAN | 25/25 | 0.00 t/s (Deadlock) | - | 100% (Hung) | Failed |

### Technical Rationale for ARM Mali CPU-NEON Fallback
The proprietary ARM Mali Vulkan driver (`/vendor/lib64/hw/vulkan.mali.so` via `/dev/mali0`) enforces aggressive kernel-level dynamic power management (DVFS) tied to `SurfaceFlinger` display composition. During headless compute dispatches in terminal or CLI environments (lacking an active window swapchain), the Mali driver interprets the absence of display flips as an idle state, downclocking GPU compute cores and failing to signal in-flight completion fences. This causes host threads waiting on `vkWaitForFences` or `vkQueueWaitIdle` to deadlock indefinitely.

To guarantee rock-solid system stability and eliminate device lockups, the **AMEVA SmartRouter** automatically identifies ARM Mali GPUs, bypasses the unstable headless Vulkan path, and dispatches inference to the high-performance ARM Cortex-A78 CPU-NEON cluster.

### CLI Quickstart
```bash
# 1. Automatic hardware diagnosis and profile inspection
ameva-run doctor
ameva-run profile

# 2. Dry-run execution plan
ameva-run plan -m qwen2.5-0.5b

# 3. Direct model execution with automatic hardware acceleration
ameva-run exec -m qwen2.5-0.5b -p "Space in Korean is:" -n 32
```


## Documentation
- [Official Documentation & API Reference](https://uno-km.vercel.app/lib/ameva-vulkan-runtime/)
- [GitHub Repository](https://github.com/uno-km/ameva-runtime)

## License
Apache-2.0 License. Copyright (c) 2026 Eunho Kim (@uno-km).
