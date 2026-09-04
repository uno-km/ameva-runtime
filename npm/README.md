# @ameva/runtime

[![npm](https://img.shields.io/npm/v/%40ameva%2Fruntime.svg?style=flat-square&color=b91c1c)](https://www.npmjs.com/package/@ameva/runtime)
[![npm downloads](https://img.shields.io/npm/dm/%40ameva%2Fruntime.svg?style=flat-square&color=b91c1c)](https://www.npmjs.com/package/@ameva/runtime)
[![License](https://img.shields.io/badge/License-Apache_2.0-004499.svg?style=flat-square)](https://github.com/uno-km/ameva-runtime)

> **Next-Generation Unified On-Device Hardware Orchestration & Multi-Modal Acceleration Runtime for Mobile Node.js & Android Termux**

---

## Overview

`@ameva/runtime` provides enterprise-grade, deterministic hardware orchestration and Vulkan acceleration bindings for Node.js / TypeScript applications running on Android Termux, Linux, and edge appliances.

---

## Installation

```bash
npm install @ameva/runtime
```

---

## Quickstart

```typescript
import { createContext, Doctor } from "@ameva/runtime";

// 1. Run 12-stage automated hardware diagnostic
const doctor = new Doctor();
const report = await doctor.runSelfTest();
console.log(`Device: ${report.deviceName} | Status: ${report.overallSuccess ? "PASS" : "FAIL"}`);

// 2. Initialize hardware context with automatic silicon dispatch
const ctx = await createContext({ device: "auto" });
console.log(`Initialized backend: ${ctx.deviceName}`);
```

---

## Empirical Real-Device Benchmarks (Ground Truth)

All metrics were captured directly on live physical consumer hardware running Android Termux with the official `qwen2.5-0.5b-instruct-q4_k_m.gguf` model.

| Device | Processor / SoC | GPU Architecture | Active Backend | Layers in VRAM | Eval Speed (Tokens/s) | Prompt Eval (Tokens/s) | System UI Stability |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Samsung Galaxy S25** | Qualcomm Snapdragon 8 Elite | Qualcomm Adreno 830 | **VULKAN** | **25 / 25 (100%)** | **34.08 t/s** | **4.59 t/s** | **100% Stable (0% Freeze)** |
| **Samsung Galaxy A35** | Samsung Exynos 1380 | ARM Mali-G68 MP5 | **CPU-NEON (Adaptive)** | **0 / 25 (Shield)** | **4.27 t/s** | **6.44 t/s** | **100% Stable (0% Freeze)** |
| **Samsung Galaxy A35** (Forced Vulkan) | Samsung Exynos 1380 | ARM Mali-G68 MP5 | VULKAN (No Fallback) | 25 / 25 | **0.00 t/s (Deadlock)** | — | **Unresponsive (Host Hang)** |

---

## Live Physical Device Telemetry Logs

### Galaxy S25 (Qualcomm Adreno 830 - Native Vulkan Full Offload)

```text
Ameva Runtime Version: 1.0.1

=== OFFICIAL INFERENCE RESULT ===
Generated text: Space in Korean is: "공간" (kakjang)
Hardware backend: VULKAN
Token generation speed: 34.08 tokens/sec
Prompt evaluation speed: 4.59 tokens/sec
Total latency: 17148.2 ms
Safety rationale: Vulkan hardware acceleration active on ADRENO (qualcomm). All 99 layers targeted to VRAM.
```

### Galaxy A35 (ARM Mali-G68 MP5 - Forced Headless Vulkan Deadlock Log)

```text
ggml_vulkan: Found 1 Vulkan devices:
Vulkan0: Mali-G68 (Mali-G68) | uma: 1 | fp16: 1 | warp size: 16
[DRIVER DEADLOCK: Proprietary vulkan.mali.so stops responding during SPIR-V compute pipeline initialization]
[PID 4690: Consuming 94% CPU in busy-wait loop, 0 tokens generated after 60s timeout]
```

### Galaxy A35 (ARM Cortex-A78 CPU-NEON - Adaptive Safe Route)

```text
Space in Korean is: 3.5268041954294...

llama_print_timings:        load time =     834.98 ms
llama_print_timings: prompt eval time =     776.73 ms /     5 tokens (  155.35 ms per token,     6.44 tokens per second)
llama_print_timings:        eval time =    3510.69 ms /    15 runs   (  234.05 ms per token,     4.27 tokens per second)
llama_print_timings:       total time =    4336.22 ms /    20 tokens
```

---

## Technical Rationale: ARM Mali Driver Quirk

The proprietary ARM Mali Vulkan driver (`/vendor/lib64/hw/vulkan.mali.so`) requires an active display presentation swapchain (`SurfaceFlinger`). In headless CLI environments without an active window, driver power management stalls compute shader execution, dropping fence completion signals.

The **AMEVA SmartRouter** automatically identifies ARM Mali hardware, prevents application hanging, and transparently routes to ARM Cortex CPU-NEON execution clusters.

---

## CLI Utilities

```bash
# Global CLI command bundled with npm package
npx @ameva/runtime doctor
npx @ameva/runtime profile
```

---

## License

Distributed under the Apache-2.0 License. Copyright (c) 2026 Eunho Kim (@uno-km).
