# AMEVA-Runtime (Node.js & TypeScript)

[![npm](https://img.shields.io/npm/v/%40ameva%2Fruntime.svg?style=flat-square&color=b91c1c)](https://www.npmjs.com/package/@ameva/runtime)
[![npm downloads](https://img.shields.io/npm/dm/%40ameva%2Fruntime.svg?style=flat-square&color=b91c1c)](https://www.npmjs.com/package/@ameva/runtime)
[![License](https://img.shields.io/badge/License-Apache_2.0-004499.svg?style=flat-square)](https://github.com/uno-km/ameva-runtime)

> **모바일 및 엣지 환경을 위한 차세대 통합 온디바이스 하드웨어 오케스트레이션 및 6대 멀티모달 가속 런타임**  
> *Next-Gen Unified On-Device Hardware Orchestration & 6-Modality AI Acceleration Runtime for Mobile & Edge*

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
AMEVA Runtime is a unified on-device hardware orchestration and AI acceleration engine engineered for mobile and edge systems. It dynamically inspects SoC topology and driver environments, routing compute graphs between Qualcomm Adreno, ARM Mali, and ARM Cortex CPU-NEON.

### 1. Empirical Real-Device LLM Benchmarks (Qwen2.5-0.5B-Instruct, GGUF Q4_K_M)

| Device & Processor | GPU Architecture | Active Backend | Layers in VRAM | Generation Speed (t/s) | Prompt Speed (t/s) | UI Freeze | Acceleration |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Galaxy S25** (Snapdragon 8 Elite) | Qualcomm Adreno 830 (Vulkan 1.3) | **VULKAN** | **25/25 (100%)** | **35.80 t/s** (27.9 ms/t) | **4.53 t/s** | **0% (Stable)** | **35.8x (vs CPU)** |
| **Galaxy A35** (Exynos 1380) | ARM Mali-G68 MP5 (Vulkan 1.3) | **VULKAN (Medium MatMul)** | **25/25 (100%)** | **4.44 t/s** (225 ms/t) | **6.12 t/s** | **0% (Stable)** | **+26.9% (vs NEON)** |
| **Galaxy A35** (Exynos 1380) | Cortex-A78 CPU-NEON (3 Threads) | CPU-NEON | 0/25 | 3.55 t/s (281 ms/t) | 8.05 t/s | 0% (Stable) | Baseline |

### 2. Empirical Real-Device STT Benchmarks (Whisper Large-v3-Turbo Q5_0, 548MB)

- **Test Device**: Samsung Galaxy A35 5G (Exynos 1380, ARM Mali-G68 MP5, 8GB RAM, Android 16 Termux)
- **Audio Source**: John F. Kennedy 1-minute speech sample (`jfk_1min.wav`)

| Execution Mode | Target Hardware | Elapsed Time | GPU Clock / Load | CPU Utilization | Accuracy | Speedup |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **CPU NEON Mode** (`-dev -1`, 4 threads) | Cortex-A78 x4 cores | **816.48s (13m 36s)** | 0% (Idle) | 291% (Active) | Standard | Baseline |
| **Vulkan GPU Mode** (`-dev 0`, Mali Quirk) | Mali-G68 MP5 | **360.60s (6m 00s)** | **949 MHz (100%)** | **20~30% (Low)** | Standard | **2.26x (56% time reduction)** |

- **Galaxy S25 (Adreno 830) STT**: Completed 4,401 ms inference on speech input (JIT compiler bug resolved).

### 3. Root-Cause Defect Resolution (Ground Truth)

#### (1) ARM Mali-G68 Valhall Integer Truncation Infinite Loop Elimination
- **Defect**: Executing `mul_mm.comp` on Mali-G68 (subgroup size 16) caused GPU hangs and hardware watchdog TDR resets (`VK_ERROR_DEVICE_LOST`).
- **Root Cause**: The stride calculation `loadstride_b = gl_WorkGroupSize.x * LOAD_VEC_B / BK = 16 * 1 / 32 = 0` truncated to zero in integer division, producing an infinite loop `for (uint l = 0; l < BN; l += 0)`.
- **Resolution**: Enforced Medium MatMul kernels (`_m`, workgroup size 128, `loadstride_b = 4 > 0`) via `enforce_medium_matmul: true`, enabling stable 25/25 layer GPU offloading.

#### (2) Qualcomm Adreno 830 JIT Compiler Bug Isolation
- **Defect**: Whisper STT pipeline compilation failed on Snapdragon 8 Elite with `VK_ERROR_UNKNOWN (-13)` during `mul_mat_vec` dispatch.
- **Root Cause**: Qualcomm's Adreno JIT compiler failed register allocation when Specialization Constant `NUM_COLS >= 3`.
- **Resolution**: Bound `mul_mat_vec_max_cols = 2` for Adreno 830, achieving stable GPU inference in 4,401 ms on speech input.

### 4. 6-Modality Acceleration Roadmap

| Modality | Engine & Architecture | Status | Hardware Acceleration Mechanism |
| :--- | :--- | :---: | :--- |
| **1. LLM (Text)** | Llama.cpp (Qwen2.5, Llama 3.2) | **[v2.0.0 Stable]** | Vulkan 25/25 layer full VRAM offload (Adreno 35.8 t/s, Mali 4.44 t/s) |
| **2. STT (Speech)** | Whisper.cpp (Large-v3-Turbo) | **[v2.0.0 Stable]** | Vulkan on-device STT acceleration (Adreno 4.4s, Mali 2.26x speedup) |
| **3. Vision (VLM)** | CLIP, MobileVLM, LLaVA | **[v2.1.0 WIP]** | GGML Vulkan image encoder tensor engine binding |
| **4. Diffusion (Image)** | Stable Diffusion v1.5 / FLUX.1 | **[v2.2.0 WIP]** | On-device Vulkan UNet & DiT tensor offload engine |
| **5. TTS (Audio)** | Piper / Sherpa-ONNX / Kokoro | **[v2.3.0 WIP]** | Real-time low-latency neural TTS streaming pipeline via Vulkan/NPU |
| **6. Train (Training)** | On-Device LoRA / QLoRA | **[v2.4.0 WIP]** | Smartphone local Vulkan gradient descent backpropagation engine |

### 5. Architectural Principles
- **Fail-Fast & Zero-Silent-Fallback**: Never disguise GPU failures as CPU success. If a hardware backend fails, immediate explicit exceptions and telemetry are raised.
- **Consolidated Single Package**: Distributed cleanly via `pip install ameva-runtime` and `npm install @ameva/runtime`.


## Documentation
- [Official Documentation & API Reference](https://uno-km.vercel.app/lib/ameva-runtime/)
- [GitHub Repository](https://github.com/uno-km/ameva-runtime)

## License
Apache-2.0 License. Copyright (c) 2026 Eunho Kim (@uno-km).
