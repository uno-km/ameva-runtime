# AMEVA-Runtime

[![PyPI](https://img.shields.io/pypi/v/ameva-runtime.svg?style=flat-square&color=0369a1)](https://pypi.org/project/ameva-runtime/)
[![Python](https://img.shields.io/pypi/pyversions/ameva-runtime.svg?style=flat-square)](https://pypi.org/project/ameva-runtime/)
[![npm](https://img.shields.io/npm/v/@ameva/runtime.svg?style=flat-square&color=b91c1c)](https://www.npmjs.com/package/@ameva/runtime)
[![License](https://img.shields.io/badge/License-Apache_2.0-004499.svg?style=flat-square)](https://github.com/uno-km/ameva-runtime)

> Next-Gen Unified On-Device Hardware Orchestration & 6-Modality AI Acceleration Runtime for Mobile & Edge

---

## Architecture & Overview

AMEVA-Runtime is a hardware abstraction layer (HAL) and compute orchestration engine engineered specifically for mobile ARM64 devices (Android Termux, Linux Edge). It continuously inspects underlying silicon topology (/dev/kgsl-3d0, /dev/mali0) to route tensor execution across Qualcomm Adreno, ARM Mali, and ARM Cortex CPU-NEON backends.

### 6-Modality Acceleration Matrix

| Modality | Engine Integration | Status (v2.1.0) | Hardware Acceleration Mechanism |
| :--- | :--- | :---: | :--- |
| **1. LLM (Text)** | Llama.cpp (Qwen2.5, Llama 3.2) | **Production** | Full 25/25 layer VRAM offload (Adreno 35.80 t/s, Mali 4.44 t/s) |
| **2. STT (Speech)** | Whisper.cpp (Large-v3-Turbo) | **Production** | Vulkan compute shader acceleration (Adreno 4.4s, Mali 2.26x speedup) |
| **3. TTS (Audio)** | Sherpa-NCNN / Piper | **Production** | Pure Vulkan GPU neural synthesis (Adreno RTF 0.264x, Mali RTF 1.146x) |
| **4. Vision (VLM)** | CLIP / MobileVLM / LLaVA | **In Development** | GGML Vulkan vision encoder tensor bindings |
| **5. Diffusion (Image)** | Stable Diffusion v1.5 / FLUX.1 | **In Development** | On-device Vulkan UNet & DiT tensor offload |
| **6. Train (Training)** | On-Device LoRA / QLoRA | **In Development** | Mobile Vulkan gradient descent backpropagation |

---

## Empirical Physical Device Benchmarks

Tested on physical devices running Android 16 under Termux ARM64:

### 1. LLM Generation (Qwen2.5-0.5B-Instruct Q4_K_M)
| Target Device | Hardware Architecture | Active Backend | Layers in VRAM | Generation Speed | Prompt Processing | Speedup |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Galaxy S25** | Snapdragon 8 Elite / Adreno 830 | Vulkan 1.3 | **25/25 (100%)** | **35.80 t/s** (27.9 ms/t) | 4.53 t/s | **35.8x (vs CPU)** |
| **Galaxy A35** | Exynos 1380 / ARM Mali-G68 MP5 | Vulkan 1.3 | **25/25 (100%)** | **4.44 t/s** (225 ms/t) | 6.12 t/s | **+26.9% (vs NEON)** |
| **Galaxy A35** | Cortex-A78 CPU-NEON (3 Threads) | CPU-NEON | 0/25 | 3.55 t/s (281 ms/t) | 8.05 t/s | Baseline |

### 2. Speech-to-Text (Whisper Large-v3-Turbo Q5_0, 548MB)
| Target Device | Hardware Architecture | Backend Mode | Latency (1-min audio) | GPU Load | CPU Load | Speedup |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Galaxy A35** | Exynos 1380 / Mali-G68 MP5 | Vulkan GPU | **360.60 s (6m 00s)** | **949 MHz (100%)** | 20~30% | **2.26x (56% time saved)** |
| **Galaxy A35** | Cortex-A78 x4 Cores | CPU-NEON | 816.48 s (13m 36s) | 0% | 291% | Baseline |

### 3. Text-to-Speech (Termux-TTS v1.3.0 Vulkan)
| Target Device | Hardware Architecture | Model Tier | Audio Length | Compute Time | Real-Time Factor (RTF) | Status |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: |
| **Galaxy S25** | Snapdragon 8 Elite / Adreno 830 | `lessac-high-fp16` | 6.70 s | **6.65 s** | **0.993x** | Real-time Studio |
| **Galaxy S25** | Snapdragon 8 Elite / Adreno 830 | `lessac-medium` | 4.59 s | **1.21 s** | **0.264x** | 3.79x Faster than RT |
| **Galaxy A35** | Exynos 1380 / Mali-G68 MP5 | `lessac-medium` | 4.52 s | **5.18 s** | **1.146x** | Validated |

---

## Root-Cause Driver Solutions

1. **ARM Mali-G68 Valhall Integer Truncation**: Enforced medium tile matmul kernel dispatch (`loadstride_b = 4 > 0`), permanently eliminating shader zero-stride infinite loops on subgroup-16 hardware.
2. **Qualcomm Adreno 830 JIT Register Bug**: Bounded vector column specialization (`mul_mat_vec_max_cols = 2`), preventing compiler crash `VK_ERROR_UNKNOWN (-13)`.
3. **Zero-Silent-Fallback**: Guaranteed fail-fast architecture without silent CPU degradation upon GPU driver faults.

---

## Installation

```bash
# Python SDK
pip install ameva-runtime

# Node.js / TypeScript
npm install @ameva/runtime
```

---

## Quickstart

### Python SDK
```python
import ameva_runtime as ameva
from ameva_runtime import vulkan

# 1. Inspect on-device silicon topology
profile = ameva.detect_hardware()
print(f"SoC: {profile.soc_name} | GPU: {profile.gpu_vendor}")

# 2. Run hardware self-test
doc = vulkan.Doctor()
report = doc.run_self_test()
print(f"GPU: {report.device_name} (Passed: {report.passed_stages}/{report.total_stages})")
```

### Node.js / TypeScript
```typescript
import { Doctor, createContext } from '@ameva/runtime';

const doc = new Doctor();
const report = await doc.runSelfTest();
console.log(`Vulkan GPU: ${report.deviceName}`);
```

---

## Official Documentation & Benchmarks
- [Official Architecture & API Reference](https://uno-km.vercel.app/lib/vulkan/)
- [Ecosystem Metrics & Registry Stats](https://uno-km.vercel.app/foundation/metrics)
- [AMEVA Open-Source Foundation Portal](https://uno-km.vercel.app/foundation/index.html)

---

## License
Licensed under the Apache-2.0 License. Copyright (c) 2026 Eunho Kim ([@uno-km](https://github.com/uno-km)).
