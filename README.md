# AMEVA-Runtime

[![PyPI](https://img.shields.io/pypi/v/ameva-runtime.svg?style=flat-square&color=0369a1)](https://pypi.org/project/ameva-runtime/)
[![Python](https://img.shields.io/pypi/pyversions/ameva-runtime.svg?style=flat-square)](https://pypi.org/project/ameva-runtime/)
[![npm](https://img.shields.io/npm/v/%40ameva%2Fruntime.svg?style=flat-square&color=b91c1c)](https://www.npmjs.com/package/@ameva/runtime)
[![npm downloads](https://img.shields.io/npm/dm/%40ameva%2Fruntime.svg?style=flat-square&color=b91c1c)](https://www.npmjs.com/package/@ameva/runtime)
[![License](https://img.shields.io/badge/License-Apache_2.0-004499.svg?style=flat-square)](https://github.com/uno-km/ameva-runtime)

> **모바일 및 엣지 환경을 위한 차세대 통합 온디바이스 하드웨어 오케스트레이션 및 멀티모달 가속 런타임**  
> *Next-Gen Unified On-Device Hardware Orchestration & Multi-Modal Acceleration Runtime for Mobile & Edge*

---

## 📌 Architecture & Overview

AMEVA Runtime은 모바일 및 엣지 단말기를 위한 침묵 폴백(Zero-Silent-Fallback) 없는 차세대 통합 하드웨어 오케스트레이션 엔진입니다. 단말기별 GPU 벤더 아키텍처를 런타임에 정밀 진단하여 Qualcomm Adreno 환경에서는 네이티브 Vulkan VRAM 오프로드(Snapdragon 8 Elite 기준 35.80 tokens/sec)를 수행하고, ARM Mali 환경에서는 커널 드라이버 락업을 원천 차단하는 ARM Cortex-A78 CPU-NEON(3.55 tokens/sec)으로 자동 라우팅합니다.

### 실기기 추론 성능 실측 벤치마크 (Qwen2.5-0.5B-Instruct)

| 단말기 및 SoC | GPU 및 드라이버 환경 | 실행 백엔드 | GPU 레이어 적재 | 토큰 생성 속도 | 프롬프트 평가 속도 | 화면 프리즈 | 상대 가속도 |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Galaxy S25** (Snapdragon 8 Elite) | Qualcomm Adreno 830 (Vulkan 1.3) | **VULKAN** | **25/25 (100% VRAM)** | **35.80 t/s** (27.93 ms/t) | **4.53 t/s** | **0%** | **35.8배** |
| **Galaxy A35** (Exynos 1380) | ARM Mali-G68 MP5 (Vulkan 1.3) | **CPU-NEON** | **0/25 (안전 보호)** | **3.55 t/s** (295 ms/t) | **8.05 t/s** | **0%** | 기준선 |
| **Galaxy A35** (강제 Vulkan 시험) | ARM Mali-G68 MP5 (vulkan.mali.so) | VULKAN | 25/25 | 0.00 t/s (데드락) | - | 100% (먹통) | 실패 |

### ARM Mali GPU의 CPU-NEON 안전 라우팅 사유
ARM Mali 독점 드라이버는 화면 갱신(SurfaceFlinger Swapchain)이 없는 터미널(헤드리스 CLI) 환경에서 전력 관리 절전 진입으로 인해 `vkWaitForFences` 커널 펜스 신호를 유실하는 치명적인 데드락 결함을 가지고 있습니다. AMEVA Runtime은 이를 런타임에 자동 감지하여 Cortex-A78 CPU-NEON 클러스터로 즉각 분기함으로써 화면 먹통 없는 100% 안정 구동을 보장합니다.

### CLI 빠른 실행 가이드
```bash
# 하드웨어 진단 및 토폴로지 프로파일 확인
ameva-run doctor
ameva-run profile

# 하드웨어 분기 계획 사전 확인
ameva-run plan -m qwen2.5-0.5b

# 최적 하드웨어 백엔드로 실제 모델 추론 실행
ameva-run exec -m qwen2.5-0.5b -p "Space in Korean is:" -n 32
```


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


---

## 🚀 Installation & Quickstart

### Python (PyPI)
```bash
pip install ameva-runtime
```
```python
import ameva_runtime as ameva

# 1. Execute LLM inference directly with optimal on-device hardware dispatch
result = ameva.run(
    model="qwen2.5-0.5b",
    prompt="Space in Korean is:",
    max_tokens=32
)

print(f"Generated text: {result.text}")
print(f"Hardware backend: {result.backend_used} ({result.tokens_per_second:.2f} t/s)")

# 2. Preview hardware execution plan
plan = ameva.plan(model="qwen2.5-0.5b")
print(f"Route: {plan.backend} | NGL: {plan.ngl} | Threads: {plan.threads}")

```

### Node.js / TypeScript (npm)
```bash
npm install @ameva/runtime
```
```typescript
import { createContext, Doctor } from "@ameva/runtime";

const doctor = new Doctor();
const report = await doctor.runSelfTest();
console.log(`Topology Status: ${report.overallSuccess}, Hardware: ${report.deviceName}`);

const ctx = await createContext({ device: "auto" });
console.log(`Runtime Context Initialized on ${ctx.deviceName}`);

```

---

## 📖 Official Documentation & Benchmarks
- [Official Architecture & API Reference](https://uno-km.vercel.app/lib/ameva-vulkan-runtime/)
- [Ecosystem Metrics & Registry Stats](https://uno-km.vercel.app/foundation/metrics)
- [AMEVA Open-Source Foundation Portal](https://uno-km.vercel.app/foundation/index.html)

---

## 📄 License
Licensed under the Apache-2.0 License. Copyright (c) 2026 Eunho Kim ([@uno-km](https://github.com/uno-km)).
