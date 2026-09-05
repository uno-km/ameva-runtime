# AMEVA-Runtime

[![PyPI](https://img.shields.io/pypi/v/ameva-runtime.svg?style=flat-square&color=0369a1)](https://pypi.org/project/ameva-runtime/)
[![Python](https://img.shields.io/pypi/pyversions/ameva-runtime.svg?style=flat-square)](https://pypi.org/project/ameva-runtime/)
[![npm](https://img.shields.io/npm/v/%40ameva%2Fruntime.svg?style=flat-square&color=b91c1c)](https://www.npmjs.com/package/@ameva/runtime)
[![npm downloads](https://img.shields.io/npm/dm/%40ameva%2Fruntime.svg?style=flat-square&color=b91c1c)](https://www.npmjs.com/package/@ameva/runtime)
[![License](https://img.shields.io/badge/License-Apache_2.0-004499.svg?style=flat-square)](https://github.com/uno-km/ameva-runtime)

> **모바일 및 엣지 환경을 위한 차세대 통합 온디바이스 하드웨어 오케스트레이션 및 6대 멀티모달 가속 런타임**  
> *Next-Gen Unified On-Device Hardware Orchestration & 6-Modality AI Acceleration Runtime for Mobile & Edge*

---

## 📌 Architecture & Overview

AMEVA Runtime은 모바일 및 엣지 단말기를 위한 통합 하드웨어 오케스트레이션 및 AI 가속 엔진입니다. 런타임에 SoC 토폴로지와 GPU 벤더를 정밀 감지하여 Qualcomm Adreno, ARM Mali, CPU-NEON으로 최적 분기합니다.

### 1. 실기기 LLM 추론 성능 실측 벤치마크 (Qwen2.5-0.5B-Instruct, GGUF Q4_K_M)

| 단말기 및 SoC | GPU 및 드라이버 환경 | 실행 백엔드 | VRAM 레이어 | 생성 속도 (t/s) | 프롬프트 속도 (t/s) | 시스템 UI 프리즈 | 가속 비율 |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Galaxy S25** (Snapdragon 8 Elite) | Qualcomm Adreno 830 (Vulkan 1.3) | **VULKAN** | **25/25 (100%)** | **35.80 t/s** (27.9 ms/t) | **4.53 t/s** | **0% (안정)** | **35.8배 (vs CPU)** |
| **Galaxy A35** (Exynos 1380) | ARM Mali-G68 MP5 (Vulkan 1.3) | **VULKAN (Medium MatMul)** | **25/25 (100%)** | **4.44 t/s** (225 ms/t) | **6.12 t/s** | **0% (안정)** | **+26.9% (vs NEON)** |
| **Galaxy A35** (Exynos 1380) | Cortex-A78 CPU-NEON (3 Threads) | CPU-NEON | 0/25 | 3.55 t/s (281 ms/t) | 8.05 t/s | 0% (안정) | 기준선 |

### 2. 실기기 음성인식 (STT) 가속 실측 벤치마크 (Whisper Large-v3-Turbo Q5_0, 548MB)

- **테스트 환경**: Samsung Galaxy A35 5G (Exynos 1380, ARM Mali-G68 MP5, 8GB RAM, Android 16 Termux)
- **테스트 오디오**: JFK 1분 실제 연설 오디오 (`jfk_1min.wav`)

| 실행 모드 | 사용 하드웨어 | 소요 시간 | GPU 클럭 / 부하 | CPU 부하 | WER / 정확도 | 속도 향상 |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **CPU NEON 모드** (`-dev -1`, 4 스레드) | Cortex-A78 x4 코어 | **816.48s (13m 36s)** | 0% (유휴) | 291% (빅코어 점유) | 정상 인식 | 기준선 |
| **Vulkan GPU 모드** (`-dev 0`, Mali Quirk) | Mali-G68 MP5 | **360.60s (6m 00s)** | **949 MHz (100%)** | **20~30% (유휴)** | 정상 인식 | **2.26배 가속 (56% 시간 단축)** |

- **Galaxy S25 (Adreno 830) STT 실측**: Whisper Large-v3-Turbo 한국어 음성 4,401 ms 안정 추론 성공 (JIT 컴파일러 크래시 패치 적용).

### 3. 하드웨어 결함 규명 및 공학적 해결 내역 (Ground Truth)

#### (1) ARM Mali-G68 Valhall 연산 커널 정수 절삭 무한 루프 제거
- **결함 현상**: `mul_mm.comp` 실행 시 Mali-G68(서브그룹 크기 16)에서 GPU 연산이 멈추고 시스템 TDR 리셋(`VK_ERROR_DEVICE_LOST`)이 발생함.
- **원인 분석**: 셰이더 내 스트라이드 계산식 `loadstride_b = gl_WorkGroupSize.x * LOAD_VEC_B / BK = 16 * 1 / 32 = 0`에서 정수 나눗셈 결과가 0으로 절삭되어 내부 루프 `for (uint l = 0; l < BN; l += 0)`가 무한 반복됨.
- **해결 방안**: 워크그룹 크기 128인 미디엄 커널(`_m`, `loadstride_b = 4 > 0`) 강제 실행 규칙(`enforce_medium_matmul: true`)을 적용하여 무한 루프를 원천 제거하고 25/25 레이어 GPU 완주를 달성함.

#### (2) Qualcomm Adreno 830 컴파일러 JIT 결함 격리
- **결함 현상**: Snapdragon 8 Elite 환경에서 Whisper STT 실행 중 `mul_mat_vec` 셰이더 파이프라인 생성 시 `VK_ERROR_UNKNOWN (-13)` 크래시 발생.
- **원인 분석**: Qualcomm 독점 드라이버의 Adreno JIT 컴파일러가 특수화 상수 `NUM_COLS >= 3` 조건에서 레지스터 할당 실패를 일으킴.
- **해결 방안**: `mul_mat_vec_max_cols = 2` 파라미터 경계를 설정하여 JIT 크래시를 방지하고 한국어 음성 인식을 4,401 ms에 정상 완료함.

### 4. 6대 멀티모달 전방위 가속 로드맵 (Roadmap & Ongoing Work)

| 모달리티 | 타깃 엔진 및 모델 | 지원 상태 | 하드웨어 가속 메커니즘 |
| :--- | :--- | :---: | :--- |
| **1. LLM (Text)** | Llama.cpp (Qwen2.5, Llama 3.2) | **[v2.0.0 완료]** | Vulkan VRAM 25/25 layers 완전 오프로드 (Adreno 35.8 t/s, Mali 4.44 t/s) |
| **2. STT (Speech)** | Whisper.cpp (Large-v3-Turbo) | **[v2.0.0 완료]** | Vulkan 온디바이스 STT 가속 (Adreno 4.4s, Mali 2.26배 가속) |
| **3. Vision (VLM)** | CLIP, MobileVLM, LLaVA | **[v2.1.0 개발중]** | GGML Vulkan 이미지 비전 인코더 텐서 연산기 바인딩 |
| **4. Diffusion (Image)** | Stable Diffusion v1.5 / FLUX.1 | **[v2.2.0 개발중]** | 온디바이스 Vulkan UNet & DiT 텐서 오프로드 엔진 |
| **5. TTS (Audio)** | Piper / Sherpa-ONNX / Kokoro | **[v2.3.0 개발중]** | Vulkan/NPU 실시간 저지연 뉴럴 음성 합성 스트리밍 파이프라인 |
| **6. Train (Training)** | On-Device LoRA / QLoRA | **[v2.4.0 개발중]** | 스마트폰 로컬 Vulkan 경사하강 역전파 가속 파이프라인 |

### 5. 핵심 엔지니어링 원칙
- **Fail-Fast & Zero-Silent-Fallback**: 하드웨어 가속 실패 시 조용히 CPU로 폴백하여 사용자를 기만하는 행위를 원천 금지하며, 즉시 정확한 원인과 에러 코드를 분출합니다.
- **단일 패키지 아키텍처**: `pip install ameva-runtime` 및 `npm install @ameva/runtime` 단일 규격으로 통합 배포됩니다.


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


---

## 🚀 Installation & Quickstart

### Python (PyPI)
```bash
pip install ameva-runtime
```
```python
import ameva_runtime as ameva
from ameva_runtime import vulkan

# 1. Execute LLM inference directly with optimal on-device hardware dispatch
result = ameva.run(
    model="qwen2.5-0.5b",
    prompt="Space in Korean is:",
    max_tokens=32
)
print(f"Generated text: {result.text}")
print(f"Hardware backend: {result.backend_used} ({result.tokens_per_second:.2f} t/s)")

# 2. Hardware diagnostic inspection via Vulkan engine
doc = vulkan.Doctor()
report = doc.run_self_test(verbose=False)
print(f"GPU Target: {report.device_name} (Passed: {report.passed_stages}/{report.total_stages})")

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
- [Official Architecture & API Reference](https://uno-km.vercel.app/lib/ameva-runtime/)
- [Ecosystem Metrics & Registry Stats](https://uno-km.vercel.app/foundation/metrics)
- [AMEVA Open-Source Foundation Portal](https://uno-km.vercel.app/foundation/index.html)

---

## 📄 License
Licensed under the Apache-2.0 License. Copyright (c) 2026 Eunho Kim ([@uno-km](https://github.com/uno-km)).
