# AMEVA-Vulkan-Runtime

[![PyPI](https://img.shields.io/pypi/v/ameva-vulkan-runtime.svg?style=flat-square&color=0369a1)](https://pypi.org/project/ameva-vulkan-runtime/)
[![Python](https://img.shields.io/pypi/pyversions/ameva-vulkan-runtime.svg?style=flat-square)](https://pypi.org/project/ameva-vulkan-runtime/)
[![npm](https://img.shields.io/npm/v/ameva-vulkan-runtime.svg?style=flat-square&color=b91c1c)](https://www.npmjs.com/package/ameva-vulkan-runtime)
[![npm downloads](https://img.shields.io/npm/dm/ameva-vulkan-runtime.svg?style=flat-square&color=b91c1c)](https://www.npmjs.com/package/ameva-vulkan-runtime)
[![License](https://img.shields.io/badge/License-Apache_2.0-004499.svg?style=flat-square)](https://github.com/uno-km/ameva-vulkan-runtime)
[![Tests](https://img.shields.io/badge/Tests-100%25%20PASS-10b981.svg?style=flat-square)](https://github.com/uno-km/ameva-vulkan-runtime)

> **안드로이드 모바일 환경을 위한 통합 크로스 모달 Vulkan GPU 가속 런타임 및 하드웨어 추상화 계층 (HAL)**  
> *Unified Cross-Modal Vulkan GPU Acceleration Runtime & Hardware Abstraction Layer (HAL) for Mobile Android*

---

## 📌 Architectural Breakthrough

모바일 Android(Termux) 환경에서 멀티모달 온디바이스 AI를 구동할 때 발생하는 **Bionic-Mesa 로더 충돌(SIGABRT), Adreno 서브그룹 제어 버그, Mali 128-byte 메모리 비정렬 크래시, 패키지별 중복 바이너리 비대화**를 단일 C++20 하드웨어 추상화 계층으로 완전히 해결합니다.

```
[ 상위 6대 모달리티 에코시스템 ]
termux-stt │ termux-vision │ termux-llamacpp │ termux-bitnet │ termux-diffusion │ termux-train
     │             │               │                 │               │               │
     └─────────────┴───────────────┼─────────────────┴───────────────┴───────────────┘
                                   ▼
                   [ ameva-vulkan-runtime (Core HAL) ]
        ┌──────────────────────────┴──────────────────────────┐
        ▼                                                     ▼
 [ 12-Stage Doctor (V0~V11) ]                [ Zero-Drift Quirks Dispatcher ]
 • V0: Bionic System ICD Loader              • Adreno 830/750/740 Subgroup Fix
 • V5: Physical Device & Compute Queues      • Mali-G78/G68/G77 128-byte Strict Align
 • V10: SGEMM Float32 Accuracy Guard         • Single-Chain Bionic ICD Protection
 • V11: E2E Modality Binding & Auto-Recovery • Xclipse (AMD RDNA) Compatibility
```

---

## ⚡ 12-Stage Probing & Validation Hierarchy (V0 ~ V11)

런타임 초기화 시 12단계 정밀 자체 검증을 수행하며, 하드웨어 결함 발생 시 무손실 CPU NEON 자동 복구를 보장합니다:

| 단계 | 검증 항목 (Stage Probe) | 검증 메커니즘 & 목적 | 결함 시 복구 정책 |
| :---: | :--- | :--- | :---: |
| **V0** | `Vulkan Loader Linkage` | `/system/lib64/libvulkan.so` Bionic ICD 단일 체인 로드 | CPU NEON 직행 |
| **V1** | `Instance Creation` | `vkCreateInstance` API 버전 및 확장 프로빙 | Fail-Fast / Fallback |
| **V2** | `Physical Device Enumeration` | 가용 GPU 유무 및 드라이버 버전 식별 | CPU Fallback |
| **V3** | `Queue Family Discovery` | `VK_QUEUE_COMPUTE_BIT` 전용 큐 탐색 | Fallback |
| **V4** | `Logical Device & Features` | FP16 연산, 서브그룹 및 메모리 모델 활성화 | Fallback |
| **V5** | `Device Memory Allocation` | UMA 메모리 정렬 및 VRAM 쿼터 사전 승인 | Fallback |
| **V6** | `SPIR-V Shader Pipeline` | 컴파일된 SPIR-V 바이트코드 파이프라인 생성 | Fallback |
| **V7** | `Descriptor Set Layout` | 텐서 버퍼 디스크립터 바인딩 검증 | Fallback |
| **V8** | `Command Buffer Submission` | 비동기 Compute Queue 제출 및 펜스 동기화 | Fallback |
| **V9** | `Pipeline Execution` | 단일 워크그룹 셰이더 실행 및 동기화 | Fallback |
| **V10** | `SGEMM Accuracy Guard` | 행렬 곱셈 부동소수점 정밀도 검증 (Error < $10^{-4}$) | Fallback |
| **V11** | `E2E Modality Binding` | 6대 모달리티 실제 모델 파이프라인 바인딩 | CPU 복구 |

---

## 📱 Hardware Support Matrix (14대 실기기 프로파일)

[`profiles/validated-vulkan-profiles.json`](https://github.com/uno-km/ameva-vulkan-runtime/blob/main/profiles/validated-vulkan-profiles.json)에 공식 등록된 대표 검증 기기:

| 프로세서 (SoC) | 타겟 기기명 | 탑재 GPU | Vulkan API | 정밀도 오차 (V10) | 검증 상태 |
| :--- | :--- | :--- | :---: | :---: | :---: |
| **Snapdragon 8 Elite** | Galaxy S25 | Qualcomm Adreno 830 | **1.3** | $1.24 \times 10^{-5}$ | **VERIFIED_PRODUCTION** |
| **Snapdragon 8 Gen 3** | Galaxy S24 Ultra | Qualcomm Adreno 750 | **1.3** | $1.35 \times 10^{-5}$ | **VERIFIED_INHERITED** |
| **Snapdragon 8 Gen 2** | Galaxy S23 | Qualcomm Adreno 740 | **1.3** | $1.40 \times 10^{-5}$ | **VERIFIED_INHERITED** |
| **Exynos 2100** | Galaxy S21 5G | ARM Mali-G78 MP14 | **1.1** | $9.39 \times 10^{-5}$ | **VERIFIED_PRODUCTION** |
| **Exynos 1380** | Galaxy A35 5G | ARM Mali-G68 MP5 | **1.1** | $9.39 \times 10^{-5}$ | **VERIFIED_PRODUCTION** |
| **Exynos 1380** | Galaxy A54 5G | ARM Mali-G68 MP5 | **1.1** | $9.39 \times 10^{-5}$ | **VERIFIED_INHERITED** |
| **Exynos 1280** | Galaxy A53 5G | ARM Mali-G68 MP4 | **1.1** | $9.45 \times 10^{-5}$ | **VERIFIED_INHERITED** |
| **Exynos 990** | Galaxy S20 5G | ARM Mali-G77 MP11 | **1.1** | $1.12 \times 10^{-4}$ | **VERIFIED_INHERITED** |

---

## 🚀 Installation & Rapid Start

### Python (PyPI)
```bash
pip install ameva-vulkan-runtime
```

```python
import ameva_vulkan_runtime as avr

# 1. Run 12-Stage Hardware Validation Doctor
doctor = avr.Doctor()
report = doctor.run_self_test()
print(f"GPU Backend: {report.status} | Device: {report.device_name} (Driver: {report.driver_version})")

# 2. Acquire High-Performance Vulkan Acceleration Context
ctx = avr.create_context(device="auto", memory_limit_mb=1024)
print(f"Context Initialized: {ctx.loader_path} (API Level {ctx.vulkan_version})")

# 3. Bind to Downstream Modality Engine (STT / LLM / Diffusion / Vision / Train)
from ameva_vulkan_runtime.adapters import LlamaCppAdapter
binding = LlamaCppAdapter.bind(engine=None, report=report)
print(f"Adapter Config: {binding.config}")
```

### Node.js / TypeScript (npm)
```bash
npm install ameva-vulkan-runtime
```

```typescript
import { Doctor, createContext, LlamaCppAdapter } from "ameva-vulkan-runtime";

const doctor = new Doctor();
const report = await doctor.runSelfTest();
console.log(`GPU Status: ${report.status}, GPU: ${report.deviceName}`);

const ctx = await createContext({ device: "auto" });
console.log(`Vulkan Context Ready on ${ctx.deviceName}`);
```

---

## 🔗 Official Documentation & Ecosystem

- **공식 기술 문서**: [https://uno-km.vercel.app/lib/vulkan/](https://uno-km.vercel.app/lib/vulkan/)
- **중앙 재단 포털**: [https://uno-km.vercel.app/foundation/](https://uno-km.vercel.app/foundation/)
- **실시간 지표 대시보드**: [https://uno-km.vercel.app/foundation/metrics](https://uno-km.vercel.app/foundation/metrics)
- **GitHub 저장소**: [https://github.com/uno-km/ameva-vulkan-runtime](https://github.com/uno-km/ameva-vulkan-runtime)

---

## 📄 License

Licensed under the **Apache-2.0 License**.  
Copyright (c) 2026 Eunho Kim ([@uno-km](https://github.com/uno-km)). All rights reserved.
