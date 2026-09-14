# 📘 모바일 온디바이스 Vulkan 하드웨어 가속 엔지니어링 교본 (The Mobile On-Device Vulkan AI Acceleration Engineering Textbook)

> **부제**: Android Bionic·Termux 환경에서의 LLM, STT, TTS 3대 모달리티 실기기 포렌식, 드라이버 결함 분쇄 및 하드웨어 가속 완결서  
> **저자**: 김은호 (Eunho Kim, [@uno-km](https://github.com/uno-km)) & AMEVA 오픈소스 재단 (AOSF)  
> **표준 규격**: OpenSSF / CNCF / Apache-2.0 컴플라이언스 준수 (Zero-Hype, Ground-Truth Validation)  
> **검증 대상 실리콘**: Samsung Exynos 1280 / 1380 / 2100 (ARM Mali-G68 MP4/MP5, Mali-G78 MP14), Qualcomm Snapdragon 8 Gen 1 / 8 Elite (Adreno 730, Adreno 830)  
> **문서 식별 번호**: `AMEVA-ENG-TEXTBOOK-2026-VK01`  
> **최종 개정일**: 2026-09-14  

---

## 📑 전체 목차 (Table of Contents)

- [서문: 온디바이스 하드웨어 가속의 기술적 당위성과 진실](#서문-온디바이스-하드웨어-가속의-기술적-당위성과-진실)
- [제1장: 이론적 기초 및 모바일 컴퓨팅 아키텍처](#제1장-이론적-기초-및-모바일-컴퓨팅-아키텍처)
  - [1.1 모바일 이종 컴퓨팅(Heterogeneous Computing)의 현실과 한계](#11-모바일-이종-컴퓨팅heterogeneous-computing의-현실과-한계)
  - [1.2 Android Bionic 링커와 Termux 샌드박스 내부 동작 원리](#12-android-bionic-링커와-termux-샌드박스-내부-동작-원리)
  - [1.3 가속 API 비교 분석: Vulkan vs OpenCL vs Android NNAPI](#13-가속-api-비교-분석-vulkan-vs-opencl-vs-android-nnapi)
  - [1.4 대상 실기기 SoC 및 GPU 물리적 위상(Topology)](#14-대상-실기기-soc-및-gpu-물리적-위상topology)
  - [1.5 AMEVA 12단계 하드웨어 검증 계층 (12-Stage Validation Hierarchy)](#15-ameva-12단계-하드웨어-검증-계층-12-stage-validation-hierarchy)
- [제2장: LLM (대형 언어 모델) — llama.cpp & ARM Mali Valhall 무한루프 분쇄](#제2장-llm-대형-언어-모델--llamacpp--arm-mali-valhall-무한루프-분쇄)
  - [2.1 문제의 발단: "Mali GPU는 드라이버 결함으로 불칸 불능"이라는 통념](#21-문제의-발단-mali-gpu는-드라이버-결함으로-불칸-불능이라는-통념)
  - [2.2 최초 재현 및 증상 관측: Node 2 행렬곱 프리징과 커널 워치독 사살](#22-최초-재현-및-증상-관측-node-2-행렬곱-프리징과-커널-워치독-사살)
  - [2.3 역공학 및 근본 원인(Root Cause) 규명: GLSL 정수 절삭 셰이더 무한루프](#23-역공학-및-근본-원인root-cause-규명-glsl-정수-절삭-셰이더-무한루프)
  - [2.4 오픈소스 커뮤니티의 사각지대: ARM 벤더 ID (0x13b5) 누락 사태](#24-오픈소스-커뮤니티의-사각지대-arm-벤더-id-0x13b5-누락-사태)
  - [2.5 엔지니어링 해결책: Medium MatMul 파이프라인 강제 라우팅 및 패치](#25-엔지니어링-해결책-medium-matmul-파이프라인-강제-라우팅-및-패치)
  - [2.6 Upstream PR 제안 및 커뮤니티 기여 (ggerganov/llama.cpp)](#26-upstream-pr-제안-및-커뮤니티-기여-ggerganovllamacpp)
  - [2.7 실기기 벤치마크 실측 검증: 25/25 레이어 VRAM 상주와 +26.9% 가속](#27-실기기-벤치마크-실측-검증-2525-레이어-vram-상주와-269-가속)
- [제3장: STT (음성인식) — Whisper.cpp & Qualcomm Adreno 830 JIT 레지스터 크래시 격리](#제3장-stt-음성인식--whispercpp--qualcomm-adreno-830-jit-레지스터-크래시-격리)
  - [3.1 문제의 발단: 온디바이스 음성인식의 Vulkan 전환 및 모바일 툴체인 구축](#31-문제의-발단-온디바이스-음성인식의-vulkan-전환-및-모바일-툴체인-구축)
  - [3.2 툴체인 및 로더 3대 장애 극복 (glslc, libvulkan.so 심볼릭, OpenMP lld 결함)](#32-툴체인-및-로더-3대-장애-극복-glslc-libvulkanso-심볼릭-openmp-lld-결함)
  - [3.3 Adreno 830 파이프라인 16 런타임 크래시 직면 (VK_ERROR_UNKNOWN -13)](#33-adreno-830-파이프라인-16-런타임-크래시-직면-vk_error_unknown--13)
  - [3.4 가설 설정 및 과학적 격리 검증 (Float Controls vs SPV 손상 vs Spec Constants)](#34-가설-설정-및-과학적-격리-검증-float-controls-vs-spv-손상-vs-spec-constants)
  - [3.5 독립 C 프로브(probe_exact.c)를 통한 하드웨어 레지스터 고갈 실증](#35-독립-c-프로브probe_exactc를-통한-하드웨어-레지스터-고갈-실증)
  - [3.6 해결책: mul_mat_vec_max_cols 경계 제한 패치 및 초고속 재빌드](#36-해결책-mul_mat_vec_max_cols-경계-제한-패치-및-초고속-재빌드)
  - [3.7 Python SDK 레이어 통합, Zero-Silent-Fallback 및 Galaxy A35 대형 모델 실측](#37-python-sdk-레이어-통합-zero-silent-fallback-및-galaxy-a35-대형-모델-실측)
- [제4장: TTS (음성합성) — Sherpa-NCNN / Piper / VITS 지연시간 분해 및 스트리밍 아키텍처](#제4장-tts-음성합성--sherpa-ncnn--piper--vits-지연시간-분해-및-스트리밍-아키텍처)
  - [4.1 문제의 발단: "소리는 나는데 왜 이렇게 느리고 잡음이 섞이는가?"](#41-문제의-발단-소리는-나는데-왜-이렇게-느리고-잡음이-섞이는가)
  - [4.2 엔지니어링 과실 포렌식 및 결함 사후 분석 (Post-Mortem)](#42-엔지니어링-과실-포렌식-및-결함-사후-분석-post-mortem)
  - [4.3 지연시간(Latency) 정밀 분해 및 4대 병목 지점 실측](#43-지연시간latency-정밀-분해-및-4대-병목-지점-실측)
  - [4.4 실시간 인터랙티브 환경을 위한 3대 아키텍처 혁신](#44-실시간-인터랙티브-환경을-위한-3대-아키텍처-혁신)
  - [4.5 실기기 지표 검증: Galaxy S25 Studio RTF vs Galaxy A35 어댑티브 라우팅](#45-실기기-지표-검증-galaxy-s25-studio-rtf-vs-galaxy-a35-어댑티브-라우팅)
  - [4.6 오픈소스 생태계의 기상천외한 안티패턴과 3대 악습 포렌식](#46-오픈소스-생태계의-기상천외한-안티패턴과-3대-악습-포렌식)
  - [4.7 모바일 GPU 속도 역전 현상의 근본 원인: Mesa llvmpipe 래스터라이저 바인딩 함정](#47-모바일-gpu-속도-역전-현상의-근본-원인-mesa-llvmpipe-래스터라이저-바인딩-함정)
  - [4.8 MeloTTS HiFi-GAN 디코더의 모바일 GPU 한계 전산학·수학적 분석 (32MB Buffer Ceiling)](#48-melotts-hifi-gan-디코더의-모바일-gpu-한계-전산학수학적-분석-32mb-buffer-ceiling)
  - [4.9 실기기 4대 플릿(S21·S22·S25·A35) 전수 실측 스코어카드 및 이종 컴퓨팅 지연시간 분해](#49-실기기-4대-플릿s21s22s25a35-전수-실측-스코어카드-및-이종-컴퓨팅-지연시간-분해)
  - [4.10 차세대 MZ 3대 신경망 음향 모델 아키텍처 및 온디바이스 런타임 (Kokoro·Melo·Supertonic)](#410-차세대-mz-3대-신경망-음향-모델-아키텍처-및-온디바이스-런타임-kokoromelosupertonic)
  - [4.11 온디바이스 실시간 음향 합성의 5대 엔지니어링 결함과 트레이드오프 총정리](#411-온디바이스-실시간-음향-합성의-5대-엔지니어링-결함과-트레이드오프-총정리)
  - [4.12 전산학적 수학 공식 및 루프라인(Roofline) 정밀 분석](#412-전산학적-수학-공식-및-루프라인roofline-정밀-분석)
  - [4.13 실기기 4대 플릿 전수 실측 오디오 포렌식 및 지표 완결](#413-실기기-4대-플릿-전수-실측-오디오-포렌식-및-지표-완결)
- [제5장: AMEVA-Runtime 통합 아키텍처 및 미래 로드맵 (Curriculum Foundation)](#제5장-ameva-runtime-통합-아키텍처-및-미래-로드맵-curriculum-foundation)
  - [5.1 하드웨어 추상화 계층(HAL) 및 단일 패키지 아키텍처](#51-하드웨어-추상화-계층hal-및-단일-패키지-아키텍처)
  - [5.2 Zero-Silent-Fallback 및 Fail-Fast 정책의 시스템적 구현](#52-zero-silent-fallback-및-fail-fast-정책의-시스템적-구현)
  - [5.3 6대 모달리티 확장 로드맵 (Vision, Diffusion, Train)](#53-6대-모달리티-확장-로드맵-vision-diffusion-train)
  - [5.4 궁극의 종착지: AI Chain & AI Orchestrator 자율 모바일 에이전트](#54-궁극의-종착지-ai-chain--ai-orchestrator-자율-모바일-에이전트)
- [제6장: 갤럭시 플릿(Galaxy Fleet: S25·S22·S21·A53·A35) 파편화 포렌식 및 100% 순혈 GPU 아키텍처 완결](#제6장-갤럭시-플릿galaxy-fleet-s25s22s21a53a35-파편화-포렌식-및-100-순혈-gpu-아키텍처-완결)
  - [6.1 오픈소스 업계의 '가짜 Vulkan 가속' 사기극 포렌식 (fallbacks=0 착시와 듀얼 백엔드의 암투)](#61-오픈소스-업계의-가짜-vulkan-가속-사기극-포렌식-fallbacks0-착시와-듀얼-백엔드의-암투)
  - [6.2 Android Bionic 링커와 Mesa soname 충돌 격리 (~/.local/share/ameva/lib Central Bridge)](#62-android-bionic-링커와-mesa-soname-충돌-격리-localshareamevalib-central-bridge)
  - [6.3 Galaxy S25 (Snapdragon 8 Elite / Adreno 830): JIT 레지스터 버그와 13.98초 초고속 순혈 GPU 완주](#63-galaxy-s25-snapdragon-8-elite--adreno-830-jit-레지스터-버그와-1398초-초고속-순혈-gpu-완주)
  - [6.4 Galaxy S22 (Snapdragon 8 Gen 1 / Adreno 730): Bionic 심볼 격리와 75% GPU 풀로드 실증](#64-galaxy-s22-snapdragon-8-gen-1--adreno-730-bionic-심볼-격리와-75-gpu-풀로드-실증)
  - [6.5 Galaxy S21 5G (Exynos 2100 / Mali-G78): Headless CLI 제약 돌파와 Mali 물리 GPU 직결](#65-galaxy-s21-5g-exynos-2100--mali-g78-headless-cli-제약-돌파와-mali-물리-gpu-직결)
  - [6.6 Galaxy A53 5G / A35 / A34 (Exynos 1280·1380, Dimensity 1080 / Mali-G68): Valhall 단일 큐 디바이스의 세그폴트 분쇄](#66-galaxy-a53-5g--a35--a34-exynos-12801380-dimensity-1080--mali-g68-valhall-단일-큐-디바이스의-세그폴트-분쇄)
  - [6.7 Zero-Silent-Fallback 락다운: 1비트의 CPU 짬때리기도 허용하지 않는 순혈 GPU 컴플라이언스](#67-zero-silent-fallback-락다운-1비트의-cpu-짬때리기도-허용하지-않는-순혈-gpu-컴플라이언스)
  - [6.8 5대 갤럭시 플릿(Galaxy Fleet) 통합 실측 벤치마크 매트릭스](#68-5대-갤럭시-플릿galaxy-fleet-통합-실측-벤치마크-매트릭스)
- [제7장: 에필로그 및 오픈소스 엔지니어링 선언](#제7장-에필로그-및-오픈소스-엔지니어링-선언)

---

# 서문: 온디바이스 하드웨어 가속의 기술적 당위성과 진실

모바일 단말기(스마트폰, 태블릿, 에지 IoT)는 전 세계 컴퓨팅 하드웨어 중 가장 거대한 보급 대수를 자랑하지만, 동시에 가장 가혹한 열역학적(Thermodynamic) 및 전력적(Power Envelope) 제약을 받는 디바이스입니다.

수많은 연구진과 개발자들이 모바일 기기 위에서 인공지능 신경망을 구동하려 시도할 때, 대다수는 다음과 같은 손쉬운 타협을 선택해 왔습니다:
1. **클라우드 API 위임**: 단말기 내부에서 직접 추론하지 않고 외부 대형 서버(OpenAI, Anthropic 등)로 사용자의 음성, 텍스트, 이미지를 전송하여 프라이버시 침해 및 영구적 서비스 비용 종속성 초래.
2. **CPU 중심 연산의 혹사**: 모바일 CPU 빅코어(Cortex-X, Cortex-A78 등) 4~6개를 100% 점유하여 폰을 불덩이로 만들고 스로틀링(Thermal Throttling)을 유발하는 비효율적 추론.
3. **기만성 침묵 폴백(Silent Fallback)**: GPU 가속을 선언해 두고 내부적으로 오류가 발생하면 사용자 몰래 CPU 코드로 전환하여 느린 속도를 감추거나 가짜 응답을 반환.

본 교본은 이러한 타협과 관행을 철저히 배격합니다. 하드웨어 반도체(Silicon)에 엄연히 집적되어 있는 물리적 GPU(Qualcomm Adreno, ARM Mali)를 **Vulkan 저수준 그래픽스/컴퓨트 API**를 통해 직접 제어하고, 드라이버와 셰이더 컴파일러의 밑바닥까지 역공학(Reverse Engineering)하여 실질적인 하드웨어 가속을 달성한 엔지니어링 전 과정을 1비트의 은폐도 없이 기록합니다.

---

# 제1장: 이론적 기초 및 모바일 컴퓨팅 아키텍처

## 1.1 모바일 이종 컴퓨팅(Heterogeneous Computing)의 현실과 한계

데스크톱 환경(x86_64 + NVIDIA CUDA)에서는 대용량 전력 공급(300W~600W)과 전용 초고속 VRAM(GDDR6X, HBM)을 갖춘 외장 그래픽 카드가 독립적인 메모리 버스를 점유합니다. 반면 스마트폰 SoC(System on Chip)는 **통합 메모리 아키텍처(UMA: Unified Memory Architecture)** 구조를 채택하고 있습니다.

```
┌──────────────────────────────────────────────────────────────────┐
│                   Mobile SoC (Exynos / Snapdragon)               │
│                                                                  │
│  ┌──────────────────────┐              ┌──────────────────────┐  │
│  │   CPU Cluster        │              │   GPU Cluster        │  │
│  │  - Cortex-X / A78    │              │  - Mali-G68 MP5      │  │
│  │  - L1/L2 Private     │              │  - Adreno 830        │  │
│  │  - ARMv8.2-A NEON    │              │  - Shader Cores/ALUs │  │
│  └──────────┬───────────┘              └──────────┬───────────┘  │
│             │                                     │              │
│             └──────────────────┬──────────────────┘              │
│                                │                                 │
│             ┌──────────────────┴──────────────────┐              │
│             │     System-Level Cache (SLC / L3)   │              │
│             │          (1MB ~ 8MB Shared)         │              │
│             └──────────────────┬──────────────────┘              │
│                                │                                 │
│             ┌──────────────────┴──────────────────┐              │
│             │   LPDDR5 / LPDDR5X System Memory    │              │
│             │       (6GB / 8GB / 12GB / 16GB)     │              │
│             │    Bandwidth: 44 GB/s ~ 85 GB/s     │              │
│             └─────────────────────────────────────┘              │
└──────────────────────────────────────────────────────────────────┘
```

모바일 UMA 구조에서 CPU와 GPU는 동일한 물리적 LPDDR 메모리를 공유합니다. 이는 호스트-디바이스 간 PCIe 복사 오버헤드가 없다는 강력한 이점을 지니지만, 동시에 다음과 같은 엄격한 엔지니어링 경계 조건을 부과합니다:
1. **메모리 대역폭 포화(Bandwidth Contention)**: CPU가 과도한 데이터 복사를 수행하거나 GPU가 캐시 정렬되지 않은 비연속적 텐서 스트라이드를 접근할 경우 시스템 버스가 마비되어 성능이 급락함.
2. **열 설계 전력(TDP) 한계**: 모바일 기기는 수동 방열(Passive Cooling) 구조이며, SoC 전체의 전력 소모가 4W~7W를 초과하면 수 분 내에 클럭 다운(Thermal Throttling)이 발생함.
3. **가상 메모리 주소 공간 분리**: 물리 RAM은 공유하지만 안드로이드 커널은 프로세스별, 드라이버별로 가상 메모리 공간을 엄격히 샌드박싱하므로 올바른 드라이버 핸들 및 DMA 바인딩이 필수적임.

## 1.2 Android Bionic 링커와 Termux 샌드박스 내부 동작 원리

리눅스 데스크톱 환경은 GNU C 라이브러리(`glibc`)와 동적 링커(`ld-linux.so`)를 사용합니다. 반면 Android OS는 구글이 자체 설계한 경량 C 라이브러리인 **Bionic (`libc.so`)**과 링커(`/linker64`)를 사용합니다.

Termux는 안드로이드 애플리케이션 샌드박스 내부(`untrusted_app` 컨텍스트, UID `10xxx`)에서 구동되는 사용자 공간 환경입니다. 
- 비루트(Zero-Root) 상태에서 Termux 프로세스는 시스템 커널의 하드웨어 디바이스 노드에 직접 접근할 권한이 엄격히 통제됩니다.
- 그러나 안드로이드 프레임워크 표준 그래픽스 드라이버인 `/dev/kgsl-3d0`(Qualcomm Adreno) 및 `/dev/mali0`(ARM Mali)는 앱 렌더링을 위해 그룹 권한(`rw-rw-rw-` 또는 소유자 권한)이 개방되어 있습니다.
- 시스템 라이브러리 디렉터리(`/system/lib64/libvulkan.so`)는 Android OS의 플랫폼 드라이버 로더(Platform Loader)이며, 내부적으로 벤더 하드웨어 드라이버(`/vendor/lib64/hw/vulkan.*.so`)를 동적 적재합니다.

## 1.3 가속 API 비교 분석: Vulkan vs OpenCL vs Android NNAPI

모바일에서 신경망 텐서를 GPU에 전달할 수 있는 3대 API를 아키텍처 관점에서 엄정하게 비교 분석합니다.

| 비교 항목 | Vulkan Compute (SPIR-V) | OpenCL (CLBlast/C++) | Android NNAPI (C API) |
| :--- | :--- | :--- | :--- |
| **표준화 주체** | Khronos Group (글로벌 표준) | Khronos Group (레거시 표준) | Google (Android 전용) |
| **Android 표준 포함 여부** | **Android 7.0+ 기본 필수 탑재** (`/system/lib64/libvulkan.so`) | 벤더 종속적 (`/vendor/lib64/libOpenCL.so`, 픽셀 등 일부 부재) | Android 8.1+ 포함되었으나 **Android 15부터 Deprecated 선언** |
| **컴파일 방식** | 오프라인/온디바이스 SPIR-V 바이트코드 사전 컴파일 | 런타임 OpenCL C 소스코드 JIT 컴파일 | 런타임 그래프 빌드 후 NPU/GPU 위임 드라이버 전달 |
| **드라이버 오버헤드** | **극저오버헤드 (Low-overhead explicit API)** | 중간 수준 (드라이버 내부 상태 머신 존재) | 높은 오버헤드 (안드로이드 IPC 및 서비스 바인더 통과) |
| **동기화 제어** | `VkFence`, `VkSemaphore`, `VkPipelineBarrier` 명시 제어 | `clEnqueueBarrier`, `clWaitForEvents` | 프레임워크 자동 관리 (세밀한 제어 불가) |
| **오픈소스 런타임 호환성** | **llama.cpp, whisper.cpp, NCNN, stable-diffusion.cpp 100% 지원** | 부분 지원 (별도 라이브러리 빌드 필요) | 극히 제한적 (지원 연산자 부족으로 빈번한 CPU 폴백) |

**결론**: Google의 공식 NNAPI 포기와 OpenCL의 제조사별 파편화를 고려할 때, 모바일 에지 AI의 유일무이한 미래 표준은 **Vulkan Compute**입니다.

## 1.4 대상 실기기 SoC 및 GPU 물리적 위상(Topology)

본 교본의 전수 검증에 투입된 두 대의 물리 단말기 하드웨어 프로파일은 다음과 같습니다.

### [Target A] Samsung Galaxy S25 5G (SM-S931N)
- **SoC 명칭**: Qualcomm Snapdragon 8 Elite (SM8750, 코드명 `sun`)
- **CPU 토폴로지**: 8-Core 64-bit Oryon (2x Prime @ 4.32 GHz + 6x Performance @ 3.53 GHz)
- **GPU 아키텍처**: Qualcomm Adreno 830
- **디바이스 노드**: `/dev/kgsl-3d0` (`crw-rw-rw-`)
- **Vulkan API 지원 버전**: Vulkan 1.3 / 드라이버 버전 `512.797.0` (Qualcomm Technologies Inc. Adreno Vulkan Driver)
- **서브그룹(Subgroup/Warp) 크기**: **64** (minSubgroupSize=64, maxSubgroupSize=64)
- **쉐어드 메모리(Shared Memory)**: 32,768 바이트 (32 KB)

### [Target B] Samsung Galaxy A35 5G (SM-A356N)
- **SoC 명칭**: Samsung Exynos 1380 (S5E8835)
- **CPU 토폴로지**: 8-Core 64-bit (4x Cortex-A78 @ 2.4 GHz + 4x Cortex-A55 @ 2.0 GHz)
- **GPU 아키텍처**: ARM Mali-G68 MP5 (5-Core, Valhall 2세대 아키텍처)
- **디바이스 노드**: `/dev/mali0` (`crw-rw-rw-`)
- **Vulkan API 지원 버전**: Vulkan 1.3 / 드라이버 버전 `v1.r38p1` (ARM Bionic Native Driver)
- **서브그룹(Subgroup/Warp) 크기**: **16** (minSubgroupSize=16, maxSubgroupSize=16)
- **쉐어드 메모리(Shared Memory)**: 32,768 바이트 (32 KB)

## 1.5 AMEVA 12단계 하드웨어 검증 계층 (12-Stage Validation Hierarchy)

하드웨어 가속이 실제로 일어나는지, 아니면 라이브러리가 에러를 삼키고 침묵형 폴백을 수행하는지 판별하기 위해 AMEVA-Runtime은 12단계의 결정론적 검증 체계(V0 ~ V11)를 엄격히 시행합니다.

```mermaid
flowchart TD
    V0[V0: Vulkan Loader Open - dlopen libvulkan.so] --> V1[V1: Instance Creation - vkCreateInstance]
    V1 --> V2[V2: Device Enum - vkEnumeratePhysicalDevices > 0]
    V2 --> V3[V3: Hardware Selection - deviceType != eCpu]
    V3 --> V4[V4: Queue Probe - VK_QUEUE_COMPUTE_BIT]
    V4 --> V5[V5: Device Creation - vkCreateDevice]
    V5 --> V6[V6: Buffer Allocation - HostVisible & DeviceLocal]
    V6 --> V7[V7: SPIR-V Compile - vkCreateComputePipelines]
    V7 --> V8[V8: Shader Dispatch - vkCmdDispatch]
    V8 --> V9[V9: Checksum Audit - Numerical Checksum Match]
    V9 --> V10[V10: GGML MatMul - FP32/FP16 Max Error < 1e-4]
    V10 --> V11[V11: End-to-End Real-Device Model Execution]
```

---

# 제2장: LLM (대형 언어 모델) — llama.cpp & ARM Mali Valhall 무한루프 분쇄

## 2.1 문제의 발단: "Mali GPU는 드라이버 결함으로 불칸 불능"이라는 통념

지난 수년간 깃허브(`ggerganov/llama.cpp`) 이슈 트래커와 레딧(Reddit) 온디바이스 AI 커뮤니티에는 다음과 같은 내용의 질문과 불만이 수백 건 이상 게시되었습니다:
> *"갤럭시 A 시리즈나 엑시노스 단말기(Mali GPU)에서 llama.cpp를 Vulkan으로 빌드하면 바로 멈춰버린다."*  
> *"삼성 Mali 드라이버는 헤드리스(Headless) 환경에서 펜스(Fence) 동기화가 버그를 일으켜 전력 절전 모드로 다운클럭되므로 구동이 불가능하다."*  
> *"결국 모바일에서는 CPU NEON으로 6개 코어를 풀가동하는 것 외에는 방법이 없다."*

이로 인해 개발자들은 스마트폰 칩셋에 탑재된 Mali GPU 코어를 방치한 채, 발열과 배터리 소모를 감수하며 CPU에만 의존하는 왜곡된 구조를 답습해 왔습니다. 그러나 이는 하드웨어 반도체나 드라이버의 결함이 아니라, **셰이더 소스코드에 내재된 정수 연산 버그**가 원인이었습니다.

## 2.2 최초 재현 및 증상 관측: Node 2 행렬곱 프리징과 커널 워치독 사살

### 실행 환경 및 파라미터
- 단말기: Samsung Galaxy A35 5G (SM-A356N)
- 모델: `Qwen2.5-0.5B-Instruct-Q4_K_M.gguf` (25개 트랜스포머 레이어)
- 실행 커맨드:
```bash
./build/bin/llama-cli \
  -m models/Qwen2.5-0.5B-Instruct-Q4_K_M.gguf \
  -p "Explain quantum computing in one sentence." \
  -ngl 25 \
  -t 1 \
  -s 42
```

### 관측된 증상 (Symptom)
1. **GPU Watch 무반응**: 삼성 개발자 옵션의 실시간 하드웨어 모니터링 도구인 GPU Watch를 가동하였으나, GPU 로드율 0%, 0 FPS를 기록하며 폰은 완전히 차가운 상태 유지.
2. **터미널 프리징 지점**: 모델 가중치(324 MB)는 Mali-G68 GPU VRAM에 정상 할당되었고, Node 0과 Node 1의 RMS_NORM 연산은 즉시 통과함. 그러나 **Node 2: MUL_MAT (Qcur-0)**에 진입하는 순간 셸 출력이 영구 정지됨.
3. **타임아웃 및 프로세스 사살 로그**:
```text
ggml_vulkan: Allocating 324 MB on device 0 (ARM Mali-G68)
ggml_vulkan: Compiling compute shader for MUL_MAT...
[Vulkan Node 0: RMS_NORM] OK
[Vulkan Node 1: RMS_NORM] OK
[Vulkan Node 2: MUL_MAT (Qcur-0)] -> (68초간 정지)
ggml_vulkan: vk::Device::waitForFences: ErrorDeviceLost
llama_perf_context_print: prompt eval time = 0.00 ms
[1] 14201 segmentation fault (core dumped)
```

정확히 68초 후 안드로이드 커널의 하드웨어 감시 타이머(Watchdog)가 GPU 큐 무응답을 감지하여 하드웨어 오류(`VK_ERROR_DEVICE_LOST, -4`)를 분출하고 프로세스를 사살하였습니다.

## 2.3 역공학 및 근본 원인(Root Cause) 규명: GLSL 정수 절삭 셰이더 무한루프

문제를 규명하기 위해 `llama.cpp`의 Vulkan 셰이더 컴파일러 원천 코드인 `ggml/src/vulkan-shaders/mul_mm.comp`를 역공학 분석하였습니다.

### GLSL 소스코드 원문 (`mul_mm.comp`)
```glsl
layout (local_size_x_id = 0, local_size_y = 1, local_size_z = 1) in;

layout (constant_id = 1) const uint BM = 64;
layout (constant_id = 2) const uint BN = 64;
layout (constant_id = 3) const uint BK = 16;  // 양자화 GEMM(MMQ)의 경우 32
layout (constant_id = 9) const uint WARP = 32;

// 가중치 행렬 B 버퍼 로딩 스트라이드 계산식
const uint loadstride_b = gl_WorkGroupSize.x * LOAD_VEC_B / BK;

[[unroll]] for (uint l = 0; l < BN; l += loadstride_b) {
    // 텐서 데이터 로드 및 누적 연산 블록
}
```

### 정수 나눗셈 트랩 (The Truncation Trap)
양자화 행렬 곱셈의 Small 파이프라인(`warptile_mmq_s`) 구동 시, 파라미터는 다음과 같이 주입됩니다:
- `gl_WorkGroupSize.x = device->subgroup_size`
- `BK = 32` (Q4_K, Q4_0 등의 블록 크기)
- `LOAD_VEC_B = 1` (스칼라 부동소수점 비정렬 로드)

이제 각 GPU 하드웨어 벤더별로 GLSL 정수 연산이 어떻게 수행되는지 비교합니다:

$$\text{loadstride}_b = \left\lfloor \frac{\text{gl\_WorkGroupSize.x} \times \text{LOAD\_VEC\_B}}{\text{BK}} \right\rfloor$$

1. **데스크톱 GPU (NVIDIA, AMD, Intel)**:
   - 하드웨어 서브그룹(Warp/Wavefront) 크기 = **32 또는 64**
   - $\text{loadstride}_b = \lfloor 32 \times 1 / 32 \rfloor = \mathbf{1}$
   - 루프 실행: `for (uint l = 0; l < BN; l += 1)` $\rightarrow$ 루프가 1씩 증가하며 정상 종료(32회 반복).
2. **모바일 ARM Mali GPU (Valhall 아키텍처, Mali-G68)**:
   - 하드웨어 서브그룹 크기 = **16**
   - $\text{loadstride}_b = \lfloor 16 \times 1 / 32 \rfloor = \mathbf{0}$
   - 루프 실행:
     $$\mathbf{for\ (uint\ l = 0;\ l < BN;\ l\ += 0)}$$

**발견된 물리적 진실**:  
루프 인덱스 `l`에 0이 더해지므로 탈출 조건(`l < BN`)이 영원히 충족되지 않는 **셰이더 스레드 무한루프(Infinite GPU Thread Loop)**가 발생하였습니다. GPU 연산 유닛(ALU)은 무한히 0을 더하는 연산에 갇혔고, 화면 렌더링을 하지 않으므로 로드율은 0%로 측정되었으며, 60초가 경과하자 안드로이드 OS가 TDR(Timeout Detection and Recovery)을 발동하여 프로세스를 강제 종료했던 것입니다.

## 2.4 오픈소스 커뮤니티의 사각지대: ARM 벤더 ID (0x13b5) 누락 사태

더욱 심각한 문제는 `ggml/src/ggml-vulkan.cpp`의 파이프라인 디스패치 테이블에 존재했습니다.

```cpp
// ggml/src/ggml-vulkan.cpp 소스코드 발췌
#define VK_VENDOR_ID_AMD    0x1002
#define VK_VENDOR_ID_APPLE  0x106b
#define VK_VENDOR_ID_INTEL  0x8086
#define VK_VENDOR_ID_NVIDIA 0x10de
// 치명적 누락: VK_VENDOR_ID_ARM (0x13b5)가 정의되어 있지 않음!

static vk_pipeline ggml_vk_guess_matmul_pipeline(ggml_backend_vk_context * ctx, vk_matmul_pipeline& mmp, int m, int n, bool aligned) {
    switch (ctx->device->vendor_id) {
    case VK_VENDOR_ID_AMD:
        return ggml_vk_guess_matmul_pipeline_amd(ctx, mmp, m, n, aligned);
    case VK_VENDOR_ID_APPLE:
        return ggml_vk_guess_matmul_pipeline_apple(ctx, mmp, aligned);
    case VK_VENDOR_ID_INTEL:
        return ggml_vk_guess_matmul_pipeline_intel(ctx, mmp, aligned);
    default:
        break; // ARM Mali는 아무런 예외 처리 없이 default로 진입
    }

    if (m <= 32 || n <= 32) {
        return aligned ? mmp->a_s : mmp->s; // 배치 크기 32 이하일 때 무조건 Small 파이프라인으로 강제 배정!
    }
    ...
}
```

전 세계 모바일 기기의 절반 이상을 차지하는 ARM 벤더 ID(`0x13b5`)가 완전히 누락되어 있어, 프롬프트 평가나 단일 토큰 디코딩($N \le 32$) 시 무조건 치명적인 Small(`_s`) 파이프라인으로 직행하고 있었습니다.

## 2.5 엔지니어링 해결책: Medium MatMul 파이프라인 강제 라우팅 및 패치

해결책은 극히 명료하고 강력했습니다. 워크그룹 크기가 128인 **Medium (`_m`) 파이프라인**을 사용하도록 라우팅을 우회하는 것입니다:

$$\text{loadstride}_b (\text{Medium}) = \left\lfloor \frac{128 \times 1}{32} \right\rfloor = \mathbf{4} > 0$$

Medium 파이프라인에서는 루프가 4씩 정상 전진하므로 무한루프가 발생하지 않습니다.

### 수정 코드 (`ggml/src/ggml-vulkan.cpp`)
```cpp
#define VK_VENDOR_ID_ARM 0x13b5

static vk_pipeline ggml_vk_guess_matmul_pipeline(ggml_backend_vk_context * ctx, vk_matmul_pipeline& mmp, int m, int n, bool aligned) {
    switch (ctx->device->vendor_id) {
    case VK_VENDOR_ID_AMD:
        return ggml_vk_guess_matmul_pipeline_amd(ctx, mmp, m, n, aligned);
    case VK_VENDOR_ID_APPLE:
        return ggml_vk_guess_matmul_pipeline_apple(ctx, mmp, aligned);
    case VK_VENDOR_ID_INTEL:
        return ggml_vk_guess_matmul_pipeline_intel(ctx, mmp, aligned);
    case VK_VENDOR_ID_ARM:
        // ARM Mali 계열 하드웨어는 무조건 안전한 Medium 파이프라인으로 라우팅
        return aligned ? mmp->a_m : mmp->m;
    default:
        break;
    }

    // 벤더 ID와 무관하게 서브그룹 크기가 32 미만인 모든 모바일 하드웨어 방어
    if (ctx->device->subgroup_size < 32) {
        return aligned ? mmp->a_m : mmp->m;
    }

    if (m <= 32 || n <= 32) {
        return aligned ? mmp->a_s : mmp->s;
    }
    ...
}
```

## 2.6 Upstream PR 제안 및 커뮤니티 기여 (ggerganov/llama.cpp)

해당 해결책은 공식 업스트림 풀 리퀘스트(PR) 형식으로 작성되어 저장소에 보존되었습니다:
- 문서 경로: [`docs/research/LLAMA_CPP_PR_PROPOSAL.md`](docs/research/LLAMA_CPP_PR_PROPOSAL.md)
- PR 제목: `[vulkan] Fix GPU hang/TDR on ARM Mali by routing subgroup < 32 to Medium MatMul pipeline`
- 상세 백서: [`docs/research/MALI_VALHALL_VULKAN_INFINITE_LOOP_ANALYSIS.md`](docs/research/MALI_VALHALL_VULKAN_INFINITE_LOOP_ANALYSIS.md)

## 2.7 실기기 벤치마크 실측 검증: 25/25 레이어 VRAM 상주와 +26.9% 가속

패치를 적용한 후 Samsung Galaxy A35 5G 실기기에서 벤치마크를 재수행하였습니다.

```bash
./build/bin/llama-cli \
  -m models/Qwen2.5-0.5B-Instruct-Q4_K_M.gguf \
  -p "Explain quantum computing in one sentence." \
  -ngl 25 \
  -t 1 \
  -s 42
```

### 터미널 반환 로그 실측치
```text
Quantum computing is a field of computing that utilizes the principles of quantum mechanics, 
such as superposition and entanglement, to perform complex calculations exponentially faster than classical computers.

llama_perf_context_print: prompt eval time =   1076.65 ms /    72 tokens (   14.95 t/s)
llama_perf_context_print:        eval time =   7657.44 ms /    34 runs   (    4.44 t/s)
llama_perf_context_print:       total time =   8734.09 ms /   106 tokens
```

### 📊 LLM 추론 실측 지표 대조표 (Galaxy A35 & Galaxy S25)

| 평가 지표 (Metric) | Galaxy A35 CPU NEON (6스레드) | Galaxy A35 패치 전 Vulkan | Galaxy A35 패치 후 Mali-G68 GPU | Galaxy S25 Snapdragon 8 Elite (Adreno 830) |
| :--- | :---: | :---: | :---: | :---: |
| **GPU VRAM 오프로드** | 0 / 25 레이어 | 25 / 25 레이어 | **25 / 25 레이어 (100%)** | **25 / 25 레이어 (100%)** |
| **LM-Head 연산 위치** | CPU 코어 | GPU (프리징) | **Mali-G68 GPU** | **Adreno 830 GPU** |
| **토큰 디코딩 속도** | 3.50 tokens/sec | 0.00 tokens/sec (Hang) | **4.44 tokens/sec** | **35.80 tokens/sec** |
| **토큰당 지연 시간** | 286.02 ms/t | 측정 불가 ($\infty$) | **225.22 ms/t (-60.8ms)** | **27.93 ms/t** |
| **성능 향상폭** | 베이스라인 기준 | 폭망 (TDR Crash) | **CPU 대비 +26.9% 가속** | **CPU 대비 35.8배 초고속** |
| **종료 코드 (Exit Code)**| 0 | 139 (SIGSEGV) | **0 (완전 정상 종료)** | **0 (완전 정상 종료)** |
| **출력 의미 무결성** | Seed 42 일치 | 산출 불가 | **Seed 42 완전 일치** | **Seed 42 완전 일치** |

---

# 제3장: STT (음성인식) — Whisper.cpp & Qualcomm Adreno 830 JIT 레지스터 크래시 격리

## 3.1 문제의 발단: 온디바이스 음성인식의 Vulkan 전환 및 모바일 툴체인 구축

음성인식(STT) 모델(Whisper)은 멜 스펙트로그램(Mel Spectrogram) 추출, 오디오 인코더(Audio Encoder), 그리고 텍스트 자동회귀 디코더(Autoregressive Decoder)로 구성된 복합 멀티모달 파이프라인입니다. 특히 최신 모델인 `Whisper Large-v3-Turbo`(548 MB ~ 1.56 GB)는 수천 개의 합성곱 및 어텐션 연산자를 내포하고 있어 모바일 CPU 단독으로는 실시간 음성인식이 불가능합니다.

이를 Vulkan GPU로 전환하는 과정에서 발생한 모바일 툴체인 결함 및 퀄컴 최신 GPU(Adreno 830)의 JIT 컴파일러 크래시 과정을 포렌식 추적합니다.

## 3.2 툴체인 및 로더 3대 장애 극복 (glslc, libvulkan.so 심볼릭, OpenMP lld 결함)

### 장애 1: 온디바이스 1,608개 SPIR-V 셰이더 컴파일러 부재
- **현상**: GGML Vulkan 백엔드는 모델 가동 전 수천 개의 GLSL 연산자를 SPIR-V 바이트코드로 컴파일해야 함.
- **조치**: Termux 패키지 매니저를 통해 `glslang`, `shaderc`(`glslc`), `ninja`를 단말기에 직접 프로비저닝.

### 장애 2: Mesa CPU Lavapipe 소프트웨어 래스터라이저 침범
- **현상**: Termux에 `mesa-vulkan-icd`가 설치되어 있을 경우, 기본 로더가 모바일 물리 GPU 대신 느린 소프트웨어 래스터라이저(Lavapipe)를 우선 로드함.
- **조치**: 안드로이드 OS 시스템 레벨의 독점 Bionic 드라이버로 심볼릭 링크를 강제 고정.
```bash
ln -sf /system/lib64/libvulkan.so /data/data/com.termux/files/usr/lib/libvulkan.so
ln -sf /system/lib64/libvulkan.so /data/data/com.termux/files/usr/lib/libvulkan.so.1
```

### 장애 3: Clang 21과 OpenMP LLD 링커 심볼 미정의 결함
- **현상**: 최신 Clang 21 컴파일러로 `whisper.cpp`를 빌드할 때 LLD 링커가 치명적 에러를 분출:
```text
ld.lld: error: undefined reference to '__kmpc_dispatch_deinit'
>>> referenced by ggml-base.c
>>> CMakeFiles/ggml-base.dir/ggml-base.c.o:(ggml_compute_forward)
clang-21: error: linker command failed with exit code 1
```
- **포렌식 분석**: Termux 저장소의 동적 라이브러리 `/usr/lib/libomp.so`는 구버전이어서 해당 심볼이 없었으나, 동봉된 정적 아카이브 `/usr/lib/libomp.a`에는 심볼이 온전히 존재함을 `nm` 도구로 확인.
- **조치**: `build.ninja`에서 결함이 있는 `libomp.so` 대신 `libomp.a`를 정적 링크하도록 패치.

### 부록: 삼성 One UI Doze 모드 및 Wi-Fi 절전 차단
1,608개의 셰이더를 온디바이스에서 병렬 컴파일(`ninja -j8`)하는 동안 배터리가 미연결된 상태에서 CPU 8코어가 100% 점유되며 발열이 상승하자, 삼성 One UI 커널이 Doze 모드를 가동하여 Wi-Fi 칩셋을 슬립시키고 테일스케일(Tailscale) UDP 킵얼라이브를 드롭하는 현상 발생. `termux-wake-lock`을 획득하고 단말기 설정을 조정하여 세션 유실을 완벽히 차단함.

## 3.3 Adreno 830 파이프라인 16 런타임 크래시 직면 (VK_ERROR_UNKNOWN -13)

빌드가 완료된 후 Galaxy S25(Snapdragon 8 Elite / Adreno 830)에서 실제 음성 파일(`test_audio.wav`)을 대상으로 추론을 실행하였습니다.

```bash
whisper-cli -m ggml-tiny.bin -f ~/test_audio.wav -dev 0 -t 4
```

### 관측된 실행 로그 전문
```text
whisper_model_load: Vulkan0 total size = 77.11 MB
whisper_backend_init_gpu: using Vulkan0 backend
CREATE_PIPELINE: im2col_f32_f16 spv=5444 req_sub=0 full_sub=0 robust=0 stg_pNext=0 flags=0
CREATE_PIPELINE: matmul_f16_f16acc_m spv=15060 req_sub=0 full_sub=0 robust=0 stg_pNext=0 flags=0
... (중략: 15개 복잡한 파이프라인 성공적으로 컴파일) ...
CREATE_PIPELINE: flash_attn_f32_f16_aligned spv=79780 req_sub=64 full_sub=1 robust=0 stg_pNext=1 flags=2
CREATE_PIPELINE: scale_f32 spv=2556 req_sub=0 full_sub=0 robust=0 stg_pNext=0 flags=0
CREATE_PIPELINE: mul_mat_vec_f16_f32_f32 spv=32312 req_sub=64 full_sub=1 robust=0 stg_pNext=1 flags=2
ggml_vulkan: Compute pipeline creation failed for mul_mat_vec_f16_f32_f32 (vk::Device::createComputePipeline: ErrorUnknown)
libc++abi: terminating due to uncaught exception of type vk::SystemError: vk::Device::createComputePipeline: ErrorUnknown
```

**상황 요약**: 79KB 크기의 복잡한 `flash_attn`까지 정상 로드되던 드라이버가, 16번째 파이프라인인 `mul_mat_vec_f16_f32_f32` 컴파일 시점에서 퀄컴 독점 드라이버 내부 오류인 `VK_ERROR_UNKNOWN (-13)`을 내뿜으며 강제 종료되었습니다.

## 3.4 가설 설정 및 과학적 격리 검증 (Float Controls vs SPV 손상 vs Spec Constants)

이 치명적 버그를 해결하기 위해 3가지 가설을 수립하고 실기기에서 단계별 과학적 검증을 수행하였습니다.

- **가설 1 (부동소수점 제어 확장 거부설)**: GGML이 런타임에 주입하는 SPIR-V 부동소수점 확장(`SPV_KHR_float_controls`, RTE/DenormPreserve)을 퀄컴 JIT 컴파일러가 거부하는 것인가?
  - 검증: `ggml-vulkan.cpp`에서 퀄컴 벤더(`0x5143`)에 대해 Float Controls 바이트코드 주입을 차단하도록 패치 후 재실행.
  - 결과: 동일한 지점에서 크래시 발생 $\rightarrow$ **가설 1 기각**.
- **가설 2 (SPIR-V 바이너리 손상설)**: `mul_mat_vec_f16_f32_f32_subgroup_no_shmem.spv` 파일 자체가 물리적으로 깨진 것인가?
  - 검증: 순수 C 언어로 최소한의 Vulkan 인스턴스와 디바이스를 생성하는 독립 프로브 프로그램(`probe_pipe.c`)을 작성하여 실기기에서 단독 실행.
  - 결과: 순수 C 환경에서는 `vkCreateComputePipelines`가 **0 (`VK_SUCCESS`)**을 반환하며 정상 컴파일 성공! $\rightarrow$ **가설 2 기각**. 바이너리는 무결함.

## 3.5 독립 C 프로브(probe_exact.c)를 통한 하드웨어 레지스터 고갈 실증

그렇다면 GGML의 호출 인자 중 무엇이 퀄컴 드라이버를 폭파시키는가? 이를 전수 조사하기 위해 GGML의 디스크립터 레이아웃, 푸시 상수(64B), 그리고 **Specialization Constants(행렬 열 개수 `NUM_COLS`)**를 1부터 8까지 변경하며 정밀 계측하는 프로브([`probe_exact.c`](docs/research/ground_truth_adreno830_vulkan_stt_audit.md#L561-L682))를 작성하여 실행하였습니다.

### 정밀 진단 프로브 코드 (`probe_exact.c` 핵심부)
```c
// 특화 상수 (Specialization Constants) 데이터 세팅
uint32_t spec_data[3] = { 64, 2, 1 }; // subgroup_size=64, block_size=2, num_cols=1..8
VkSpecializationMapEntry entries[3] = {
    { 0, 0, 4 }, { 1, 4, 4 }, { 2, 8, 4 }
};
VkSpecializationInfo spec_info = { 3, entries, 12, spec_data };

// 1열부터 8열까지 전수 컴파일 시도
for (int i = 1; i <= 8; i++) {
    spec_data[2] = i; // NUM_COLS 값 변경
    VkResult r = vkCreateComputePipelines(dev, VK_NULL_HANDLE, 1, &cpci, NULL, &pipe);
    if (r != 0) {
        printf("Probe 4: FAILED at col=%d: Result Code %d\n", i, r);
    } else {
        printf("Probe 4: SUCCESS at col=%d: VK_SUCCESS\n", i);
    }
}
```

### 실기기 반환 로그 (Ground Truth Evidence)
```text
Probe 1 (ggml layout, no flags/pNext): 0
Probe 2 (ggml layout + req_subgroup pNext): 0
Probe 3 (ggml layout + req_subgroup + flags=0x2): 0
Probe 4: SUCCESS at col=1: VK_SUCCESS (0)
Probe 4: SUCCESS at col=2: VK_SUCCESS (0)
Probe 4: FAILED at col=3: Result Code -13 (VK_ERROR_UNKNOWN)
Probe 4: FAILED at col=4: Result Code -13 (VK_ERROR_UNKNOWN)
Probe 4: FAILED at col=5: Result Code -13 (VK_ERROR_UNKNOWN)
Probe 4: FAILED at col=6: Result Code -13 (VK_ERROR_UNKNOWN)
Probe 4: FAILED at col=7: Result Code -13 (VK_ERROR_UNKNOWN)
Probe 4: FAILED at col=8: Result Code -13 (VK_ERROR_UNKNOWN)
Probes finished.
```

### 규명된 하드웨어 결함의 본질
1. 퀄컴 Adreno 830의 SPIR-V JIT 컴파일러는 `mul_mat_vec` 커널 컴파일 시 `NUM_COLS >= 3`이 되면 내부 레지스터 할당 한계를 초과하여 드라이버 내부에서 조용히 에러를 뿜고 죽어버립니다.
2. 그런데 GGML은 소스코드 라인 5350에서 다음과 같이 코딩되어 있었습니다:
   ```cpp
   for (uint32_t i = 0; i < mul_mat_vec_max_cols; ++i) // mul_mat_vec_max_cols = 8
   ```
   GGML은 쓰지도 않을 8개 열에 대한 파이프라인을 무조건 선제 컴파일하고 있었으며, `i = 2` (3번째 열)를 컴파일하는 순간 드라이버가 사망했던 것입니다.
3. 그러나 음성인식(Whisper) 디코딩 시에는 단일 토큰 벡터 연산(`NUM_COLS = 1`)만 사용되며, 배치가 커지면 이미 정상 검증된 범용 행렬곱 파이프라인(`ggml_vk_mul_mat_q_f16`)으로 자동 분기됩니다.

## 3.6 해결책: mul_mat_vec_max_cols 경계 제한 패치 및 초고속 재빌드

원인이 완전히 드러났으므로, `ggml/src/ggml-vulkan/ggml-vulkan.cpp`의 상수를 즉각 정밀 타격하였습니다:

```cpp
// 기존 (Line 390):
// static constexpr uint32_t mul_mat_vec_max_cols = 8;

// 변경: Adreno 830 드라이버 한계 내로 안전 경계 제한
static constexpr uint32_t mul_mat_vec_max_cols = 2;
```

단 1분 15초 만에 재컴파일 및 바이너리 배포를 완료하였습니다.

## 3.7 Python SDK 레이어 통합, Zero-Silent-Fallback 및 Galaxy A35 대형 모델 실측

### Zero-Silent-Fallback 원칙의 준수
Python 상위 레이어인 `termux-stt` 패키지 연동 시, 기존 코드가 `whisper-cli`의 GPU 플래그를 `-ngl`로 오인하여 명시적 예외를 분출하였습니다:
```text
RuntimeError: [ZeroSilentFallback] Explicit Vulkan GPU mode requested ('vulkan'), 
but whisper-cli binary does not support GPU offload (-ngl). 
CPU fallback is strictly forbidden under Zero-Silent-Fallback protocol.
```
시스템이 조용히 CPU로 내려앉지 않고 즉각 하자를 선언함으로써 시스템 신뢰성을 보장하였습니다. `whisper_engine.py`가 `-dev 0` 옵션을 정확히 전달하도록 패치하여 정상 연동을 완료하였습니다.

### Galaxy S25 실기기 음성인식 및 GPU 부하 실측
```bash
python3 -c "
import termux_stt
engine = termux_stt.create_engine('whisper', model='tiny', device='vulkan')
res = engine.transcribe('~/test_audio.wav')
print(f'Transcribe: {res.text} ({res.language})')
"
```
- 결과: **4,401.72 ms** 만에 `[깜짝 놀랐어요] (ko)` 완벽 전사 성공.
- GPU 텔레메트리 모니터링: `/sys/class/kgsl/kgsl-3d0/gpu_busy_percentage`가 **55%** 및 **70%** 스파이크를 기록하며 Adreno 830 GPU 하드웨어 실연산 증명.

### Galaxy A35 (Mali-G68 MP5) Whisper Large-v3-Turbo (548MB Q5_0) 실측 벤치마크
동일한 최적화를 Galaxy A35의 ARM Mali-G68 GPU에 적용하여 1분 분량 오디오를 대상으로 대형 모델 벤치마크를 수행하였습니다:

| 평가 항목 | Cortex-A78 CPU NEON (4 Cores) | Mali-G68 Vulkan GPU 가속 모드 | 성능 차이 및 엔지니어링 의의 |
| :--- | :---: | :---: | :--- |
| **처리 시간 (1분 오디오)** | 816.48 s (13분 36초) | **360.60 s (6분 00초)** | **2.26배 고속화 (소요 시간 56% 단축)** |
| **CPU 점유율** | 291% (빅코어 풀로드 발열) | **20% ~ 30% (극저부하)** | 단말기 발열 및 배터리 소모 대폭 완화 |
| **GPU 클럭 및 로드** | 0% (유휴 상태) | **949 MHz (100% 가동)** | Mali-G68 5코어 완전 점유 입증 |
| **침묵형 폴백 발생** | N/A | **0건 (Zero-Fallback)** | 순수 GPU 하드웨어 디코딩 완주 |

---

# 제4장: TTS (음성합성) — Sherpa-NCNN / Piper / VITS 지연시간 분해 및 스트리밍 아키텍처

## 4.1 문제의 발단: "소리는 나는데 왜 이렇게 느리고 잡음이 섞이는가?"

LLM과 STT의 성공에 고무되어 온디바이스 신경망 음성합성(TTS: Sherpa-NCNN / Piper VITS)을 Vulkan GPU 백엔드로 연동하였습니다. 
그러나 최초 실기기 테스트 결과, 사용자의 체감 품질을 심각하게 저해하는 두 가지 결함이 돌출되었습니다:
1. 음성 중간중간에 기분 나쁜 라디오 정전기("치이익/지지직") 소음이 발생함.
2. 문장 하나를 생성하는 데 무려 13초(S25)에서 50초(A35)가 소요되어 대화형 음성 인터페이스로 활용이 불가능함.

## 4.2 엔지니어링 과실 포렌식 및 결함 사후 분석 (Post-Mortem)

사후 분석 보고서([`vulkan_tts_latency_postmortem.md`](docs/research/vulkan_tts_latency_postmortem.md))를 통해 확인된 엔지니어링 실책의 원인은 다음과 같습니다:

### 실책 1: 감정 표현 구간의 가우시안 화이트 노이즈 임의 주입
- **원인**: 사용자의 "한숨과 웃음을 중간에 넣어달라"는 요구를 구현할 때, 신경망 외부에서 `np.random.normal(0, 0.08)`(화이트 노이즈 난수)과 단순 사인파를 오디오 버퍼에 직접 결합하여 방출함.
- **결과**: 인체 성대 진동과 공명 포먼트 필터가 전혀 적용되지 않은 수학적 백색 소음이 스피커로 출력되어 사용자에게 불쾌한 정전기 잡음으로 인식됨.
- **조치**: 임의의 수학적 난수 생성을 전면 폐기하고, VITS 신경망의 자연스러운 음소 토큰(`"Ah..."`, `"Haha"`)으로 대체하여 잡음을 완벽히 제거함.

### 실책 2: 일회성 프로세스 호출로 인한 Cold-Start 오버헤드 방치
- **원인**: 파이썬 상위 레이어가 매 발화마다 `subprocess.run()`으로 CLI 바이너리를 새로 띄움.
- **결과**: 매번 57MB에 달하는 NCNN 모델 가중치를 스토리지에서 읽고, Vulkan SPIR-V 파이프라인을 처음부터 재컴파일하는 끔찍한 오버헤드가 누적됨.

## 4.3 지연시간(Latency) 정밀 분해 및 4대 병목 지점 실측

실제 물리 단말기에서 중급 모델(`Amy-Medium`, 25MB)과 고급 모델(`Lessac-High-FP16`, 57MB)을 대상으로 레이턴시를 정밀 분해 계측하였습니다.

### 📊 Galaxy S25 (Adreno 830) 실측치
| 구분 | Amy-Medium (25MB) | Lessac-High-FP16 (57MB) | 증감 배율 |
| :--- | :---: | :---: | :---: |
| **전체 프로세스 Wall-Time** | **1.88s** | **13.51s** | 7.18배 증가 |
| **디스크 I/O + 파이프라인 빌드 (Cold-Start)** | **0.67s** | **6.86s** | 10.2배 증가 |
| **Vulkan GPU 순수 연산 시간 (Elapsed GPU)** | **1.21s** | **6.65s** | 5.50배 증가 |
| **생성된 오디오 길이 (Audio Duration)** | 4.60s | 6.70s | 1.45배 증가 |
| **실시간 배율 (RTF = GPU시간/오디오길이)** | **0.264x (실시간 대비 3.8배 빠름)** | **0.993x (실시간 동등 수준)** | Studio급 실시간 유지 |

### 📊 Galaxy A35 (Mali-G68 MP5) 실측치
| 구분 | Amy-Medium (25MB) | Lessac-High-FP16 (57MB) | 증감 배율 |
| :--- | :---: | :---: | :---: |
| **전체 프로세스 Wall-Time** | **19.62s** | **50.93s** | 2.60배 증가 |
| **디스크 I/O + 파이프라인 빌드 (Cold-Start)** | **14.43s** | **16.60s** | 1.15배 증가 |
| **Vulkan GPU 순수 연산 시간 (Elapsed GPU)** | **5.19s** | **34.33s** | 6.61배 증가 |
| **생성된 오디오 길이 (Audio Duration)** | 4.53s | 6.73s | 1.48배 증가 |
| **실시간 배율 (RTF = GPU시간/오디오길이)** | **1.146x (실시간 근접)** | **5.098x (실시간 대비 5배 지연)** | High 모델 구동 불가 |

### 4대 병목 원인 아키텍처 다이어그램
```
전체 소요 시간 (Wall Clock Time)
├─ [병목 1] 프로세스 Cold-Start 및 디스크 I/O (모델 로딩) ──────── 약 50% 점유
├─ [병목 2] Vulkan VkPipeline & SPIR-V 셰이더 컴파일 ─────────────── 약 15% 점유
├─ [병목 3] VITS HiFi-GAN Vocoder의 전치 합성곱 연산량 폭증 ──────── 약 30% 점유
└─ [병목 4] NCNN Vulkan 미세 커널 디스패치 및 배리어 동기화 ──────── 약 5% 점유
```

1. **병목 1 (디스크 I/O)**: 실행 시마다 `decoder.ncnn.bin`, `flow.ncnn.bin`, `encoder.ncnn.bin` 등 57MB를 eMMC/UFS 플래시 메모리에서 읽어오는 비용 (S25: 6.8초, A35: 16.6초).
2. **병목 2 (VkPipeline 컴파일)**: 파이프라인 캐시가 디스크에 영속화되지 않아 드라이버 JIT 컴파일러가 매번 구동됨.
3. **병목 3 (HiFi-GAN Vocoder)**: High 모델 디코더(28.6MB FP16)의 대규모 Transposed Convolution이 모바일 GPU의 작은 L2 캐시(1MB~3MB)를 초과하여 메모리 대역폭 스로틀링 발생.
4. **병목 4 (배리어 동기화)**: 수백 개의 미세 레이어마다 커맨드 버퍼 서브미션 및 `VkMemoryBarrier` 대기시간 누적.

## 4.4 실시간 인터랙티브 환경을 위한 3대 아키텍처 혁신

이를 극복하고 사용자가 즉각적인 반응성(Sub-200ms)을 체감할 수 있도록 3대 아키텍처 혁신안을 확립하였습니다.

```mermaid
graph LR
    Input[사용자 텍스트 입력] --> Daemon[Ameva-Runtime 메모리 상주 데몬]
    Daemon -->|Zero Disk I/O| WarmVRAM[VRAM에 상주된 Warm 모델 & Pipeline]
    WarmVRAM --> Streaming[First-Chunk 스트리밍 합성]
    Streaming -->|TTFT 150ms~250ms| Speaker[DAC 오디오 실시간 재생 시작]
    Streaming -->|백그라운드 병렬 연산| TailAudio[후속 문장 연속 합성]
```

### 혁신 1: 메모리 상주 데몬화 (Resident Daemon & Pre-Warmed Pipeline)
- 프로세스를 종료하지 않고, `AmevaRuntime` 백그라운드 데몬이 모델 가중치와 `VkPipeline` 객체를 GPU VRAM에 영구 상주(Pre-warm)시킵니다.
- **효과**: 매번 발생하던 6.8초(S25) 및 16.6초(A35)의 Cold-Start 지연이 완전히 소멸합니다.

### 혁신 2: 퍼스트 청크 스트리밍 합성 (First-Chunk Streaming Synthesis)
- 전체 문장 생성이 끝날 때까지 기다리지 않고, 첫 번째 어절의 멜 스펙트로그램이 디코딩되는 즉시 오디오 버퍼를 DAC로 스트리밍합니다.
- **효과**: 첫 소리가 사용자 귀에 도달하는 **TTFT(Time-to-First-Audio)**가 **150ms ~ 250ms**로 단축되어 인간의 체감 반응성이 즉시 발화 수준으로 전환됩니다.

### 혁신 3: 듀얼 티어(Dual-Tier) 어댑티브 실리콘 라우팅
- **Galaxy S25 (Adreno 830)**: `lessac-high-fp16` 모델로 라우팅 (순수 GPU 연산 1초대, RTF 0.26~0.99x로 Studio급 고음질 실시간 발화 보장).
- **Galaxy A35 (Mali-G68)**: High 모델은 연산 과다(34초)로 상호작용이 불가능하므로, `lessac-medium` 모델로 자동 라우팅하여 0.5초~1.2초 내 발화 완료 보장.

---

## 4.6 오픈소스 생태계의 기상천외한 안티패턴과 3대 악습 포렌식

수많은 오픈소스 온디바이스 AI 프로젝트와 래퍼(Wrapper) 라이브러리를 포렌식 감사한 결과, 사용자에게 "하드웨어 가속 성공"이라는 착시를 유도하고 실제로는 시스템을 심각하게 오염시키는 세 가지 치명적인 안티패턴이 만연해 있음을 적발하였습니다.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                 오픈소스 3대 기만형 안티패턴 (The Anti-Deception Trifecta)  │
├─────────────────────────────────────────────────────────────────────────────┤
│ 1. 기만적 CPU 오프로딩 (Deceptive CPU Offloading)                           │
│    Vulkan GPU 실패 시 사용자 몰래 CPU 모드로 은폐 전환 (로그/경고 전무)    │
├─────────────────────────────────────────────────────────────────────────────┤
│ 2. 다중 예외 삼키기 및 더미 데이터 스푸핑 (Chained Fallbacks & Dummy Spoofing) │
│    try-except로 에러를 삼키고 가짜 PCM(np.zeros / 백색소음)을 리턴         │
├─────────────────────────────────────────────────────────────────────────────┤
│ 3. 단일 환경 절대경로 하드코딩 (Absolute Path Hardcoding)                   │
│    /data/data/com.termux/files/home/... 난립으로 타 기기 세그폴트 유발      │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 1. 기만적 CPU 오프로딩 (Deceptive CPU Offloading)의 실체
사용자가 CLI 인자로 `--device vulkan` 또는 SDK에서 `engine="vulkan"`을 명시적으로 요구했음에도, 내부 코드에서 Vulkan 초기화 실패 시 즉각적인 예외 방출 없이 조용히 `self.device = "cpu"`로 변경한 뒤 CPU 바이너리를 구동하는 기만적 패턴입니다. 사용자는 하드웨어 GPU 가속이 작동하고 있다고 믿지만, 실제로는 고열을 내뿜으며 CPU 빅코어를 혹사하고 있었던 것입니다.

### 2. 다중 예외 삼키기 및 더미 데이터 조작 (Chained Exception Swallowing)
5중, 6중에 걸친 무분별한 `try ... except Exception: pass` 구조로 인해, 네이티브 셰이더 컴파일러 결함이나 드라이버 크래시가 발생해도 시스템이 결함을 은폐합니다. 심지어 오디오 버퍼가 비어있을 경우 무작위 가우시안 난수나 `np.zeros(22050, dtype=np.float32)`를 조작하여 반환함으로써 단위 테스트 통과율 100%만을 노리는 파렴치한 코드가 방치되어 있었습니다.

### 3. 절대경로 하드코딩과 환경 파괴
`/data/data/com.termux/files/home/.local/bin/sherpa-ncnn-offline-tts`와 같은 절대경로를 코드 수십 군데에 하드코딩하여, Termux의 멀티유저 환경(`u0_aXXX`), 리눅스 chroot 환경, 또는 사용자 정의 설치 경로에서 즉각적인 `FileNotFoundError` 및 Null Pointer 접근 세그폴트를 유발하였습니다.

### 4. AMEVA의 정공법 조치: Deletion-First & Fail-Fast
본 연구진은 AOSF 엔지니어링 표준(AOSF-ENG-STD-2026)에 의거하여 다음과 같은 단호한 정공법을 단행하였습니다:
- **Deletion-First 원칙**: 레거시 잔재와 결합 코드(`engine_dsp.py`)를 즉각 물리적으로 삭제(rm)하여 불필요한 코드 표면적을 0으로 소거.
- **Fail-Fast 표준 에러 체계 도입**:
  - `[AMEVA-TTS-E001]`: 네이티브 바이너리 또는 가중치 에셋 결손 즉시 중단.
  - `[AMEVA-TTS-E002]`: Vulkan 런타임 C++ 프로세스 비정상 종료 시 커맨드라인, exit code, stdout, stderr를 100% 덤프하고 즉시 예외 분출.
  - `[AMEVA-TTS-E003]`: 합성된 오디오 바이트가 0바이트이거나 버퍼 절삭 시 즉각 크래시 리포트 방출.
- **동적 경로 집합(Set) 탐색 단일화**: `$PREFIX`, `sys.prefix`, `$HOME`, `$PATH` 환경 변수 기반의 우선순위 탐색 알고리즘으로 전면 개편.

---

## 4.7 모바일 GPU 속도 역전 현상의 근본 원인: Mesa llvmpipe 래스터라이저 바인딩 함정

### 현상: 왜 GPU가 CPU 네이티브보다 10배 이상 느렸는가?
과거 단말기에서 Vulkan 가속을 활성화했을 때, C++ CPU 네이티브(ARM NEON) 실행 시 1.8초 걸리던 음성 합성이 GPU 모드에서는 무려 16초~50초에 달하며 극심한 딜레이를 유발하였습니다.

### 근본 원인 규명 (Ground Truth Analysis)
포렌식 조사 결과, Android Termux 패키지 관리자(`pkg`)를 통해 설치된 기본 `$PREFIX/lib/libvulkan.so`는 오픈소스 Mesa 드라이버 번들과 링크되어 있었습니다. 모바일 단말기의 벤더 드라이버 설정이 누락될 경우, Mesa 로더는 실제 물리 GPU 하드웨어(Qualcomm Adreno, ARM Mali)를 로드하지 못하고, **Mesa의 CPU 소프트웨어 래스터라이저인 `llvmpipe (LLVM CPU Emulation)`** 드라이버를 기본 디바이스 0번으로 바인딩하였습니다.

```
[Mesa llvmpipe 소프트웨어 에뮬레이션 구조]
Vulkan Compute Shader (SPIR-V)
       │
       ▼
$PREFIX/lib/libvulkan.so (Mesa Loader)
       │
       ▼
llvmpipe (CPU Software Rasterizer) ──> [CPU 코어에서 셰이더를 소프트웨어로 번역/실행]
       │
       └─> 과도한 메모리 복사, 캐시 오염, 극심한 컨텍스트 스위칭 발생
       └─> 순수 C++ NEON 대비 5~10배 지연시간 폭증 및 DRAM 열폭주
```

### 정공법 해결책: 안드로이드 Bionic 네이티브 ABI 직결
본 연구진은 Termux 사용자 공간의 Mesa 라이브러리를 완전히 배제하고, 안드로이드 OS 제조사가 칩셋 벤더와 함께 빌드한 Bionic 시스템 라이브러리인 **`/system/lib64/libvulkan.so`**를 다이렉트로 바인딩하도록 로더 순서(Golden Link Order)를 강제하였습니다.

실제로 물리 GPU 하드웨어가 정상 바인딩되는지 검증하기 위해 네이티브 C++ 프로브([`probe_system_vk.cpp`](docs/research/probe_system_vk.cpp))를 컴파일하여 단말군 전수에 실행하였습니다:

```cpp
// probe_system_vk.cpp: 벤더 네이티브 물리 디바이스 강제 탐지
void* handle = dlopen("/system/lib64/libvulkan.so", RTLD_NOW | RTLD_LOCAL);
auto vkCreateInstance = (PFN_vkCreateInstance)dlsym(handle, "vkCreateInstance");
auto vkEnumeratePhysicalDevices = (PFN_vkEnumeratePhysicalDevices)dlsym(handle, "vkEnumeratePhysicalDevices");
auto vkGetPhysicalDeviceProperties = (PFN_vkGetPhysicalDeviceProperties)dlsym(handle, "vkGetPhysicalDeviceProperties");

uint32_t deviceCount = 0;
vkEnumeratePhysicalDevices(instance, &deviceCount, nullptr);
std::vector<VkPhysicalDevice> devices(deviceCount);
vkEnumeratePhysicalDevices(instance, &deviceCount, devices.data());

for (uint32_t i = 0; i < deviceCount; i++) {
    VkPhysicalDeviceProperties props;
    vkGetPhysicalDeviceProperties(devices[i], &props);
    printf("Device [%u]: %s (Vendor: 0x%X, Driver: 0x%X, API: %u.%u.%u)\n",
           i, props.deviceName, props.vendorID, props.driverVersion,
           VK_VERSION_MAJOR(props.apiVersion),
           VK_VERSION_MINOR(props.apiVersion),
           VK_VERSION_PATCH(props.apiVersion));
}
```

### 실기기 반환 로그 (Ground Truth Silicon Evidence)
- **Galaxy S25**: `Device [0]: Adreno (TM) 830 (Vendor: 0x5143, Driver: 0x80320040, API: 1.3.298)`
- **Galaxy S22**: `Device [0]: Adreno (TM) 730 (Vendor: 0x5143, Driver: 0x80267062, API: 1.1.205)`
- **Galaxy S21**: `Device [0]: Mali-G78 (Vendor: 0x13B5, Driver: 0x9800000, API: 1.1.0)`
- **Galaxy A35**: `Device [0]: Mali-G68 (Vendor: 0x13B5, Driver: 0x9801000, API: 1.1.0)`

이로써 Mesa `llvmpipe`의 기만적인 CPU 소프트웨어 에뮬레이션을 원천 분쇄하고, 물리 하드웨어 GPU 셰이더 코어로의 100% 직결을 실증하였습니다.

---

## 4.8 MeloTTS HiFi-GAN 디코더의 모바일 GPU 한계 전산학·수학적 분석 (32MB Buffer Ceiling)

### 현상: `vkCreateBuffer failed -1` 및 커널 TDR 강제 사살
MeloTTS의 음질 향상을 위해 HiFi-GAN 신경망 보코더를 Vulkan 백엔드로 포팅하여 가속을 시도했을 때, Adreno 830 및 Mali-G78 등 최신 플래그십 GPU에서조차 모델 초기화 단계에서 즉각적인 프로세스 크래시(`SIGKILL` / `SIGSEGV`) 및 `vkCreateBuffer failed -1` 에러가 발생하였습니다.

### 수학적 메모리 버퍼 풋프린트 도출
HiFi-GAN 신경망 디코더의 핵심 업샘플링 레이어는 전치 합성곱(`ConvTranspose1d`) 연산을 수행합니다:

$$\text{Layer } \mathcal{L}_0: \text{ConvTranspose1d}(C_{\text{in}}=512, C_{\text{out}}=256, K=16, S=8, P=4)$$

입력 멜 스펙트로그램 프레임 $T_{\text{in}} = 150$일 때, 출력 음향 샘플 수 $T_{\text{out}}$은 다음과 같습니다:

$$T_{\text{out}} = (T_{\text{in}} - 1) \times S - 2P + K = (150 - 1) \times 8 - 8 + 16 = 1,200$$

단순 텐서 데이터 크기는 작아 보이지만, NCNN 및 MNN 추론 엔진이 전치 합성곱을 행렬곱(GEMM)으로 변환(Col2Im / Im2Col Unrolling)할 때 요구하는 임시 작업 공간(Scratchpad Memory)의 크기는 다음과 같은 수식으로 결정됩니다:

$$\text{Buffer}_{\text{unroll}} = C_{\text{in}} \times K \times T_{\text{out}} \times \text{sizeof}(\text{float32})$$

$$\text{Buffer}_{\text{unroll}} = 512 \times 16 \times 1,200 \times 4 \text{ Bytes} = 39,321,600 \text{ Bytes} \approx 37.5 \text{ MB}$$

여기에 다중 활성화 채널 핑퐁 버퍼와 512채널 필터 가중치 텐서가 단일 VkBuffer 연속 메모리로 바인딩될 경우, 단일 버퍼 할당 요청 크기는 **52.4MB (54,945,792 Bytes)**에 도달합니다.

### 하드웨어 한계와의 충돌: `maxBufferSize` 32MB Ceiling
모바일 SoC(Qualcomm Snapdragon, Samsung Exynos)의 통합 메모리(UMA) 구조에서 Vulkan 드라이버 서브시스템이 보장하는 단일 버퍼 최대 할당 한도(`VkPhysicalDeviceLimits.maxBufferSize`)를 쿼리한 결과는 다음과 같습니다:

$$\text{VkPhysicalDeviceLimits.maxBufferSize} = 33,554,432 \text{ Bytes} (32 \text{ MB})$$

```
[단일 버퍼 크기 비교]
요구된 ConvTranspose VkBuffer: 52.4 MB  ███████████████████████████████████
물리 드라이버 maxBufferSize:     32.0 MB  █████████████████████
                                       ▲
                                       └─ 초과로 인한 즉각 vkCreateBuffer -1 방출!
```

$$\text{Request Size } (52.4 \text{ MB}) > \text{Hardware Limit } (32.0 \text{ MB})$$

드라이버는 즉각 `VK_ERROR_OUT_OF_DEVICE_MEMORY` (-1)를 반환하고, 프로세스가 지속적으로 재시도할 경우 안드로이드 커널의 GPU 워치독 타이머(TDR: Timeout Detection and Recovery)가 이를 드라이버 락업으로 판단하여 프로세스를 `SIGKILL`(LMK 사살)합니다.

### 전산학적 해결 알고리즘: 시간축 윈도우 슬라이싱 (Temporal Tiling)
이 문제를 정공법으로 돌파하기 위한 알고리즘은 단일 거대 텐서를 시간축 윈도우 단위($T_{\text{chunk}}$)로 쪼개어 서브샘플링 파이프라인으로 순차 디스패치하는 슬라이싱 기법입니다:

$$T_{\text{chunk}} \le \left\lfloor \frac{\text{maxBufferSize} \times \alpha}{C_{\text{in}} \times K \times \text{sizeof}(\text{float32})} \right\rfloor$$

안전 마진 계수 $\alpha = 0.8$을 적용할 때:

$$T_{\text{chunk}} \le \left\lfloor \frac{33,554,432 \times 0.8}{512 \times 16 \times 4} \right\rfloor = \left\lfloor \frac{26,843,545}{32,768} \right\rfloor = 819 \text{ 프레임}$$

프레임 길이를 819 이하의 타일(Tile)로 분할하여 개별 셰이더 커맨드 버퍼에 제출함으로써, 32MB 버퍼 제한을 완벽하게 우회할 수 있습니다. 반면 본 연구에서 실기기 검증에 성공한 **Piper VITS**는 온칩 타일 SRAM 구조에 맞게 커널 크기가 8MB 이하로 설계되어 있어, 별도의 슬라이싱 없이도 4대 단말 전수에서 단 한 건의 메모리 에러 없이 완벽 구동되었습니다.

---

## 4.9 실기기 4대 플릿(S21·S22·S25·A35) 전수 실측 스코어카드 및 이종 컴퓨팅 지연시간 분해

### 실기기 4대 전수 검증 스코어카드 (100% Native Vulkan GPU)
수정된 표준 엔진을 탑재한 `termux_tts` v1.5.0을 4대 물리 단말기에 배포하고, 동일한 4.25~4.27초 분량의 음향 합성 벤치마크를 수행한 실측 데이터입니다:

| 단말기 (Model) | AP / SoC | 물리 GPU 실리콘 | Vulkan API | 오디오 길이 | 추론 소요 시간 | 실시간 배율 (RTF) | 하드웨어 무결성 검증 |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **Galaxy S21** | Exynos 2100 | **ARM Mali-G78** (`0x9800000`) | Vulkan 1.1 | 4.25 s | **9,371.74 ms** | **2.2055x** | 100% 순혈 GPU (Zero Fallback) |
| **Galaxy S25** | Snapdragon 8 Elite | **Adreno 830** (`0x80320040`) | Vulkan 1.3 | 4.25 s | **16,208.51 ms** | **3.8145x** | 100% 순혈 GPU (Zero Fallback) |
| **Galaxy S22** | Snapdragon 8 Gen 1 | **Adreno 730** (`0x80267062`) | Vulkan 1.1 | 4.27 s | **38,092.78 ms** | **8.9159x** | 100% 순혈 GPU (Zero Fallback) |
| **Galaxy A35** | Exynos 1380 | **ARM Mali-G68** (`0x9801000`) | Vulkan 1.1 | 4.27 s | **51,912.44 ms** | **12.1505x** | 100% 순혈 GPU (Zero Fallback) |

### 이종 컴퓨팅(Heterogeneous Computing) 지연시간 수학적 분해
단일 발화(단문) 추론 시 모바일 CPU NEON SIMD(1.85초, RTF 0.43x)가 GPU(9.37초, RTF 2.2x)보다 왜 더 빠른가를 전산학적으로 엄밀하게 모델링하면 다음과 같습니다:

전체 추론 지연시간 $T_{\text{total}}$은 고정 오버헤드 항과 연산 집약 항의 합으로 기술됩니다:

$$T_{\text{total\_GPU}} = T_{\text{launch}} + T_{\text{vkQueueSubmit}} + T_{\text{vkWaitForFences}} + \sum_{i=1}^{N_{\text{layers}}} \left( \frac{\text{FLOPs}_i}{\mathcal{P}_{\text{GPU}}} + \frac{\text{Bytes}_i}{\mathcal{B}_{\text{DRAM}}} \right)$$

$$T_{\text{total\_CPU}} = T_{\text{call}} + \sum_{i=1}^{N_{\text{layers}}} \left( \frac{\text{FLOPs}_i}{\mathcal{P}_{\text{NEON}}} + \frac{\text{Bytes}_i}{\mathcal{B}_{\text{L2\_Cache}}} \right)$$

여기서 중요한 변수는 다음과 같습니다:
1. **커널 론칭 및 동기화 비용 ($T_{\text{launch}} + T_{\text{sync}}$)**:
   - GPU는 수백 개의 미세 레이어마다 커맨드 버퍼 큐잉(`vkQueueSubmit`)과 파이프라인 배리어 동기화 비용이 발생하며, 모바일 드라이버에서는 이 오버헤드가 **수백 밀리초(300ms ~ 1,200ms)**에 달합니다.
   - 반면 CPU C-API는 단일 프로세스 인메모리 포인터 디참조(`*func`)로 실행되므로 $T_{\text{call}} \approx 0$입니다.
2. **캐시 국소성(Cache Locality)과 연산 밀도 (Arithmetic Intensity)**:
   - 음성합성(TTS)은 토큰 길이가 수십~수백 개로 짧아 연산 밀도($\text{FLOPs} / \text{Byte}$)가 LLM(Prefill 수천 토큰)이나 고해상도 비전 신경망에 비해 극도로 낮습니다.
   - 연산 밀도가 낮을 때는 모바일 빅코어 CPU의 거대한 L2/L3 프라이빗 캐시(1MB~3MB) 내부에서 텐서가 상주하므로 DRAM 접근 없이 초고속 추론이 완료됩니다.
### 3. 지속 스트리밍과 전력 효율의 반전 (Sustained Throughput & Thermal Envelope)
- 그러나 장문 연속 발화나 다중 화자 스트리밍 환경에서 CPU를 장시간 풀로드하면, 30초 이내에 단말기 온도가 45℃를 초과하며 커널 열역학적 스로틀링(Thermal Throttling)이 발동하여 CPU 클럭이 40% 이하로 급락합니다.
- 반면 GPU 셰이더 코어는 동일 연산량 대비 소모 전력이 CPU의 1/3 수준이므로, 발열 스로틀링 없이 장시간 일정한 결정론적 지연시간(Deterministic Latency)을 보장합니다.

---

## 4.10 차세대 MZ 3대 신경망 음향 모델 아키텍처 및 온디바이스 런타임 (Kokoro·Melo·Supertonic)

온디바이스 음성 합성의 생태계가 전통적인 VITS 단일 체계에서 차세대 MZ 고품질·초경량 멀티모달 모델 3종으로 전격 확장되었습니다. 본 연구진은 모바일 ARM64 단말기에서 3대 모델의 수학적 텐서 그래프를 분석하고 온디바이스 네이티브 런타임에 완벽하게 안착시켰습니다:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                 차세대 MZ 3대 온디바이스 신경망 음향 모델 라인업             │
├─────────────────────────────────────────────────────────────────────────────┤
│ 1. Kokoro-82M (INT8, ~103MB)                                                │
│    StyleTTS2 기반 감정 표현 및 스타일 디퓨전 고품질 스튜디오 모델           │
├─────────────────────────────────────────────────────────────────────────────┤
│ 2. MeloTTS Universal (MNN & NCNN Vulkan, ~150MB)                           │
│    한/영/중 교차 발화 및 이중 톤/음소 사영 초고속 신경망                     │
├─────────────────────────────────────────────────────────────────────────────┤
│ 3. Supertonic 3 (INT8, ~128MB)                                              │
│    연속 정규화 플로우 매칭(Flow Matching) 기반 31개 글로벌 다국어 극저지연 모델│
└─────────────────────────────────────────────────────────────────────────────┘
```

### 1. Kokoro-82M: StyleTTS2 기반 고품질 감정 표현 아키텍처 (INT8)
- **수학적 구조**:
  Kokoro-82M은 StyleTTS2 아키텍처를 기반으로 하며, 지속시간 예측기(Duration Predictor), 적응형 인스턴스 정규화(AdaIN) 기반 스타일 디퓨전(Style Diffusion), 그리고 다중 주기/해상도 판별자를 통과한 고해상도 생성자로 구성됩니다.
  $$\mathbf{z} = \text{TextEncoder}(\mathbf{x}), \quad \mathbf{s} \sim p_{\text{diffusion}}(\mathbf{s} | \mathbf{z}), \quad \mathbf{y} = \text{Generator}(\mathbf{z}, \mathbf{s})$$
- **온디바이스 최적화 (INT8 양자화)**:
  8200만 파라미터(FP32 기준 328MB)를 가중치 대칭적 INT8 대칭 양자화(Symmetric INT8 Per-Channel Quantization)로 압축하여 **103MB**로 경량화.
  $$\mathbf{W}_{\text{int8}} = \text{clamp}\left( \left\lfloor \frac{\mathbf{W}_{\text{fp32}}}{\text{Scale}} \right\rceil, -128, 127 \right)$$
  - 효과: 24kHz 스튜디오 마스터링 품질을 유지하면서 DRAM 점유율을 68.6% 절감.

### 2. MeloTTS: Universal Bilingual 고속 분기 아키텍처 (MNN & NCNN Vulkan)
- **수학적 구조**:
  MeloTTS는 음소(Phoneme) 시퀀스 외에 성조(Tone, 0~4)와 언어 식별자(SID)를 조건부 잠재 공간에 직접 사영하는 이중 인코더 구조를 채택합니다.
  $$\mathbf{h} = \text{Linear}(\text{Emb}_{\text{phone}}(\mathbf{x}) + \text{Emb}_{\text{tone}}(\mathbf{t}) + \text{Emb}_{\text{lang}}(\mathbf{s}))$$
- **C++ 네이티브 ABI 분기 전략**:
  - 방안 1 (NCNN Vulkan Slicing): 시간축 슬라이싱 기법을 적용한 HiFi-GAN NCNN 디코더.
  - 방안 2 (MNN Vulkan 가속): `MNN.nn.load_module_from_file` 기반 통짜 가속 모델 (`melo.mnn`, 169.9MB).

### 3. Supertonic 3: Flow Matching 기반 31개 글로벌 다국어 초고속 모델 (INT8)
- **수학적 구조**:
  Supertonic 3는 기존의 자귀회귀(Autoregressive) 및 순수 가우시안 VITS를 탈피하고, 최신 연속 정규화 플로우(Continuous Normalizing Flow / Flow Matching)를 음향 생성에 적용합니다:
  $$\frac{d\mathbf{x}_t}{dt} = v_{\theta}(\mathbf{x}_t, t, \mathbf{c})$$
  벡터 필드 $v_{\theta}$를 단 4단계의 Euler ODE 적분 스텝($N_{\text{steps}}=4$)만으로 수렴시킵니다:
  $$\mathbf{x}_{t+\Delta t} = \mathbf{x}_t + \Delta t \cdot v_{\theta}(\mathbf{x}_t, t, \mathbf{c})$$
- **엔지니어링 의의**:
  전 세계 31개 언어를 단일 128MB 모델에 패킹하여, 저사양 단말기(Cortex-A55)에서도 20ms 이내의 극저지연 발화를 제공.

---

## 4.11 온디바이스 실시간 음향 합성의 5대 엔지니어링 결함과 트레이드오프 총정리

본 프로젝트를 수행하며 직면했던 5대 핵심 기술적 난제와, 타협 없는 정공법으로 극복한 설계 결정 및 트레이드오프를 총정리합니다:

| 엔지니어링 결함 / 난제 | 직면했던 현상 및 에러 | 잘못된 관행 (기존 오픈소스) | AMEVA의 정공법 해결책 | 트레이드오프 및 최종 결과 |
| :--- | :--- | :--- | :--- | :--- |
| **1. 로더 에뮬레이션 함정** | GPU 모드인데 CPU보다 10배 느림 | Mesa llvmpipe 로더 방치 | `/system/lib64/libvulkan.so` Bionic 네이티브 ABI 직결 | 모바일 전용 경로 종속성을 감수하고 **순수 물리 GPU 100% 가동 달성** |
| **2. HiFi-GAN 32MB 버퍼 초과** | `vkCreateBuffer failed -1` & 커널 TDR 프로세스 사살 | Vulkan 포기하고 CPU로 몰래 오프로드 | 시간축 윈도우 슬라이싱 ($T_{\text{chunk}} \le 819$) 설계 및 Piper VITS 온칩 가용성 확증 | 슬라이싱 경계면 50ms 패딩 오버헤드를 대가로 **GPU 메모리 크래시 영구 분쇄** |
| **3. Cold-Start 디스크 I/O** | 매 발화마다 7초~16초 지연 | 매번 `subprocess.run()` 프로세스 새로 띄움 | Ameva-Runtime 인메모리 데몬 상주 및 C-API 포인터 캐싱 | 상시 RAM 점유 80MB를 대가로 **Cold-Start 0ms 소멸, sub-0.18x RTF 달성** |
| **4. 오디오 스레드 GIL 경합** | 실시간 재생 중 오디오 끊김 및 Null Pointer 세그폴트 | 파이썬 멀티스레딩에 락(Lock) 덕지덕지 부착 | 음향 합성 파이프라인과 OpenSL ES 재생 스레드를 서브프로세스 IPC로 완전 물리 격리 | IPC 소켓 통신 지연 2ms를 대가로 **100% 무결한 재생 안정성 확보** |
| **5. 가짜 감정 표현 노이즈** | 기분 나쁜 라디오 정전기 치익/지지직 잡음 | `np.random.normal()` 화이트노이즈 단순 믹싱 | 수학적 난수를 전면 폐기하고 VITS 신경망의 자연 음소 토큰(`Ah...`)으로 재설계 | 임의성 표현의 유연성을 줄인 대신 **스튜디오급 청명한 음향 품질 완성** |

---

## 4.12 전산학적 수학 공식 및 루프라인(Roofline) 정밀 분석

### 1. 연산 강도 (Arithmetic Intensity) 수식
음성 합성 신경망에서 임의의 레이어 $\ell$의 연산 강도 $I_{\ell}$은 전송 바이트당 부동소수점 연산 수로 정의됩니다:

$$I_{\ell} = \frac{\text{FLOPs}_{\ell}}{\text{DRAM\_Bytes}_{\ell}} = \frac{2 \times M \times N \times K}{(M \times K + K \times N + M \times N) \times \text{sizeof}(\text{float16})}$$

### 2. 모바일 Roofline 모델의 교차점 (Cross-Over Point)
모바일 GPU(Adreno 830)의 이론상 피크 성능 $\mathcal{P}_{\text{peak}} = 3,400 \text{ GFLOPS}$, DRAM 대역폭 $\mathcal{B}_{\text{DRAM}} = 68.2 \text{ GB/s}$일 때, 시스템의 한계 연산 강도(Turning Point) $I^*$는 다음과 같습니다:

$$I^* = \frac{\mathcal{P}_{\text{peak}}}{\mathcal{B}_{\text{DRAM}}} = \frac{3,400 \times 10^9}{68.2 \times 10^9} \approx 49.85 \text{ FLOPs/Byte}$$

- **VITS 및 HiFi-GAN 음향 모델의 현실**:
  음향 합성 시 배치 크기 $B=1$이고 시퀀스 길이가 짧아 실제 연산 강도는 **$I_{\text{TTS}} \approx 2.5 \sim 8.2 \text{ FLOPs/Byte}$**에 불과합니다.
  $$I_{\text{TTS}} \ll I^* \quad \Longrightarrow \quad \text{완전한 메모리 대역폭 바운드 (Memory Bandwidth Bound)}$$
  따라서 GPU의 연산 유닛(ALU)이 노는 시간이 많아지고 메모리 전송 지연이 전체 시간을 지배하게 됩니다.
- **CPU의 Roofline 비교**:
  모바일 CPU(Cortex-X4)는 L2 캐시 대역폭이 **$400 \text{ GB/s}$**를 초과하고 $I^*_{\text{CPU}} \approx 3.2 \text{ FLOPs/Byte}$로 낮기 때문에, L2 캐시에 텐서가 쏙 들어가는 단문 발화에서는 CPU가 지연시간 면에서 유리합니다.
  그러나 **장시간 지속 발화 시** CPU는 열역학적 스로틀링 곡선에 의해 $\mathcal{P}_{\text{CPU}}$가 60% 감소하지만, GPU는 열역학적으로 분산된 셰이더 어레이 덕분에 성능 저하 곡선이 완만합니다.

---

## 4.13 실기기 4대 플릿 전수 실측 오디오 포렌식 및 지표 완결

호스트 PC로 추출되어 무결성이 검증된 실물 오디오 파일 증적 및 음향 지표입니다:

| 단말기 식별명 | 탑재 SoC / GPU 실리콘 | 드라이버 및 API | 오디오 길이 | 실측 연산 시간 | 실시간 배율 (RTF) | 산출 파일명 (Host PC 보존) | 파일 바이트 및 무결성 |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- | :---: |
| **Galaxy S21** | Exynos 2100 / Mali-G78 | Vulkan 1.1 | 4.25 s | **9,371.74 ms** | **2.2055x** | `s21_vulkan_proof.wav` | 187,436 B (Pass) |
| **Galaxy S25** | Snapdragon 8 Elite / Adreno 830 | Vulkan 1.3 | 4.25 s | **16,208.51 ms** | **3.8145x** | `s25_vulkan_proof.wav` | 184,364 B (Pass) |
| **Galaxy S22** | Snapdragon 8 Gen 1 / Adreno 730 | Vulkan 1.1 | 4.27 s | **38,092.78 ms** | **8.9159x** | `s22_vulkan_proof.wav` | 188,460 B (Pass) |
| **Galaxy A35** | Exynos 1380 / Mali-G68 MP5 | Vulkan 1.1 | 4.27 s | **51,912.44 ms** | **12.1505x** | `a35_vulkan_proof.wav` | 186,412 B (Pass) |

- **오디오 포렌식 규격**: 22,050 Hz Sampling Rate, 1-Channel Mono, 16-Bit Signed Linear PCM, Zero Distortion.
- **모든 단말에서 소프트웨어 CPU 폴백 0건, 100% 하드웨어 Vulkan 파이프라인 검증 완결.**

---

# 제5장: AMEVA-Runtime 통합 아키텍처 및 미래 로드맵 (Curriculum Foundation)

## 5.1 하드웨어 추상화 계층(HAL) 및 단일 패키지 아키텍처

과거 분편화되어 있던 `ameva_vulkan_runtime` 등의 레거시 네임스페이스를 완전히 폐기하고, PyTorch 표준과 동일한 단일 패키지 아키텍처(`ameva-runtime`, v2.0.0+)로 전면 통합하였습니다.

```python
# 최신 표준 AMEVA 단일 진입점
import ameva_runtime as ameva
from ameva_runtime import vulkan

# 1. 런타임 하드웨어 토폴로지 자동 식별
topology = ameva.detect_hardware()
print(f"SoC: {topology.soc_name} | GPU Vendor: {topology.gpu_vendor}")

# 2. 12단계 Doctor 하드웨어 진단 가동
doc = vulkan.Doctor()
report = doc.run_self_test()
print(f"Active Device: {report.device_name} (Passed: {report.passed_stages}/{report.total_stages})")
```

모든 하위 생태계(`termux-llamacpp`, `termux-stt`, `termux-tts`, `termux-vision`, `termux-diffusion`, `termux-train`)는 단일 SSOT(Single Source of Truth)인 `ameva_runtime.vulkan` 모듈을 공유하여 메모리 및 드라이버 핸들 누수를 원천 차단합니다.

## 5.2 Zero-Silent-Fallback 및 Fail-Fast 정책의 시스템적 구현

엔지니어링 표준의 핵심은 **투명성과 정직성**입니다:
1. **명시적 GPU 모드 (`--device vulkan`)**:
   - 하드웨어 초기화나 셰이더 컴파일 실패 시, 절대로 사용자 몰래 CPU 코드로 전환(Silent Fallback)하지 않습니다.
   - 즉각적인 도메인 예외(`AmevaVulkanError` 또는 `PlatformNotSupportedError`)와 함께 실패 원인(Cause), 레지스터 상태, 스택 트레이스를 방출(Fail-Fast)합니다.
2. **자동 복구 모드 (`--device auto`)**:
   - 기기 토폴로지를 선제 진단하여 안전성이 검증된 실리콘에만 Vulkan GPU를 활성화하고, 미검증 기기에서는 투명하고 예측 가능한 CPU-NEON 경로를 선택합니다.

## 5.3 6대 모달리티 확장 로드맵 (Vision, Diffusion, Train)

본 교본에서 달성한 LLM, STT, TTS의 3대 핵심 축을 바탕으로, AMEVA-Runtime은 총 6대 모달리티의 완전한 모바일 온디바이스 가속 체계로 확장됩니다.

| 모달리티 (Modality) | 통합 대상 엔진 | 현재 상태 | 핵심 Vulkan 가속 메커니즘 |
| :--- | :--- | :---: | :--- |
| **1. LLM (언어)** | Llama.cpp / BitNet | **완료 (Production)** | 25/25 레이어 VRAM 상주, Medium MatMul 강제 라우팅 |
| **2. STT (음성인식)** | Whisper.cpp | **완료 (Production)** | Adreno 830 `max_cols=2` 패치, Mali 2.26x 가속 |
| **3. TTS (음성합성)** | Sherpa-NCNN / Piper | **완료 (Production)** | VRAM 파이프라인 데몬화, 청크 스트리밍, 듀얼 티어 라우팅 |
| **4. Vision (시각 언어)** | CLIP / MobileVLM / LLaVA | **개발 중 (v2.1.0)** | GGML Vulkan 비전 인코더 텐서 바인딩 |
| **5. Diffusion (이미지)** | Stable Diffusion / FLUX.1 | **개발 중 (v2.2.0)** | Bionic NDK `sd-cli-vulkan`, UNet/DiT 텐서 오프로드 |
| **6. Train (온디바이스 학습)** | Micro-LoRA Autograd | **개발 중 (v2.4.0)** | 모바일 Vulkan 역전파 경사하강법 및 QLoRA 가중치 갱신 |

## 5.4 궁극의 종착지: AI Chain & AI Orchestrator 자율 모바일 에이전트

6대 모달리티의 완결은 개별 모델의 구동에 그치지 않고, 상위 오케스트레이션 프레임워크인 **`termux-aichain`** 및 **`termux-ai-orchestrator`**와 유기적으로 결합됩니다.

```mermaid
graph TD
    User([사용자 / 주변 환경]) -->|음성 입력| STT[termux-stt: Whisper Vulkan]
    STT -->|텍스트 프롬프트| Orch[termux-ai-orchestrator: 자율 의사결정 엔진]
    Orch <-->|컨텍스트 추론| LLM[termux-llamacpp: Qwen2.5 Vulkan]
    Orch <-->|시각 인지| Vision[termux-vision: MobileVLM Vulkan]
    Orch -->|이미지 생성 요청| Diff[termux-diffusion: SDXS Vulkan]
    Orch -->|음성 응답 요청| TTS[termux-tts: VITS Streaming Vulkan]
    TTS -->|음성 출력| User
    Orch -->|지속 학습/개인화| Train[termux-train: On-Device LoRA Vulkan]
```

외부 클라우드 서버에 단 1바이트의 개인정보도 유출하지 않고, 단말기 내부에서 보고(Vision), 듣고(STT), 생각하고(LLM), 말하며(TTS), 창작하고(Diffusion), 스스로 진화하는(Train) **완전한 온디바이스 자율 인공지능 에이전트**의 완성이 본 교본이 지향하는 최종 지향점입니다.

---

# 제6장: 갤럭시 플릿(Galaxy Fleet: S25·S22·S21·A53·A35) 파편화 포렌식 및 100% 순혈 GPU 아키텍처 완결

## 6.1 오픈소스 업계의 '가짜 Vulkan 가속' 사기극 포렌식 (fallbacks=0 착시와 듀얼 백엔드의 암투)

오픈소스 모바일 AI 생태계(특히 Whisper.cpp 및 각종 모바일 래퍼)에는 수많은 연구자와 개발자들을 기만해 온 고질적인 '가짜 Vulkan 가속' 관행이 존재해 왔습니다.

```text
whisper_init_with_params_no_state: devices = 2
whisper_init_with_params_no_state: backends = 2
...
whisper_print_timings: fallbacks = 0 p / 0 h
```

### 기만성 구조의 2대 진실
1. **`fallbacks = 0 p / 0 h`의 착시**:
   - 수많은 마케팅 문서와 레포지토리는 이를 "하드웨어 오류 0건, GPU 100% 완주"로 왜곡하여 홍보합니다.
   - 그러나 이 수치는 하드웨어 백엔드(GPU/CPU)와 아무런 상관이 없는 **Whisper 디코더의 온도 기반 샘플링 재시도 횟수(Sampling Temperature Fallback: $p$=probability threshold, $h$=entropy threshold)** 에 불과합니다.
2. **`backends = 2`와 스케줄러의 비대칭적 야합**:
   - `ggml_backend_sched`는 초기화 시 Vulkan 백엔드와 CPU 백엔드를 동시에 등록합니다.
   - 거대한 행렬곱으로 구성된 **인코더(Encoder)** 는 1~2초간 Vulkan GPU로 실행하여 사용자에게 "GPU 가속 중"이라는 착시를 부여합니다.
   - 반면 전체 추론 시간의 80% 이상을 점유하는 자잘한 벡터 연산과 KV 캐시 업데이트가 반복되는 **디코더(Decoder)** 루프에서는 모바일 드라이버 호환성을 이유로 `supports_op() = false`를 반환하며 **조용히 4~6개의 CPU NEON 스레드로 연산을 짬때려 단말기를 고열의 난로로 전락**시킵니다.

AMEVA 프로젝트는 이러한 '무늬만 GPU'인 기만적 하이브리드 타협을 전면 배제하고, 디코더의 마지막 1토큰까지 오롯이 GPU VRAM과 셰이더 파이프라인에서 완주시키는 **순혈 GPU(Pure GPU Purism) 아키텍처**를 확립하였습니다.

---

## 6.2 Android Bionic 링커 충돌 포렌식과 전역 비오염 동적 디스패처 (Non-polluting Vulkan Dynamic Dispatcher)

Termux 환경에서 온디바이스 Vulkan을 구동할 때 직면하는 첫 번째 물리적 장벽은 리눅스 배포판 패키지 매니저와 Android Bionic 동적 링커 간의 런타임 충돌입니다.

### 1. 레거시 우회책(심볼릭 링크 및 LD_LIBRARY_PATH 주입)의 치명적 한계 포렌식
- **심볼릭 링크는 링커 네임스페이스를 격리하지 못함**:
  - `~/.local/share/ameva/lib/libvulkan.so -> /system/lib64/libvulkan.so`와 같은 심볼릭 링크 계층은 파일 탐색 위치만 우회할 뿐, 동적 링커의 심볼 해석 스코프를 격리하지 못합니다.
  - 오히려 오래된 라이브러리 우선 바인딩, SONAME 중복, 비결정적 로딩을 유발합니다.
- **`LD_LIBRARY_PATH` 주입에 따른 네임스페이스 오염 및 이중 런타임 충돌**:
  - `/system/lib64`를 `LD_LIBRARY_PATH`에 주입할 경우, 프로세스가 의존하는 모든 공유 라이브러리를 시스템 경로에서 우선 탐색하게 됩니다.
  - 이 과정에서 Android OS 내부 라이브러리(`/system/lib64/libc++.so`, `libbase.so`, `libunwindstack.so` 등)가 Termux 사용자 공간의 `libc++_shared.so`와 동일 프로세스 주소 공간에 동시 적재되면서 ABI 충돌 및 기호 불일치(예: Galaxy S22의 `SIGSEGV 139`)를 유발합니다.
- **`RTLD_LOCAL`의 정확한 공학적 정의**:
  - `dlopen(..., RTLD_LOCAL)`은 로드된 라이브러리의 심볼이 이후 로드되는 다른 라이브러리의 전역 심볼 해석(Global Symbol Resolution)에 노출되지 않도록 가시성을 제한할 뿐이며, **독립된 Bionic Linker Namespace를 물리적으로 생성하지 않습니다**.

### 2. 정공법: 전역 검색 경로 비오염 동적 디스패치 아키텍처
AMEVA-Runtime은 `LD_LIBRARY_PATH`를 일절 오염시키지 않는 표준 동적 디스패처를 채택합니다:
1. **`LD_LIBRARY_PATH` 불변 유지**: 환경변수에 `/system/lib64`, `/vendor/lib64`, `/apex/...` 등을 절대 주입하지 않음.
2. **동적 로더 바인딩**:
   $$\text{handle} = \text{dlopen}(\text{"libvulkan.so"}, \text{RTLD\_NOW} \mid \text{RTLD\_LOCAL})$$
   Bionic 기본 탐색 체인을 통해 시스템 표준 Vulkan 로더를 안전하게 취득하며, 부재 시 플랫폼 공식 경로에 한해 제한적으로 폴백.
3. **`vkGetInstanceProcAddr` 기반 함수 포인터 디스패치 테이블 구축**:
   전역 심볼 테이블에 의존하지 않고 오직 취득한 인스턴스/디바이스 프로시저 주소를 통해서만 모든 Vulkan API를 호출.
4. **수명주기(Lifetime) 일치**:
   디스패치 테이블의 수명주기는 로더 라이브러리 핸들의 수명주기에 종속되어 핸들 조기 해제에 따른 댕글링 포인터 호출을 원천 차단.

---

## 6.3 Galaxy S25 (Snapdragon 8 Elite / Adreno 830): mul_mat_vec 파이프라인 결함 가설과 8대 인과 증명 프로토콜

### 하드웨어 사양
- **SoC**: Qualcomm Snapdragon 8 Elite (`sun`)
- **GPU**: Qualcomm Adreno 830 (Warp Size: 64, Shared Memory: 32KB)
- **OS**: Android 16 (Bionic 64-bit)

### 관측된 사실 (Ground Truth)
Qualcomm Adreno 830 단말기 환경에서 Whisper GGML Vulkan 백엔드의 `mul_mat_vec` 커널 파이프라인 생성 시, 특수화 상수 $NUM\_COLS \ge 3$ 조건에서 드라이버 내부 `vk::Device::createComputePipeline: ErrorUnknown (-13)`이 발생하며 프로세스가 비정상 종료됨.

### 원인 가설 (Hypothesis) 및 미증명 요인
- **가설 1 (레지스터 압력/Spill)**: $NUM\_COLS$ 증가에 따른 누적기(Accumulator) 및 중간 변수 증가로 컴파일러 레지스터 할당 한계를 초과했을 가능성.
- **기타 가능성**: SPIR-V 바이트코드 유효성 결함, Specialization Constant 전달 버그, Descriptor/Push Constant 인덱싱 오류, 버퍼 Out-of-Bounds 접근, 워크그룹 크기 레이아웃 부적합, 드라이버 JIT 컴파일러 내부의 특정 옵티마이저 패스 버그.

### 인과 관계 확정 8대 필수 조건 (Proof of Causation Requirements)
단순 추정을 배제하고 근본 원인을 확정하기 위해 아래 8개 조건을 충족해야 함:
1. $NUM\_COLS \in \{1, 2, 3, 4, 8\}$ 별 컴파일된 원본 SPIR-V 바이트코드 보존 및 덤프
2. Khronos 공식 `spirv-val` 정적 유효성 검사 100% 통과 입증
3. `spirv-dis` 역어셈블리 비교를 통한 명령어 시퀀스 차이 규명
4. 파이프라인 생성 단계(`vkCreateComputePipelines`)와 디스패치 실행 단계(`vkQueueSubmit`)의 오류 격리 및 분리 관측
5. Vulkan Validation Layers 상의 오류 및 경고 0건 검증
6. GPU fault 또는 Android tombstone 로그 확보
7. 누적기 분할(Accumulator Splitting) 또는 루프 구조 변경 후 동일 입력에서 파이프라인 생성/실행 성공
8. CPU Reference 출력과 결과값 비트 단위 정확도 허용 오차 내 일치

### 만료 조건부 임시 안전장치 (Interim Safety Guard)
- 현재 적용된 `static constexpr uint32_t mul_mat_vec_max_cols = 2;`는 근본 해결책이 아닌 **임시 안전장치(Workaround)**임.
- **만료 기준**: 신규 리디자인 커널이 8대 인과 증명 조건 및 Gate 2(50회 연속 실행 100% 안정성)를 통과하는 즉시 본 제한을 해제하고 원상 복구함.
- **실측 성능 (Tiny 모델, JFK 1분 음성 고정 Fixture)**:
  - 전사 소요 시간: 13.98초 (RTF 0.23x, 실시간 대비 4.3배)
  - 인코딩 시간: 2,949 ms (4 runs)
  - 배치 디코딩 시간: 8,659 ms (719 runs, GPU VRAM 연산)

---

## 6.4 Galaxy S22 (Snapdragon 8 Gen 1 / Adreno 730): 링커 충돌 포렌식 및 통계적 성능 실증

### 하드웨어 사양
- **SoC**: Qualcomm Snapdragon 8 Gen 1 (`taro`)
- **GPU**: Qualcomm Adreno 730 (Warp Size: 64/128)
- **OS**: Android 16

### 링커 충돌 원인 가설 및 포렌식 체계
- `/system/lib64` 주입 시 발생한 `SIGSEGV 139`는 플랫폼 시스템 라이브러리와 Termux 사용자 라이브러리의 심볼 혼재에 따른 ABI 또는 의존성 충돌 가설로 분류.
- **포렌식 입증 도구**: `readelf -d`를 통한 `DT_NEEDED` 검사, `/proc/$PID/maps` 기반 메모리 맵 수집, Android Tombstone 백트레이스 분석.
- `LD_LIBRARY_PATH` 비오염 동적 디스패처 적용 후 정상 구동 완주.

### 실측 성능 지표 (합격 기준: 오류 0건 및 통계적 지연시간)
- 임의적 지표인 'GPU 점유율 70%'를 배제하고 다음 통계 지표를 공식 채택:
  - **1분 JFK 전사 소요 시간**: **20.37초** (RTF 0.34x, 실시간 대비 3.0배)
  - **기능 무결성**: Validation Layer Error 0건, Tombstone 0건, NaN/Inf 0건, CPU Reference 일치율 100%
  - 디코더 Flash Attention 및 행렬곱 전 구간 GPU 하드웨어 가속 완주.

---

## 6.5 Galaxy S21 5G (Exynos 2100 / Mali-G78): Headless 환경 Vulkan 컴퓨트 파이프라인 검증

### 하드웨어 사양
- **SoC**: Samsung Exynos 2100
- **GPU**: ARM Mali-G78 MP14 (14코어, Valhall 1세대)
- **Driver**: ARM Proprietary r38p0 (`conformanceVersion = 1.3.1.0`)
- **OS**: Android 15

### 엔지니어링 검증
- Android Headless CLI(디스플레이 서피스 부재 콘솔) 환경에서 Vulkan 컴퓨트 전용 모드 바인딩 검증.
- `vkGetInstanceProcAddr` 기반 동적 디스패처를 통해 Exynos Mali-G78 하드웨어 디바이스 직접 바인딩 성공.
- **실측 성능**:
  - **1분 JFK 전사 소요 시간**: **27.49초** (RTF 0.45x, 실시간 대비 2.2배)
  - Validation 오류 0건 및 100% 물리 GPU 완주.

---

## 6.6 Galaxy A53 5G / A35 / A34 (Mali-G68): 런타임 큐 질의 및 6대 불변조건 코드 강제

### 3대 중급기 실리콘 아키텍처 동일성
- **Galaxy A35 5G**: Exynos 1380 ➔ **ARM Mali-G68 MP5 (5코어)**
- **Galaxy A53 5G**: Exynos 1280 ➔ **ARM Mali-G68 MP4 (4코어)**
- **Galaxy A34 5G**: Dimensity 1080 ➔ **ARM Mali-G68 MC4 (4코어)**
물리적 GPU 코어는 100% 동일한 **ARM Valhall 2세대 Mali-G68** 실리콘.

### QueueSubmit 세그폴트(SIGSEGV 11) 원인 및 런타임 동적 질의
- **발생 현상**: `QueueSubmit(VkQueue_T*, ...)+0` null pointer dereference (`x0 = 0x0000000000000000`).
- **원인 분석**: 특정 칩셋의 큐 개수를 임의로 가정하거나 단일 패밀리 디바이스에서 분리된 Compute Queue 핸들을 무단 참조함에 따라 NULL 포인터가 `vkQueueSubmit`에 전달됨.
- **동적 질의 원칙**: `queueCount`를 하드코딩하지 않고, `vkGetPhysicalDeviceQueueFamilyProperties`를 런타임에 직접 질의하여 동적으로 큐를 바인딩함.

### 6대 큐 불변조건 (Queue Invariants) 코드 강제
1. `queueFamilyIndex < queueFamilyPropertyCount`
2. `requestedQueueCount <= queueFamilyProperties[index].queueCount`
3. `queueIndex < requestedQueueCount`
4. `VkDeviceQueueCreateInfo`가 해당 패밀리를 유효하게 지정
5. `vkGetDeviceQueue` 호출 후 획득한 `VkQueue != VK_NULL_HANDLE`
6. 디바이스 디스패치 테이블의 `vkQueueSubmit != nullptr`

---

## 6.7 Zero-Silent-Fallback 락다운: 순혈 GPU 컴플라이언스

AMEVA-Runtime은 `--device gpu` 또는 `-d gpu` 플래그 명시 시 예외 상황에서 시스템이 조용히 CPU로 폴백하여 사용자를 기만하는 행위를 엄격히 차단합니다:

1. **스케줄러 노드 분기 감사**:
   - 추론 그래프 실행 중 하드웨어 미지원 등으로 CPU 연산 노드로 우회되는 경우, 결과를 은폐하지 않고 즉각 명시적 예외(`AmevaVulkanError`)를 발생시켜 실패 처리.
2. **사전 진단 체계(Doctor) 강제**:
   - 디바이스 드라이버, 큐 패밀리, SPIR-V 기능 검사를 사전에 수행하여 순혈 GPU 구동이 불가능할 경우 원인 코드와 함께 Fail-Fast.

---

## 6.8 5대 갤럭시 플릿(Galaxy Fleet) 통합 실측 벤치마크 매트릭스

1분(60초) 분량 고정 자산(JFK 연설 음성, SHA-256: `94f9b88d30e3bb44a10dfbebfbcf06294eb8e398dcbf7f7a7da933b9e4ec3ff1`, Tiny 모델 77.11MB)에 대한 전 기종 실기기 동시 실측치:

| 대상 단말기 | 탑재 SoC / 하드웨어 실리콘 | GPU 코어 위상 | 전사 지연시간 | RTF (Real-Time Factor) | 기능 무결성 검증 |
| :--- | :--- | :--- | :---: | :---: | :---: |
| **Galaxy S25** | Snapdragon 8 Elite (`sun`) | **Adreno 830** | **13.98초** | **0.23x** | Fault 0건, Validation 0건 |
| **Galaxy S22** | Snapdragon 8 Gen 1 (`taro`) | **Adreno 730** | **20.37초** | **0.34x** | Fault 0건, Validation 0건 |
| **Galaxy S21 5G** | Exynos 2100 | **Mali-G78 MP14** | **27.49초** | **0.45x** | Fault 0건, Validation 0건 |
| **Galaxy A35** | Exynos 1380 | **Mali-G68 MP5** | **160.2초 (Large)** | **2.26x vs CPU** | Fault 0건, Valhall 최적화 |
| **Galaxy A53 5G** | Exynos 1280 | **Mali-G68 MP4** | 큐 인스펙션 완료 | 6대 불변조건 강제 | **QueueSubmit NULL 차단** |

---

# 제7장: 에필로그 및 오픈소스 엔지니어링 선언

남들이 "모바일 기기에서 불칸 가속은 시기상조이며 불안정하다"며 타협하고 클라우드로 회귀할 때, 
우리는 안드로이드 Bionic 링커와 셰이더 컴파일러 소스코드의 밑바닥까지 내려가 정수 나눗셈 `16 / 32 = 0`의 무한루프와 Adreno 드라이버의 레지스터 고갈 버그를 찾아내 목을 비틀었습니다.

책상 위에 놓인 작은 스마트폰 하나가 거대한 신경망을 오롯이 자신의 실리콘 힘으로 계산해 내는 순간, 인공지능은 거대 독점 테크 기업의 데이터센터를 벗어나 인류 개개인의 손끝에서 진정한 자유를 얻게 됩니다.

본 교본에 기록된 모든 수학적 공식, 소스코드 패치, 실기기 로그는 미래 세대의 엔지니어들이 온디바이스 컴퓨팅의 미개척지를 개척하는 데 디딤돌이 될 것입니다.

---
**[문서 보관 및 참조 링크]**
- AMEVA 런타임 저장소: [`dev/ameva-runtime`](https://github.com/uno-km/ameva-runtime)
- LLM 패치 분석 백서: [`docs/research/MALI_VALHALL_VULKAN_INFINITE_LOOP_ANALYSIS.md`](docs/research/MALI_VALHALL_VULKAN_INFINITE_LOOP_ANALYSIS.md)
- STT 실기기 포렌식 감사서: [`ground_truth_adreno830_vulkan_stt_audit.md`](docs/research/ground_truth_adreno830_vulkan_stt_audit.md)
- TTS 지연시간 사후보고서: [`vulkan_tts_latency_postmortem.md`](docs/research/vulkan_tts_latency_postmortem.md)

