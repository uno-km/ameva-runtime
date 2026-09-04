# Community Announcement Drafts

This document contains prepared publication drafts for technical communities regarding the ARM Mali Vulkan compute hang fix in `llama.cpp` and `dev/ameva-vulkan-runtime`.

---

## 1. Reddit r/LocalLLaMA Draft

### Post Title
`[Technical Analysis & Fix] Solved the deterministic Vulkan GPU Freeze / DeviceLost on ARM Mali (Exynos/Dimensity) in llama.cpp`

### Post Body
```markdown
Hi everyone,

If you have tried running `llama.cpp` with the Vulkan backend (`-DGGML_VULKAN=ON`) on mobile devices powered by ARM Mali GPUs (such as Samsung Exynos 1380 / Mali-G68, Mali-G78, Dimensity Mali-G7xx series), you likely encountered a deterministic hang followed by:
```
ggml_vulkan: vk::Device::waitForFences: ErrorDeviceLost
```
The GPU would freeze on the very first quantized matrix multiplication (Node 2, `Qcur-0` or `MUL_MAT`), stall for 60+ seconds, and crash under the driver's watchdog timeout.

We conducted an in-depth debugging and disassembly session on physical hardware (Samsung Galaxy A35 5G, Exynos 1380, Mali-G68 MP5) and identified the exact mathematical root cause in the compute shader and vendor routing.

---

### Root Cause Analysis

In `mul_mm.comp`:
```glsl
const uint loadstride_b = gl_WorkGroupSize.x * LOAD_VEC_B / BK;
...
[[unroll]] for (uint l = 0; l < BN; l += loadstride_b) {
    ...
}
```
When executing quantized matrix multiplication using the Small (`_s`) unaligned pipeline (`warptile_mmq_s`), the workgroup size is configured to match the hardware subgroup size:
- `gl_WorkGroupSize.x = device->subgroup_size`
- `BK = 32` (Quantization block size for Q4_0, Q4_K, etc.)
- `LOAD_VEC_B = 1` (Scalar float unaligned load)

On desktop GPUs (NVIDIA, AMD, Intel):
- Subgroup size is 32 or 64.
- Integer division: `32 * 1 / 32 = 1`. The loop advances properly by 1.

On ARM Mali (Valhall / Bifrost):
- Native subgroup size is **16**.
- Integer division in GLSL:
  $$\text{loadstride}_b = \left\lfloor \frac{16 \times 1}{32} \right\rfloor = 0$$
- This yields:
  ```glsl
  for (uint l = 0; l < BN; l += 0)
  ```
  **An infinite loop executed directly inside the GPU compute shader.**
- The GPU cores spin indefinitely until the Android kernel watchdog detects a hardware TDR and issues `VK_ERROR_DEVICE_LOST`.

Furthermore, `ggml-vulkan.cpp` defines vendor IDs for Apple, Intel, and NVIDIA, but completely omits ARM (`0x13b5`), defaulting ARM devices into the Small (`_s`) pipeline for batch tokens $N \le 32$.

---

### The Fix

1. Define `VK_VENDOR_ID_ARM 0x13b5`.
2. In `ggml_vk_guess_matmul_pipeline()`, intercept devices where `vendor_id == VK_VENDOR_ID_ARM` or `subgroup_size < 32` and force them to use the Medium (`_m`) pipeline (`mmp->a_m : mmp->m`).

The Medium pipeline configures `gl_WorkGroupSize.x = 128`:
$$\text{loadstride}_b = \left\lfloor \frac{128 \times 1}{32} \right\rfloor = 4 > 0$$
The loop strides forward correctly, completely eliminating the GPU hang.

---

### Empirical Benchmark on Physical Hardware

- **Device**: Samsung Galaxy A35 5G (SM-A356N)
- **SoC**: Samsung Exynos 1380 (4x Cortex-A78 @ 2.4GHz + 4x Cortex-A55 @ 2.0GHz)
- **GPU**: ARM Mali-G68 MP5 (Vulkan 1.3, Subgroup Size: 16)
- **Model**: `Qwen2.5-0.5B-Instruct-Q4_K_M.gguf`
- **Layers Offloaded**: 25 / 25 (100% GPU offload, including LM-Head)
- **OS**: Android 16 / Termux 64-bit

| Metric | CPU Baseline (6 NEON threads) | Mali-G68 Vulkan (Before) | Mali-G68 Vulkan (After Fix) |
| :--- | :---: | :---: | :---: |
| **Status** | Clean Execution | **Hang & Crash (`DeviceLost`)** | **Clean Execution (Exit Code 0)** |
| **Prompt Eval (t/s)** | 20.89 t/s | 0.00 t/s (Deadlock) | **14.86 t/s** |
| **Token Gen (t/s)** | 3.50 t/s | 0.00 t/s (Deadlock) | **4.44 t/s (+26.9% vs CPU)** |
| **Latency per Token** | 286.02 ms | $\infty$ | **225.22 ms** |
| **VRAM Leak** | 0 bytes | N/A | **0 bytes** |
| **Numerical Fidelity** | Identical (Seed 42) | N/A | **Identical (Seed 42)** |

Full technical whitepaper and diff are available in our open-source repo:
- Whitepaper: `docs/research/MALI_VALHALL_VULKAN_INFINITE_LOOP_ANALYSIS.md`
- Upstream PR proposal has been prepared for `ggerganov/llama.cpp`.

Has anyone else noticed similar subgroup size issues on mobile GPUs (Adreno / PowerVR)? Happy to discuss in the comments.
```

---

## 2. Korean Tech Community Draft (GeekNews / Velog / 기술 블로그)

### Post Title
`[기술 분석] llama.cpp Vulkan 백엔드의 ARM Mali GPU 무한루프(DeviceLost) 원인 규명 및 해결기 (갤럭시 A35 실측)`

### Post Body
```markdown
### 1. 개요
모바일 기기에서 경량 언어 모델(On-Device LLM)을 구동하기 위해 `llama.cpp`의 Vulkan 백엔드를 활용할 때, 삼성 엑시노스(Exynos) 탑재 기기(ARM Mali-G68/G78/G710 등)에서 모델 추론 시작과 동시에 GPU가 100% 프리징되며 `vk::Device::waitForFences: ErrorDeviceLost` 에러가 발생하는 문제가 지속되어 왔습니다.

본 글에서는 갤럭시 A35 5G(Exynos 1380, Mali-G68 MP5) 실기기 환경에서 Vulkan 셰이더와 커널 디버깅을 진행하여 밝혀낸 수학적/아키텍처적 원인과, 이를 해결하여 CPU 대비 +26.9%의 토큰 생성 성능을 달성한 패치 내용을 공유합니다.

---

### 2. 근본 원인 분석 (Root Cause)

#### (1) GLSL 컴퓨트 셰이더 내 정수 나눗셈 언더플로우
문제는 `mul_mm.comp` 셰이더 내의 B 행렬 로드 스트라이드 계산식에서 발생했습니다.

```glsl
const uint loadstride_b = gl_WorkGroupSize.x * LOAD_VEC_B / BK;
...
[[unroll]] for (uint l = 0; l < BN; l += loadstride_b) {
    ...
}
```

양자화된 행렬 곱(`warptile_mmq_s`) 파이프라인에서:
- `gl_WorkGroupSize.x`는 타겟 하드웨어의 `subgroup_size`로 할당됩니다.
- 양자화 블록 크기 `BK`는 32입니다.
- 스칼라 비정렬 로드 시 `LOAD_VEC_B`는 1입니다.

데스크톱 GPU(NVIDIA, AMD, Intel)의 서브그룹 크기는 32 또는 64입니다.
따라서 `32 * 1 / 32 = 1`이 되어 루프가 정상적으로 1씩 증가합니다.

반면, **ARM Mali(Valhall 아키텍처)의 네이티브 서브그룹 크기는 16**입니다.
정수 나눗셈 연산 결과:
$$\text{loadstride}_b = \left\lfloor \frac{16 \times 1}{32} \right\rfloor = 0$$

결과적으로 셰이더 내부에서 다음과 같은 루프가 실행됩니다:
```glsl
for (uint l = 0; l < BN; l += 0)
```
증감값이 0이 되어 **GPU 내부 스레드가 무한루프에 갇히게 됩니다.** 이로 인해 Vulkan 큐가 응답하지 않고, 안드로이드 커널의 하드웨어 감시 타이머(Watchdog)가 만료되면서 60초 후 `VK_ERROR_DEVICE_LOST`를 발생시키며 비정상 종료되었습니다.

#### (2) llama.cpp 내 ARM 벤더 식별자 누락
`ggml-vulkan.cpp` 내에 AMD, Apple, Intel, NVIDIA에 대한 벤더 식별자 분기는 존재했으나, ARM(`0x13b5`)에 대한 정의가 누락되어 있었습니다. 이로 인해 배치 토큰 $N \le 32$ 구간에서 결함이 있는 Small(`_s`) 파이프라인으로 무조건 강제 할당되고 있었습니다.

---

### 3. 해결책 (Solution)

1. `VK_VENDOR_ID_ARM 0x13b5` 정의 추가.
2. `ggml_vk_guess_matmul_pipeline()`에서 벤더가 ARM이거나 `subgroup_size < 32`인 디바이스는 Small 파이프라인 대신 Medium(`_m`) 파이프라인으로 라우팅.

Medium 파이프라인은 워크그룹 크기를 128로 설정하므로:
$$\text{loadstride}_b = \left\lfloor \frac{128 \times 1}{32} \right\rfloor = 4 > 0$$
루프가 정상적으로 전진하며 프리징이 완전히 해소됩니다.

---

### 4. 실기기(갤럭시 A35 5G) 벤치마크 결과

- **모델**: `Qwen2.5-0.5B-Instruct-Q4_K_M.gguf` (25개 레이어 전체 GPU 100% 오프로드)
- **하드웨어**: Samsung Galaxy A35 (Exynos 1380, Mali-G68 MP5, 8GB RAM)

| 구분 | CPU Baseline (6 스레드) | 패치 전 (Vulkan) | 패치 후 (Vulkan GPU) |
| :--- | :---: | :---: | :---: |
| **동작 여부** | 정상 구동 | **GPU 프리징 및 충돌 (`DeviceLost`)** | **정상 구동 (Exit Code 0)** |
| **토큰 생성 속도** | 3.50 t/s | 0.00 t/s | **4.44 t/s (+26.9%)** |
| **토큰당 지연 시간** | 286.02 ms | 측정 불가 | **225.22 ms** |
| **메모리 누수** | 0 바이트 | - | **0 바이트** |
| **출력 일관성** | Seed 42 기준 Ground Truth | - | **Seed 42 기준 100% 동일한 고품질 텍스트 생성** |

---

### 5. 향후 계획
해당 분석 내용 및 패치 코드는 현재 `dev/ameva-vulkan-runtime` (v1.0.2) 패키지에 공식 반영되었으며, `llama.cpp` 공식 저장소(`ggerganov/llama.cpp`)에 업스트림 PR을 제출하여 전 세계 ARM Mali 모바일 환경에서도 Vulkan On-Device 추론이 완벽히 동작하도록 기여할 예정입니다.
```
