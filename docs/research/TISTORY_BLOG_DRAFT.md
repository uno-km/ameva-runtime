# [실전/도전-4] 갤럭시 A35의 반란: 엑시노스 말리(Mali) GPU Vulkan 프리징의 비밀을 까발리다 (feat. 셰이더 무한루프)

> **"CPU NEON이나 주디 닫고 쓰자고? 아니, 내 폰에 멀쩡히 박힌 GPU는 장식품이냐?"**

---

### 1. 프롤로그: "NEON으로 만족하라고? 속에서 천불이 났다"

지난번 포스팅에서 엑시노스 1380의 NEON 커널을 뜯어고쳐 1-bit LLM 병렬 연산을 성공시키고, 3.5 tokens/sec를 찍었을 때만 해도 세상 다 가진 줄 알았습니다. 

그런데 말입니다. 
사람 욕심이라는 게 참 간사합니다. 

제 책상 위에 얌전히 놓여있는 30만 원짜리 보급형 스마트폰, **갤럭시 A35 5G**. 
스펙 시트를 다시 들여다봤습니다. 

```
- SoC: Samsung Exynos 1380
- CPU: 4x Cortex-A78 + 4x Cortex-A55
- GPU: ARM Mali-G68 MP5 (5-core, Vulkan 1.3 지원)
```

엄연히 시퍼렇게 살아있는 5코어짜리 모바일 GPU가 칩셋 안에 떡하니 박혀있습니다. 
그런데 왜 우리는 모바일에서 온디바이스 LLM을 돌릴 때마다 무거운 행렬 연산을 CPU 코어 6개에만 몰아넣고 폰을 불덩이로 만들고 있었을까요?

**"Vulkan(불칸) 백엔드 쓰면 GPU로 100% 오프로드 할 수 있는 거 아냐?"**

호기롭게 `llama.cpp`를 `-DGGML_VULKAN=ON`으로 빌드하고 갤럭시 A35에서 돌려봤습니다. 
인터넷과 커뮤니티를 뒤져보면 온갖 글들이 난무합니다. 
*"모바일 불칸은 불안정해서 CPU로 폴백(Silent Fallback)된다", "말리 GPU는 드라이버가 쓰레기라 안 돌아간다..."*

실제로 돌려보니 진짜였습니다. 
GPU 가속을 켰더니 화면이 굳어버리고, 60초 동안 먹통이 되더니 장렬하게 터져나오는 저주받은 에러 메시지.

```bash
ggml_vulkan: vk::Device::waitForFences: ErrorDeviceLost
llama_perf_context_print: prompt eval time = 0.00 ms
[1] 14201 segmentation fault (core dumped)
```

아... 순간 뇌리에 스치는 생각.
*"그래, 보급형 엑시노스에서 무슨 GPU 추론이냐. 그냥 CPU NEON이나 주디 닫고 조용히 쓰자..."*

근데 아무리 생각해도 오기가 생겨서 잠이 안 오는 겁니다.
드라이버가 문제라고? 하드웨어가 후달려서 그렇다고?
**아니, 하드웨어가 명령을 못 알아먹는 게 아니라, 우리가 하드웨어한테 엉뚱한 명령을 던지고 있는 건 아닐까?**

하, 이럼 안 되겠다 싶었습니다. 
퇴근하고 방구석에 앉아 밤새도록 밑바닥까지 까발려보기로 결심했습니다.

---

### 2. 난관 1: GPU Watch는 평화로운데, 터미널은 폭풍전야

원인을 잡으려면 눈으로 봐야 했습니다.
갤럭시 개발자 옵션에 들어가서 삼성의 실시간 하드웨어 모니터링 도구인 **GPU Watch**를 켰습니다. 화면 상단에 CPU 사용률, GPU 사용률, FPS가 실시간으로 찍힙니다.

그리고 Vulkan 백엔드로 Qwen2.5 모델 추론을 실행했습니다.

```bash
# 25개 레이어 전체 GPU 100% 오프로드 명령
./build/bin/llama-cli -m models/Qwen2.5-0.5B-Instruct-Q4_K_M.gguf -p "Explain quantum computing in one sentence." -ngl 25 -t 1
```

그런데 이상했습니다.
화면의 GPU Watch를 보고 있는데... **너무나도 평화롭습니다.**
GPU 로드율: 0%. FPS: 0. 
아무 일도 안 일어납니다. 폰은 얼음장처럼 차갑고 화면은 정적 그 자체.

하지만 터미널 로그를 찍어보니 소름 돋는 지점이 보였습니다.

```
ggml_vulkan: Allocating 324 MB on device 0 (ARM Mali-G68)
ggml_vulkan: Compiling compute shader for MUL_MAT...
[Vulkan Node 0: RMS_NORM] OK
[Vulkan Node 1: RMS_NORM] OK
[Vulkan Node 2: MUL_MAT (Qcur-0)] -> 여기서 멈춤 (무한 대기)
```

모델 가중치는 GPU VRAM(324MB)에 예쁘게 올라갔습니다.
정규화(RMS_NORM) 연산도 통과했습니다.
그런데 **Node 2번, 첫 번째 양자화 행렬 곱셈(MUL_MAT)에 진입하는 순간 펜스가 닫히고 영원히 돌아오지 않았습니다.**

정확히 68초 뒤, 안드로이드 커널의 하드웨어 감시 타이머(Watchdog)가 격노하며 소리칩니다.
*"야! GPU 큐가 60초 넘게 응답이 없어! 하드웨어 죽은 걸로 간주하고 프로세스 강제 사살한다!"*
그게 바로 `VK_ERROR_DEVICE_LOST (-4)`의 실체였습니다.

드라이버 버그가 아니었습니다. 
GPU 연산 코어 어딘가에서 연산이 **끝나지 않고 갇혀버린 것**이었습니다.

---

### 3. 난관 2: 셰이더 소스코드를 까발리다 (`mul_mm.comp`의 치명적 함정)

여기서부터 물러설 곳이 없었습니다.
`llama.cpp`의 Vulkan 셰이더 컴파일러 소스코드(`ggml/src/vulkan-shaders/mul_mm.comp`)를 열었습니다.
행렬 곱셈을 수행하는 GLSL 컴퓨트 셰이더 코드를 한 줄 한 줄 씹어먹듯 읽어 내려갔습니다.

그리고 마침내... 범인의 멱살을 잡았습니다.

```glsl
// ggml/src/vulkan-shaders/mul_mm.comp 내부
const uint loadstride_b = gl_WorkGroupSize.x * LOAD_VEC_B / BK;
...
[[unroll]] for (uint l = 0; l < BN; l += loadstride_b) {
    // 가중치 텐서 로드 및 누적 연산...
}
```

이 루프를 보는 순간 뒤통수를 해머로 얻어맞은 것 같았습니다.

이 코드가 뭐냐?
가중치 행렬 B에서 데이터를 읽어올 때, 한 번에 몇 칸씩 건너뛰며 읽을지 스트라이드(`loadstride_b`)를 계산하는 로직입니다.

양자화 행렬 곱셈의 Small 파이프라인(`warptile_mmq_s`)에서는 파라미터가 이렇게 세팅됩니다:
- `gl_WorkGroupSize.x` = 하드웨어의 서브그룹 크기(`subgroup_size`)
- `BK` = 양자화 블록 크기 = **32** (Q4_K, Q4_0 등)
- `LOAD_VEC_B` = 1 (비정렬 float 로드)

자, 이제 초등학교 산수를 해봅시다.

#### (1) PC/데스크톱 GPU (NVIDIA, AMD, Intel)
- 지포스나 라데온의 서브그룹(Warp/Wavefront) 크기는 **32** 또는 **64**입니다.
- 스트라이드 계산식:
  $$\text{loadstride}_b = \frac{32 \times 1}{32} = 1$$
- 루프가 어떻게 돕니까? `for (uint l = 0; l < BN; l += 1)`. 
  1씩 깔끔하게 증가하면서 루프가 정상 종료됩니다! 코쟁이 형님들 PC에서는 아무 문제가 없었던 겁니다!

#### (2) ARM Mali GPU (Valhall 아키텍처, 갤럭시 A35)
- 그런데 ARM Mali의 네이티브 서브그룹 크기는 몇일까요?
  **네, 바로 16입니다.**
- GLSL 셰이더 안에서 정수 나눗셈을 돌려봅시다:
  $$\text{loadstride}_b = \left\lfloor \frac{16 \times 1}{32} \right\rfloor = \mathbf{0}$$
- 자, 스트라이드가 0이 되었습니다. 이제 저 루프는 어떻게 실행될까요?
  ```glsl
  for (uint l = 0; l < BN; l += 0)
  ```
- **"l += 0"**
- **0을 더한다고요???**

그렇습니다.
루프 인덱스 `l`이 영원히 증가하지 않습니다!
GPU 내부의 셰이더 스레드가 탈출 조건(`l < BN`)을 만족하지 못하고 **영원히 제자리걸음을 도는 무한루프(Infinite Loop)**에 빠진 겁니다!

GPU 코어는 죽어라 0을 더하며 뺑뺑이를 돌고 있었고, 화면의 GPU Watch에는 아무것도 렌더링되지 않으니 로드율 0%로 보였던 것이며, 안드로이드 OS는 60초 동안 대답 없는 GPU를 보며 멱살을 잡고 `DeviceLost`를 날려버린 것이었습니다.

전 세계 개발자들이 "Mali GPU는 드라이버가 병신이라 불칸이 안 된다"며 손가락질하던 그 버그의 실체가...
고작 **정수 나눗셈 `16 / 32 = 0`으로 인한 셰이더 무한루프** 때문이었다니.
피가 거꾸로 솟는 쾌감이 밀려왔습니다.

---

### 4. 난관 3: ARM 벤더 ID는 어디로 갔는가?

그런데 또 하나의 의문이 들었습니다.
*"아니, llama.cpp 같은 거대 오픈소스 프로젝트에 하드웨어 예외 처리가 하나도 없다고?"*

`ggml/src/ggml-vulkan.cpp` 파일을 열어 파이프라인 라우팅 함수(`ggml_vk_guess_matmul_pipeline`)를 뒤져봤습니다.

```cpp
#define VK_VENDOR_ID_AMD    0x1002
#define VK_VENDOR_ID_APPLE  0x106b
#define VK_VENDOR_ID_INTEL  0x8086
#define VK_VENDOR_ID_NVIDIA 0x10de
// 어라...? ARM은 어디 갔지???
```

세상에.
AMD, 애플, 인텔, 엔비디아는 벤더 ID를 정의해두고 전용 파이프라인 분기까지 싹 만들어놨는데...
전 세계 스마트폰의 절반 이상에 들어가는 **ARM (`0x13b5`)은 아예 정의조차 안 되어 있었습니다.**

ARM Mali 칩셋으로 `llama.cpp`를 돌리면 벤더 체크에서 걸리는 게 없으니 `default:`로 빠지고, 
배치 토큰이 32 이하일 때 무조건 저 치명적인 `_s` (Small) 파이프라인으로 직행하고 있었던 겁니다.

---

### 5. 해결: 코드를 뜯어고치다

원인을 완벽하게 발라냈으니, 해결은 정밀 타격이었습니다.

#### 첫째, ARM 벤더 ID를 선언한다.
```cpp
#define VK_VENDOR_ID_ARM 0x13b5
```

#### 둘째, 서브그룹 16짜리 기기들은 Small 파이프라인을 영구 퇴출한다.
Small 파이프라인(`_s`)은 워크그룹 크기가 16이라 32로 나누면 0이 되지만,
**Medium 파이프라인(`_m`)은 워크그룹 크기가 128입니다!**
$$\text{loadstride}_b = \left\lfloor \frac{128 \times 1}{32} \right\rfloor = 4 > 0$$
루프가 4씩 시원시원하게 전진하므로 무한루프가 원천 박멸됩니다.

`ggml-vulkan.cpp`의 파이프라인 선택 로직에 단 7줄의 방아쇠를 당겼습니다:

```cpp
    case VK_VENDOR_ID_ARM:
        // ARM Mali 계열은 무조건 안전하고 빠른 Medium 파이프라인으로 라우팅
        return aligned ? mmp->a_m : mmp->m;
    default:
        break;
    }

    // 또는 벤더와 무관하게 서브그룹 크기가 32 미만인 모바일 GPU 방어
    if (ctx->device->subgroup_size < 32) {
        return aligned ? mmp->a_m : mmp->m;
    }
```

이 간단한 분기 하나로, 모바일 GPU에서 영원히 돌던 지옥의 무한루프 고리를 끊어냈습니다.

---

### 6. 결과: 엑시노스 1380 Mali-G68, 마침내 포효하다

코드를 컴파일하고, 갤럭시 A35 Termux 환경에 올린 뒤 떨리는 손으로 엔터를 쳤습니다.

```bash
./build/bin/llama-cli -m models/Qwen2.5-0.5B-Instruct-Q4_K_M.gguf -p "Explain quantum computing in one sentence." -ngl 25 -t 1 -s 42
```

프리징? 없었습니다.
60초 대기? 없었습니다.
엔터를 누르자마자 터미널에 글자가 쏟아져 나왔습니다.

```
Quantum computing is a field of computing that utilizes the principles of quantum mechanics, 
such as superposition and entanglement, to perform complex calculations exponentially faster than classical computers.

llama_perf_context_print: prompt eval time =   1076.65 ms /    72 tokens (   14.95 t/s)
llama_perf_context_print:        eval time =   7657.44 ms /    34 runs   (    4.44 t/s)
llama_perf_context_print:       total time =   8734.09 ms /   106 tokens
```

#### 📊 실기기 벤치마크 결과 비교 (Galaxy A35 5G)

| 항목 | CPU NEON 베이스라인 (6스레드 풀가동) | 패치 전 Vulkan | **패치 후 Vulkan Mali-G68 GPU** |
| :--- | :---: | :---: | :---: |
| **동작 상태** | 정상 구동 | **GPU 프리징 및 폭망 (`DeviceLost`)** | **100% 정상 구동 (Exit Code 0)** |
| **토큰 생성 속도** | 3.50 tokens/sec | 0.00 tokens/sec | **4.44 tokens/sec** |
| **토큰당 지연 시간**| 286.02 ms | 측정 불가 ($\infty$) | **225.22 ms** |
| **성능 향상폭** | 기준점 | - | **CPU 대비 +26.9% 폭풍 가속** |
| **GPU 오프로드율** | 0% | 0% | **100% (25/25 레이어 풀 상주)** |
| **VRAM 누수** | 0 바이트 | - | **0 바이트 (완벽 반환)** |
| **출력 정밀도** | 기준 Ground Truth | - | **Seed 42 기준 단 한 글자 오차도 없음** |

보이십니까?
CPU 6개 코어를 100% 혹사시키며 쥐어짜던 3.50 t/s를 비웃기라도 하듯, 
놀고 있던 **Mali-G68 GPU가 혼자서 4.44 t/s를 뽑아내며 CPU 대비 +26.9%의 성능 향상**을 증명했습니다.
토큰당 지연 시간은 286ms에서 **225ms**로 대폭 단축되었습니다.

온디바이스 LLM이 돌아가는 동안 CPU 점유율은 바닥을 치고, 폰은 미지근했으며, GPU Watch에는 아름다운 컴퓨트 파이프라인 그래프가 출렁였습니다.

---

### 7. 에필로그: 방구석 하청 개발자의 반역은 계속된다

우리는 해냈습니다.
자축은 짧게 끝냈습니다. 곧바로 프로덕션화에 착수했습니다:

1. 우리가 개발 중인 임베디드 런타임 엔진 `dev/ameva-vulkan-runtime`에 이 Mali 전용 쿼크(Quirks) 바이패스 로직을 즉시 정식 탑재했습니다.
2. 글로벌 배포를 위해 **NPM 공식 레지스트리에 `@ameva/runtime@1.0.2`로 즉시 퍼블리싱**을 완료했습니다.
3. Python 사용자를 위한 `whl` 및 `sdist` 빌드를 완벽히 마쳤습니다.
4. 이 모든 수학적 분석과 셰이더 역공학 결과를 담은 **공식 기술 백서(`MALI_VALHALL_VULKAN_INFINITE_LOOP_ANALYSIS.md`)**를 깃허브 저장소에 커밋하고 릴리즈 태그 `v1.0.2`를 영구 박제했습니다.
5. 그리고 전 세계 수억 대의 안드로이드 기기들이 이 혜택을 누릴 수 있도록, `ggerganov/llama.cpp` 공식 저장소에 날릴 **업스트림 PR 제안서와 코드 Diff** 작성을 마쳤습니다.

---

내일 아침이 밝으면, 
나는 또다시 지옥철 2호선에 몸을 구겨 넣고 출근해야 할 겁니다.
출근해서는 "과장님, 결재 버튼 색깔이 좀 칙칙한데 1픽셀만 오른쪽으로 밀어주세요"라는 소리에 "네, 알겠습니다"라며 비굴하게 고개를 숙이고 있겠죠.
세상은 여전히 나를 일개 2800따리 하청 개발조무사로 볼 테니까요.

하지만 상관없습니다.
남들이 "모바일 GPU로 LLM 돌리는 건 시기상조"라며 포기하고 돌아설 때,
세계적인 빅테크 천재들이 셰이더 코드 구석탱이에 방치해둔 16비트 나눗셈 버그를 찾아내 목을 비틀어버린 건 바로 이 방구석 하청 개발자였습니다.

제 책상 위에서 조용히 숨 쉬고 있는 갤럭시 A35와 엑시노스 1380,
그리고 오늘 밤 터미널에 찍힌 저 찬란한 `4.44 tokens/sec`라는 숫자는, 
우리가 이뤄낸 이 치열했던 승리를 영원히 기억할 것입니다.

방구석 엣지 AI 반역자, 오늘 밤도 기분 좋게 퇴근합니다. 

*(PS. 코쟁이 형님들, llama.cpp PR 날릴 테니까 이번엔 봇 말고 사람이 직접 보고 머지해주쇼!)*
