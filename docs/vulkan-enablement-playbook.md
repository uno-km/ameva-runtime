# 📘 Vulkan Enablement Playbook & 12-Stage Validation Hierarchy

## 1. Architectural Principles

1. **Zero Firmware-Specific Hardcoding**:
   - `/vendor/lib64/hw/vulkan.*.so`와 같은 특정 드라이버 절대 경로 하드코딩이나 문자열 비교(`name == "Mali-G78"`)를 전면 배제하고 오직 런타임 Vulkan API 기능 질의(`vkEnumeratePhysicalDevices`, `vkGetPhysicalDeviceProperties`)에 기반하여 동작합니다.
2. **Single Loader Chain Pinning via dladdr**:
   - Termux 사용자 공간 패키지(`mesa`, `vulkan-tools`)와 Android 벤더 시스템 드라이버(`/system/lib64/libvulkan.so`) 간 함수 포인터 디스패치 테이블 충돌을 차단하기 위해 `dladdr` 텔레메트리로 단일 체인을 강제 고정합니다.
3. **Fail-Fast & Zero-Data-Loss Auto Recovery**:
   - `--device auto`: 초기화 또는 연산 실패 시 서비스 중단 없이 CPU NEON 엔진으로 투명 복구.
   - `--device vulkan`: 명시적 GPU 모드 실패 시 침묵형 Fallback 없이 즉시 `PlatformNotSupportedError` 반환.

---

## 2. 12-Stage Validation Hierarchy (V0 – V11)

| 단계 | 명칭 | 검증 대상 연산 | 성공 기준 |
| :--- | :--- | :--- | :--- |
| **V0** | `Vulkan Loader Open` | `dlopen("libvulkan.so")` 시스템 라이브러리 로드 | 유효한 핸들 반환 |
| **V1** | `Instance Creation` | `vkCreateInstance()` 호출 | `VK_SUCCESS` 반환 |
| **V2** | `Device Enum` | `vkEnumeratePhysicalDevices()` 호출 | 물리 GPU 개수 > 0 |
| **V3** | `Hardware Selection`| GPU 디바이스 타입 검사 | `deviceType != eCpu` |
| **V4** | `Queue Probe` | Compute 전용 큐 패밀리 탐색 | `VK_QUEUE_COMPUTE_BIT` 식별 |
| **V5** | `Device Creation` | `vkCreateDevice()` 논리 디바이스 생성 | 필수 확장 탑재 생성 성공 |
| **V6** | `Buffer Alloc` | Host-Visible / Device-Local 메모리 바인딩 | 버퍼 쓰기/읽기 일관성 보장 |
| **V7** | `SPIR-V Compile` | SPIR-V 셰이더 모듈 및 파이프라인 빌드 | 파이프라인 핸들 정상 생성 |
| **V8** | `Shader Dispatch` | `vkCmdDispatch()` 커맨드 버퍼 제출 | 타임아웃/행(Hang) 없이 완료 |
| **V9** | `Checksum Audit` | 출력 버퍼 수치 검증 | 예상 계산값과 바이트 일치 |
| **V10**| `GGML MatMul` | `ggml-vulkan` FP32/FP16 행렬 곱셈 | 최대 절대 오차 < 1e-4 |
| **V11**| `End-to-End SDXS`| 256x256 1-step 실모델 이미지 생성 | 완전한 PNG 파일 및 해시 검증 |
