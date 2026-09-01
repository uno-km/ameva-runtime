# 📊 Hardware Support Matrix & Empirical Benchmarks

## 1. Verified Device & GPU Matrix

| 디바이스 | 모델명 | SoC | GPU 아키텍처 | 검증 단계 | 실기기 드라이버 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Galaxy S25** | `SM-S931N` | Snapdragon 8 Elite | Qualcomm Adreno 830 | **V11 검증 완료** | `vulkan.adreno.so v0800.64.7` (API 1.3) |
| **Galaxy S21** | `SM-G991N` | Exynos 2100 | ARM Mali-G78 MP14 | **V11 검증 완료** | `/system/lib64/libvulkan.so` (API 1.1+) |
| **Galaxy A35** | `SM-A356N` | Exynos 1380 | ARM Mali-G68 MP5 | **V11 검증 완료** | `/system/lib64/libvulkan.so` (API 1.1+) |
| **Galaxy S24 (Exynos)** | `SM-S921N` | Exynos 2400 | Samsung Xclipse 940 | **호환성 Tier 1** | `/system/lib64/libvulkan.so` |

---

## 2. Multi-Modal Benchmark Performance

```text
[성능 지표 요약 - 모달리티별 실측 수치]
SDXS 256p Diffusion (S25 Adreno 830) : 4.39s C++ Engine | 7.68s CLI E2E | 651.92 MB VRAM
SDXS 256p Diffusion (A35 Mali-G68)   : 9.22s UNet       | 15.72s CLI E2E| 651.92 MB VRAM (1.43x vs CPU)
Whisper STT Base (A35 Mali-G68)      : RTF 0.28 (Vulkan) vs RTF 1.20 (CPU NEON) -> 4.2x 가속
LLaMA-3.2 1B (S25 Adreno 830)        : 26.4 tokens/sec (Vulkan) vs 8.5 tokens/sec (CPU) -> 3.1x 가속
```
