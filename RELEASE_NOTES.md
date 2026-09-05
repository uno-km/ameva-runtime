# AMEVA-Runtime Release Notes

All notable changes and milestones for `ameva-runtime` will be documented in this file.
This project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html) and Apache-2.0 governance.

---

## [v2.0.1] - 2026-09-05
### Zero-Silent-Fallback Hardening & Ecosystem Alignment

#### Highlights
- **Strict Fail-Fast Enforcement**: Eradicated all silent fallback return paths and unhandled exception swallowing across sibling modality adapters.
- **Unified Distribution**: Full synchronization across PyPI (`pip install ameva-runtime`) and npm (`npm install @ameva/runtime`).
- **Sibling Ecosystem Synchronized**: `termux-bitnet` (1.1.4), `termux-diffusion` (1.4.4), `termux-llamacpp` (1.2.3), `termux-stt` (1.1.6), `termux-train` (1.1.4), `termux-tts` (1.1.4), `termux-vision` (1.1.3), `termux-aichain` (1.1.3).

#### Distribution
- PyPI: `pip install ameva-runtime`
- npm: `npm install @ameva/runtime`
- Web Docs: [https://uno-km.vercel.app/lib/vulkan/](https://uno-km.vercel.app/lib/vulkan/)

---

## [v2.0.0] - 2026-09-05
### Major Architecture Milestone — Single Package Unification, Mali Valhall MatMul Loop Elimination & STT 2.26x Acceleration

#### Highlights
- **ARM Mali Valhall MatMul Zero-Stride Infinite Loop Elimination**:
  - Root Cause: In `mul_mm.comp`, devices with subgroup size 16 (Valhall architecture) computed `loadstride_b = gl_WorkGroupSize.x * LOAD_VEC_B / BK = 16 * 1 / 32 = 0`, producing an infinite loop `for (uint l = 0; l < BN; l += 0)` and hardware watchdog TDR aborts (`VK_ERROR_DEVICE_LOST`).
  - Resolution: Enforced Medium kernels (`_m`, workgroup size 128, `loadstride_b = 4 > 0`) via `enforce_medium_matmul: true`. Galaxy A35 achieved **4.44 tokens/sec** with 25/25 layers (100%) GPU offload (+26.9% faster than CPU) with 0 freezes.
- **Whisper STT 2.26x Real-Device GPU Acceleration**:
  - Validated Whisper Large-v3-Turbo on Samsung Galaxy A35 completing in **360.60s (6m 00s)** vs CPU NEON **816.48s (13m 36s)** — a **2.26x acceleration (56% time reduction)** with 0 fallbacks, while reducing CPU load from 291% to 20~30%.
- **Qualcomm Adreno 830 JIT Bug Isolation**:
  - Handled Qualcomm Adreno JIT compiler crash (`VK_ERROR_UNKNOWN -13`) when Specialization Constant `NUM_COLS >= 3` by bounding `mul_mat_vec_max_cols = 2`, enabling stable GPU inference on Galaxy S25 in 4,401 ms.
- **PyTorch-Style Single Package Architecture**:
  - Unified repository and distribution under `ameva-runtime` (v2.0.0), housing specialized Vulkan acceleration in `from ameva_runtime import vulkan` with dynamic single-source-of-truth versioning.
- **Complete Sibling Ecosystem Migration**:
  - Migrated `termux-stt`, `termux-vision`, `termux-llamacpp`, `termux-diffusion`, `termux-bitnet`, `termux-tts`, and `termux-train` to directly import `from ameva_runtime import vulkan`.
- **6-Modality Vulkan Acceleration Roadmap**:
  - `LLM`: [v2.0.0 Completed] Llama.cpp Q4_K_M 25/25 layer full VRAM offload (Adreno 830: 35.80 t/s, Mali-G68: 4.44 t/s).
  - `STT`: [v2.0.0 Completed] Whisper.cpp on-device Vulkan acceleration (Adreno 830: 4,401 ms, Mali-G68: 360.60s / 2.26x speedup).
  - `Vision`: [v2.1.0 In Progress] CLIP, MobileVLM, LLaVA Vulkan GGML tensor binding.
  - `Diffusion`: [v2.2.0 In Progress] Stable Diffusion v1.5 / Turbo & FLUX.1 on-device UNet/DiT tensor offload.
  - `TTS`: [v2.3.0 In Progress] Piper, Sherpa-ONNX, Kokoro low-latency neural TTS streaming.
  - `Train`: [v2.4.0 In Progress] On-device LoRA / QLoRA Vulkan gradient descent backpropagation.

#### Distribution
- PyPI: `pip install ameva-runtime`
- npm: `npm install @ameva/runtime`
- Web Docs: [https://uno-km.vercel.app/lib/vulkan/](https://uno-km.vercel.app/lib/vulkan/)

---

## [v1.0.0] - 2026-09-01
### Production Initial Release — Unified Multi-Modal Acceleration

#### 🚀 Highlights
- **Universal Multi-Modal Acceleration Core**: Unified C++20 Vulkan Hardware Abstraction Layer (HAL) accelerating STT, Vision, LLM, Diffusion, and Autograd Training.
- **Single Loader Chain Pinning**: Strict Bionic ICD `/system/lib64/libvulkan.so` dispatching, completely preventing Termux Mesa symbol collisions and SIGABRT.
- **12-Stage Probing & Auto-Recovery (V0~V11)**: Granular diagnostic hierarchy validating GPU capability from `dlopen` to SGEMM precision and E2E model pipelines with transparent CPU NEON auto-recovery.
- **Zero-Drift Hardware Quirks**: Out-of-the-box hardware bug mitigation for Qualcomm Adreno 830/750/740, ARM Mali-G78/G68/G77 (128-byte alignment), and Samsung Xclipse.
- **Cross-Platform Dual Bindings**: Full Python CFFI/ctypes and Node.js/TypeScript N-API SDK.

#### 📦 Distribution
- PyPI: `pip install ameva-runtime`
- npm: `npm install @ameva/runtime`
- Web Docs: [https://uno-km.vercel.app/lib/vulkan/](https://uno-km.vercel.app/lib/vulkan/)
