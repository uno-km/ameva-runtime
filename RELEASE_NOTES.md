# AMEVA-Runtime Release Notes

All notable changes and milestones for `ameva-runtime` will be documented in this file.
This project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html) and Apache-2.0 governance.

## [v2.6.0-alpha.1] - 2026-09-10
### Controlled Subprocess Execution Engine & Public Alpha Release

#### Version Architecture SSOT
- **Node.js Package**: `@ameva/runtime v2.6.0-alpha.1` (maintains monotonic SemVer progression from published v2.5.0)
- **Python Distribution**: `2.5.0` (independent distribution baseline)
- **Node-API Bridge ABI**: Version 1
- **Native Vulkan HAL ABI**: Version 1.2.0
- **Telemetry Schema**: Version 1 (FD 3 JSONL)
- **Hardware Certificate Schema**: Version 2

#### Highlights
- **Controlled Subprocess Execution Engine (Phase N2)**:
  - Hardened single-shot execution layer designed for on-device AI inference (`executeSubprocess`, `validateExecutable`, `toSubprocessOptions`).
  - Strict security invariants: `shell: false`, absolute executable paths, argv array invocation, environment allowlisting & injection attack prevention (`BLOCKED_ENV_KEYS`).
  - Independent stream bounds (`maxStdoutBytes`, `maxStderrBytes`, `maxTelemetryBytes`). Exceeding bounds triggers `OUTPUT_LIMIT_EXCEEDED` with multi-stage termination.
  - First-Cause-Wins deterministic termination model across timeouts, abort signals, and limit violations.
  - Two-phase lifecycle termination: `SIGTERM` followed by grace period escalation to `SIGKILL` on POSIX.
  - Dedicated Structured Telemetry channel on FD 3 (`AMEVA_TELEMETRY_FD=3`) with conflict detection locking (`telemetry_conflict`).
- **Packaged Native Addon & Host Runtime Dependency**:
  - Included prebuilt native C ABI addon for `android-arm64` (`prebuilds/android-arm64/ameva_native.node`) verified on Snapdragon 8 Elite / Adreno 830.
  - **Runtime Dependency Notice**: Dynamically links against Termux `$PREFIX/lib/libc++_shared.so` and requires the standard Termux environment layout (not a standalone Bionic-only binary).
  - Strict non-blocking package import: `require('@ameva/runtime')` succeeds on all platforms.
  - Zero-silent-fallback fail-fast policy: on platforms without native addon support (e.g. Windows x64), native GPU calls and `Doctor.runSelfTest()` strictly reject with `PlatformNotSupportedError`, while pure JavaScript APIs and the Subprocess Engine remain fully operational.
  - Hardened native addon diagnostic inspection API: `nativeBridge.getNativeAddonInfo()`. Environment override requires explicit opt-in (`AMEVA_ALLOW_NATIVE_OVERRIDE=1`).
- **Empirical Real-Device Validation (Samsung Galaxy S25 / Snapdragon 8 Elite / Adreno 830)**:
  - Verified inside clean isolated `node_modules` environment (`s25-release-smoke`).
  - 10-Gate Packaged Smoke Verification: 10/10 PASS.
  - Native Doctor hardware probe: Stages V0-V9 PASS (10/12 stages, `computeCertified: true`, `recommendedBackend: "vulkan"`).
  - 13-Gate Subprocess Lifecycle Suite: 62/62 assertions PASS (including SIGTERM -> SIGKILL escalation and 1,000-run cycle completion).
- **Compliance & Artifact Integrity**:
  - Pure standard Apache-2.0 license text in `LICENSE` and separated attribution in `NOTICE`.
  - Added comprehensive `THIRD_PARTY_NOTICES.md` and `source-provenance.json`.
  - Prebuild manifest (`manifest.json`) recording artifact SHA-256, Node-API level 8, and runtime dependencies.
  - Generated external release manifest: `release/SHA256SUMS`.

#### Known Limitations
- Single-shot execution only; persistent resident worker daemon is deferred to Phase N3.
- `modelCertified` status is strictly `false` (V10/V11 model graph execution is deferred to dedicated modality engines).
- Memory trend in uncollected 1,000 rapid invocations is classified as `INCONCLUSIVE` (allocator heap retention vs external buffer delta 0.00MB).

---

## [v2.5.0] - 2026-09-07
### Native BitNet 1.58-bit Vulkan Compute Acceleration, Permanent VRAM Residency & Zero-Copy Pipeline

#### Highlights
- **Native BitNet 1.58-bit Vulkan Compute Pipeline (`src/core/vulkan_bitnet_engine.cpp`)**:
  - Implemented full native Vulkan compute runtime targeting mobile GPUs: ARM Mali (Bifrost/Valhall) and Qualcomm Adreno (6xx/7xx/8xx).
  - Dedicated SPIR-V compute kernels for BitNet 1.58-bit ternary GEMV (`bitnet_gemv_i2_s.comp`), in-place rotary position embeddings (`rope.comp`), decode multi-head attention (`attention_decode.comp`), SwiGLU activation (`swiglu_silu.comp`), RMSNorm (`rmsnorm_norm.comp`), and residual summation (`residual_add.comp`).
- **Llama.cpp-Style Permanent Model VRAM Residency**:
  - Pre-allocates unified GPU storage buffers for all 30 transformer layers (498 MB) and FP16 LM Head (626 MB) at initialization.
  - Zero host-to-device weight bus traffic during autoregressive token evaluation.
- **Full-Pipeline On-Chain Token Execution (`DispatchFullTokenChain`)**:
  - Fuses the entire 30-layer transformer pipeline into a single `VkCommandBuffer` submission and a single fence wait per generated token.
  - Eliminates 99.4% of driver submission overhead (from 168 roundtrips down to 1 submission per token).
- **FP16 LM Head GPU Compute Shader Offload (`bitnet_gemv_f16.comp`)**:
  - Offloads the $128,256 \times 2,560$ (626.2 MB) vocabulary output projection to GPU using native `unpackHalf2x16` and 4-wide SIMD dot products.
  - Resolves the major CPU bottleneck on Exynos 1380 (128.8 ms -> 48.9 ms, saving ~80 ms per token).
  - Verified with Cosine Similarity 1.000000 and 0.00 logits max difference against CPU reference.
- **Empirical Real-Device Benchmarks (Microsoft BitNet-b1.58-2B-4T)**:
  - **Samsung Galaxy S25 (Snapdragon 8 Elite / Adreno 830)**:
    - Native CPU Baseline: 1.396 t/s
    - Vulkan GPU Full Pipeline: **17.558 t/s** (**12.58x total speedup**)
    - Prompt Evaluation Time: **205.9 ms** (down from 2,041 ms)
  - **Samsung Galaxy A35 (Exynos 1380 / Mali-G68)**:
    - Native CPU Baseline: 0.584 t/s
    - Vulkan GPU Full Pipeline: **3.471 t/s** (**5.94x total speedup**)
    - Prompt Evaluation Time: **1,552.8 ms** (down from 8,775 ms)
- **Unified Python BitNet Adapter (`python/ameva_runtime/adapters/bitnet.py`)**:
  - First-class high-level orchestration interface: `from ameva_runtime.adapters.bitnet import BitnetAdapter`.
  - Automatically loads and routes pre-compiled SPIR-V shaders and manages GPU context lifecycle.
- **Official Ecosystem Alignment**:
  - Direct integration and verified support for `termux-bitnet` v1.4.0.

#### Distribution
- PyPI: `pip install ameva-runtime`
- npm: `npm install @ameva/runtime`
- Web Docs: [https://uno-km.vercel.app/lib/vulkan/](https://uno-km.vercel.app/lib/vulkan/)

---

## [v2.4.0] - 2026-09-07
### Qualcomm Snapdragon 8 Elite (Adreno 830) Full-GPU VLM Acceleration & Sibling Alignment

#### Highlights
- **Adreno 830 Full-GPU VLM Offloading**:
  - Validated 25/25 layer full GPU offload for Moondream2 1.8B f16 on Galaxy S25 Adreno 830 (15.00 tok/s generation, 2,706 MiB VRAM, 0.00 MiB CPU Mapped VRAM).
- **Kernel Watchdog & ErrorDeviceLost Defense**:
  - `VisionAdapter.get_execution_environment()` now automatically injects `GGML_VULKAN_SKIP_CHECKS="999999999"` to eliminate mobile GPU host shader check latency.
  - Micro-batch prefill chunking (`-b 64 -ub 64`) preventing Qualcomm KGSL watchdog timeouts during large ViT prefill (729 tokens).
- **Universal Parameter Alignment**:
  - Unlocked dynamic parameter forwarding and removed artificial thread clamping in VLM execution pipeline.
  - Synchronized with `termux-vision` v1.4.0.

#### Distribution
- PyPI: `pip install ameva-runtime`
- npm: `npm install @ameva/runtime`
- Web Docs: [https://uno-km.vercel.app/lib/vulkan/](https://uno-km.vercel.app/lib/vulkan/)

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
