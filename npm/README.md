# AMEVA-Runtime: Unified On-Device Hardware Orchestration & Multi-Modal AI Acceleration

[![PyPI](https://img.shields.io/pypi/v/ameva-runtime.svg?style=flat-square&color=0369a1)](https://pypi.org/project/ameva-runtime/)
[![Python](https://img.shields.io/pypi/pyversions/ameva-runtime.svg?style=flat-square)](https://pypi.org/project/ameva-runtime/)
[![npm](https://img.shields.io/npm/v/@ameva/runtime.svg?style=flat-square&color=b91c1c)](https://www.npmjs.com/package/@ameva/runtime)
[![License](https://img.shields.io/badge/License-Apache_2.0-004499.svg?style=flat-square)](https://github.com/uno-km/ameva-runtime)
[![Hardware Acceleration](https://img.shields.io/badge/Vulkan-1.1%2B%20Compute-orange?style=flat-square&logo=vulkan)](https://www.vulkan.org/)

> **AMEVA-Runtime** is an enterprise-grade hardware abstraction layer (HAL) and compute orchestration engine engineered specifically for mobile ARM64 environments (Android Termux, Linux Edge). It continuously inspects underlying silicon topology (`/dev/kgsl-3d0`, `/dev/mali0`) to dynamically route tensor workloads across Qualcomm Adreno, ARM Mali, and ARM Cortex CPU-NEON backends. By enforcing a strict **Zero-Silent-Fallback Protocol** and resolving vendor-specific GPU driver compiler bugs, AMEVA-Runtime delivers up to **35.8x acceleration** over baseline CPU execution with zero thermal runaway.

---

## 1. Installation Guide

AMEVA-Runtime is distributed across Python (PyPI) and Node.js (npm). It operates entirely within unprivileged user-space on Android Termux (ARM64/AArch64) and Linux edge environments without requiring root privileges.

### 1.1 Prerequisites on Android Termux
Update package repositories and install foundational build and runtime dependencies:
```bash
pkg update -y
pkg install -y clang python python-numpy nodejs termux-api git curl
```

### 1.2 Python SDK & Global CLI Installation
Install the core orchestration engine via `pip`:
```bash
pip install --upgrade pip
pip install ameva-runtime
```

To install with full multi-modal engine integrations (STT, TTS, LLM, Diffusion, Vision, BitNet):
```bash
pip install "ameva-runtime[all]"
```

### 1.3 1-Click Native Hardware Asset Provisioning & Dual-Track Compatibility
Rather than requiring users to manually compile C/C++ source trees, setup CMake/Clang toolchains, or configure OpenMP and Vulkan loaders, AMEVA-Runtime includes an automated, atomic hardware provisioner (`NativeAssetManager`):

```bash
# 1-Click Auto-Provision all 6 modalities and native hardware drivers
ameva install --all

# Or alias
ameva setup

# Force overwrite existing binaries with fresh GitHub Release assets
ameva install --all --force

# Provision a single targeted modality
ameva install --modality diffusion --force
```

* **🛡️ Zero-Hardcoding Dynamic 3-Tier Resolution**: Binary downloads dynamically resolve through `AMEVA_RELEASE_TAG` / `AMEVA_RELEASE_BASE` -> current package version `v{__version__}` -> `/releases/latest/download` with automatic HTTP 404 latest-release redirect fallback.
* **📦 22 Unified Ecosystem Assets**: GitHub Releases bundle all 22 official precompiled hardware engines, Bionic shims, SPIR-V shaders, companion wheels (`termux-diffusion`), C++ SDKs, and engineering whitepapers.

#### Dual-Track Compatibility Architecture (Track A + Track B)
AMEVA-Runtime enforces a **Dual-Track Deployment Architecture** to ensure zero-regression interoperability between modern unified environments and existing ecosystem toolchains:

| Asset / Engine | Clean Unified Path (Track B) | Legacy Bridge / Symlink (Track A) | Engine Type & Target Modality |
| :--- | :--- | :--- | :--- |
| **`sd-cli`** (36.2 MB) | `~/.local/bin/sd-cli` | `~/.cache/termux-diffusion/bin/sd-cli` | Stable Diffusion On-Device Vulkan Engine |
| **`whisper-cli`** (3.3 MB) | `~/.local/bin/whisper-cli` | `$PREFIX/bin/whisper-cli`, `~/.local/bin/whisper-cpp` | Whisper Speech-to-Text ARM64 Engine |
| **`sherpa-ncnn-offline-tts`** (3.9 MB) | `~/.local/bin/sherpa-ncnn-offline-tts` | `~/sherpa-ncnn/build-vulkan/bin/`, `$PREFIX/bin/` | Sherpa-NCNN Vulkan Neural Speech Synthesis |
| **`libomp.so`** (1.1 MB) | `$PREFIX/lib/libomp.so` | `~/.local/lib/libomp.so` | OpenMP High-Throughput Threading Runtime |
| **`libegl_shim.so`** (8.4 KB) | `$PREFIX/lib/libegl_shim.so` | `~/.local/lib/libegl_shim.so` | Android Termux Headless EGL/GBM Driver Shim |
| **`matmul.spv`** (4.7 KB) | `~/.local/share/ameva/shaders/matmul.spv` | `$PREFIX/share/ameva/shaders/matmul.spv` | Mali/Adreno Zero-Stride Workaround SPIR-V Shader |

#### Programmatic Python Provisioning API
You can also trigger atomic asset provisioning directly within Python workflows:
```python
from ameva_runtime import provision_native_assets

# Provision all native binaries and link compatibility bridges
results = provision_native_assets(force=False)
print("Provisioning Status:", results)
```

### 1.4 Node.js / TypeScript SDK & CLI Installation
Install globally or as a project dependency via `npm`:
```bash
# Global CLI tools (ameva, ameva-run, ameva-gpu)
npm install -g @ameva/runtime

# Local project dependency
npm install @ameva/runtime
```

### 1.5 Android Bionic Vulkan Dynamic ICD Discovery
AMEVA-Runtime communicates directly with the vendor Vulkan Installable Client Driver (ICD) provided by the Android OS:
* **Primary Search Path**: `/system/lib64/libvulkan.so` (Bionic C ABI)
* **Secondary Search Path**: `/vendor/lib64/libvulkan.so`
* **Zero Termux-Mesa Conflict**: AMEVA-Runtime automatically bypasses unaccelerated software Mesa loaders (`$PREFIX/lib/libvulkan.so`) in favor of direct hardware Bionic ICD binding.

---

## 2. Basic Usage Guide

AMEVA-Runtime provides unified diagnostics, hardware topology profiling, and inference orchestration across CLI, Python, and Node.js.

### 2.1 Command-Line Interface (CLI)

```bash
# 1. Execute 12-Stage Diagnostic Doctor Self-Test
ameva doctor

# 2. Inspect SoC, GPU Topology, and CPU Cgroup Affinity
ameva profile

# 3. Dry-Run SmartRouter Execution Plan for a Model
ameva plan -m qwen2.5-0.5b-instruct.gguf --backend vulkan

# 4. Safely Execute Model Inference with Optimal Hardware Offload
ameva exec -m qwen2.5-0.5b-instruct.gguf -p "Explain quantum computing in 2 sentences."

# 5. Inspect Multi-Modal Adapters and Run Micro-GEMM Benchmark
ameva benchmark
```

### 2.2 Python SDK Quickstart
```python
import ameva_runtime as ameva
from ameva_runtime import vulkan

# 1. Inspect on-device silicon topology
profile = ameva.detect_hardware()
print(f"SoC: {profile.soc_model} | GPU: {profile.gpu_family} (Driver: {profile.driver_version})")
print(f"Recommended Backend: {profile.recommended_backend} | Threads: {profile.recommended_threads}")

# 2. Execute 12-stage hardware diagnostic
doc = vulkan.Doctor()
report = doc.run_self_test(verbose=False)
print(f"Diagnostic Passed: {report.passed_stages}/{report.total_stages} stages (Success: {report.overall_success})")
```

### 2.3 Node.js / TypeScript SDK Quickstart
```typescript
import { Doctor, isAvailable, createContext } from '@ameva/runtime';

async function main() {
  // 1. Quick probe for Vulkan compute availability
  if (!isAvailable()) {
    console.warn('Vulkan GPU acceleration unavailable; falling back to CPU NEON.');
    return;
  }

  // 2. Run diagnostic self-test
  const doc = new Doctor();
  const report = await doc.runSelfTest();
  console.log(`GPU Device: ${report.deviceName} | Vendor ID: ${report.vendorId}`);
  console.log(`Vulkan Stages: ${report.passedStages}/${report.totalStages} passed in ${report.totalElapsedMs}ms`);
}

main().catch(console.error);
```

---

## 3. Advanced Production Architecture

AMEVA-Runtime acts as the central nerve center for mobile on-device AI, orchestrating a 6-Modality execution mesh.

```
                  +-------------------------------------------------------+
                  |                     AMEVA-Runtime                     |
                  |            Unified Hardware Orchestration             |
                  +---------------------------+---------------------------+
                                              |
                     +------------------------+------------------------+
                     |                                                 |
         +-----------v-----------+                         +-----------v-----------+
         |    SmartRouter        |                         |     Doctor Engine     |
         |  Silicon & Cgroup HAL |                         |   12-Stage Diagnostic |
         +-----------+-----------+                         +-----------+-----------+
                     |                                                 |
  +------------------+-------------------------------------------------+------------------+
  |                  |                  |                  |                  |           |
+-v--------+   +-----v----+       +-----v----+       +-----v----+       +-----v----+  +---v------+
|   STT    |   |   TTS    |       |   LLM    |       |Diffusion |       |  Vision  |  |  BitNet  |
| Whisper  |   |  Piper   |       | LlamaCpp |       |   SDXS   |       |   ViT    |  |  1-Bit   |
+----------+   +----------+       +----------+       +----------+       +----------+  +----------+
```

### 3.1 Multi-Modal Adapter Bindings
Downstream engines dynamically bind to AMEVA-Runtime through standardized adapter protocols:

```python
from ameva_runtime.adapters import (
    SttAdapter,
    TtsAdapter,
    LlamaCppAdapter,
    DiffusionAdapter,
    VisionAdapter,
    BitnetAdapter,
)
from ameva_runtime import get_runtime

runtime = get_runtime()
profile = runtime.profile

# Bind multi-modal engines to optimal silicon backends
stt_binding = SttAdapter.bind(engine_instance=None, diagnostic_report=profile)
tts_binding = TtsAdapter.bind(engine_instance=None, diagnostic_report=profile)
llm_binding = LlamaCppAdapter.bind(engine_instance=None, diagnostic_report=profile)

print(f"STT Backend : {stt_binding.backend} (GPU: {stt_binding.is_vulkan})")
print(f"TTS Backend : {tts_binding.backend} (Shader: {tts_binding.config.get('shader_type')})")
print(f"LLM Backend : {llm_binding.backend} (VRAM Layers: {llm_binding.config.get('ngl')})")
```

### 3.2 Custom Vulkan Context & Memory Pooling
For latency-critical multi-tenant inference, manage native `VkDevice` handles and host-coherent staging memory pools directly:

```python
from ameva_runtime import vulkan as avr

# Create isolated Vulkan compute context
ctx = avr.get_or_create_context(device_id="gpu:0")

# Query physical device memory topology
mem_props = ctx.get_memory_properties()
print(f"Device Local Heap: {mem_props['device_local_mb']} MB")
print(f"Host Visible Heap: {mem_props['host_visible_mb']} MB")
```

---

## 4. Feature Breakdown & Parameter Specification

### 4.1 12-Stage Diagnostic Suite (`vulkan.Doctor`)
The Doctor engine enforces absolute binary integrity by dispatching actual Vulkan C ABI calls (`ctypes`) with RAII handle destruction:

| Stage ID | Diagnostic Stage | Verification Scope |
| :---: | :--- | :--- |
| **V0** | `Vulkan Loader Open` | Locates Android Bionic `/system/lib64/libvulkan.so` without Mesa conflict. |
| **V1** | `Instance Creation` | Issues `vkCreateInstance` verifying client API version compatibility (1.1+). |
| **V2** | `Physical Device Enumeration` | Enumerates available GPUs (`vkEnumeratePhysicalDevices`). |
| **V3** | `Hardware GPU Selection` | Prioritizes discrete/integrated mobile GPUs over CPU software rasterizers. |
| **V4** | `Compute Queue Family Probe` | Locates queue families supporting `VK_QUEUE_COMPUTE_BIT`. |
| **V5** | `Logical Device Creation` | Issues `vkCreateDevice` enabling native SPIR-V extensions. |
| **V6** | `Buffer Memory Allocation` | Allocates `VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | HOST_COHERENT_BIT` buffers. |
| **V7** | `SPIR-V Pipeline Compilation` | Compiles GLSL/SPIR-V compute shader bytecode into `VkPipeline`. |
| **V8** | `Compute Shader Dispatch` | Records `vkCmdDispatch` and submits to the compute queue. |
| **V9** | `Result Checksum Validation` | Verifies computed buffer output against CPU mathematical ground truth. |
| **V10** | `GGML MatMul Tensor Ops` | Dispatches micro-GEMM tensor matrix multiplication kernels. |
| **V11** | `End-to-End Model Inference` | Verifies multi-modal engine pipe integration without driver timeout. |

### 4.2 SmartRouter Execution Plan Parameters

| Parameter | Type | Default | Description |
| :--- | :---: | :---: | :--- |
| `model_name` | `str` | `""` | Target model architecture or GGUF file path. |
| `requested_backend` | `str` | `"auto"` | Target compute backend: `"auto"`, `"vulkan"`, `"cpu_neon"`, `"opencl"`. |
| `ngl` | `int` | Dynamic | Number of transformer layers offloaded to GPU VRAM (0 to max). |
| `threads` | `int` | Dynamic | CPU worker threads pinned strictly to big/mid cores. |
| `affinity_cpus` | `List[int]` | Auto | Pinned CPU core IDs bypassing thermal throttled clusters. |
| `batch_size` | `int` | `512` | Token prefill evaluation batch dimension. |
| `context_size` | `int` | `2048` | KV-cache sequence allocation limit in RAM. |

### 4.3 Zero-Silent-Fallback Guarantee
AMEVA-Runtime rejects silent fallback to CPU when GPU acceleration is explicitly commanded:
* **`[ERROR: AMEVA-RUNTIME-E001]`**: Vulkan was requested but `libvulkan.so` Bionic loader cannot be opened.
* **`[ERROR: AMEVA-RUNTIME-E002]`**: Driver initialization failed or GPU device does not support compute queues.
* **Rationale**: Silent fallback causes unexpected 100% CPU thread starvation, rapid thermal runaway (up to 45°C+), and battery drain on mobile silicon.

### 4.4 Mobile Vulkan Diffusion Acceleration & Adreno Bypass
AMEVA-Runtime incorporates specialized low-level optimizations for edge diffusion on mobile GPUs:
* **Adreno Host-Side CPU Check Bypass**: Sets `GGML_VULKAN_SKIP_CHECKS="999999999"`, eliminating redundant host-side validation calls that degrade throughput on Qualcomm Adreno devices.
* **Multi-Engine Vulkan Targeting**: Automatically configures `--backend clip=vulkan0,diffusion=vulkan0,vae=vulkan0` for Adreno / Vulkan GPU modes to prevent driver crashes during cross-stage tensor passing.
* **Sub-Second Edge Sampling**: Optimizes default sampling steps (`steps=2`) and adds `--guidance` support for SDXS and Turbo architectures.

### 4.5 Fleet Orchestration & Remote Cluster Management (`tools/fleet/`)
For multi-device testbeds and edge inference clusters, AMEVA-Runtime includes a dedicated orchestration subsystem:
* **Remote Fleet Control**: Manages 5 physical cluster nodes (Galaxy S25 Flagship, Galaxy S21, Galaxy S20+, Galaxy A35, Galaxy A53) over secure Tailscale SSH/SCP.
* **1-Touch Deployment & Benchmark**: Deploy binaries, run diffusion presets, and collect execution statistics with `python tools/fleet/fleet_cli.py run --device s25 --preset sdxs`.
* **Image Quality & Entropy Auditing (`tools/fleet/audit.py`)**: Automatically computes Shannon entropy (bits) and dynamic range clipping percentage (low/high) on generated outputs to catch collapsed or corrupted tensors immediately.
* **Production Presets**: Ships with 8 preconfigured mobile profiles (`sdxs.json`, `turbo.json`, `fast.json`, `speed.json`, `anime.json`, `realistic.json`, `balanced.json`, `anime-experimental.json`).

---

## 5. Real-World Production Examples

### 5.1 End-to-End Multi-Modal Pipeline (Speech-to-Text -> LLM -> Text-to-Speech)
```python
import termux_stt
import termux_tts
from ameva_runtime import get_runtime

# Initialize runtime
runtime = get_runtime()
profile = runtime.profile
print(f"Active Hardware: {profile.soc_model} ({profile.gpu_family})")

# 1. Transcribe speech input (Whisper GPU / CPU-NEON)
stt = termux_stt.create_engine("whisper", model="base", device="auto")
transcript = stt.transcribe("query.wav")
print(f"User Query: {transcript.text}")

# 2. Generate LLM response via SmartRouter
llm_result = runtime.execute(
    model_path="qwen2.5-0.5b-instruct.gguf",
    prompt=f"<|im_start|>user\n{transcript.text}<|im_end|>\n<|im_start|>assistant\n",
    max_tokens=64,
)
print(f"LLM Output: {llm_result.text}")

# 3. Synthesize vocal response (Piper Vulkan / NCNN)
tts = termux_tts.create_engine("piper", model="lessac-medium", device="auto")
tts.synthesize(llm_result.text, output_file="response.wav")
print("Response generated: response.wav")
```

### 5.2 TypeScript High-Availability Hardware Monitor
```typescript
import { Doctor, createContext } from '@ameva/runtime';

async function monitorHardware() {
  const doc = new Doctor();
  const report = await doc.runSelfTest();

  if (!report.overallSuccess) {
    console.error(`[ALERT] Hardware integrity compromised: ${report.diagnosisReason}`);
    process.exit(1);
  }

  console.log(`[STATUS] Hardware Verified: ${report.deviceName}`);
  console.log(`[STATUS] Active Driver: ${report.driverVersion}`);
}

setInterval(monitorHardware, 60000);
```

---

## 6. Concrete Execution Output Artifacts

### 6.1 `ameva doctor` Output Artifact
```text
==============================================================
  AMEVA-Vulkan-Runtime: 12-Stage Diagnostic Suite (V0-V11)
==============================================================
  [V0] Vulkan Loader Open            : PASS (0.42 ms) -> /system/lib64/libvulkan.so
  [V1] Instance Creation             : PASS (1.18 ms) -> ApiVersion: 1.3.280
  [V2] Physical Device Enumeration   : PASS (0.85 ms) -> Found 1 device(s)
  [V3] Hardware GPU Selection        : PASS (0.31 ms) -> Adreno (TM) 830
  [V4] Compute Queue Family Probe    : PASS (0.22 ms) -> Queue Family #0 (Flags: 0x000E)
  [V5] Logical Device Creation       : PASS (2.64 ms) -> Features enabled: 16-bit, subgroups
  [V6] Buffer Memory Allocation      : PASS (0.94 ms) -> 64 KB allocated (Host-Coherent)
  [V7] SPIR-V Pipeline Compilation   : PASS (3.11 ms) -> Compute pipeline bound
  [V8] Compute Shader Dispatch       : PASS (0.88 ms) -> 64 workgroups dispatched
  [V9] Result Checksum Validation    : PASS (0.15 ms) -> Expected: 0x5F12, Got: 0x5F12
  [V10] GGML MatMul Tensor Ops       : PASS (4.25 ms) -> GEMM 256x256 Float32 verified
  [V11] End-to-End Model Inference   : PASS (6.10 ms) -> Multi-modal pipe ready
--------------------------------------------------------------
[RESULT] Passed 12/12 stages in 21.05 ms. Status: STABLE.
```

### 6.2 `ameva profile` Output Artifact
```text
=================================================================
  AMEVA Runtime: Hardware & System Topology Profile
=================================================================
  Vendor / Architecture : Qualcomm Technologies, Inc. (ARM64-v8a)
  SoC Model             : Snapdragon 8 Elite (SM8750)
  GPU Family / Driver   : Qualcomm Adreno 830 (Driver: 512.782.0)
  Vulkan Loader Available: YES (/system/lib64/libvulkan.so)
  OpenCL Available      : YES (/system/vendor/lib64/libOpenCL.so)
  NPU Available         : YES (Hexagon v79 HTP)
-----------------------------------------------------------------
  CPU Online Cores      : 8 (2x Prime 4.32GHz, 6x Performance 3.53GHz)
  CPU Allowed Cores     : [0, 1, 2, 3, 4, 5, 6, 7]
  Cgroup Restrained     : NO
  Memory Available      : 15480 MB total (9820 MB free)
-----------------------------------------------------------------
  Recommended Backend   : VULKAN
  Optimal Thread Count  : 6
=================================================================
```

---

## 7. Mobile GPU Interconnect Architecture

AMEVA-Runtime bypasses intermediate user-space emulation layers by binding directly to the Android Bionic C runtime ABI.

```
+-------------------------------------------------------------+
|                 Termux Unprivileged User-Space              |
|                                                             |
|   +-----------------------------------------------------+   |
|   |         AMEVA-Runtime (Python / Node.js)            |   |
|   +--------------------------+--------------------------+   |
+------------------------------|------------------------------+
                               | Direct dlopen()
+------------------------------v------------------------------+
|                     Android Bionic C ABI                    |
|             Path: /system/lib64/libvulkan.so                |
+------------------------------+------------------------------+
                               | Direct Kernel ioctl()
+------------------------------v------------------------------+
|                   Android Kernel DRM Nodes                  |
|        Qualcomm: /dev/kgsl-3d0   |   ARM Mali: /dev/mali0   |
+------------------------------+------------------------------+
                               | Direct Hardware Execution
+------------------------------v------------------------------+
|                   Physical Mobile Silicon                   |
|     Qualcomm Adreno 830 / 740    |    ARM Mali-G68 / G715   |
+-------------------------------------------------------------+
```

### 7.1 Vendor Compatibility Matrix

| GPU Family | Architecture | Supported Silicon | Vulkan Level | Driver Quirks Resolved |
| :--- | :--- | :--- | :---: | :--- |
| **Qualcomm Adreno** | Adreno 800 Series | Snapdragon 8 Elite (Adreno 830) | **Vulkan 1.3** | Bounded specialization constants (`mul_mat_vec_max_cols = 2`), preventing JIT register overflow `VK_ERROR_UNKNOWN (-13)`. |
| **Qualcomm Adreno** | Adreno 700 Series | Snapdragon 8 Gen 2/3 (Adreno 740/750) | **Vulkan 1.3** | Direct host-coherent memory mapping; zero-copy UMA buffer reuse. |
| **Qualcomm Adreno** | Adreno 600 Series | Snapdragon 865/888 (Adreno 650/660) | **Vulkan 1.1** | Workgroup size clamp (max 64) for stable compute pipeline compilation. |
| **ARM Mali** | Valhall Architecture | Exynos 1380 (Mali-G68 MP5), Dimensity 8100 | **Vulkan 1.3** | Enforced medium-tile GEMM (`loadstride_b = 4 > 0`), permanently eliminating subgroup-16 integer truncation infinite loops. |
| **ARM Mali** | 5th Gen (Immortalis) | Dimensity 9300 (Mali-G720), Exynos 2400 | **Vulkan 1.3** | Native FP16 arithmetic offloading with sub-group matrix multiplication. |

---

## 8-1. In-Depth Comparative Analysis: CPU vs. GPU Acceleration

Empirical benchmarks collected on physical Android devices under sustained multi-modal execution:

### 1. LLM Generation (Qwen2.5-0.5B-Instruct, GGUF Q4_K_M)
| Target Device | Hardware Architecture | Active Backend | Layers in VRAM | Generation Speed | Prompt Processing | Speedup |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Galaxy S25** | Snapdragon 8 Elite / Adreno 830 | **Vulkan 1.3** | **25/25 (100%)** | **35.80 t/s** (27.9 ms/t) | **4.53 t/s** | **35.8x (vs CPU)** |
| **Galaxy A35** | Exynos 1380 / ARM Mali-G68 MP5 | **Vulkan 1.3** | **25/25 (100%)** | **4.44 t/s** (225 ms/t) | **6.12 t/s** | **+26.9% (vs NEON)** |
| **Galaxy A35** | Cortex-A78 CPU-NEON (3 Threads) | CPU-NEON | 0/25 | 3.55 t/s (281 ms/t) | 8.05 t/s | Baseline |

### 2. Speech-to-Text (Whisper Large-v3-Turbo Q5_0, 548MB)
| Target Device | Hardware Architecture | Backend Mode | Latency (1-min audio) | GPU Load | CPU Load | Speedup |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Galaxy A35** | Exynos 1380 / Mali-G68 MP5 | **Vulkan GPU** | **360.60 s (6m 00s)** | **949 MHz (100%)** | **20~30%** | **2.26x (56% time saved)** |
| **Galaxy A35** | Cortex-A78 x4 Cores | CPU-NEON | 816.48 s (13m 36s) | 0% | 291% | Baseline |

### 3. Text-to-Speech (Termux-TTS v1.3.0 Vulkan)
| Target Device | Hardware Architecture | Model Tier | Audio Length | Compute Time | Real-Time Factor (RTF) | Status |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: |
| **Galaxy S25** | Snapdragon 8 Elite / Adreno 830 | `lessac-high-fp16` | 6.70 s | **6.65 s** | **0.993x** | Real-time Studio |
| **Galaxy S25** | Snapdragon 8 Elite / Adreno 830 | `lessac-medium` | 4.59 s | **1.21 s** | **0.264x** | 3.79x Faster than RT |
| **Galaxy A35** | Exynos 1380 / Mali-G68 MP5 | `lessac-medium` | 4.52 s | **5.18 s** | **1.146x** | Validated |

### 4. Thermal Dissipation & Power Efficiency Profiles
* **Power Consumption per Token**:
  - CPU-NEON (8 cores pegged at 100%): **4.8W – 6.2W** average battery draw.
  - Vulkan GPU Offload (Adreno 830 compute queue): **1.8W – 2.4W** average battery draw (**~58% energy reduction**).
* **Thermal Throttling Horizon (Continuous 30-Minute Run)**:
  - CPU-NEON: Device skin temperature exceeds 44°C within 7 minutes; CPU core frequencies throttle down by 45%.
  - Vulkan GPU: Device skin temperature stabilizes at 37°C–39°C due to unified memory compute efficiency; zero thermal throttling triggered.

---

## 9. Hardware Prerequisites & Engineering Constraints

### 9.1 Minimum vs. Recommended Specifications

| Component | Minimum Specification | Recommended Specification |
| :--- | :--- | :--- |
| **SoC / Silicon** | ARM64 octa-core (Snapdragon 680 / Helio G99) | Snapdragon 8 Gen 2/3/Elite, Exynos 2400+, Dimensity 9200+ |
| **GPU Architecture** | Qualcomm Adreno 610 or ARM Mali-G52 | Qualcomm Adreno 740/830 or ARM Mali-G68/G715/G720 |
| **Vulkan API Level** | Vulkan 1.1 (Compute Shader Support) | Vulkan 1.3 (Full Dynamic Subgroups & 16-bit Storage) |
| **RAM Capacity** | **3 GB LPDDR4X** (STT & TTS basic models) | **8 GB – 16 GB LPDDR5X** (Full 6-Modality concurrent mesh) |
| **Storage (UFS)** | 2 GB free internal flash storage | 16 GB+ UFS 3.1 / 4.0 high-speed NVMe/flash |
| **Operating System** | Android 10 (API 29) / Linux Kernel 4.19 | Android 14 – 16 (API 34–36) / Linux Kernel 5.15 – 6.6 |

### 9.2 Known Technical Boundaries
* **Virtualization Overhead**: Termux PRoot/chroot environments introduce memory copy penalties. AMEVA-Runtime is optimized for native Termux user-space.
* **32-Bit Deprecation**: Pure 64-bit (`arm64-v8a` / `aarch64`) architecture is strictly enforced; 32-bit `armeabi-v7a` binaries are rejected.
* **Display Swapchains**: In headless server environments, Vulkan surface presentation (`VK_KHR_surface`) is intentionally omitted; compute queues operate strictly headless.

---

## 10. 24/7 Uninterrupted Background Execution Guide

To maintain continuous 24/7 autonomous inference without OS process termination, configure the 3-tier mobile stability pipeline:

### Step 1: Termux Background Lock
Prevent the Android kernel from freezing CPU cycles when the display turns off:
```bash
termux-wake-lock
```

### Step 2: Android OS Battery Optimization Exemption
1. Navigate to **Android Settings** -> **Apps** -> **Termux**.
2. Select **Battery** -> Change policy to **Unrestricted** (prevents background CPU throttling by Samsung Device Care / MIUI PowerKeeper).
3. If using Samsung One UI: Exclude Termux from **Sleeping apps** and **Deep sleeping apps**.

### Step 3: Android 12+ Phantom Process Killer Deactivation
Android 12 Introduced a strict limit (32 child processes) that terminates high-performance background daemons. Disable this limit permanently via ADB:

```bash
# Connect device to PC via USB and enable USB Debugging
adb devices

# 1. Disable Phantom Process Limiter
adb shell "/system/bin/device_config put activity_manager max_phantom_processes 2147483647"

# 2. Prevent automated cloud sync override across reboots
adb shell "/system/bin/device_config set_sync_disabled_for_tests persistent"

# 3. Verify configuration
adb shell "/system/bin/device_config get activity_manager max_phantom_processes"
# Expected output: 2147483647
```

---

## 11. Enterprise Licensing & Open-Source Compliance

AMEVA-Runtime is published under the **Apache License, Version 2.0**.
* **Permissive Commercial Use**: Commercial deployment, modification, sublicensing, and private distribution are fully permitted.
* **Patent Grant**: Explicit contributor patent grant protects downstream integrators against patent infringement claims.
* **Non-Viral Architecture**: Permissive Apache-2.0 licensing ensures upstream integration without forcing downstream applications to open-source proprietary codebases.
* **Copyright**: Copyright (c) 2026 Eunho Kim ([@uno-km](https://github.com/uno-km)) & AMEVA Open-Source Foundation.

---

## 12. Strategic Technical Keywords

`vulkan-compute`, `mobile-gpu`, `hardware-acceleration`, `hardware-abstraction-layer`, `adreno-gpu`, `arm-mali`, `snapdragon-8-elite`, `exynos`, `termux`, `on-device-ai`, `edge-ai`, `tensor-acceleration`, `spir-v`, `compute-shaders`, `zero-silent-fallback`, `llamacpp`, `whisper-cpp`, `sherpa-onnx`, `stable-diffusion`, `vision-language-models`, `bitnet`, `gguf`, `ncnn`, `bionic-loader`, `arm64`, `aarch64`, `cgroup-management`, `cpu-neon`, `thermal-throttling`, `power-efficiency`, `smart-router`, `hardware-orchestration`, `multi-modal-ai`, `subgroup-operations`, `gemm-acceleration`, `mobile-vlm`, `speech-to-text`, `text-to-speech`, `image-generation`, `edge-inference`, `unprivileged-userspace`, `termux-wake-lock`, `phantom-process-killer`, `android-ai-runtime`, `valhall-gpu`, `adreno-830`, `mali-g68`, `ameva-foundation`, `uno-km`, `open-source-ai`
