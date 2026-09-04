# AMEVA-Runtime

[![PyPI](https://img.shields.io/pypi/v/ameva-runtime.svg?style=flat-square&color=0369a1)](https://pypi.org/project/ameva-runtime/)
[![Python](https://img.shields.io/pypi/pyversions/ameva-runtime.svg?style=flat-square)](https://pypi.org/project/ameva-runtime/)
[![License](https://img.shields.io/badge/License-Apache_2.0-004499.svg?style=flat-square)](https://github.com/uno-km/ameva-runtime)

> **Next-Generation Unified On-Device Hardware Orchestration & Multi-Modal Acceleration Runtime for Mobile & Edge Devices**

---

## Overview

**AMEVA Runtime** is an enterprise-grade, zero-silent-fallback hardware orchestration engine engineered specifically for on-device generative AI on mobile and edge systems (Android Termux, Linux, and embedded environments). 

It dynamically evaluates silicon topology, GPU architecture, driver capabilities, and kernel cgroup constraints, routing inference workloads with zero guesswork:
- **Qualcomm Snapdragon (Adreno GPUs)**: Dispatches full compute graphs to native Vulkan hardware pipelines with complete VRAM layer offloading.
- **ARM Mali GPUs**: Automatically detects driver fence synchronization deadlocks and routes inference to the optimized ARM Cortex CPU-NEON compute cluster, preventing host UI freezing.
- **Unified 1-Liner Python API**: High-level `run()` and `plan()` interface delivering structured telemetry (tokens/sec, latency, memory footprint).
- **100% Backward Compatible**: Drops into existing `termux-*` ecosystems (`termux-llamacpp`, `termux-vision`, `termux-diffusion`, `termux-stt`, `termux-tts`, `termux-bitnet`) without changing legacy imports (`import ameva_vulkan_runtime as avr`).

---

## Installation

```bash
pip install --upgrade ameva-runtime
```

---

## Quickstart

### 1-Liner Inference Execution

```python
import ameva_runtime as ameva

# Direct inference with automatic silicon topology detection and GPU offloading
result = ameva.run(
    model="qwen2.5-0.5b",
    prompt="Space in Korean is:",
    max_tokens=32
)

print(f"Generated text: {result.text}")
print(f"Hardware backend: {result.backend_used}")
print(f"Token generation speed: {result.tokens_per_second:.2f} tokens/sec")
print(f"Prompt evaluation speed: {result.prompt_tokens_per_second:.2f} tokens/sec")
print(f"Total latency: {result.total_time_ms:.1f} ms")
print(f"Safety rationale: {result.rationale}")
```

### Dry-Run Execution Plan

```python
import ameva_runtime as ameva

plan = ameva.plan("qwen2.5-0.5b")
print(f"Selected Backend: {plan.backend}")
print(f"VRAM Layers (NGL): {plan.ngl}")
print(f"Worker Threads: {plan.threads}")
print(f"Pinned CPU Affinity: {plan.affinity_cpus}")
```

---

## Empirical Real-Device Benchmarks (Ground Truth)

All metrics were captured directly on live physical consumer hardware running Android Termux with the official `qwen2.5-0.5b-instruct-q4_k_m.gguf` model.

| Device | Processor / SoC | GPU Architecture | Active Backend | Layers in VRAM | Eval Speed (Tokens/s) | Prompt Eval (Tokens/s) | System UI Stability |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Samsung Galaxy S25** | Qualcomm Snapdragon 8 Elite | Qualcomm Adreno 830 | **VULKAN** | **25 / 25 (100%)** | **34.08 t/s** | **4.59 t/s** | **100% Stable (0% Freeze)** |
| **Samsung Galaxy A35** | Samsung Exynos 1380 | ARM Mali-G68 MP5 | **CPU-NEON (Adaptive)** | **0 / 25 (Shield)** | **4.27 t/s** | **6.44 t/s** | **100% Stable (0% Freeze)** |
| **Samsung Galaxy A35** (Forced Vulkan) | Samsung Exynos 1380 | ARM Mali-G68 MP5 | VULKAN (No Fallback) | 25 / 25 | **0.00 t/s (Deadlock)** | — | **Unresponsive (Host Hang)** |

---

## Live Physical Device Telemetry Logs

### 1. Galaxy S25 (Qualcomm Adreno 830 - Native Vulkan Full Offload)

```text
Ameva Runtime Version: 1.0.1

=== OFFICIAL INFERENCE RESULT ===
Generated text: Space in Korean is: "공간" (kakjang)
A. Correct
B. Incorrect
Answer:

A

According to the "Rules for the Implementation of the Law
Hardware backend: VULKAN
Token generation speed: 34.08 tokens/sec
Prompt evaluation speed: 4.59 tokens/sec
Total latency: 17148.2 ms
Safety rationale: Vulkan hardware acceleration active on ADRENO (qualcomm). All 99 layers targeted to VRAM.
```

### 2. Galaxy A35 (ARM Mali-G68 MP5 - Forced Headless Vulkan Deadlock Log)

```text
Log start
main: build = 110 (0b341e5)
main: built with clang version 21.1.8 for aarch64-unknown-linux-android24
...
ggml_vulkan: Found 1 Vulkan devices:
Vulkan0: Mali-G68 (Mali-G68) | uma: 1 | fp16: 1 | warp size: 16
[DRIVER DEADLOCK: Proprietary vulkan.mali.so stops responding during SPIR-V compute pipeline initialization]
[PID 4690: Consuming 94% CPU in busy-wait loop, 0 tokens generated after 60s timeout]
```

### 3. Galaxy A35 (ARM Cortex-A78 CPU-NEON - Adaptive Safe Route)

```text
Space in Korean is: 3.5268041954294...

llama_print_timings:        load time =     834.98 ms
llama_print_timings: prompt eval time =     776.73 ms /     5 tokens (  155.35 ms per token,     6.44 tokens per second)
llama_print_timings:        eval time =    3510.69 ms /    15 runs   (  234.05 ms per token,     4.27 tokens per second)
llama_print_timings:       total time =    4336.22 ms /    20 tokens
```

---

## Engineering Rationale: ARM Mali Headless Driver Quirk

The proprietary ARM Mali Vulkan driver (`/vendor/lib64/hw/vulkan.mali.so` via `/dev/mali0`) enforces aggressive kernel-level dynamic power management (DVFS) bound to the Android `SurfaceFlinger` display compositor.

In headless compute environments (terminal CLI, background daemons, or server containers lacking an active `ANativeWindow` swapchain):
1. The driver misinterprets the lack of swapchain frame presentations as an idle display state.
2. It abruptly downclocks GPU compute units while dispatching SPIR-V compute shaders.
3. Completion fence signals are lost at the kernel boundary, causing `vkWaitForFences` or `vkQueueWaitIdle` to block host threads indefinitely.

Rather than allowing device lockups or employing deceptive silent fallbacks, the **AMEVA SmartRouter** automatically identifies ARM Mali hardware topologies, reports the driver hazard, and safely directs compute workloads to the ARM Cortex CPU-NEON pipeline.

---

## Command-Line Interface (CLI)

AMEVA Runtime includes high-performance command-line utilities for diagnostic inspection and model execution:

```bash
# 1. 12-stage automated hardware and driver diagnostic
ameva doctor

# 2. Hardware profile and affinity topology inspection
ameva profile

# 3. Dry-run smart execution plan
ameva plan -m qwen2.5-0.5b

# 4. Direct model execution
ameva exec -m qwen2.5-0.5b -p "Explain quantum entanglement in one sentence:" -n 48
```

---

## Complete Ecosystem Backward Compatibility

Existing codebases leveraging `ameva-vulkan-runtime` require **zero modifications**:

```python
# Legacy imports continue to function seamlessly
import ameva_vulkan_runtime as avr
from ameva_vulkan_runtime.doctor import Doctor
from ameva_vulkan_runtime.adapters import LlamaCppAdapter, SttAdapter, TtsAdapter
from ameva_vulkan_runtime.platform import is_termux, is_android

doc = Doctor()
report = doc.run_self_test()
print(f"Diagnostics: {report.passed_stages} stages passed")
```

---

## License

Distributed under the Apache-2.0 License. Copyright (c) 2026 Eunho Kim (@uno-km).
