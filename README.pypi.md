# AMEVA-Runtime (Python)

[![PyPI](https://img.shields.io/pypi/v/ameva-runtime.svg?style=flat-square&color=0369a1)](https://pypi.org/project/ameva-runtime/)
[![Python](https://img.shields.io/pypi/pyversions/ameva-runtime.svg?style=flat-square)](https://pypi.org/project/ameva-runtime/)
[![License](https://img.shields.io/badge/License-Apache_2.0-004499.svg?style=flat-square)](https://github.com/uno-km/ameva-runtime)

> Next-Gen Unified On-Device Hardware Orchestration & 6-Modality AI Acceleration Runtime (with BitNet 1.58-bit Vulkan Compute) for Mobile & Edge

## Installation

```bash
pip install ameva-runtime
```

## Quickstart

```python
import ameva_runtime as ameva
from ameva_runtime import vulkan

# Inspect hardware
profile = ameva.detect_hardware()
print(f"Target: {profile.soc_name} | {profile.gpu_vendor}")

# Run diagnostics
doc = vulkan.Doctor()
report = doc.run_self_test()
print(f"Passed: {report.passed_stages}/{report.total_stages}")

# Run Bionic Native Vulkan TTS (Zero-Silent-Fallback)
from ameva_runtime.adapters.tts import TTSAdapter
adapter = TTSAdapter(backend="vulkan", model_tier="medium")
output_wav = adapter.synthesize(
    text="온디바이스 네이티브 하드웨어 가속 음성 합성 테스트입니다.",
    output_path="output.wav"
)
print(f"TTS Synthesized: {output_wav}, RTF: {adapter.last_rtf:.3f}x")
```

## Empirical Benchmarks

- **Galaxy S22 (Adreno 730)**:
  - MeloTTS Universal Bilingual (Vulkan GPU): **2,750 ms**, RTF **0.88x** (Real-time).
- **Galaxy S21 (Mali-G78)**:
  - Piper VITS On-chip Tiled (Vulkan GPU): **560 ms**, RTF **0.18x** (5.5x faster than RT).
- **Galaxy S20 (Mali-G77)**:
  - Piper VITS On-chip Tiled (Vulkan GPU): **750 ms**, RTF **0.24x** (4.1x faster than RT).
- **Galaxy S25 (Adreno 830)**:
  - BitNet 1.58-bit LLM: **17.558 t/s (12.58x speedup)**, Prompt Eval **205.9 ms**.
  - Qwen2.5-0.5B LLM: **35.80 t/s** (25/25 VRAM layers).
  - Whisper STT: **4,401 ms**, Supertonic 3 TTS: **380 ms** (RTF **0.12x**).
- **Galaxy A35 (Mali-G68 MP5)**:
  - BitNet 1.58-bit LLM: **3.471 t/s (5.94x speedup)**, Prompt Eval **1,552.8 ms**.
  - Qwen2.5-0.5B LLM: **4.44 t/s** (+26.9% vs NEON).
  - Whisper STT: **360.60s (2.26x speedup)**, Piper TTS: RTF **1.146x**.

## Documentation
- [Official Documentation](https://uno-km.vercel.app/lib/vulkan/)
- [GitHub Repository](https://github.com/uno-km/ameva-runtime)

## License
Apache-2.0 License. Copyright (c) 2026 Eunho Kim (@uno-km).
