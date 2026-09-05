# AMEVA-Runtime (Python)

[![PyPI](https://img.shields.io/pypi/v/ameva-runtime.svg?style=flat-square&color=0369a1)](https://pypi.org/project/ameva-runtime/)
[![Python](https://img.shields.io/pypi/pyversions/ameva-runtime.svg?style=flat-square)](https://pypi.org/project/ameva-runtime/)
[![License](https://img.shields.io/badge/License-Apache_2.0-004499.svg?style=flat-square)](https://github.com/uno-km/ameva-runtime)

> Next-Gen Unified On-Device Hardware Orchestration & 6-Modality AI Acceleration Runtime for Mobile & Edge

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
```

## Empirical Benchmarks

- **Galaxy S25 (Adreno 830)**: LLM 35.80 t/s (VRAM 25/25 layers), Whisper STT 4,401 ms, TTS RTF 0.264x (medium) / 0.993x (high-fp16).
- **Galaxy A35 (Mali-G68 MP5)**: LLM 4.44 t/s (+26.9% vs NEON), Whisper STT 360.60s (2.26x speedup), TTS RTF 1.146x.

## Documentation
- [Official Documentation](https://uno-km.vercel.app/lib/vulkan/)
- [GitHub Repository](https://github.com/uno-km/ameva-runtime)

## License
Apache-2.0 License. Copyright (c) 2026 Eunho Kim (@uno-km).
