"""
6-Modality Acceleration Adapters (Modular Subsystem)

Re-exports individual modality adapters for:
- SttAdapter (termux-stt / whisper.cpp)
- DiffusionAdapter (termux-diffusion / stable-diffusion.cpp)
- BitnetAdapter (termux-bitnet / 1.58-bit LLM)
- LlamaCppAdapter (termux-llamacpp / GGUF LLM)
- TtsAdapter (termux-tts / Piper & VITS)
- VisionAdapter (termux-vision / LLaVA ViT)
"""
from __future__ import annotations

from .base import _is_vulkan_report, _make_cpu_fallback, _ADRENO_VENDOR_ID, _MALI_VENDOR_ID
from .bitnet import BitnetAdapter
from .diffusion import DiffusionAdapter
from .llamacpp import LlamaCppAdapter
from .stt import SttAdapter
from .tts import TtsAdapter
from .vision import VisionAdapter
from ..protocol import BindingResult, IVulkanConsumer

__all__ = [
    "SttAdapter",
    "DiffusionAdapter",
    "BitnetAdapter",
    "LlamaCppAdapter",
    "TtsAdapter",
    "VisionAdapter",
    "BindingResult",
    "IVulkanConsumer",
    "_is_vulkan_report",
    "_make_cpu_fallback",
]
