"""
Backward compatibility re-export for ameva_runtime.adapters.
All canonical adapter implementations reside in ameva_runtime.adapters (SSOT).
"""
from __future__ import annotations

from ameva_runtime.adapters import (
    BaseAdapter,
    resolve_diagnostic_report,
    SttAdapter,
    DiffusionAdapter,
    BitnetAdapter,
    LlamaCppAdapter,
    TtsAdapter,
    VisionAdapter,
    BindingResult,
    get_vulkan_env,
    find_system_vulkan_driver_dir,
    _is_vulkan_report,
    _make_cpu_fallback,
    _make_cpu_binding,
    check_vulkan_availability_or_raise,
)

import sys
from ameva_runtime.adapters import stt, tts, diffusion, llamacpp, bitnet, vision

sys.modules[__name__ + ".stt"] = stt
sys.modules[__name__ + ".tts"] = tts
sys.modules[__name__ + ".diffusion"] = diffusion
sys.modules[__name__ + ".llamacpp"] = llamacpp
sys.modules[__name__ + ".bitnet"] = bitnet
sys.modules[__name__ + ".vision"] = vision

