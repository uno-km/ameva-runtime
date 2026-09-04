"""
DiffusionAdapter — termux-diffusion (Stable Diffusion / Onnx / NCNN) Acceleration Adapter
"""
from __future__ import annotations

import logging
from typing import Any

from ..protocol import BindingResult
from ..detector import detect_hardware, HardwareProfile
from .base import _make_adaptive_binding

logger = logging.getLogger("ameva_runtime.adapters.diffusion")


class DiffusionAdapter:
    """termux-diffusion acceleration adapter."""

    module_name = "termux-diffusion"

    @staticmethod
    def bind(engine: Any = None, profile: Any = None) -> BindingResult:
        if not isinstance(profile, HardwareProfile):
            profile = detect_hardware()

        is_vk = profile.recommended_backend == "vulkan"
        config: dict = {
            "module": DiffusionAdapter.module_name,
            "device_name": profile.gpu_family,
            "backend": "vulkan" if is_vk else "cpu_neon",
        }

        if engine is not None:
            try:
                if hasattr(engine, "device"):
                    engine.device = "vulkan" if is_vk else "cpu"
            except Exception as e:
                logger.warning("Failed to configure diffusion engine: %s", e)

        return _make_adaptive_binding(
            module=DiffusionAdapter.module_name,
            profile=profile,
            config=config,
            backend="vulkan" if is_vk else "cpu_neon",
            status="BOUND_VULKAN" if is_vk else "BOUND_CPU_NEON",
        )

    @staticmethod
    def unbind() -> None:
        pass
