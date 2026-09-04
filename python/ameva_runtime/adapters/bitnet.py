"""
BitnetAdapter — termux-bitnet (1-bit LLM) Acceleration Adapter
"""
from __future__ import annotations

import logging
from typing import Any

from ..protocol import BindingResult
from ..detector import detect_hardware, HardwareProfile
from .base import _make_adaptive_binding

logger = logging.getLogger("ameva_runtime.adapters.bitnet")


class BitnetAdapter:
    """termux-bitnet acceleration adapter."""

    module_name = "termux-bitnet"

    @staticmethod
    def bind(engine: Any = None, profile: Any = None) -> BindingResult:
        if not isinstance(profile, HardwareProfile):
            profile = detect_hardware()

        config: dict = {
            "module": BitnetAdapter.module_name,
            "device_name": profile.gpu_family,
            "backend": "cpu_neon",  # BitNet is natively optimized for CPU 1-bit LUTs
            "threads": profile.recommended_threads,
        }

        if engine is not None:
            try:
                if hasattr(engine, "threads"):
                    engine.threads = profile.recommended_threads
            except Exception as e:
                logger.warning("Failed to configure BitNet engine: %s", e)

        return _make_adaptive_binding(
            module=BitnetAdapter.module_name,
            profile=profile,
            config=config,
            backend="cpu_neon",
            status="BOUND_CPU_NEON",
        )

    @staticmethod
    def unbind() -> None:
        pass
