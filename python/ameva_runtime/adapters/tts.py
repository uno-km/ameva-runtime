"""
TtsAdapter — termux-tts (Piper / Kokoro) Acceleration Adapter
"""
from __future__ import annotations

import logging
from typing import Any

from ..protocol import BindingResult
from ..detector import detect_hardware, HardwareProfile
from ..router import SmartRouter, ExecutionPlan
from .base import _make_adaptive_binding

logger = logging.getLogger("ameva_runtime.adapters.tts")


class TtsAdapter:
    """termux-tts acceleration and hardware execution adapter."""

    module_name = "termux-tts"

    @staticmethod
    def bind(
        engine: Any = None,
        profile: Any = None,
        requested_backend: str | None = None,
    ) -> BindingResult:
        if not isinstance(profile, HardwareProfile):
            profile = detect_hardware()

        router = SmartRouter(profile)
        plan: ExecutionPlan = router.route_for_tts(
            requested_backend=requested_backend,
        )

        is_vk = plan.is_gpu_accelerated
        config: dict = {
            "module": TtsAdapter.module_name,
            "device_name": profile.gpu_family,
            "backend": plan.backend,
            "threads": plan.threads,
            "cli_flags": plan.cli_flags,
            "env_overrides": plan.env_overrides,
            "diagnosis": plan.diagnosis,
        }

        # Apply environment variables
        import os
        for k, v in plan.env_overrides.items():
            os.environ[k] = v

        if engine is not None:
            try:
                if hasattr(engine, "device"):
                    engine.device = "vulkan" if is_vk else "cpu"
                if hasattr(engine, "threads"):
                    engine.threads = plan.threads
                if hasattr(engine, "backend"):
                    engine.backend = plan.backend
            except Exception as e:
                logger.warning("Failed to configure TTS engine: %s", e)

        return _make_adaptive_binding(
            module=TtsAdapter.module_name,
            profile=profile,
            config=config,
            backend=plan.backend,
            status="BOUND_VULKAN" if is_vk else "BOUND_CPU_NEON",
        )

    @staticmethod
    def unbind() -> None:
        pass
