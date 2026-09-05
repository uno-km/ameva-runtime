"""
SttAdapter — termux-stt (Whisper) Acceleration Adapter
"""
from __future__ import annotations

import logging
from typing import Any

from ..protocol import BindingResult
from ..detector import detect_hardware, HardwareProfile
from ..router import SmartRouter, ExecutionPlan
from .base import _make_adaptive_binding

logger = logging.getLogger("ameva_runtime.adapters.stt")


class SttAdapter:
    """termux-stt acceleration adapter."""

    module_name = "termux-stt"

    @staticmethod
    def bind(
        engine: Any = None,
        profile: Any = None,
        requested_backend: str | None = None,
    ) -> BindingResult:
        if not isinstance(profile, HardwareProfile):
            profile = detect_hardware()

        router = SmartRouter(profile)
        plan: ExecutionPlan = router.route_for_stt(
            requested_backend=requested_backend,
        )

        is_vk = plan.is_gpu_accelerated
        config: dict = {
            "module": SttAdapter.module_name,
            "device_name": profile.gpu_family,
            "backend": plan.backend,
            "threads": plan.threads,
            "cli_flags": plan.cli_flags,
            "env_overrides": plan.env_overrides,
            "diagnosis": plan.diagnosis,
        }

        if engine is not None:
            try:
                if hasattr(engine, "device"):
                    engine.device = "vulkan" if is_vk else "cpu"
                if hasattr(engine, "threads"):
                    engine.threads = plan.threads
            except Exception as e:
                logger.warning("Failed to configure STT engine: %s", e)

        return _make_adaptive_binding(
            module=SttAdapter.module_name,
            profile=profile,
            config=config,
            backend=plan.backend,
            status="BOUND_VULKAN" if is_vk else "BOUND_CPU_NEON",
        )

    @staticmethod
    def unbind() -> None:
        pass

