"""
ameva_runtime.adapters.llamacpp
===============================
Adapter for termux-llamacpp (GGUF LLM) inference.
Integrates with SmartRouter to provide Zero-Crash adaptive execution.
"""
from __future__ import annotations

import logging
import os
from typing import Any

from ..protocol import BindingResult
from ..detector import detect_hardware, HardwareProfile
from ..router import SmartRouter, ExecutionPlan
from .base import _make_adaptive_binding

logger = logging.getLogger("ameva_runtime.adapters.llamacpp")


class LlamaCppAdapter:
    """termux-llamacpp execution and acceleration adapter."""

    module_name = "termux-llamacpp"

    @staticmethod
    def bind(engine: Any = None, profile: Any = None) -> BindingResult:
        if isinstance(profile, HardwareProfile):
            prof = profile
        else:
            prof = detect_hardware()

        router = SmartRouter(prof)
        plan: ExecutionPlan = router.route_for_llm()

        config: dict = {
            "module": LlamaCppAdapter.module_name,
            "backend": plan.backend,
            "ngl": plan.ngl,
            "threads": plan.threads,
            "cli_flags": plan.cli_flags,
            "allowed_cpus": plan.allowed_cpus,
            "is_gpu": plan.is_gpu_accelerated,
            "diagnosis": plan.diagnosis,
        }

        # Apply environment overrides
        for k, v in plan.env_overrides.items():
            os.environ[k] = v

        if engine is not None:
            try:
                if isinstance(engine, dict):
                    engine["ngl"] = plan.ngl
                    engine["threads"] = plan.threads
                    engine["backend"] = plan.backend
                    engine.setdefault("cli_flags", []).extend(plan.cli_flags)
                    engine.setdefault("env", {}).update(plan.env_overrides)
                elif hasattr(engine, "config"):
                    cfg = engine.config
                    if isinstance(cfg, dict):
                        cfg["ngl"] = plan.ngl
                        cfg["n_gpu_layers"] = plan.ngl
                        cfg["threads"] = plan.threads
                        cfg["backend"] = plan.backend
                    else:
                        if hasattr(cfg, "ngl"):
                            cfg.ngl = plan.ngl
                        if hasattr(cfg, "n_gpu_layers"):
                            cfg.n_gpu_layers = plan.ngl
                        if hasattr(cfg, "threads"):
                            cfg.threads = plan.threads
                        if hasattr(cfg, "device"):
                            cfg.device = "vulkan" if plan.is_gpu_accelerated else "cpu"
            except Exception as e:
                logger.warning("Failed to inject configuration into engine: %s", e)

        return _make_adaptive_binding(
            module=LlamaCppAdapter.module_name,
            profile=profile,
            config=config,
            backend=plan.backend,
            status="BOUND_VULKAN" if plan.is_gpu_accelerated else "BOUND_CPU_NEON",
        )

    @staticmethod
    def unbind() -> None:
        pass
