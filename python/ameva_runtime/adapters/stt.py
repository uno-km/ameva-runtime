"""
SttAdapter — termux-stt (whisper.cpp / sherpa-onnx) Vulkan Acceleration Adapter
"""
from __future__ import annotations

import logging
import os
from typing import Any

from ..doctor import DiagnosticReport
from ..exceptions import AmevaRuntimeError, PlatformNotSupportedError
from ..protocol import BindingResult
from .base import (
    _is_vulkan_report,
    _make_cpu_binding,
    check_vulkan_availability_or_raise,
    resolve_diagnostic_report,
    get_vulkan_env,
    BaseAdapter,
)

logger = logging.getLogger("ameva_runtime.adapters.stt")


def _calculate_whisper_layers(engine: Any) -> int:
    """Dynamically determines optimal GPU offload layers based on Whisper model architecture."""
    if engine is not None:
        if hasattr(engine, "n_layers") and getattr(engine, "n_layers", 0) > 0:
            return int(engine.n_layers)
        model_name = str(getattr(engine, "model", "") or getattr(engine, "model_name", "")).lower()
        if "tiny" in model_name:
            return 4
        elif "base" in model_name:
            return 6
        elif "small" in model_name:
            return 12
        elif "medium" in model_name:
            return 24
        elif "large" in model_name:
            return 32
    return 32


class SttAdapter(BaseAdapter):
    """termux-stt (whisper.cpp / sherpa-onnx) Vulkan acceleration adapter."""

    module_name = "termux-stt"

    @classmethod
    def get_execution_environment(cls, base_env: dict[str, str] | None = None) -> dict[str, str]:
        """Provides verified execution environment adhering to Golden Link Order."""
        return get_vulkan_env(base_env)

    @staticmethod
    def bind(
        engine: Any = None,
        report: Any = None,
        profile: Any = None,
        requested_backend: str | None = None,
        **kwargs: Any,
    ) -> BindingResult:
        report = resolve_diagnostic_report(report, profile)
        is_vk = _is_vulkan_report(report)
        if requested_backend in ("cpu", "cpu_neon"):
            is_vk = False
        else:
            req_check = requested_backend
            if engine is not None:
                req_dev = str(getattr(engine, "device", "") or getattr(getattr(engine, "config", None), "device", "")).lower()
                if req_dev in ("gpu", "vulkan"):
                    req_check = "vulkan"
            check_vulkan_availability_or_raise(
                SttAdapter.module_name,
                report,
                is_vk,
                req_check,
            )
        config: dict = {
            "module": SttAdapter.module_name,
            "device_name": report.device_name,
            "vendor_id": report.vendor_id,
        }

        if is_vk:
            ngl = _calculate_whisper_layers(engine)
            env = SttAdapter.get_execution_environment()
            is_mali = (getattr(report, "vendor_id", 0) == 0x13B5 or "mali" in str(getattr(report, "device_name", "")).lower() or getattr(profile, "gpu_family", "") == "mali")
            if is_mali:
                env["GGML_VK_FORCE_MEDIUM_MATMUL"] = "1"
                env["GGML_VK_DISABLE_F16"] = "1"
            config.update({
                "backend": "vulkan",
                "encoder_fp16": True,
                "gpu_layers": ngl,
                "vulkan_flag": True,
                "env_overrides": env,
            })
            if engine is not None:
                if hasattr(engine, "device"):
                    engine.device = "vulkan"
                if hasattr(engine, "threads"):
                    engine.threads = getattr(profile, "recommended_threads", 4)
                try:
                    if hasattr(engine, "config"):
                        if not hasattr(engine.config, "extra") or engine.config.extra is None:
                            engine.config.extra = {}
                        engine.config.extra["gpu_layers"] = ngl
                        engine.config.extra["use_vulkan"] = True
                        logger.info(
                            "[ameva-runtime:SttAdapter] Injected Whisper Vulkan flags to config.extra"
                            " (layers=%d, device=%s)", ngl, report.device_name
                        )
                    elif hasattr(engine, "set_vulkan"):
                        engine.set_vulkan(True, gpu_layers=ngl)
                    else:
                        logger.warning(
                            "[ameva-runtime:SttAdapter] Engine has neither config nor set_vulkan. Type: %s",
                            type(engine).__name__,
                        )
                except Exception as e:
                    logger.error("[ameva-runtime:SttAdapter] Engine binding error: %s", e)
                    raise AmevaRuntimeError(
                        f"[ameva-runtime:SttAdapter] WhisperEngine Vulkan binding failure: {e}"
                    ) from e

            return BindingResult(
                module=SttAdapter.module_name,
                backend="vulkan",
                is_vulkan=True,
                device_name=report.device_name,
                vendor_id=report.vendor_id,
                config=config,
                status="BOUND_VULKAN",
            )
        else:
            cpu_cores = os.cpu_count() or 8
            optimal_threads = max(1, cpu_cores // 2)
            threads_to_use = getattr(profile, "recommended_threads", optimal_threads)
            config["threads"] = threads_to_use
            if engine is not None:
                if hasattr(engine, "device"):
                    engine.device = "cpu"
                if hasattr(engine, "threads"):
                    engine.threads = threads_to_use
            return _make_cpu_binding(
                SttAdapter.module_name,
                report,
                config,
                reason="Explicit CPU requested" if requested_backend in ("cpu", "cpu_neon") else "Vulkan unavailable",
            )

    @staticmethod
    def unbind(engine: Any = None) -> None:
        logger.info("[ameva-runtime:SttAdapter] Unbinding adapter and resetting resources.")
        if engine is not None:
            try:
                if hasattr(engine, "config"):
                    if hasattr(engine.config, "extra") and isinstance(engine.config.extra, dict):
                        engine.config.extra.pop("gpu_layers", None)
                        engine.config.extra.pop("use_vulkan", None)
                if hasattr(engine, "set_vulkan"):
                    engine.set_vulkan(False, gpu_layers=0)
            except Exception as e:
                logger.debug("[ameva-runtime:SttAdapter] Ignored exception during unbind: %s", e)
