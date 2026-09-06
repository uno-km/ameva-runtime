"""
BitnetAdapter — termux-bitnet (1.58-bit LLM) Vulkan Acceleration Adapter
"""
from __future__ import annotations

import logging
import os
from typing import Any

from ..doctor import DiagnosticReport
from ..exceptions import AmevaRuntimeError
from ..protocol import BindingResult
from .base import (
    _is_vulkan_report,
    _make_cpu_binding,
    check_vulkan_availability_or_raise,
    _MALI_VENDOR_ID,
    resolve_diagnostic_report,
    BaseAdapter,
)

logger = logging.getLogger("ameva_runtime.adapters.bitnet")


def _calculate_bitnet_layers(engine: Any) -> int:
    """Dynamically determines optimal GPU offload layers for BitNet 1.58-bit models."""
    if engine is not None:
        if hasattr(engine, "n_gpu_layers") and getattr(engine, "n_gpu_layers", 0) > 0:
            return int(engine.n_gpu_layers)
        if hasattr(engine, "config") and hasattr(engine.config, "n_gpu_layers") and engine.config.n_gpu_layers > 0:
            return int(engine.config.n_gpu_layers)
        if hasattr(engine, "n_layers") and getattr(engine, "n_layers", 0) > 0:
            return int(engine.n_layers)
        model_name = str(getattr(engine, "model_name", "") or getattr(engine, "model", "")).lower()
        if "0.7b" in model_name:
            return 16
        elif "1.3b" in model_name:
            return 24
        elif "3b" in model_name or "2.7b" in model_name:
            return 32
    return 32


class BitnetAdapter(BaseAdapter):
    """termux-bitnet (BitNet 1.58-bit i2_s) Vulkan acceleration adapter."""

    module_name = "termux-bitnet"

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
            check_vulkan_availability_or_raise(
                BitnetAdapter.module_name,
                report,
                is_vk,
                requested_backend,
            )

        config: dict = {
            "module": BitnetAdapter.module_name,
            "device_name": report.device_name,
            "vendor_id": report.vendor_id,
        }

        if is_vk:
            ngl = _calculate_bitnet_layers(engine)
            mali_align_required = (report.vendor_id == _MALI_VENDOR_ID or
                                   "Mali" in (report.device_name or ""))

            config.update({
                "backend": "vulkan",
                "n_gpu_layers": ngl,
                "mali_128byte_align": mali_align_required,
                "flash_attn": True,
            })

            if engine is not None:
                try:
                    if hasattr(engine, "config"):
                        engine.config.n_gpu_layers = ngl
                        engine.config.flash_attn = True
                        logger.info(
                            "[ameva-runtime:BitnetAdapter] Set config.n_gpu_layers=%d"
                            " (device=%s, mali_align=%s)", ngl, report.device_name, mali_align_required
                        )
                    else:
                        logger.warning(
                            "[ameva-runtime:BitnetAdapter] Engine has no config attribute. Type: %s",
                            type(engine).__name__
                        )
                except Exception as e:
                    logger.error("[ameva-runtime:BitnetAdapter] Binding error: %s", e)
                    raise AmevaRuntimeError(
                        f"[ameva-runtime:BitnetAdapter] BitNetEngine Vulkan binding failure: {e}"
                    ) from e

            return BindingResult(
                module=BitnetAdapter.module_name,
                backend="vulkan",
                is_vulkan=True,
                device_name=report.device_name,
                vendor_id=report.vendor_id,
                config=config,
                status="BOUND",
            )
        else:
            config["n_threads"] = max(1, (os.cpu_count() or 8) // 2)
            config["kernel"] = "neon_dotprod"
            if engine is not None and hasattr(engine, "config"):
                engine.config.n_gpu_layers = 0
            return _make_cpu_binding(
                BitnetAdapter.module_name,
                report,
                config,
                reason="Explicit CPU requested" if requested_backend in ("cpu", "cpu_neon") else "Vulkan unavailable",
            )

    @staticmethod
    def unbind(engine: Any = None) -> None:
        logger.info("[ameva-runtime:BitnetAdapter] Unbinding adapter and resetting resources.")
        if engine is not None:
            try:
                if hasattr(engine, "config"):
                    engine.config.n_gpu_layers = 0
                    if hasattr(engine.config, "flash_attn"):
                        engine.config.flash_attn = False
            except Exception as e:
                logger.debug("[ameva-runtime:BitnetAdapter] Ignored exception during unbind: %s", e)
