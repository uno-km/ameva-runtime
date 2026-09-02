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
from .base import _is_vulkan_report, _make_cpu_fallback, _MALI_VENDOR_ID

logger = logging.getLogger("ameva_vulkan_runtime.adapters.bitnet")


class BitnetAdapter:
    """termux-bitnet (BitNet 1.58-bit i2_s) Vulkan 가속 바인딩 어댑터."""

    module_name = "termux-bitnet"

    @staticmethod
    def bind(engine: Any, report: DiagnosticReport) -> BindingResult:
        is_vk = _is_vulkan_report(report)
        config: dict = {
            "module": BitnetAdapter.module_name,
            "device_name": report.device_name,
            "vendor_id": report.vendor_id,
        }

        if is_vk:
            ngl = 33
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
                            "[ameva-vulkan-runtime:BitnetAdapter] config.n_gpu_layers=%d 설정 완료"
                            " (device=%s, mali_align=%s)", ngl, report.device_name, mali_align_required
                        )
                    else:
                        logger.warning(
                            "[ameva-vulkan-runtime:BitnetAdapter] engine.config 속성 없음. 타입: %s",
                            type(engine).__name__
                        )
                except Exception as e:
                    logger.error("[ameva-vulkan-runtime:BitnetAdapter] 바인딩 오류: %s", e)
                    raise AmevaRuntimeError(
                        f"[ameva-vulkan-runtime:BitnetAdapter] BitNetEngine Vulkan 바인딩 실패: {e}"
                    ) from e

            return BindingResult(
                module=BitnetAdapter.module_name, backend="vulkan", is_vulkan=True,
                device_name=report.device_name, vendor_id=report.vendor_id,
                config=config, status="BOUND",
            )
        else:
            config["n_threads"] = max(1, (os.cpu_count() or 8) // 2)
            config["kernel"] = "neon_dotprod"
            if engine is not None and hasattr(engine, "config"):
                engine.config.n_gpu_layers = 0
            return _make_cpu_fallback(BitnetAdapter.module_name, report, config)

    @staticmethod
    def unbind(engine: Any = None) -> None:
        logger.info("[ameva-vulkan-runtime:BitnetAdapter] 바인딩 해제 및 리소스 초기화.")
        if engine is not None:
            try:
                if hasattr(engine, "config"):
                    engine.config.n_gpu_layers = 0
                    if hasattr(engine.config, "flash_attn"):
                        engine.config.flash_attn = False
            except Exception as e:
                logger.debug("[ameva-vulkan-runtime:BitnetAdapter] unbind 중 무시된 예외: %s", e)
