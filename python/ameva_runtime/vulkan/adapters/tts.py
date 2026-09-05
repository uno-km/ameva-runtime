"""
TtsAdapter — termux-tts (Piper / VITS HiFi-GAN) Vulkan Acceleration Adapter
"""
from __future__ import annotations

import logging
from typing import Any

from ..doctor import DiagnosticReport
from ..exceptions import AmevaRuntimeError
from ..protocol import BindingResult
from .base import _is_vulkan_report, _make_cpu_fallback

logger = logging.getLogger("ameva_vulkan_runtime.adapters.tts")


class TtsAdapter:
    """termux-tts (Piper TTS / VITS HiFi-GAN) Vulkan 가속 바인딩 어댑터."""

    module_name = "termux-tts"

    @staticmethod
    def bind(engine: Any, report: DiagnosticReport) -> BindingResult:
        is_vk = _is_vulkan_report(report)
        config: dict = {
            "module": TtsAdapter.module_name,
            "device_name": report.device_name,
            "vendor_id": report.vendor_id,
        }

        if is_vk:
            config.update({
                "backend": "vulkan",
                "transposed_conv_vulkan": True,
                "fp16_vocoder": True,
            })

            if engine is not None:
                try:
                    if hasattr(engine, "use_vulkan"):
                        engine.use_vulkan = True
                    if hasattr(engine, "fp16") and hasattr(engine, "device") and getattr(engine, "device") != "cpu":
                        engine.fp16 = True
                    logger.info(
                        "[ameva-vulkan-runtime:TtsAdapter] Piper/VITS Vulkan 바인딩 완료"
                        " (device=%s)", report.device_name
                    )
                except Exception as e:
                    logger.error("[ameva-vulkan-runtime:TtsAdapter] 바인딩 오류: %s", e)
                    raise AmevaRuntimeError(
                        f"[ameva-vulkan-runtime:TtsAdapter] TTS Vulkan 바인딩 실패: {e}"
                    ) from e

            return BindingResult(
                module=TtsAdapter.module_name, backend="vulkan", is_vulkan=True,
                device_name=report.device_name, vendor_id=report.vendor_id,
                config=config, status="BOUND",
            )
        else:
            return _make_cpu_fallback(TtsAdapter.module_name, report, config)

    @staticmethod
    def unbind(engine: Any = None) -> None:
        logger.info("[ameva-vulkan-runtime:TtsAdapter] 바인딩 해제 및 리소스 초기화.")
        if engine is not None:
            try:
                if hasattr(engine, "use_vulkan"):
                    engine.use_vulkan = False
                if hasattr(engine, "fp16"):
                    engine.fp16 = False
            except Exception as e:
                logger.debug("[ameva-vulkan-runtime:TtsAdapter] unbind 중 무시된 예외: %s", e)
