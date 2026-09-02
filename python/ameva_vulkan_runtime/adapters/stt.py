"""
SttAdapter — termux-stt (whisper.cpp / sherpa-onnx) Vulkan Acceleration Adapter
"""
from __future__ import annotations

import logging
import os
from typing import Any

from ..doctor import DiagnosticReport
from ..exceptions import AmevaRuntimeError
from ..protocol import BindingResult
from .base import _is_vulkan_report, _make_cpu_fallback

logger = logging.getLogger("ameva_vulkan_runtime.adapters.stt")


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


class SttAdapter:
    """termux-stt (whisper.cpp / sherpa-onnx) Vulkan 가속 바인딩 어댑터."""

    module_name = "termux-stt"

    @staticmethod
    def bind(engine: Any, report: DiagnosticReport) -> BindingResult:
        is_vk = _is_vulkan_report(report)
        config: dict = {
            "module": SttAdapter.module_name,
            "device_name": report.device_name,
            "vendor_id": report.vendor_id,
        }

        if is_vk:
            ngl = _calculate_whisper_layers(engine)
            config.update({
                "backend": "vulkan",
                "encoder_fp16": True,
                "gpu_layers": ngl,
                "vulkan_flag": True,
            })
            if engine is not None:
                try:
                    if hasattr(engine, "config"):
                        if not hasattr(engine.config, "extra") or engine.config.extra is None:
                            engine.config.extra = {}
                        engine.config.extra["gpu_layers"] = ngl
                        engine.config.extra["use_vulkan"] = True
                        logger.info(
                            "[ameva-vulkan-runtime:SttAdapter] WhisperEngine.config.extra 에 "
                            "Vulkan 플래그 주입 완료 (layers=%d, device=%s)",
                            ngl, report.device_name
                        )
                    elif hasattr(engine, "set_vulkan"):
                        engine.set_vulkan(True, gpu_layers=ngl)
                    else:
                        logger.warning(
                            "[ameva-vulkan-runtime:SttAdapter] engine 에 config 또는 set_vulkan 속성이 없습니다. "
                            "엔진 타입: %s — 구성 정보만 반환합니다.", type(engine).__name__
                        )
                except Exception as e:
                    logger.error("[ameva-vulkan-runtime:SttAdapter] 엔진 바인딩 중 오류: %s", e)
                    raise AmevaRuntimeError(
                        f"[ameva-vulkan-runtime:SttAdapter] WhisperEngine Vulkan 바인딩 실패: {e}"
                    ) from e

            return BindingResult(
                module=SttAdapter.module_name, backend="vulkan", is_vulkan=True,
                device_name=report.device_name, vendor_id=report.vendor_id,
                config=config, status="BOUND",
            )
        else:
            cpu_cores = os.cpu_count() or 8
            optimal_threads = max(1, cpu_cores // 2)
            config["threads"] = optimal_threads
            if engine is not None and hasattr(engine, "threads"):
                engine.threads = optimal_threads
            return _make_cpu_fallback(SttAdapter.module_name, report, config)

    @staticmethod
    def unbind(engine: Any = None) -> None:
        logger.info("[ameva-vulkan-runtime:SttAdapter] 바인딩 해제 및 리소스 초기화.")
        if engine is not None:
            try:
                if hasattr(engine, "config"):
                    if hasattr(engine.config, "extra") and isinstance(engine.config.extra, dict):
                        engine.config.extra.pop("gpu_layers", None)
                        engine.config.extra.pop("use_vulkan", None)
                if hasattr(engine, "set_vulkan"):
                    engine.set_vulkan(False, gpu_layers=0)
            except Exception as e:
                logger.debug("[ameva-vulkan-runtime:SttAdapter] unbind 중 무시된 예외: %s", e)
