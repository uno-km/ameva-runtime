"""
VisionAdapter — termux-vision (LLaVA ViT / YOLO) Vulkan Acceleration Adapter
"""
from __future__ import annotations

import logging
from typing import Any

from ..doctor import DiagnosticReport
from ..exceptions import AmevaRuntimeError
from ..protocol import BindingResult
from .base import _is_vulkan_report, _make_cpu_fallback

logger = logging.getLogger("ameva_vulkan_runtime.adapters.vision")


class VisionAdapter:
    """termux-vision (LLaVA ViT / SmolVLM / YOLO) Vulkan 가속 바인딩 어댑터."""

    module_name = "termux-vision"

    @staticmethod
    def bind(engine: Any, report: DiagnosticReport) -> BindingResult:
        is_vk = _is_vulkan_report(report)
        config: dict = {
            "module": VisionAdapter.module_name,
            "device_name": report.device_name,
            "vendor_id": report.vendor_id,
        }

        if is_vk:
            config.update({
                "backend": "vulkan",
                "vit_acceleration": True,
                "patch_embedding_vulkan": True,
                "ameva_loader_path": report.loader_path,
            })

            if engine is not None:
                try:
                    try:
                        from termux_vision.csrc import backend as vk_backend
                        vision_lib_path = getattr(vk_backend, "_vk_lib_path", None)
                        if vision_lib_path and vision_lib_path != report.loader_path:
                            logger.warning(
                                "[ameva-vulkan-runtime:VisionAdapter] Vulkan ICD 경로 불일치 감지. "
                                "ameva=%s | vision=%s. "
                                "termux-vision 자체 libfast_cv_vk.so 스택을 유지합니다.",
                                report.loader_path, vision_lib_path
                            )
                        else:
                            logger.info(
                                "[ameva-vulkan-runtime:VisionAdapter] Vulkan ICD 경로 일치 확인: %s",
                                report.loader_path
                            )
                    except ImportError:
                        logger.warning(
                            "[ameva-vulkan-runtime:VisionAdapter] termux_vision 패키지를 import 할 수 없습니다. "
                            "vision 이 설치되어 있는지 확인하세요."
                        )

                    if hasattr(engine, "device"):
                        engine.device = "vulkan"
                    if hasattr(engine, "use_vulkan"):
                        engine.use_vulkan = True
                    logger.info(
                        "[ameva-vulkan-runtime:VisionAdapter] LLaVA/YOLO Vulkan ViT 바인딩 완료."
                    )
                except Exception as e:
                    logger.error("[ameva-vulkan-runtime:VisionAdapter] 바인딩 오류: %s", e)
                    raise AmevaRuntimeError(
                        f"[ameva-vulkan-runtime:VisionAdapter] Vision Vulkan 바인딩 실패: {e}"
                    ) from e

            return BindingResult(
                module=VisionAdapter.module_name, backend="vulkan", is_vulkan=True,
                device_name=report.device_name, vendor_id=report.vendor_id,
                config=config, status="BOUND",
            )
        else:
            config["vit_acceleration"] = False
            return _make_cpu_fallback(VisionAdapter.module_name, report, config)

    @staticmethod
    def unbind(engine: Any = None) -> None:
        logger.info("[ameva-vulkan-runtime:VisionAdapter] 바인딩 해제 및 리소스 초기화.")
        if engine is not None:
            try:
                if hasattr(engine, "use_gpu"):
                    engine.use_gpu = False
                if hasattr(engine, "use_vulkan"):
                    engine.use_vulkan = False
                if hasattr(engine, "device"):
                    engine.device = "cpu"
            except Exception as e:
                logger.debug("[ameva-vulkan-runtime:VisionAdapter] unbind 중 무시된 예외: %s", e)
