"""
VisionAdapter — termux-vision (LLaVA ViT / YOLO) Vulkan Acceleration Adapter
"""
from __future__ import annotations

import logging
from typing import Any

from .base import (
    _is_vulkan_report,
    _make_cpu_binding,
    check_vulkan_availability_or_raise,
    DiagnosticReport,
    BindingResult,
    resolve_diagnostic_report,
    get_vulkan_env,
    BaseAdapter,
)
from ..exceptions import AmevaRuntimeError

logger = logging.getLogger("ameva_runtime.adapters.vision")


class VisionAdapter(BaseAdapter):
    """termux-vision (LLaVA ViT / SmolVLM / YOLO) Vulkan acceleration adapter."""

    module_name = "termux-vision"

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
            check_vulkan_availability_or_raise(
                VisionAdapter.module_name,
                report,
                is_vk,
                requested_backend,
            )

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
                                "[ameva-runtime:VisionAdapter] Vulkan ICD path discrepancy detected: "
                                "ameva=%s | vision=%s. "
                                "Preserving termux-vision internal libfast_cv_vk.so stack.",
                                report.loader_path, vision_lib_path
                            )
                        else:
                            logger.info(
                                "[ameva-runtime:VisionAdapter] Confirmed matching Vulkan ICD path: %s",
                                report.loader_path
                            )
                    except ImportError:
                        logger.warning(
                            "[ameva-runtime:VisionAdapter] Cannot import termux_vision package. "
                            "Ensure vision package is installed."
                        )

                    if hasattr(engine, "device"):
                        engine.device = "vulkan"
                    if hasattr(engine, "use_vulkan"):
                        engine.use_vulkan = True
                    logger.info(
                        "[ameva-runtime:VisionAdapter] LLaVA/YOLO Vulkan ViT binding complete."
                    )
                except Exception as e:
                    logger.error("[ameva-runtime:VisionAdapter] Binding error: %s", e)
                    raise AmevaRuntimeError(
                        f"[ameva-runtime:VisionAdapter] Vision Vulkan binding failure: {e}"
                    ) from e

            return BindingResult(
                module=VisionAdapter.module_name,
                backend="vulkan",
                is_vulkan=True,
                device_name=report.device_name,
                vendor_id=report.vendor_id,
                config=config,
                status="BOUND",
            )
        else:
            config["vit_acceleration"] = False
            return _make_cpu_binding(
                VisionAdapter.module_name,
                report,
                config,
                reason="Explicit CPU requested" if requested_backend in ("cpu", "cpu_neon") else "Vulkan unavailable",
            )

    @staticmethod
    def unbind(engine: Any = None) -> None:
        logger.info("[ameva-runtime:VisionAdapter] Unbinding adapter and resetting resources.")
        if engine is not None:
            try:
                if hasattr(engine, "use_gpu"):
                    engine.use_gpu = False
                if hasattr(engine, "use_vulkan"):
                    engine.use_vulkan = False
                if hasattr(engine, "device"):
                    engine.device = "cpu"
            except Exception as e:
                logger.debug("[ameva-runtime:VisionAdapter] Ignored exception during unbind: %s", e)

    @classmethod
    def build_cli_args(
        cls,
        executable: str,
        text_model_path: str,
        vision_model_path: str,
        image_path: str,
        prompt_file: str,
        target_backend: str = "auto",
        threads: Any = "auto",
        context_limit: int = 2048,
        max_tokens: int = 150,
        temperature: float = 0.2,
        repeat_penalty: Optional[float] = 1.2,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        presence_penalty: Optional[float] = None,
        frequency_penalty: Optional[float] = None,
        seed: Optional[int] = None,
        device_name: Optional[str] = None,
        ngl_override: Optional[int] = None,
    ) -> list[str]:
        """Assembles verified VLM CLI argument list conforming to modern llama-cli specifications."""
        import os
        if threads == "auto" or threads is None:
            cpu_count = os.cpu_count() or 8
            thread_val = str(max(1, min(4, cpu_count // 2 if cpu_count > 4 else cpu_count)))
        else:
            thread_val = str(threads)

        if ngl_override is not None:
            ngl_val = str(ngl_override)
        elif target_backend == "cpu":
            ngl_val = "0"
        elif target_backend in ("vulkan", "gpu"):
            ngl_val = "99"
        else:
            ngl_val = "99"

        cmd = [
            str(executable),
            "-m", str(text_model_path),
            "--mmproj", str(vision_model_path),
            "--image", str(image_path),
            "-f", str(prompt_file),
            "-t", thread_val,
            "-c", str(context_limit),
            "-n", str(max_tokens),
            "--temp", str(temperature),
            "-ngl", ngl_val,
            "--simple-io",
        ]

        if target_backend in ("vulkan", "gpu") and device_name:
            cmd.extend(["--device", device_name])

        if repeat_penalty is not None:
            cmd.extend(["--repeat-penalty", str(repeat_penalty)])
        if top_p is not None:
            cmd.extend(["--top-p", str(top_p)])
        if top_k is not None:
            cmd.extend(["--top-k", str(top_k)])
        if presence_penalty is not None:
            cmd.extend(["--presence-penalty", str(presence_penalty)])
        if frequency_penalty is not None:
            cmd.extend(["--frequency-penalty", str(frequency_penalty)])
        if seed is not None:
            cmd.extend(["-s", str(seed)])

        return cmd
