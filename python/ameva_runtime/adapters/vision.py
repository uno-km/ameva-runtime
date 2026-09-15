"""
VisionAdapter — termux-vision (LLaVA ViT / YOLO) Vulkan Acceleration Adapter
"""
from __future__ import annotations

import logging
import os
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


def _get_optimal_threads() -> int:
    cpu_count = os.cpu_count() or 8
    return max(1, cpu_count // 2 if cpu_count > 4 else cpu_count)


class VisionAdapter(BaseAdapter):
    """termux-vision (LLaVA ViT / SmolVLM / YOLO) Vulkan acceleration adapter."""

    module_name = "termux-vision"
    device_signatures = ("vulkan", "fast_cv_vk", "libvulkan.so")

    @classmethod
    def get_execution_environment(
        cls,
        base_env: dict[str, str] | None = None,
        tune_mali: bool = False,
        tune_adreno: bool = False,
    ) -> dict[str, str]:
        """Provides verified execution environment adhering to Golden Link Order without check bypass."""
        return super().get_execution_environment(
            base_env,
            tune_mali=tune_mali,
            tune_adreno=tune_adreno,
        )

    @classmethod
    def _mutate_modality_attributes(
        cls,
        engine: Any,
        snapshot: dict[str, Any],
        backend: str,
        report: Any,
        profile: Any,
        **kwargs: Any,
    ) -> None:
        """Tier 2: Vision-specific ViT / fast_cv_vk mutation."""
        if engine is None:
            return
        is_vk = (backend == "vulkan")
        if is_vk:
            cls._set_engine_property(engine, snapshot, "use_vulkan", True)
            cls._set_engine_property(engine, snapshot, "use_gpu", True)
            cls._set_engine_property(engine, snapshot, "vit_acceleration", True)
        else:
            cls._set_engine_property(engine, snapshot, "use_vulkan", False)
            cls._set_engine_property(engine, snapshot, "use_gpu", False)
            cls._set_engine_property(engine, snapshot, "vit_acceleration", False)

    @classmethod
    def bind(
        cls,
        engine: Any = None,
        report: Any = None,
        profile: Any = None,
        requested_backend: str | None = None,
        **kwargs: Any,
    ) -> BindingResult:
        report = resolve_diagnostic_report(report, profile)
        req_norm = cls.normalize_backend(requested_backend)
        is_vk = _is_vulkan_report(report)

        if req_norm in ("vulkan", "gpu"):
            check_vulkan_availability_or_raise(
                cls.module_name,
                report,
                is_vk,
                "vulkan",
            )
            is_vk = True
        elif req_norm in ("cpu", "cpu_neon"):
            is_vk = False
        else:
            # 'auto': adaptive routing based on hardware diagnostic report
            pass

        config: dict = {
            "module": cls.module_name,
            "device_name": report.device_name,
            "vendor_id": report.vendor_id,
            "requested_backend": req_norm,
        }

        snapshot: dict[str, Any] = {}
        threads = _get_optimal_threads()

        if is_vk:
            config.update({
                "backend": "vulkan",
                "vit_acceleration": True,
                "patch_embedding_vulkan": True,
                "ameva_loader_path": getattr(report, "loader_path", ""),
            })

            try:
                cls._mutate_common_attributes(engine, snapshot, "vulkan", threads)
                cls._mutate_modality_attributes(engine, snapshot, "vulkan", report, profile, **kwargs)
                logger.info("[%s] LLaVA/YOLO Vulkan ViT binding complete.", cls.module_name)
            except Exception as e:
                logger.error("[%s] Binding error: %s", cls.module_name, e)
                raise AmevaRuntimeError(f"[{cls.module_name}] Vision Vulkan binding failure: {e}") from e

            return BindingResult(
                module=cls.module_name,
                backend="vulkan",
                is_vulkan=True,
                device_name=report.device_name,
                vendor_id=report.vendor_id,
                config=config,
                status="BOUND",
                restore_state=snapshot,
            )
        else:
            config["vit_acceleration"] = False
            cls._mutate_common_attributes(engine, snapshot, "cpu", threads)
            cls._mutate_modality_attributes(engine, snapshot, "cpu", report, profile, **kwargs)
            res = _make_cpu_binding(
                cls.module_name,
                report,
                config,
                reason="Explicit CPU requested" if req_norm in ("cpu", "cpu_neon") else "Vulkan unavailable",
            )
            # Attach snapshot to CPU binding as well for non-destructive restore
            return BindingResult(
                module=res.module,
                backend=res.backend,
                is_vulkan=res.is_vulkan,
                device_name=res.device_name,
                vendor_id=res.vendor_id,
                config=res.config,
                status=res.status,
                restore_state=snapshot,
            )

    @classmethod
    def unbind(cls, engine: Any = None, binding_result: Optional[BindingResult] = None) -> None:
        """Restores engine state strictly and ensures GPU compute flags are reset to safe baseline."""
        super().unbind(engine, binding_result)
        if engine is not None:
            if hasattr(engine, "device"):
                try:
                    engine.device = "cpu"
                except Exception:
                    pass
            if hasattr(engine, "use_gpu"):
                try:
                    engine.use_gpu = False
                except Exception:
                    pass

    @classmethod
    def build_cli_args(
        cls,
        executable: str,
        text_model_path: str,
        vision_model_path: str,
        image_path: str,
        prompt_file: Optional[str] = None,
        prompt: Optional[str] = None,
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
        chat_template: Optional[str] = "auto",
        no_warmup: bool = True,
        pure_gpu: bool = True,
        fit_off: bool = True,
        batch_size: int = 64,
        ubatch_size: Optional[int] = None,
        flash_attn: bool = False,
        no_mmproj_offload: bool = True,
        simple_io: bool = False,
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
        elif target_backend in ("vulkan", "gpu", "vulkan-force"):
            ngl_val = "99"
        else:
            ngl_val = "99"

        cmd = [
            str(executable),
            "-m", str(text_model_path),
            "--mmproj", str(vision_model_path),
            "--image", str(image_path),
        ]

        if prompt_file:
            cmd.extend(["-f", str(prompt_file)])
        elif prompt:
            cmd.extend(["-p", f"'{prompt}'"])

        cmd.extend([
            "-t", thread_val,
            "-c", str(context_limit),
            "-n", str(max_tokens),
            "--temp", str(temperature),
            "-ngl", ngl_val,
            "-b", str(batch_size),
            "-ub", str(ubatch_size if ubatch_size is not None else batch_size),
        ])

        if simple_io:
            cmd.append("--simple-io")

        if not flash_attn:
            cmd.extend(["-fa", "off"])

        if no_mmproj_offload:
            cmd.append("--no-mmproj-offload")

        if chat_template:
            if chat_template == "auto":
                m_str = str(text_model_path).lower()
                if "moondream" in m_str:
                    resolved_template = "vicuna"
                elif "qwen" in m_str:
                    resolved_template = "chatml"
                elif "smolvlm" in m_str:
                    resolved_template = "smolvlm"
                elif "deepseek" in m_str:
                    resolved_template = "deepseek"
                else:
                    resolved_template = "vicuna"
            else:
                resolved_template = chat_template
            cmd.extend(["--chat-template", resolved_template])

        if no_warmup:
            cmd.append("--no-warmup")

        if target_backend in ("vulkan", "gpu", "vulkan-force"):
            if pure_gpu:
                cmd.extend(["-ot", "token_embd.weight=Vulkan0"])
            if fit_off:
                cmd.extend(["-fit", "off"])
            if device_name:
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
