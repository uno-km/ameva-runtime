"""
DiffusionAdapter — termux-diffusion (stable-diffusion.cpp) Vulkan Acceleration Adapter
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from ..doctor import DiagnosticReport
from ..exceptions import AmevaRuntimeError
from ..protocol import BindingResult
from .base import _is_vulkan_report, _make_cpu_fallback

logger = logging.getLogger("ameva_vulkan_runtime.adapters.diffusion")


@dataclass
class VulkanDriverInfo:
    """Standardized Vulkan driver handle for termux-diffusion."""
    library_path: str
    usable: bool = True


class DiffusionAdapter:
    """termux-diffusion (stable-diffusion.cpp) Vulkan 가속 바인딩 어댑터."""

    module_name = "termux-diffusion"

    @staticmethod
    def bind(engine: Any, report: DiagnosticReport) -> BindingResult:
        is_vk = _is_vulkan_report(report)
        config: dict = {
            "module": DiffusionAdapter.module_name,
            "device_name": report.device_name,
            "vendor_id": report.vendor_id,
        }

        if is_vk:
            config.update({
                "backend": "vulkan",
                "vulkan_lib_path": report.loader_path,
                "unet_tiling": True,
                "sd_vulkan_flag": True,
                "coopmat_off": True,  # Adreno 830 NV coopmat2 bug avoidance
            })

            if engine is not None:
                try:
                    if hasattr(engine, "hw_profile"):
                        engine.hw_profile.vulkan_available = True
                        engine.hw_profile.vulkan_driver = VulkanDriverInfo(
                            library_path=report.loader_path,
                            usable=True,
                        )
                        logger.info(
                            "[ameva-vulkan-runtime:DiffusionAdapter] hw_profile.vulkan_driver 패치 완료: %s",
                            report.loader_path
                        )
                    elif hasattr(engine, "set_vulkan_lib"):
                        engine.set_vulkan_lib(report.loader_path)
                    else:
                        logger.warning(
                            "[ameva-vulkan-runtime:DiffusionAdapter] engine 에 hw_profile 또는 "
                            "set_vulkan_lib 속성이 없습니다. 타입: %s", type(engine).__name__
                        )
                except Exception as e:
                    logger.error("[ameva-vulkan-runtime:DiffusionAdapter] 바인딩 오류: %s", e)
                    raise AmevaRuntimeError(
                        f"[ameva-vulkan-runtime:DiffusionAdapter] sd.cpp Vulkan 바인딩 실패: {e}"
                    ) from e

            return BindingResult(
                module=DiffusionAdapter.module_name, backend="vulkan", is_vulkan=True,
                device_name=report.device_name, vendor_id=report.vendor_id,
                config=config, status="BOUND",
            )
        else:
            config["offload_to_cpu"] = True
            return _make_cpu_fallback(DiffusionAdapter.module_name, report, config)

    @staticmethod
    def unbind(engine: Any = None) -> None:
        logger.info("[ameva-vulkan-runtime:DiffusionAdapter] 바인딩 해제 및 리소스 초기화.")
        if engine is not None:
            try:
                if hasattr(engine, "hw_profile"):
                    engine.hw_profile.vulkan_available = False
                    engine.hw_profile.vulkan_driver = None
                if hasattr(engine, "set_vulkan_lib"):
                    engine.set_vulkan_lib(None)
            except Exception as e:
                logger.debug("[ameva-vulkan-runtime:DiffusionAdapter] unbind 중 무시된 예외: %s", e)

    @classmethod
    def build_cli_args(
        cls,
        executable: str,
        model_path: str,
        prompt: str,
        output_path: str,
        width: int = 512,
        height: int = 512,
        steps: int = 4,
        cfg_scale: float = 1.0,
        threads: Any = "auto",
        target_backend: str = "auto",
        sampling_method: Optional[str] = None,
        seed: Optional[int] = None,
        vae_path: Optional[str] = None,
    ) -> list[str]:
        """최신 sd-cli 규격에 부합하는 안전하고 검증된 Diffusion CLI 인자 목록을 조립합니다."""
        import os
        if threads == "auto" or threads is None:
            cpu_count = os.cpu_count() or 8
            thread_val = str(max(1, min(4, cpu_count // 2 if cpu_count > 4 else cpu_count)))
        else:
            thread_val = str(threads)

        cmd = [
            str(executable),
            "-m", str(model_path),
            "-p", str(prompt),
            "-o", str(output_path),
            "-W", str(width),
            "-H", str(height),
            "--steps", str(steps),
            "-t", thread_val,
            "--cfg-scale", str(cfg_scale),
        ]

        if target_backend == "cpu":
            cmd.extend(["--backend", "cpu", "--params-backend", "cpu"])
        elif target_backend in ("vulkan", "gpu"):
            cmd.extend(["--backend", "vulkan", "--params-backend", "vulkan"])
        else:
            # Auto detection: On Mali devices, default to safe CPU SIMD
            cmd.extend(["--backend", "cpu", "--params-backend", "cpu"])

        if sampling_method:
            cmd.extend(["--sampling-method", str(sampling_method)])
        if seed is not None:
            cmd.extend(["-s", str(seed)])
        if vae_path:
            cmd.extend(["--vae", str(vae_path)])

        return cmd
