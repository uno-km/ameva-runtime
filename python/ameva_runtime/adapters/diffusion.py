"""
DiffusionAdapter — termux-diffusion (stable-diffusion.cpp) Vulkan Acceleration Adapter
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any, Optional

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

logger = logging.getLogger("ameva_vulkan_runtime.adapters.diffusion")


@dataclass
class VulkanDriverInfo:
    """Standardized Vulkan driver handle for termux-diffusion."""
    library_path: str
    usable: bool = True


class DiffusionAdapter(BaseAdapter):
    """termux-diffusion (stable-diffusion.cpp) Vulkan acceleration adapter."""

    module_name = "termux-diffusion"

    @staticmethod
    def resolve_shim_path() -> Optional[str]:
        """Locate verified mobile Vulkan HAL shim binary (libegl_shim.so)."""
        env_path = os.environ.get("AMEVA_VULKAN_SHIM")
        if env_path and os.path.isfile(env_path):
            return os.path.abspath(env_path)

        prefix = os.environ.get("PREFIX", "/data/data/com.termux/files/usr")
        std_path = os.path.join(prefix, "lib", "libegl_shim.so")
        if os.path.isfile(std_path):
            return os.path.abspath(std_path)
        return None

    @classmethod
    def get_execution_environment(
        cls,
        base_env: Optional[dict[str, str]] = None,
        **kwargs: Any,
    ) -> dict[str, str]:
        """Assemble environment variables adhering to Golden Link Order with mobile Vulkan HAL shim."""
        env = super().get_execution_environment(base_env, **kwargs)
        shim_path = cls.resolve_shim_path()
        if shim_path:
            existing_preload = env.get("LD_PRELOAD", "").strip()
            if shim_path not in existing_preload:
                env["LD_PRELOAD"] = f"{shim_path}:{existing_preload}" if existing_preload else shim_path
        return env

    @classmethod
    def get_execution_env(cls, extra_env: Optional[dict[str, str]] = None) -> dict[str, str]:
        """Backward-compatible alias for get_execution_environment."""
        return cls.get_execution_environment(extra_env)

    @classmethod
    def _mutate_modality_attributes(
        cls,
        engine: Any,
        snapshot: dict[str, Any],
        report: DiagnosticReport,
        **kwargs: Any,
    ) -> None:
        """Tier 2: Diffusion specific mutation (hw_profile / set_vulkan_lib)."""
        if hasattr(engine, "hw_profile"):
            hp = engine.hw_profile
            if "hw_profile.vulkan_available" not in snapshot:
                snapshot["hw_profile.vulkan_available"] = getattr(hp, "vulkan_available", False)
            if "hw_profile.vulkan_driver" not in snapshot:
                snapshot["hw_profile.vulkan_driver"] = getattr(hp, "vulkan_driver", None)
            hp.vulkan_available = True
            hp.vulkan_driver = VulkanDriverInfo(
                library_path=report.loader_path,
                usable=True,
            )
            logger.info(
                "[ameva-runtime:DiffusionAdapter] Patched hw_profile.vulkan_driver: %s",
                report.loader_path,
            )
        elif hasattr(engine, "set_vulkan_lib"):
            if "set_vulkan_lib" not in snapshot:
                snapshot["set_vulkan_lib"] = True
            engine.set_vulkan_lib(report.loader_path)
        else:
            logger.warning(
                "[ameva-runtime:DiffusionAdapter] Engine has neither hw_profile nor set_vulkan_lib. Type: %s",
                type(engine).__name__,
            )

    @classmethod
    def bind(
        cls,
        engine: Any = None,
        report: Any = None,
        profile: Any = None,
        requested_backend: str | None = None,
        model_name: str = "sdxs-512-dreamshaper",
        **kwargs: Any,
    ) -> BindingResult:
        report = resolve_diagnostic_report(report, profile)
        is_vk = _is_vulkan_report(report)
        if requested_backend in ("cpu", "cpu_neon"):
            is_vk = False
        else:
            check_vulkan_availability_or_raise(
                cls.module_name,
                report,
                is_vk,
                requested_backend,
            )

        is_mali = report.vendor_id == 0x13B5 or "mali" in str(report.device_name).lower()
        is_adreno = report.vendor_id == 0x5143 or "adreno" in str(report.device_name).lower()

        config: dict[str, Any] = {
            "module": cls.module_name,
            "device_name": report.device_name,
            "vendor_id": report.vendor_id,
            "model_name": model_name,
        }
        snapshot: dict[str, Any] = {}

        if is_vk:
            config.update({
                "backend": "vulkan",
                "vulkan_lib_path": report.loader_path,
                "unet_tiling": True,
                "sd_vulkan_flag": True,
                "is_mali": is_mali,
                "is_adreno": is_adreno,
                "needs_shim": is_mali or is_adreno,
                "shim_path": cls.resolve_shim_path(),
            })

            if engine is not None:
                try:
                    threads = getattr(profile, "recommended_threads", 4) if profile else 4
                    cls._mutate_common_attributes(engine, snapshot, backend="vulkan", threads=threads)
                    cls._mutate_modality_attributes(engine, snapshot, report, **kwargs)
                except Exception as e:
                    cls._restore_engine_properties(engine, snapshot)
                    logger.error("[ameva-runtime:DiffusionAdapter] Binding error: %s", e)
                    raise AmevaRuntimeError(
                        f"[ameva-runtime:DiffusionAdapter] sd.cpp Vulkan binding failure: {e}"
                    ) from e

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
            config["offload_to_cpu"] = True
            if engine is not None:
                cls._mutate_common_attributes(engine, snapshot, backend="cpu", threads=4)
                if hasattr(engine, "hw_profile"):
                    hp = engine.hw_profile
                    if "hw_profile.vulkan_available" not in snapshot:
                        snapshot["hw_profile.vulkan_available"] = getattr(hp, "vulkan_available", False)
                    hp.vulkan_available = False
            return _make_cpu_binding(
                cls.module_name,
                report,
                config,
                reason="Explicit CPU requested" if requested_backend in ("cpu", "cpu_neon") else "Vulkan unavailable",
                restore_state=snapshot,
            )

    @classmethod
    def unbind(cls, engine: Any = None, binding_result: BindingResult | None = None) -> None:
        """Restores engine state to pre-binding configuration without silent exception swallowing."""
        logger.info("[ameva-runtime:DiffusionAdapter] Unbinding adapter and restoring resources.")
        if engine is None:
            return

        snapshot = binding_result.restore_state if binding_result else None
        if snapshot:
            if hasattr(engine, "hw_profile"):
                hp = engine.hw_profile
                if "hw_profile.vulkan_available" in snapshot:
                    hp.vulkan_available = snapshot["hw_profile.vulkan_available"]
                if "hw_profile.vulkan_driver" in snapshot:
                    hp.vulkan_driver = snapshot["hw_profile.vulkan_driver"]
            if hasattr(engine, "set_vulkan_lib") and "set_vulkan_lib" in snapshot:
                try:
                    engine.set_vulkan_lib(None)
                except Exception as e:
                    logger.warning("[ameva-runtime:DiffusionAdapter] Failed to reset vulkan_lib: %s", e)
            super().unbind(engine, binding_result)
        else:
            try:
                if hasattr(engine, "hw_profile"):
                    engine.hw_profile.vulkan_available = False
                    engine.hw_profile.vulkan_driver = None
                if hasattr(engine, "set_vulkan_lib"):
                    engine.set_vulkan_lib(None)
            except Exception as e:
                logger.warning("[ameva-runtime:DiffusionAdapter] Exception during fallback unbind: %s", e)

    @classmethod
    def build_cli_args(
        cls,
        executable: str,
        model_path: str,
        prompt: str,
        output_path: str,
        width: int = 512,
        height: int = 512,
        steps: int = 2,
        cfg_scale: float = 1.0,
        guidance: Optional[float] = None,
        threads: Any = "auto",
        target_backend: str = "auto",
        sampling_method: Optional[str] = None,
        seed: Optional[int] = None,
        vae_path: Optional[str] = None,
        vae_tiling: bool = False,
        backend: Optional[str] = None,
        is_adreno: bool = False,
    ) -> list[str]:
        """최신 sd-cli 규격에 부합하는 안전하고 검증된 Diffusion CLI 인자 목록을 조립합니다."""
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

        if guidance is not None:
            cmd.extend(["--guidance", str(guidance)])

        if target_backend == "cpu":
            cmd.append("--offload-to-cpu")
        elif backend:
            cmd.extend(["--backend", str(backend)])
        elif is_adreno or target_backend in ("vulkan", "gpu"):
            cmd.extend(["--backend", "clip=vulkan0,diffusion=vulkan0,vae=vulkan0"])

        if vae_tiling:
            cmd.append("--vae-tiling")
        if sampling_method:
            cmd.extend(["--sampling-method", str(sampling_method)])
        if seed is not None and int(seed) >= 0:
            cmd.extend(["--seed", str(seed)])
        if vae_path:
            cmd.extend(["--vae", str(vae_path)])

        return cmd
