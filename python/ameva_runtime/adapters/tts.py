"""
TtsAdapter — termux-tts (Piper / VITS HiFi-GAN) Vulkan Acceleration Adapter.
Complete Enterprise Hardware Delegation Adapter matching DiffusionAdapter standards.
"""
from __future__ import annotations

import logging
import os
import shutil
from pathlib import Path
from typing import Any, Optional

from .base import (
    _is_vulkan_report,
    _make_cpu_binding,
    check_vulkan_availability_or_raise,
    DiagnosticReport,
    BindingResult,
    resolve_diagnostic_report,
    BaseAdapter,
)
from ..exceptions import AmevaRuntimeError

logger = logging.getLogger("ameva_runtime.adapters.tts")


class TtsAdapter(BaseAdapter):
    """termux-tts (Piper TTS / VITS HiFi-GAN) Vulkan acceleration adapter."""

    module_name = "termux-tts"

    CANDIDATE_BINARIES = [
        "sherpa-ncnn-offline-tts",
        str(Path.home() / ".local" / "bin" / "sherpa-ncnn-offline-tts"),
        str(Path.home() / "sherpa-ncnn" / "build-vulkan" / "bin" / "sherpa-ncnn-offline-tts"),
        "/data/data/com.termux/files/home/.local/bin/sherpa-ncnn-offline-tts",
        "/data/data/com.termux/files/home/sherpa-ncnn/build-vulkan/bin/sherpa-ncnn-offline-tts",
        "/data/data/com.termux/files/usr/bin/sherpa-ncnn-offline-tts",
    ]

    STANDARD_MODEL_DIRS = [
        Path.home() / ".cache" / "termux-tts" / "models",
        Path.home() / "ncnn-vits-piper-en_US-lessac-high-fp16",
        Path.home() / "ncnn-vits-piper-en_US-amy-medium",
        Path("/data/data/com.termux/files/home/ncnn-vits-piper-en_US-lessac-high-fp16"),
        Path("/data/data/com.termux/files/home/ncnn-vits-piper-en_US-amy-medium"),
        Path("/data/data/com.termux/files/home/.cache/termux-tts/models"),
    ]

    @staticmethod
    def resolve_binary_path() -> Optional[str]:
        """Locate verified native Vulkan TTS binary (sherpa-ncnn-offline-tts)."""
        env_bin = os.environ.get("AMEVA_TTS_BINARY")
        if env_bin and os.path.isfile(env_bin) and (os.access(env_bin, os.X_OK) or os.name == "nt"):
            return os.path.abspath(env_bin)

        for cand in TtsAdapter.CANDIDATE_BINARIES:
            found = shutil.which(cand) if not os.path.isabs(cand) else cand
            if found and os.path.isfile(found) and (os.access(found, os.X_OK) or os.name == "nt"):
                return os.path.abspath(found)
        return None

    @staticmethod
    def resolve_model_dir(tier: str = "high") -> Optional[str]:
        """Locate verified VITS NCNN model directory based on requested tier."""
        tier = (tier or "high").lower()
        preferred_keyword = "amy-medium" if tier == "medium" else "lessac-high"

        search_dirs: list[Path] = []
        for s in TtsAdapter.STANDARD_MODEL_DIRS:
            if s.exists():
                if preferred_keyword in s.name.lower():
                    search_dirs.insert(0, s)
                else:
                    search_dirs.append(s)
                for child in s.glob("ncnn-vits*"):
                    if child.is_dir():
                        if preferred_keyword in child.name.lower():
                            search_dirs.insert(0, child)
                        else:
                            search_dirs.append(child)

        for d in search_dirs:
            if (d / "config.json").exists() and (d / "decoder.ncnn.bin").exists():
                return str(d.resolve())
        return None

    @classmethod
    def get_execution_env(
        cls,
        extra_env: Optional[dict[str, str]] = None,
        is_mali: bool = False,
    ) -> dict[str, str]:
        """Assemble environment variables including Android driver preloads and Mali DSP flags."""
        env = dict(os.environ)
        if extra_env:
            env.update(extra_env)

        # Android Bionic Vulkan loader search path
        if os.path.exists("/system/lib64/libvulkan.so"):
            cur_ld = env.get("LD_LIBRARY_PATH", "")
            if not cur_ld.startswith("/system/lib64"):
                env["LD_LIBRARY_PATH"] = f"/system/lib64:{cur_ld}".rstrip(":")

        if is_mali:
            env["AMEVA_VK_DSP_ACCEL"] = "1"

        return env

    @classmethod
    def build_cli_args(
        cls,
        executable: str,
        model_dir: str,
        text: str,
        output_filename: str,
        threads: int = 1,
        use_vulkan: bool = True,
    ) -> list[str]:
        """Build standard CLI arguments for sherpa-ncnn-offline-tts execution."""
        return [
            str(executable),
            f"--vits-model-dir={model_dir}",
            f"--use-vulkan-compute={1 if use_vulkan else 0}",
            f"--num-threads={threads}",
            f"--output-filename={output_filename}",
            str(text).strip(),
        ]

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
                TtsAdapter.module_name,
                report,
                is_vk,
                requested_backend,
            )

        is_mali = report.vendor_id == 0x13B5 or "mali" in str(report.device_name).lower()
        is_adreno = report.vendor_id == 0x5143 or "adreno" in str(report.device_name).lower()

        # Adaptive dual-tier model selection
        model_tier = "medium" if is_mali else "high"

        config: dict = {
            "module": TtsAdapter.module_name,
            "device_name": report.device_name,
            "vendor_id": report.vendor_id,
            "is_mali": is_mali,
            "is_adreno": is_adreno,
            "model_tier": model_tier,
        }

        if is_vk:
            config.update({
                "backend": "vulkan",
                "vulkan_lib_path": getattr(report, "loader_path", "/system/lib64/libvulkan.so"),
                "binary_path": TtsAdapter.resolve_binary_path(),
                "model_dir": TtsAdapter.resolve_model_dir(model_tier),
                "dsp_accel": is_mali,
                "subgroup64": is_adreno,
            })

            if engine is not None:
                try:
                    if hasattr(engine, "device"):
                        engine.device = "vulkan"
                    if hasattr(engine, "model_tier"):
                        engine.model_tier = model_tier
                    if hasattr(engine, "backend"):
                        engine.backend = "vulkan"
                    if hasattr(engine, "use_vulkan"):
                        engine.use_vulkan = True
                    if hasattr(engine, "threads"):
                        engine.threads = getattr(profile, "recommended_threads", 4)
                    logger.info(
                        "[TtsAdapter] VITS Vulkan GPU bound successfully (device=%s, tier=%s)",
                        report.device_name, model_tier
                    )
                except Exception as e:
                    logger.error("[TtsAdapter] Binding error: %s", e)
                    raise AmevaRuntimeError(f"[TtsAdapter] TTS Vulkan binding failure: {e}") from e

            return BindingResult(
                module=TtsAdapter.module_name,
                backend="vulkan",
                is_vulkan=True,
                device_name=report.device_name,
                vendor_id=report.vendor_id,
                config=config,
                status="BOUND_VULKAN",
            )
        else:
            config["offload_to_cpu"] = True
            config["model_tier"] = "balanced"
            if engine is not None:
                try:
                    if hasattr(engine, "device"):
                        engine.device = "cpu"
                    if hasattr(engine, "model_tier"):
                        engine.model_tier = "balanced"
                except Exception:
                    pass
            return _make_cpu_binding(
                TtsAdapter.module_name,
                report,
                config,
                reason="Explicit CPU requested" if requested_backend in ("cpu", "cpu_neon") else "Vulkan unavailable",
            )

    @staticmethod
    def unbind(engine: Any = None) -> None:
        logger.info("[TtsAdapter] Unbinding and resetting resources.")
        if engine is not None:
            try:
                if hasattr(engine, "device"):
                    engine.device = "cpu"
                if hasattr(engine, "use_vulkan"):
                    engine.use_vulkan = False
            except Exception as e:
                logger.debug("[TtsAdapter] Ignored exception during unbind: %s", e)
