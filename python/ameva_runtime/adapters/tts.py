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
    get_vulkan_env,
    BaseAdapter,
)
from ..exceptions import AmevaRuntimeError

logger = logging.getLogger("ameva_runtime.adapters.tts")


class TtsAdapter(BaseAdapter):
    """termux-tts (Piper TTS / VITS HiFi-GAN) Vulkan acceleration adapter."""

    module_name = "termux-tts"

    @staticmethod
    def _get_base_prefixes() -> list[Path]:
        prefixes = [Path.home()]
        prefix_env = os.environ.get("PREFIX")
        if prefix_env:
            prefixes.append(Path(prefix_env).parent / "home")
            prefixes.append(Path(prefix_env))
        return prefixes

    @classmethod
    def get_candidate_binaries(cls, binary_name: str) -> list[str]:
        candidates = [binary_name]
        for p in cls._get_base_prefixes():
            candidates.extend([
                str(p / "bin" / binary_name),
                str(p / ".local" / "bin" / binary_name),
                str(p / binary_name),
            ])
        return candidates

    CANDIDATE_VULKAN_BINARIES = [
        "sherpa-ncnn-offline-tts",
        "melo-mnn-cli",
        str(Path.home() / ".local" / "bin" / "sherpa-ncnn-offline-tts"),
        str(Path.home() / ".local" / "bin" / "melo-mnn-cli"),
        str(Path.home() / "bin" / "sherpa-ncnn-offline-tts"),
        str(Path.home() / "bin" / "melo-mnn-cli"),
    ]

    CANDIDATE_CPU_BINARIES = [
        "sherpa-onnx-offline-tts",
        str(Path.home() / ".local" / "bin" / "sherpa-onnx-offline-tts"),
        str(Path.home() / "bin" / "sherpa-onnx-offline-tts"),
    ]

    CANDIDATE_MNN_BINARIES = [
        "melo-mnn-cli",
        "mnn",
        str(Path.home() / ".local" / "bin" / "melo-mnn-cli"),
        str(Path.home() / "bin" / "melo-mnn-cli"),
    ]

    CANDIDATE_BINARIES = CANDIDATE_VULKAN_BINARIES

    STANDARD_MODEL_DIRS = [
        Path.home() / ".cache" / "termux-tts" / "models",
        Path.home() / "ncnn-vits-piper-en_US-lessac-high-fp16",
        Path.home() / "ncnn-vits-piper-en_US-amy-medium",
        Path.home() / "melo-vits-ncnn",
        Path.home() / "kokoro-int8-en-v0_19",
        Path.home() / "sherpa-onnx-supertonic-3-tts-int8-2026-05-11",
        Path.home() / "vits-melo-tts-zh_en",
        Path.home() / "vits-mimic3-ko_KO-kss_low",
        Path.home() / "melo-mnn",
    ]

    @staticmethod
    def resolve_binary_path(backend: str = "vulkan") -> Optional[str]:
        """Locate verified native TTS binary based on target backend (vulkan, mnn, or cpu)."""
        backend = (backend or "vulkan").lower()
        env_bin = os.environ.get("AMEVA_TTS_BINARY")
        if env_bin and os.path.isfile(env_bin) and (os.access(env_bin, os.X_OK) or os.name == "nt"):
            return os.path.abspath(env_bin)

        if backend == "mnn":
            candidates = TtsAdapter.CANDIDATE_MNN_BINARIES
        elif backend == "vulkan":
            candidates = TtsAdapter.CANDIDATE_VULKAN_BINARIES
        else:
            candidates = TtsAdapter.CANDIDATE_CPU_BINARIES

        for cand in candidates:
            found = shutil.which(cand) if not os.path.isabs(cand) else cand
            if found and os.path.isfile(found) and (os.access(found, os.X_OK) or os.name == "nt"):
                return os.path.abspath(found)
        return None

    @staticmethod
    def resolve_model_dir(tier: str = "high", model_type: str = "auto") -> Optional[str]:
        """Locate verified TTS model directory based on requested tier and architecture."""
        tier = (tier or "high").lower()
        model_type = (model_type or "auto").lower()

        search_dirs: list[Path] = []
        for s in TtsAdapter.STANDARD_MODEL_DIRS:
            if s.exists():
                search_dirs.append(s)

        # 1. Next-gen model specific resolution
        if model_type == "kokoro":
            for d in search_dirs:
                if (d / "model.int8.onnx").exists() or (d / "model.onnx").exists():
                    if (d / "voices.bin").exists():
                        return str(d.resolve())
            return None

        if model_type == "supertonic":
            for d in search_dirs:
                if (d / "duration_predictor.int8.onnx").exists() or (d / "duration_predictor.onnx").exists():
                    return str(d.resolve())
            return None

        if model_type in ("melo_vulkan", "melo_ncnn"):
            for d in search_dirs:
                if (d / "melo_decoder.ncnn.param").exists() and (d / "melo_decoder.ncnn.bin").exists():
                    return str(d.resolve())
                if (d / "decoder.ncnn.param").exists() and "melo" in d.name.lower():
                    return str(d.resolve())
            return None

        if model_type in ("melo_mnn", "mnn"):
            for d in search_dirs:
                if (d / "melo.mnn").exists():
                    return str(d.resolve())
                for child in d.glob("*.mnn"):
                    return str(child.parent.resolve())
            return None

        if model_type == "melo":
            # Plan 1 Priority: NCNN Vulkan Sliced Decoder
            for d in search_dirs:
                if (d / "melo_decoder.ncnn.param").exists() and (d / "melo_decoder.ncnn.bin").exists():
                    return str(d.resolve())
            # Plan 2 Priority: MNN Vulkan
            for d in search_dirs:
                if (d / "melo.mnn").exists():
                    return str(d.resolve())
            # Fallback: CPU ONNX model
            for d in search_dirs:
                if (d / "model.onnx").exists() and (d / "lexicon.txt").exists():
                    return str(d.resolve())
            return None

        if model_type in ("kss", "korean"):
            for d in search_dirs:
                if (d / "ko_KO-kss_low.onnx").exists():
                    return str(d.resolve())
            return None

        # 2. VITS NCNN Vulkan tier resolution (default)
        preferred_keyword = "amy-medium" if tier == "medium" else "lessac-high"
        sorted_dirs: list[Path] = []
        for s in search_dirs:
            if preferred_keyword in s.name.lower():
                sorted_dirs.insert(0, s)
            else:
                sorted_dirs.append(s)
            for child in s.glob("ncnn-vits*"):
                if child.is_dir():
                    if preferred_keyword in child.name.lower():
                        sorted_dirs.insert(0, child)
                    else:
                        sorted_dirs.append(child)

        for d in sorted_dirs:
            if (d / "config.json").exists() and (d / "decoder.ncnn.bin").exists():
                return str(d.resolve())
        return None

    @classmethod
    def get_execution_env(
        cls,
        extra_env: Optional[dict[str, str]] = None,
        is_mali: bool = False,
    ) -> dict[str, str]:
        """Assemble environment variables conforming to Golden Link Order with Mali DSP flags."""
        base_env = dict(os.environ)
        if extra_env:
            base_env.update(extra_env)
        env = get_vulkan_env(base_env)

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

        # User-selected tier without heuristic device-specific forced overrides
        model_tier = (kwargs.get("tier") or kwargs.get("model_tier") or "high").lower()

        config: dict = {
            "module": TtsAdapter.module_name,
            "device_name": report.device_name,
            "vendor_id": report.vendor_id,
            "is_mali": is_mali,
            "is_adreno": is_adreno,
            "model_tier": model_tier,
        }

        model_type = kwargs.get("model_type", "auto")

        if is_vk:
            effective_model_type = "vits"
            if model_type in ("melo", "melo_vulkan", "melo_ncnn", "melo_mnn"):
                dir_ncnn = TtsAdapter.resolve_model_dir(tier=model_tier, model_type="melo_vulkan")
                dir_mnn = TtsAdapter.resolve_model_dir(tier=model_tier, model_type="melo_mnn")
                if model_type in ("melo_vulkan", "melo_ncnn"):
                    effective_model_type = "melo_vulkan"
                elif model_type in ("melo_mnn", "mnn"):
                    effective_model_type = "melo_mnn"
                elif dir_ncnn is not None:
                    # Plan 1 priority (HiFi-GAN NCNN Vulkan Slicing)
                    effective_model_type = "melo_vulkan"
                elif dir_mnn is not None:
                    # Plan 2 fallback (MNN Vulkan)
                    effective_model_type = "melo_mnn"
                else:
                    # Default to Plan 1 for fail-fast
                    effective_model_type = "melo_vulkan"

            target_backend = "mnn" if "mnn" in effective_model_type else "vulkan"
            target_binary = TtsAdapter.resolve_binary_path(target_backend)
            target_model_dir = TtsAdapter.resolve_model_dir(tier=model_tier, model_type=effective_model_type)

            config.update({
                "backend": "vulkan",
                "vulkan_lib_path": getattr(report, "loader_path", "/system/lib64/libvulkan.so"),
                "binary_path": target_binary,
                "model_dir": target_model_dir,
                "model_type": effective_model_type,
                "strategy": "ncnn_sliced" if effective_model_type == "melo_vulkan" else ("mnn_vulkan" if effective_model_type == "melo_mnn" else "sherpa_vits"),
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
            config["model_type"] = model_type
            config["binary_path"] = TtsAdapter.resolve_binary_path("cpu")
            config["model_dir"] = TtsAdapter.resolve_model_dir(tier="balanced", model_type=model_type)
            if engine is not None:
                try:
                    if hasattr(engine, "device"):
                        engine.device = "cpu"
                    if hasattr(engine, "model_tier"):
                        engine.model_tier = "balanced"
                    if hasattr(engine, "model_type"):
                        engine.model_type = model_type
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
