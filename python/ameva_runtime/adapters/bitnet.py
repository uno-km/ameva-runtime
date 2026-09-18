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
from .base import (
    _is_vulkan_report,
    _make_cpu_binding,
    check_vulkan_availability_or_raise,
    _MALI_VENDOR_ID,
    resolve_diagnostic_report,
    get_vulkan_env,
    BaseAdapter,
    resolve_authoritative_binary,
)

logger = logging.getLogger("ameva_runtime.adapters.bitnet")


def _calculate_bitnet_layers(engine: Any) -> int:
    """Dynamically determines optimal GPU offload layers for BitNet 1.58-bit models."""
    if engine is not None:
        if hasattr(engine, "n_gpu_layers") and getattr(engine, "n_gpu_layers", 0) > 0:
            return int(engine.n_gpu_layers)
        if hasattr(engine, "config") and hasattr(engine.config, "n_gpu_layers") and engine.config.n_gpu_layers > 0:
            return int(engine.config.n_gpu_layers)
        if hasattr(engine, "n_layers") and getattr(engine, "n_layers", 0) > 0:
            return int(engine.n_layers)
        model_name = str(getattr(engine, "model_name", "") or getattr(engine, "model", "")).lower()
        if "0.7b" in model_name:
            return 16
        elif "1.3b" in model_name:
            return 24
        elif "3b" in model_name or "2.7b" in model_name:
            return 33
    return 33


class BitnetAdapter(BaseAdapter):
    """termux-bitnet (BitNet 1.58-bit i2_s) Vulkan acceleration adapter."""

    module_name = "termux-bitnet"

    @classmethod
    def get_execution_environment(
        cls,
        base_env: dict[str, str] | None = None,
        tune_mali: bool = False,
        **kwargs: Any,
    ) -> dict[str, str]:
        """Provides verified BitNet execution environment conforming to Golden Link Order."""
        return super().get_execution_environment(base_env, tune_mali=tune_mali, **kwargs)

    @classmethod
    def _mutate_modality_attributes(
        cls,
        engine: Any,
        snapshot: dict[str, Any],
        report: DiagnosticReport,
        **kwargs: Any,
    ) -> None:
        """Tier 2: BitNet specific mutation."""
        ngl = _calculate_bitnet_layers(engine)
        mali_align_required = (report.vendor_id == _MALI_VENDOR_ID or
                               "Mali" in (report.device_name or ""))

        cls._set_engine_property(engine, snapshot, "n_gpu_layers", ngl)
        cls._set_engine_property(engine, snapshot, "ngl", ngl)
        cls._set_engine_property(engine, snapshot, "flash_attn", True)
        if mali_align_required:
            cls._set_engine_property(engine, snapshot, "mali_128byte_align", True)

        if isinstance(engine, list):
            if "cli_args_len" not in snapshot:
                snapshot["cli_args_len"] = len(engine)
            if "-ngl" not in engine and "--n-gpu-layers" not in engine:
                engine.extend(["-ngl", str(ngl)])
            if "-fa" not in engine and "--flash-attn" not in engine:
                engine.append("-fa")

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

        config: dict[str, Any] = {
            "module": cls.module_name,
            "device_name": report.device_name,
            "vendor_id": report.vendor_id,
        }
        snapshot: dict[str, Any] = {}

        if is_vk:
            ngl = _calculate_bitnet_layers(engine)
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
                    threads = getattr(profile, "recommended_threads", 4) if profile else 4
                    cls._mutate_common_attributes(engine, snapshot, backend="vulkan", threads=threads)
                    cls._mutate_modality_attributes(engine, snapshot, report, **kwargs)
                except Exception as e:
                    cls._restore_engine_properties(engine, snapshot)
                    logger.error("[%s] Binding error: %s", cls.module_name, e)
                    raise AmevaRuntimeError(
                        f"[{cls.module_name}] BitNetEngine Vulkan binding failure: {e}"
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
            cpu_threads = max(1, (os.cpu_count() or 8) // 2)
            config["n_threads"] = cpu_threads
            config["kernel"] = "neon_dotprod"
            if engine is not None:
                cls._mutate_common_attributes(engine, snapshot, backend="cpu", threads=cpu_threads)
                cls._set_engine_property(engine, snapshot, "n_gpu_layers", 0)
                cls._set_engine_property(engine, snapshot, "ngl", 0)

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
        super().unbind(engine, binding_result)

    @classmethod
    def resolve_binary_path(cls, binary_name: str = "termux-bitnet-cli") -> str:
        """Resolves authoritative BitNet CLI binary path ($PREFIX/bin/termux-bitnet-cli)."""
        return str(resolve_authoritative_binary(binary_name, "bitnet"))

    @classmethod
    def resolve_library_path(cls, lib_name: str = "libtermux_bitnet.so") -> str:
        """Resolves authoritative BitNet C ABI shared library ($PREFIX/lib/libtermux_bitnet.so)."""
        import os
        from pathlib import Path
        prefix = Path(os.environ.get("PREFIX", "/data/data/com.termux/files/usr"))
        primary_so = prefix / "lib" / lib_name
        if primary_so.is_file():
            return str(primary_so.resolve())

        home = Path(os.environ.get("HOME", os.path.expanduser("~")))
        isolated_so = home / ".local" / "share" / "ameva" / "current" / "bitnet" / "lib" / lib_name
        if isolated_so.is_file():
            return str(isolated_so.resolve())

        raise AmevaRuntimeError(
            f"[{cls.module_name}] Authoritative BitNet shared library not found at '{primary_so}'. "
            f"Remediation: Provision official AMEVA assets using 'python -m ameva_runtime.installer --asset bitnet'."
        )

    @classmethod
    def build_cli_args(
        cls,
        executable: str,
        model_path: str,
        prompt: str | None = None,
        prompt_file: str | None = None,
        target_backend: str = "auto",
        threads: Any = "auto",
        context_limit: int = 2048,
        batch_size: int = 512,
        ubatch_size: int = 512,
        max_tokens: int = 128,
        temperature: float = 0.7,
        top_p: float = 0.95,
        top_k: int = 40,
        min_p: float = 0.05,
        typical_p: float = 1.0,
        repeat_penalty: float = 1.15,
        repeat_last_n: int = 64,
        freq_penalty: float = 0.0,
        presence_penalty: float = 0.0,
        seed: int = 0,
        ngl_override: int | None = None,
        flash_attn: bool = False,
        system_prompt: str | None = None,
        stop_tokens: str | None = None,
        verbose: bool = False,
    ) -> list[str]:
        """Assembles verified BitNet CLI argument list conforming to modern specifications."""
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
            ngl_val = "33"
        else:
            ngl_val = "33"

        cmd = [
            str(executable),
            "-m", str(model_path),
            "-t", thread_val,
            "-c", str(context_limit),
            "-b", str(batch_size),
            "-ub", str(ubatch_size),
            "-n", str(max_tokens),
            "--temp", str(temperature),
            "--top-p", str(top_p),
            "--top-k", str(top_k),
            "--min-p", str(min_p),
            "--typical", str(typical_p),
            "--repeat-penalty", str(repeat_penalty),
            "--repeat-last-n", str(repeat_last_n),
            "--freq-penalty", str(freq_penalty),
            "--presence-penalty", str(presence_penalty),
            "-ngl", ngl_val,
            "--simple-io",
            "--no-warmup",
        ]

        if prompt_file:
            cmd.extend(["-f", str(prompt_file)])
        elif prompt:
            cmd.extend(["-p", str(prompt)])

        if seed != 0:
            cmd.extend(["-s", str(seed)])

        if target_backend in ("vulkan", "gpu"):
            cmd.extend(["--device", "vulkan"])
        elif target_backend == "cpu":
            cmd.extend(["--device", "cpu"])

        if flash_attn:
            cmd.append("-fa")

        if system_prompt:
            cmd.extend(["--system-prompt", str(system_prompt)])

        if stop_tokens:
            for st in stop_tokens.split(","):
                st = st.strip()
                if st:
                    cmd.extend(["-r", st])

        if verbose:
            cmd.append("--verbose")

        return cmd
