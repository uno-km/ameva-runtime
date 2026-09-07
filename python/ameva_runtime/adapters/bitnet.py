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
    BaseAdapter,
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
            return 32
    return 32


class BitnetAdapter(BaseAdapter):
    """termux-bitnet (BitNet 1.58-bit i2_s) Vulkan acceleration adapter."""

    module_name = "termux-bitnet"

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
                BitnetAdapter.module_name,
                report,
                is_vk,
                requested_backend,
            )

        config: dict = {
            "module": BitnetAdapter.module_name,
            "device_name": report.device_name,
            "vendor_id": report.vendor_id,
        }

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
                    if isinstance(engine, dict):
                        engine.setdefault("ngl", ngl)
                        engine.setdefault("n_gpu_layers", ngl)
                        engine["device"] = "vulkan"
                        engine.setdefault("flash_attn", True)
                    elif hasattr(engine, "config"):
                        cfg = engine.config
                        if isinstance(cfg, dict):
                            cfg["n_gpu_layers"] = ngl
                            cfg["device"] = "vulkan"
                            cfg["flash_attn"] = True
                        else:
                            if hasattr(cfg, "n_gpu_layers"):
                                cfg.n_gpu_layers = ngl
                            if hasattr(cfg, "device"):
                                cfg.device = "vulkan"
                            if hasattr(cfg, "flash_attn"):
                                cfg.flash_attn = True
                        logger.info(
                            "[ameva-runtime:BitnetAdapter] Set config.n_gpu_layers=%d"
                            " (device=%s, mali_align=%s)", ngl, report.device_name, mali_align_required
                        )
                    elif hasattr(engine, "n_gpu_layers"):
                        engine.n_gpu_layers = ngl
                        if hasattr(engine, "device"):
                            engine.device = "vulkan"
                        if hasattr(engine, "flash_attn"):
                            engine.flash_attn = True
                    elif isinstance(engine, list):
                        if "-ngl" not in engine and "--n-gpu-layers" not in engine:
                            engine.extend(["-ngl", str(ngl)])
                        if "--device" not in engine and "-d" not in engine:
                            engine.extend(["--device", "vulkan"])
                        if "-fa" not in engine and "--flash-attn" not in engine:
                            engine.append("-fa")
                except Exception as e:
                    logger.error("[ameva-runtime:BitnetAdapter] Binding error: %s", e)
                    raise AmevaRuntimeError(
                        f"[ameva-runtime:BitnetAdapter] BitNetEngine Vulkan binding failure: {e}"
                    ) from e

            return BindingResult(
                module=BitnetAdapter.module_name,
                backend="vulkan",
                is_vulkan=True,
                device_name=report.device_name,
                vendor_id=report.vendor_id,
                config=config,
                status="BOUND",
            )
        else:
            config["n_threads"] = max(1, (os.cpu_count() or 8) // 2)
            config["kernel"] = "neon_dotprod"
            if engine is not None:
                if isinstance(engine, dict):
                    engine["n_gpu_layers"] = 0
                    engine["device"] = "cpu"
                elif hasattr(engine, "config"):
                    cfg = engine.config
                    if isinstance(cfg, dict):
                        cfg["n_gpu_layers"] = 0
                        cfg["device"] = "cpu"
                    else:
                        if hasattr(cfg, "n_gpu_layers"):
                            cfg.n_gpu_layers = 0
                        if hasattr(cfg, "device"):
                            cfg.device = "cpu"
                elif hasattr(engine, "n_gpu_layers"):
                    engine.n_gpu_layers = 0
                    if hasattr(engine, "device"):
                        engine.device = "cpu"
            return _make_cpu_binding(
                BitnetAdapter.module_name,
                report,
                config,
                reason="Explicit CPU requested" if requested_backend in ("cpu", "cpu_neon") else "Vulkan unavailable",
            )

    @staticmethod
    def unbind(engine: Any = None) -> None:
        logger.info("[ameva-runtime:BitnetAdapter] Unbinding adapter and resetting resources.")
        if engine is not None:
            try:
                if isinstance(engine, dict):
                    engine["n_gpu_layers"] = 0
                    engine["device"] = "cpu"
                elif hasattr(engine, "config"):
                    cfg = engine.config
                    if isinstance(cfg, dict):
                        cfg["n_gpu_layers"] = 0
                        cfg["device"] = "cpu"
                    else:
                        if hasattr(cfg, "n_gpu_layers"):
                            cfg.n_gpu_layers = 0
                        if hasattr(cfg, "device"):
                            cfg.device = "cpu"
                        if hasattr(cfg, "flash_attn"):
                            cfg.flash_attn = False
                elif hasattr(engine, "n_gpu_layers"):
                    engine.n_gpu_layers = 0
                    if hasattr(engine, "device"):
                        engine.device = "cpu"
            except Exception as e:
                logger.debug("[ameva-runtime:BitnetAdapter] Ignored exception during unbind: %s", e)

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
