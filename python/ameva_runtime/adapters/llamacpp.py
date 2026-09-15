"""
LlamaCppAdapter — termux-llamacpp (GGUF LLM) Vulkan Acceleration Adapter
"""
from __future__ import annotations

import logging
import os
from typing import Any

from .base import (
    _is_vulkan_report,
    _make_cpu_binding,
    check_vulkan_availability_or_raise,
    _MALI_VENDOR_ID,
    DiagnosticReport,
    BindingResult,
    resolve_diagnostic_report,
    get_vulkan_env,
    BaseAdapter,
)
from ..exceptions import AmevaRuntimeError, PlatformNotSupportedError

logger = logging.getLogger("ameva_runtime.adapters.llamacpp")

CPU_FALLBACK_PATTERNS = (
    "no gpu found",
    "using cpu backend",
    "falling back to cpu",
    "fallback to cpu",
    "vulkan initialization failed",
    "failed to initialize vulkan",
    "no vulkan device",
    "backend unavailable",
    "ggml_vulkan: failed",
    "ggml_vulkan: cannot",
)

VULKAN_INIT_PATTERNS = (
    "ggml_vulkan: using device",
    "ggml_vulkan: found 1 vulkan device",
    "ggml_vulkan: found",
)


def verify_vulkan_llm_output(output: str) -> None:
    """Strict Zero-Silent-Fallback: Validates inference output stream against CPU fallback.
    
    Raises:
        PlatformNotSupportedError: If CPU fallback or absence of Vulkan initialization is detected.
    """
    lowered = output.lower()
    for pat in CPU_FALLBACK_PATTERNS:
        if pat in lowered:
            raise PlatformNotSupportedError(
                f"[termux-llamacpp] CPU fallback detected under Vulkan mode: '{pat}'.\n"
                f"Execution halted strictly under Zero-Silent-Fallback policy."
            )
    if not any(pat in lowered for pat in VULKAN_INIT_PATTERNS):
        raise PlatformNotSupportedError(
            "[termux-llamacpp] No positive Vulkan initialization logged during LLM inference.\n"
            "Execution halted strictly without silent CPU fallback."
        )


def _calculate_llama_layers(engine: Any = None, requested_ngl: int | None = None) -> int:
    """Determines GPU offload layers for GGUF LLMs.
    
    Zero-String-Heuristics Principle:
    - If user explicitly specified requested_ngl, enforce it unconditionally (caller assumes full responsibility for OOM).
    - If engine config specifies ngl, use that value directly.
    - If no layer count was specified, default to 999 (delegates to upstream llama.cpp automatic clamp).
    """
    if requested_ngl is not None:
        return int(requested_ngl)
    if engine is not None:
        if isinstance(engine, dict) and "ngl" in engine and engine["ngl"] is not None:
            return int(engine["ngl"])
        if hasattr(engine, "n_gpu_layers") and getattr(engine, "n_gpu_layers", None) is not None:
            return int(engine.n_gpu_layers)
        if hasattr(engine, "ngl") and getattr(engine, "ngl", None) is not None:
            return int(engine.ngl)
        if hasattr(engine, "n_layers") and getattr(engine, "n_layers", None) is not None:
            return int(engine.n_layers)
    return 999


class LlamaCppAdapter(BaseAdapter):
    """termux-llamacpp (llama.cpp GGUF) Vulkan acceleration adapter."""

    module_name = "termux-llamacpp"

    @classmethod
    def get_execution_environment(
        cls,
        base_env: dict[str, str] | None = None,
    ) -> dict[str, str]:
        """Provides verified execution environment adhering to Golden Link Order."""
        env = get_vulkan_env(base_env)
        env.setdefault("GGML_VULKAN_SKIP_CHECKS", "999999999")
        return env

    @staticmethod
    def bind(
        engine: Any = None,
        report: Any = None,
        profile: Any = None,
        requested_backend: str | None = None,
        requested_ngl: int | None = None,
        **kwargs: Any,
    ) -> BindingResult:
        report = resolve_diagnostic_report(report, profile)
        is_vk = _is_vulkan_report(report)

        # Strict Rule: Only 'auto' or None allows adaptive CPU routing when Vulkan is unavailable.
        # Explicit 'vulkan' or 'gpu' strictly refuses any CPU fallback and raises PlatformNotSupportedError immediately.
        req_norm = str(requested_backend or "").lower().strip()
        if req_norm in ("vulkan", "gpu"):
            check_vulkan_availability_or_raise(
                LlamaCppAdapter.module_name,
                report,
                is_vk,
                "vulkan",
            )
            is_vk = True
        elif req_norm in ("cpu", "cpu_neon"):
            is_vk = False
        else:
            # req_norm is 'auto' or None: strictly adaptive routing based on hardware profile
            pass

        config: dict = {
            "module": LlamaCppAdapter.module_name,
            "device_name": report.device_name,
            "vendor_id": report.vendor_id,
        }

        if is_vk:
            cpu_cores = os.cpu_count() or 8
            big_cores = max(1, cpu_cores // 2)
            ngl = _calculate_llama_layers(engine, requested_ngl=requested_ngl)
            is_mali = (report.vendor_id == _MALI_VENDOR_ID or "mali" in str(report.device_name or "").lower())
            if is_mali or (report and getattr(report, "overall_success", False)):
                config["system_icd_prioritized"] = True
                config["bridge_active"] = False

            config.update({
                "backend": "vulkan",
                "ngl": ngl,
                "device_flag": "vulkan",
                "flash_attn": True,
                "ctx_size": 2048,
                "threads": big_cores,
                "mali_align": is_mali,
            })

            if engine is not None:
                try:
                    if isinstance(engine, dict):
                        engine.setdefault("ngl", ngl)
                        engine["device"] = "vulkan"
                        engine.setdefault("flash_attn", True)
                        engine.setdefault("threads", big_cores)
                        engine.setdefault("env", {})["LD_LIBRARY_PATH"] = os.environ.get("LD_LIBRARY_PATH", "")
                    elif hasattr(engine, "config"):
                        cfg = engine.config
                        if isinstance(cfg, dict):
                            cfg.setdefault("ngl", ngl)
                            cfg.setdefault("n_gpu_layers", ngl)
                            cfg["device"] = "vulkan"
                            cfg.setdefault("flash_attn", True)
                            cfg.setdefault("threads", big_cores)
                        else:
                            if hasattr(cfg, "n_gpu_layers"):
                                cfg.n_gpu_layers = ngl
                            if hasattr(cfg, "ngl"):
                                cfg.ngl = ngl
                            if hasattr(cfg, "device"):
                                cfg.device = "vulkan"
                            if hasattr(cfg, "flash_attn"):
                                cfg.flash_attn = True
                            if hasattr(cfg, "threads") and getattr(cfg, "threads", 0) == 0:
                                cfg.threads = big_cores
                    elif hasattr(engine, "ngl") or hasattr(engine, "n_gpu_layers"):
                        if hasattr(engine, "ngl") and getattr(engine, "ngl", 0) == 0:
                            engine.ngl = ngl
                        if hasattr(engine, "n_gpu_layers") and getattr(engine, "n_gpu_layers", 0) == 0:
                            engine.n_gpu_layers = ngl
                        if hasattr(engine, "device"):
                            engine.device = "vulkan"
                        if hasattr(engine, "threads") and getattr(engine, "threads", 0) == 0:
                            engine.threads = big_cores
                    elif isinstance(engine, list):
                        if "-ngl" not in engine and "--n-gpu-layers" not in engine:
                            engine.extend(["-ngl", str(ngl)])
                        if "--device" not in engine and "-dev" not in engine:
                            engine.extend(["--device", "vulkan"])
                        if "-t" not in engine and "--threads" not in engine:
                            engine.extend(["-t", str(big_cores)])
                    logger.info(
                        "[ameva-runtime:LlamaCppAdapter] Injected -ngl %d -t %d --device vulkan"
                        " (device=%s, is_mali=%s)", ngl, big_cores, report.device_name, is_mali
                    )
                except Exception as e:
                    logger.error("[ameva-runtime:LlamaCppAdapter] Binding error: %s", e)
                    raise AmevaRuntimeError(
                        f"[ameva-runtime:LlamaCppAdapter] llama.cpp Vulkan binding failure: {e}"
                    ) from e

            return BindingResult(
                module=LlamaCppAdapter.module_name,
                backend="vulkan",
                is_vulkan=True,
                device_name=report.device_name,
                vendor_id=report.vendor_id,
                config=config,
                status="BOUND",
            )
        else:
            cpu_cores = os.cpu_count() or 8
            big_cores = max(1, cpu_cores // 2)
            config["ngl"] = 0
            config["threads"] = big_cores
            if engine is not None:
                if isinstance(engine, dict):
                    engine["ngl"] = 0
                    engine.setdefault("threads", big_cores)
                elif hasattr(engine, "config"):
                    cfg = engine.config
                    if isinstance(cfg, dict):
                        cfg["ngl"] = 0
                        cfg["n_gpu_layers"] = 0
                        cfg.setdefault("threads", big_cores)
                    else:
                        if hasattr(cfg, "n_gpu_layers"):
                            cfg.n_gpu_layers = 0
                        if hasattr(cfg, "ngl"):
                            cfg.ngl = 0
                        if hasattr(cfg, "threads"):
                            cfg.threads = big_cores
                elif hasattr(engine, "ngl") or hasattr(engine, "n_gpu_layers"):
                    if hasattr(engine, "ngl"):
                        engine.ngl = 0
                    if hasattr(engine, "n_gpu_layers"):
                        engine.n_gpu_layers = 0
                    if hasattr(engine, "threads"):
                        engine.threads = big_cores
                elif isinstance(engine, list):
                    if "-t" not in engine and "--threads" not in engine:
                        engine.extend(["-t", str(big_cores)])
            return _make_cpu_binding(
                LlamaCppAdapter.module_name,
                report,
                config,
                reason="Explicit CPU requested" if requested_backend in ("cpu", "cpu_neon") else "Vulkan unavailable",
            )

    @staticmethod
    def unbind(engine: Any = None) -> None:
        logger.info("[ameva-runtime:LlamaCppAdapter] Unbinding adapter and resetting resources.")
        if engine is not None:
            try:
                if isinstance(engine, dict):
                    engine["ngl"] = 0
                    engine["device"] = "cpu"
                elif hasattr(engine, "config"):
                    cfg = engine.config
                    if isinstance(cfg, dict):
                        cfg["ngl"] = 0
                        cfg["n_gpu_layers"] = 0
                        cfg["device"] = "cpu"
                    else:
                        if hasattr(cfg, "n_gpu_layers"):
                            cfg.n_gpu_layers = 0
                        if hasattr(cfg, "ngl"):
                            cfg.ngl = 0
                        if hasattr(cfg, "device"):
                            cfg.device = "cpu"
                elif hasattr(engine, "ngl") or hasattr(engine, "n_gpu_layers"):
                    if hasattr(engine, "ngl"):
                        engine.ngl = 0
                    if hasattr(engine, "n_gpu_layers"):
                        engine.n_gpu_layers = 0
                    if hasattr(engine, "device"):
                        engine.device = "cpu"
            except Exception as e:
                logger.debug("[ameva-runtime:LlamaCppAdapter] Ignored exception during unbind: %s", e)

    @classmethod
    def build_cli_args(
        cls,
        executable: str,
        model_path: str,
        prompt: Optional[str] = None,
        prompt_file: Optional[str] = None,
        target_backend: str = "auto",
        threads: Any = "auto",
        context_limit: int = 2048,
        max_tokens: int = 256,
        temperature: float = 0.2,
        repeat_penalty: Optional[float] = 1.1,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        seed: Optional[int] = None,
        device_name: Optional[str] = None,
        ngl_override: Optional[int] = None,
        no_display_prompt: bool = True,
    ) -> list[str]:
        """Assembles verified LLM CLI argument list conforming to modern llama-cli specifications."""
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
            "-m", str(model_path),
            "-t", thread_val,
            "-c", str(context_limit),
            "-n", str(max_tokens),
            "--temp", str(temperature),
            "-ngl", ngl_val,
        ]

        if prompt_file:
            cmd.extend(["-f", str(prompt_file)])
        elif prompt:
            cmd.extend(["-p", str(prompt)])

        if target_backend in ("vulkan", "gpu") and device_name:
            cmd.extend(["--device", device_name])

        if no_display_prompt:
            cmd.append("--no-display-prompt")

        if repeat_penalty is not None:
            cmd.extend(["--repeat-penalty", str(repeat_penalty)])
        if top_p is not None:
            cmd.extend(["--top-p", str(top_p)])
        if top_k is not None:
            cmd.extend(["--top-k", str(top_k)])
        if seed is not None:
            cmd.extend(["-s", str(seed)])

        return cmd

