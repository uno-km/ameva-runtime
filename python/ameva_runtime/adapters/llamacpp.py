"""
LlamaCppAdapter — termux-llamacpp (GGUF LLM) Vulkan Acceleration Adapter
Zero-Silent-Fallback and Anti-Deception Compliant Engine Orchestrator
"""
from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional, Union

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

FULL_OFFLOAD_SENTINEL = 999  # Upstream llama.cpp sentinel value triggering automatic clamp to total model layers
DEFAULT_GPU_LAYERS = FULL_OFFLOAD_SENTINEL
ALLOWED_BACKENDS = {"auto", "vulkan", "gpu", "cpu", "cpu_neon"}

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


def _extract_cli_ngl(args: list[str]) -> Optional[int]:
    """Extracts existing -ngl or --n-gpu-layers value from CLI arguments list."""
    for flag in ("-ngl", "--n-gpu-layers"):
        if flag in args:
            idx = args.index(flag)
            if idx + 1 < len(args):
                try:
                    return int(args[idx + 1])
                except ValueError:
                    return None
    return None


def verify_vulkan_llm_output(output: str, returncode: int = 0) -> None:
    """Strict Zero-Silent-Fallback: Validates inference output stream and return code against CPU fallback.

    Raises:
        AmevaRuntimeError: If process exited with a non-zero exit code.
        PlatformNotSupportedError: If CPU fallback or absence of Vulkan initialization is detected.
    """
    if returncode != 0:
        raise AmevaRuntimeError(
            f"[termux-llamacpp] LLM execution failed with exit code {returncode}.\n"
            f"Output:\n{output}"
        )

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


def _calculate_llama_layers(
    engine: Any = None,
    requested_ngl: Optional[int] = None,
    is_vulkan: bool = True,
) -> int:
    """Determines GPU offload layers for GGUF LLMs adhering to strict Zero-Silent-Fallback.

    Rules:
    - If is_vulkan is False, offload layers must strictly be 0.
    - If caller specified requested_ngl:
        - If is_vulkan and requested_ngl <= 0: raises AmevaRuntimeError due to conflicting configuration.
        - Otherwise returns int(requested_ngl).
    - If engine config specifies an existing positive ngl, respects it.
    - Default to FULL_OFFLOAD_SENTINEL (999) for full GPU offloading under Vulkan (delegates to upstream layer clamp).
    """
    if not is_vulkan:
        return 0

    if requested_ngl is not None:
        val = int(requested_ngl)
        if val <= 0:
            raise AmevaRuntimeError(
                f"[termux-llamacpp] Conflicting configuration: Vulkan GPU backend requested "
                f"with non-positive layer offloading (requested_ngl={val})."
            )
        return val

    if engine is not None:
        existing_ngl = None
        if isinstance(engine, dict) and "ngl" in engine and engine["ngl"] is not None:
            existing_ngl = int(engine["ngl"])
        elif hasattr(engine, "config"):
            cfg = engine.config
            if isinstance(cfg, dict):
                existing_ngl = int(cfg.get("ngl") or cfg.get("n_gpu_layers") or 0)
            else:
                existing_ngl = int(getattr(cfg, "ngl", 0) or getattr(cfg, "n_gpu_layers", 0))
        elif hasattr(engine, "n_gpu_layers") and getattr(engine, "n_gpu_layers", None) is not None:
            existing_ngl = int(engine.n_gpu_layers)
        elif hasattr(engine, "ngl") and getattr(engine, "ngl", None) is not None:
            existing_ngl = int(engine.ngl)
        elif hasattr(engine, "n_layers") and getattr(engine, "n_layers", None) is not None:
            existing_ngl = int(engine.n_layers)

        if existing_ngl is not None and existing_ngl > 0:
            return existing_ngl

    return DEFAULT_GPU_LAYERS


class LlamaCppAdapter(BaseAdapter):
    """termux-llamacpp (llama.cpp GGUF) Vulkan acceleration adapter."""

    module_name = "termux-llamacpp"

    @classmethod
    def get_execution_environment(
        cls,
        base_env: Optional[Dict[str, str]] = None,
    ) -> Dict[str, str]:
        """Provides verified execution environment adhering to Golden Link Order.
        Strictly preserves Vulkan validation checks without forced bypasses.
        """
        return get_vulkan_env(base_env)

    @staticmethod
    def bind(
        engine: Any = None,
        report: Any = None,
        profile: Any = None,
        requested_backend: Optional[str] = None,
        requested_ngl: Optional[int] = None,
        **kwargs: Any,
    ) -> BindingResult:
        """Binds llama.cpp engine or config to verified hardware acceleration plan."""
        report = resolve_diagnostic_report(report, profile)
        raw_req = requested_backend
        req_norm = str(raw_req or "auto").lower().strip()

        if req_norm not in ALLOWED_BACKENDS:
            raise AmevaRuntimeError(
                f"[termux-llamacpp] Unsupported backend '{requested_backend}'. "
                f"Allowed backends: {sorted(ALLOWED_BACKENDS)}"
            )

        is_vk = _is_vulkan_report(report)

        # Strict Rule: Only 'auto' allows adaptive CPU routing when Vulkan is unavailable.
        # Explicit 'vulkan' or 'gpu' strictly refuses CPU fallback and raises PlatformNotSupportedError immediately.
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
            # 'auto': adaptive routing strictly based on hardware diagnostic report
            pass

        config: dict = {
            "module": LlamaCppAdapter.module_name,
            "device_name": report.device_name,
            "vendor_id": report.vendor_id,
            "requested_backend": req_norm,
        }

        if is_vk:
            cpu_cores = os.cpu_count() or 8
            big_cores = max(1, cpu_cores // 2)
            ngl = _calculate_llama_layers(engine, requested_ngl=requested_ngl, is_vulkan=True)
            is_mali = (report.vendor_id == _MALI_VENDOR_ID or "mali" in str(report.device_name or "").lower())
            loader_path = getattr(report, "loader_path", "")
            is_system_icd = bool(loader_path and "/system" in str(loader_path))

            config.update({
                "backend": "vulkan",
                "ngl": ngl,
                "device_flag": "vulkan",
                "flash_attn": True,
                "ctx_size": 2048,
                "threads": big_cores,
                "mali_align": is_mali,
                "bridge_active": False,
                "system_icd_prioritized": is_system_icd,
                "loader_path": loader_path,
            })

            if engine is not None:
                try:
                    if isinstance(engine, dict):
                        engine["ngl"] = ngl
                        engine["device"] = "vulkan"
                        engine.setdefault("flash_attn", True)
                        engine.setdefault("threads", big_cores)
                    elif hasattr(engine, "config"):
                        cfg = engine.config
                        if isinstance(cfg, dict):
                            cfg["ngl"] = ngl
                            cfg["n_gpu_layers"] = ngl
                            cfg["device"] = "vulkan"
                            cfg.setdefault("flash_attn", True)
                            cfg.setdefault("threads", big_cores)
                        else:
                            setattr(cfg, "n_gpu_layers", ngl)
                            setattr(cfg, "ngl", ngl)
                            setattr(cfg, "device", "vulkan")
                            setattr(cfg, "flash_attn", True)
                            if hasattr(cfg, "threads") and getattr(cfg, "threads", 0) == 0:
                                setattr(cfg, "threads", big_cores)
                    elif hasattr(engine, "ngl") or hasattr(engine, "n_gpu_layers"):
                        if hasattr(engine, "ngl"):
                            engine.ngl = ngl
                        if hasattr(engine, "n_gpu_layers"):
                            engine.n_gpu_layers = ngl
                        if hasattr(engine, "device"):
                            engine.device = "vulkan"
                        if hasattr(engine, "threads") and getattr(engine, "threads", 0) == 0:
                            engine.threads = big_cores
                    elif isinstance(engine, list):
                        existing_ngl = _extract_cli_ngl(engine)
                        if existing_ngl is not None and existing_ngl <= 0:
                            raise AmevaRuntimeError(
                                f"[{LlamaCppAdapter.module_name}] Conflicting CLI arguments: Vulkan GPU backend requested "
                                f"but existing arguments specify -ngl {existing_ngl} <= 0."
                            )
                        if existing_ngl is None:
                            engine.extend(["-ngl", str(ngl)])
                        if "--device" not in engine and "-dev" not in engine:
                            engine.extend(["--device", "vulkan"])
                        if "-t" not in engine and "--threads" not in engine:
                            engine.extend(["-t", str(big_cores)])
                    logger.info(
                        "[%s] Injected -ngl %d -t %d --device vulkan (device=%s, is_mali=%s)",
                        LlamaCppAdapter.module_name, ngl, big_cores, report.device_name, is_mali
                    )
                except Exception as e:
                    logger.error("[%s] Binding error: %s", LlamaCppAdapter.module_name, e)
                    raise AmevaRuntimeError(
                        f"[{LlamaCppAdapter.module_name}] llama.cpp Vulkan binding failure: {e}"
                    ) from e

            return BindingResult(
                module=LlamaCppAdapter.module_name,
                backend="vulkan",
                is_vulkan=True,
                device_name=report.device_name,
                vendor_id=report.vendor_id,
                config=config,
                status="CONFIGURED_VULKAN",
            )
        else:
            cpu_cores = os.cpu_count() or 8
            big_cores = max(1, cpu_cores // 2)
            config["backend"] = "cpu_neon"
            config["ngl"] = 0
            config["threads"] = big_cores

            if engine is not None:
                if isinstance(engine, dict):
                    engine["ngl"] = 0
                    engine["device"] = "cpu"
                    engine.setdefault("threads", big_cores)
                elif hasattr(engine, "config"):
                    cfg = engine.config
                    if isinstance(cfg, dict):
                        cfg["ngl"] = 0
                        cfg["n_gpu_layers"] = 0
                        cfg["device"] = "cpu"
                        cfg.setdefault("threads", big_cores)
                    else:
                        if hasattr(cfg, "n_gpu_layers"):
                            cfg.n_gpu_layers = 0
                        if hasattr(cfg, "ngl"):
                            cfg.ngl = 0
                        if hasattr(cfg, "device"):
                            cfg.device = "cpu"
                        if hasattr(cfg, "threads"):
                            cfg.threads = big_cores
                elif hasattr(engine, "ngl") or hasattr(engine, "n_gpu_layers"):
                    if hasattr(engine, "ngl"):
                        engine.ngl = 0
                    if hasattr(engine, "n_gpu_layers"):
                        engine.n_gpu_layers = 0
                    if hasattr(engine, "device"):
                        engine.device = "cpu"
                    if hasattr(engine, "threads"):
                        engine.threads = big_cores
                elif isinstance(engine, list):
                    existing_ngl = _extract_cli_ngl(engine)
                    if existing_ngl is not None and existing_ngl > 0:
                        raise AmevaRuntimeError(
                            f"[{LlamaCppAdapter.module_name}] Conflicting CLI arguments: CPU backend requested "
                            f"but existing arguments specify -ngl {existing_ngl} > 0."
                        )
                    if existing_ngl is None:
                        engine.extend(["-ngl", "0"])
                    if "-t" not in engine and "--threads" not in engine:
                        engine.extend(["-t", str(big_cores)])

            reason = (
                "Explicit CPU requested by user"
                if req_norm in ("cpu", "cpu_neon")
                else "Vulkan hardware unavailable (adaptive routing)"
            )
            return _make_cpu_binding(
                LlamaCppAdapter.module_name,
                report,
                config,
                reason=reason,
            )

    @staticmethod
    def unbind(engine: Any = None) -> None:
        """Unbinds adapter and resets engine configuration."""
        logger.info("[%s] Unbinding adapter and resetting resources.", LlamaCppAdapter.module_name)
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
                logger.warning("[%s] Unbind failed: %s", LlamaCppAdapter.module_name, e)

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
    ) -> List[str]:
        """Assembles verified LLM CLI argument list conforming to modern llama-cli specifications."""
        backend_norm = (target_backend or "auto").strip().lower()
        if backend_norm not in ALLOWED_BACKENDS:
            raise AmevaRuntimeError(
                f"[{cls.module_name}] Unsupported target_backend '{target_backend}'. Allowed: {sorted(ALLOWED_BACKENDS)}"
            )

        if threads == "auto" or threads is None:
            cpu_count = os.cpu_count() or 8
            thread_val = str(max(1, min(4, cpu_count // 2 if cpu_count > 4 else cpu_count)))
        else:
            thread_val = str(threads)

        if backend_norm in ("cpu", "cpu_neon"):
            ngl_val = "0"
        elif backend_norm in ("vulkan", "gpu"):
            if ngl_override is not None:
                if int(ngl_override) <= 0:
                    raise AmevaRuntimeError(
                        f"[{cls.module_name}] Vulkan backend requested but ngl_override is {ngl_override} <= 0."
                    )
                ngl_val = str(ngl_override)
            else:
                ngl_val = str(DEFAULT_GPU_LAYERS)
        else:
            # 'auto': defaults to DEFAULT_GPU_LAYERS unless overridden
            ngl_val = str(ngl_override if ngl_override is not None else DEFAULT_GPU_LAYERS)

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

        if backend_norm in ("vulkan", "gpu") and device_name:
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


