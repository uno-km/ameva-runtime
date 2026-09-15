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
    BaseAdapter.verify_vulkan_compute_output(
        stderr=output,
        returncode=returncode,
        expected_signatures=VULKAN_INIT_PATTERNS,
        module_name="termux-llamacpp",
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
        **kwargs: Any,
    ) -> Dict[str, str]:
        """Provides verified execution environment adhering to Golden Link Order.
        Strictly preserves Vulkan validation checks without forced bypasses.
        """
        return super().get_execution_environment(base_env, **kwargs)

    @classmethod
    def _mutate_modality_attributes(
        cls,
        engine: Any,
        snapshot: dict[str, Any],
        report: DiagnosticReport,
        requested_ngl: Optional[int] = None,
        big_cores: int = 4,
        **kwargs: Any,
    ) -> None:
        """Tier 2: llama.cpp specific mutation (ngl, flash_attn, threads)."""
        if isinstance(engine, list):
            if "cli_args_len" not in snapshot:
                snapshot["cli_args_len"] = len(engine)
            existing_ngl = _extract_cli_ngl(engine)
            if existing_ngl is not None and existing_ngl <= 0:
                raise AmevaRuntimeError(
                    f"[{cls.module_name}] Conflicting CLI arguments: Vulkan GPU backend requested "
                    f"but existing arguments specify -ngl {existing_ngl} <= 0."
                )
            ngl = _calculate_llama_layers(engine, requested_ngl=requested_ngl, is_vulkan=True)
            if existing_ngl is None:
                engine.extend(["-ngl", str(ngl)])
            if "--device" not in engine and "-dev" not in engine:
                engine.extend(["--device", "vulkan"])
            if "-t" not in engine and "--threads" not in engine:
                engine.extend(["-t", str(big_cores)])
            return

        ngl = _calculate_llama_layers(engine, requested_ngl=requested_ngl, is_vulkan=True)
        cls._set_engine_property(engine, snapshot, "ngl", ngl)
        cls._set_engine_property(engine, snapshot, "n_gpu_layers", ngl)
        cls._set_engine_property(engine, snapshot, "flash_attn", True)

    @classmethod
    def bind(
        cls,
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
                f"[{cls.module_name}] Unsupported backend '{requested_backend}'. "
                f"Allowed backends: {sorted(ALLOWED_BACKENDS)}"
            )

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
            # 'auto': adaptive routing strictly based on hardware diagnostic report
            pass

        config: dict[str, Any] = {
            "module": cls.module_name,
            "device_name": report.device_name,
            "vendor_id": report.vendor_id,
            "requested_backend": req_norm,
        }
        snapshot: dict[str, Any] = {}

        cpu_cores = os.cpu_count() or 8
        big_cores = max(1, cpu_cores // 2)

        if is_vk:
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
                    cls._mutate_common_attributes(engine, snapshot, backend="vulkan", threads=big_cores)
                    cls._mutate_modality_attributes(
                        engine,
                        snapshot,
                        report,
                        requested_ngl=requested_ngl,
                        big_cores=big_cores,
                        **kwargs,
                    )
                    logger.info(
                        "[%s] Injected -ngl %d -t %d --device vulkan (device=%s, is_mali=%s)",
                        cls.module_name, ngl, big_cores, report.device_name, is_mali
                    )
                except Exception as e:
                    cls._restore_engine_properties(engine, snapshot)
                    logger.error("[%s] Binding error: %s", cls.module_name, e)
                    raise AmevaRuntimeError(
                        f"[{cls.module_name}] llama.cpp Vulkan binding failure: {e}"
                    ) from e

            return BindingResult(
                module=cls.module_name,
                backend="vulkan",
                is_vulkan=True,
                device_name=report.device_name,
                vendor_id=report.vendor_id,
                config=config,
                status="CONFIGURED_VULKAN",
                restore_state=snapshot,
            )
        else:
            config["backend"] = "cpu_neon"
            config["ngl"] = 0
            config["threads"] = big_cores

            if engine is not None:
                if isinstance(engine, list):
                    if "cli_args_len" not in snapshot:
                        snapshot["cli_args_len"] = len(engine)
                    existing_ngl = _extract_cli_ngl(engine)
                    if existing_ngl is not None and existing_ngl > 0:
                        raise AmevaRuntimeError(
                            f"[{cls.module_name}] Conflicting CLI arguments: CPU backend requested "
                            f"but existing arguments specify -ngl {existing_ngl} > 0."
                        )
                    if existing_ngl is None:
                        engine.extend(["-ngl", "0"])
                    if "-t" not in engine and "--threads" not in engine:
                        engine.extend(["-t", str(big_cores)])
                else:
                    cls._mutate_common_attributes(engine, snapshot, backend="cpu", threads=big_cores)
                    cls._set_engine_property(engine, snapshot, "ngl", 0)
                    cls._set_engine_property(engine, snapshot, "n_gpu_layers", 0)

            reason = (
                "Explicit CPU requested by user"
                if req_norm in ("cpu", "cpu_neon")
                else "Vulkan hardware unavailable (adaptive routing)"
            )
            return _make_cpu_binding(
                cls.module_name,
                report,
                config,
                reason=reason,
                restore_state=snapshot,
            )

    @classmethod
    def unbind(cls, engine: Any = None, binding_result: BindingResult | None = None) -> None:
        """Unbinds adapter and restores exact pre-binding configuration."""
        super().unbind(engine, binding_result)

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


