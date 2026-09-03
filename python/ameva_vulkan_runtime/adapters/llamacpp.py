"""
LlamaCppAdapter — termux-llamacpp (GGUF LLM) Vulkan Acceleration Adapter
"""
from __future__ import annotations

import logging
import os
from typing import Any

from ..doctor import DiagnosticReport
from ..exceptions import AmevaRuntimeError
from ..protocol import BindingResult
from .base import _is_vulkan_report, _make_cpu_fallback, _MALI_VENDOR_ID

logger = logging.getLogger("ameva_vulkan_runtime.adapters.llamacpp")


def _calculate_llama_layers(engine: Any) -> int:
    """Dynamically determines optimal GPU offload layers for GGUF LLMs."""
    if engine is not None:
        if isinstance(engine, dict) and "ngl" in engine and engine["ngl"] > 0:
            return int(engine["ngl"])
        if hasattr(engine, "n_gpu_layers") and getattr(engine, "n_gpu_layers", 0) > 0:
            return int(engine.n_gpu_layers)
        if hasattr(engine, "ngl") and getattr(engine, "ngl", 0) > 0:
            return int(engine.ngl)
        if hasattr(engine, "n_layers") and getattr(engine, "n_layers", 0) > 0:
            return int(engine.n_layers)
        model_name = str(getattr(engine, "model", "") or getattr(engine, "model_path", "")).lower()
        if "1b" in model_name or "0.5b" in model_name:
            return 16
        elif "3b" in model_name or "2b" in model_name:
            return 24
        elif "7b" in model_name or "8b" in model_name:
            return 32
        elif "13b" in model_name or "14b" in model_name:
            return 40
        elif "70b" in model_name:
            return 80
    return 33


class LlamaCppAdapter:
    """termux-llamacpp (llama.cpp GGUF) Vulkan 가속 바인딩 어댑터."""

    module_name = "termux-llamacpp"

    @staticmethod
    def bind(engine: Any, report: DiagnosticReport) -> BindingResult:
        is_vk = _is_vulkan_report(report)
        config: dict = {
            "module": LlamaCppAdapter.module_name,
            "device_name": report.device_name,
            "vendor_id": report.vendor_id,
        }

        if is_vk:
            cpu_cores = os.cpu_count() or 8
            big_cores = max(1, cpu_cores // 2)
            ngl = _calculate_llama_layers(engine)
            is_mali = (report.vendor_id == _MALI_VENDOR_ID or "Mali" in (report.device_name or ""))

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
                        "[ameva-vulkan-runtime:LlamaCppAdapter] -ngl %d -t %d --device vulkan 주입 완료"
                        " (device=%s, is_mali=%s)", ngl, big_cores, report.device_name, is_mali
                    )
                except Exception as e:
                    logger.error("[ameva-vulkan-runtime:LlamaCppAdapter] 바인딩 오류: %s", e)
                    raise AmevaRuntimeError(
                        f"[ameva-vulkan-runtime:LlamaCppAdapter] llama.cpp Vulkan 바인딩 실패: {e}"
                    ) from e

            return BindingResult(
                module=LlamaCppAdapter.module_name, backend="vulkan", is_vulkan=True,
                device_name=report.device_name, vendor_id=report.vendor_id,
                config=config, status="BOUND",
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
            return _make_cpu_fallback(LlamaCppAdapter.module_name, report, config)

    @staticmethod
    def unbind(engine: Any = None) -> None:
        logger.info("[ameva-vulkan-runtime:LlamaCppAdapter] 바인딩 해제 및 리소스 초기화.")
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
                logger.debug("[ameva-vulkan-runtime:LlamaCppAdapter] unbind 중 무시된 예외: %s", e)

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
        ngl_override: Optional[int] = None,
        no_display_prompt: bool = True,
    ) -> list[str]:
        """최신 llama-cli 규격에 부합하는 안전하고 검증된 LLM CLI 인자 목록을 조립합니다."""
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

        if target_backend in ("vulkan", "gpu"):
            cmd.extend(["--device", "vulkan"])

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

