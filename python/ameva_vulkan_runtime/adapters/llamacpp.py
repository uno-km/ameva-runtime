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
            ngl = 33
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
