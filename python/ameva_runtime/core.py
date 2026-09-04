"""
ameva_runtime.core
==================
Core AMEVA Runtime Engine & Hardware Orchestrator.
"""
from __future__ import annotations

import logging
import os
import re
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from typing import Any, Optional, Dict, List

from .detector import detect_hardware, HardwareProfile
from .router import SmartRouter, ExecutionPlan, get_router
from .protocol import BindingResult
from .exceptions import AmevaRuntimeError

logger = logging.getLogger("ameva_runtime.core")


@dataclass
class ExecutionResult:
    """Detailed telemetry and generated text from on-device model execution."""
    text: str
    backend_used: str
    model_name: str
    prompt_tokens: int = 0
    eval_tokens: int = 0
    tokens_per_second: float = 0.0
    prompt_tokens_per_second: float = 0.0
    total_time_ms: float = 0.0
    command: List[str] = field(default_factory=list)
    rationale: str = ""
    return_code: int = 0


def resolve_model_path(model_arg: str) -> str:
    """Resolves model path from direct path or standard Termux storage locations."""
    if os.path.exists(model_arg):
        return os.path.abspath(model_arg)
    
    candidates = [
        os.path.expanduser(f"~/.termux-llama/models/{model_arg}"),
        os.path.expanduser(f"~/.termux-llama/models/{model_arg}.gguf"),
        f"/data/data/com.termux/files/home/.termux-llama/models/{model_arg}",
        f"/data/data/com.termux/files/home/.termux-llama/models/{model_arg}.gguf",
        os.path.expanduser(f"~/models/{model_arg}"),
        os.path.expanduser(f"~/models/{model_arg}.gguf"),
    ]
    for c in candidates:
        if os.path.exists(c):
            return os.path.abspath(c)
    return model_arg


def find_inference_binary() -> Optional[str]:
    """Locates llama-cli or compatible inference binary across PATH and mobile environments."""
    found_in_path = shutil.which("llama-cli")
    if found_in_path:
        return found_in_path

    search_paths = [
        "/data/data/com.termux/files/usr/bin/llama-cli",
        os.path.expanduser("~/.termux-llama/bin/llama-cli"),
        "/data/data/com.termux/files/home/.termux-llama/bin/llama-cli",
        os.path.expanduser("~/vulkan-llama/bin/llama-cli"),
        "/data/data/com.termux/files/home/vulkan-llama/bin/llama-cli",
        os.path.expanduser("~/BitNet_ms/3rdparty/llama.cpp/build-vulkan/bin/llama-cli"),
        "/data/data/com.termux/files/home/BitNet_ms/3rdparty/llama.cpp/build-vulkan/bin/llama-cli",
        os.path.expanduser("~/llama.cpp/build/bin/llama-cli"),
        "/data/data/com.termux/files/home/llama.cpp/build/bin/llama-cli",
    ]
    for p in search_paths:
        if os.path.isfile(p) and os.access(p, os.X_OK):
            return os.path.abspath(p)
    return None


def resolve_inference_environment(plan: ExecutionPlan, binary_path: str) -> Dict[str, str]:
    """Assembles all dynamic library dependencies and vendor paths into LD_LIBRARY_PATH."""
    env = os.environ.copy()
    env.update(plan.env_overrides)

    bin_dir = os.path.dirname(os.path.abspath(binary_path))
    candidate_dirs = [
        "/system/lib64",
        "/data/data/com.termux/files/usr/lib",
        bin_dir,
        os.path.abspath(os.path.join(bin_dir, "..", "lib")),
        os.path.abspath(os.path.join(bin_dir, "..", "ggml", "src")),
        os.path.abspath(os.path.join(bin_dir, "..", "src")),
        os.path.expanduser("~/vulkan-llama/ggml/src"),
        os.path.expanduser("~/vulkan-llama/src"),
    ]
    cur_ld = env.get("LD_LIBRARY_PATH", "")
    valid_paths = [p for p in candidate_dirs if os.path.isdir(p)]
    if cur_ld:
        valid_paths.append(cur_ld)
    if valid_paths:
        env["LD_LIBRARY_PATH"] = ":".join(valid_paths)
    return env


class AmevaRuntime:
    """Central orchestrator for on-device AI acceleration."""

    _instance: Optional["AmevaRuntime"] = None

    def __init__(self, profile: HardwareProfile | None = None) -> None:
        self.profile = profile or detect_hardware()
        self.router = SmartRouter(self.profile)
        logger.info(
            "[AmevaRuntime] Initialized. SoC: %s (%s) | GPU: %s | Recommended: %s",
            self.profile.soc_model, self.profile.vendor,
            self.profile.gpu_family, self.profile.recommended_backend
        )

    @classmethod
    def get_instance(cls) -> "AmevaRuntime":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def bind_engine(self, module_name: str, engine: Any = None) -> BindingResult:
        """Dynamically binds an engine instance using the appropriate modality adapter."""
        from .adapters import (
            LlamaCppAdapter,
            VisionAdapter,
            DiffusionAdapter,
            SttAdapter,
            TtsAdapter,
            BitnetAdapter,
        )

        adapter_map = {
            "termux-llamacpp": LlamaCppAdapter,
            "llamacpp": LlamaCppAdapter,
            "termux-vision": VisionAdapter,
            "vision": VisionAdapter,
            "termux-diffusion": DiffusionAdapter,
            "diffusion": DiffusionAdapter,
            "termux-stt": SttAdapter,
            "stt": SttAdapter,
            "termux-tts": TtsAdapter,
            "tts": TtsAdapter,
            "termux-bitnet": BitnetAdapter,
            "bitnet": BitnetAdapter,
        }

        adapter = adapter_map.get(module_name.lower())
        if adapter is None:
            raise AmevaRuntimeError(
                f"Unknown module '{module_name}'. Supported modules: {list(adapter_map.keys())}"
            )

        return adapter.bind(engine=engine, profile=self.profile)

    def plan_execution(self, model_name: str = "", requested_backend: str | None = None) -> ExecutionPlan:
        """Returns an ExecutionPlan for running inference."""
        return self.router.route_for_llm(model_name_or_path=model_name, requested_backend=requested_backend)

    def execute(
        self,
        model_path: str,
        prompt: str = "Hello! Who are you?",
        max_tokens: int = 64,
        temperature: float = 0.7,
        backend: str | None = None,
        stream_output: bool = True,
    ) -> ExecutionResult:
        """Executes model inference safely and returns structured performance telemetry."""
        plan = self.plan_execution(model_name=model_path, requested_backend=backend)

        llama_cli = find_inference_binary()
        if not llama_cli:
            raise AmevaRuntimeError(
                "llama-cli inference binary not found. Please install llama.cpp or termux-llama."
            )

        resolved_model = resolve_model_path(model_path)
        if not os.path.exists(resolved_model):
            raise AmevaRuntimeError(f"Target model file not found: {model_path}")

        cmd: List[str] = [
            llama_cli,
            "-m", resolved_model,
            "-p", prompt,
            "-n", str(max_tokens),
            "-t", str(plan.threads),
            "-ngl", str(plan.ngl),
            "-b", str(plan.batch_size),
            "-c", str(plan.context_size),
            "--temp", str(temperature),
        ]

        exec_env = resolve_inference_environment(plan, llama_cli)

        # Apply CPU core affinity pinning if supported
        if hasattr(os, "sched_setaffinity") and plan.affinity_cpus:
            try:
                os.sched_setaffinity(0, set(plan.affinity_cpus))
            except Exception as e:
                logger.warning("Could not set CPU affinity: %s", e)

        t0 = time.perf_counter()
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=exec_env,
            text=True,
            bufsize=1,
        )

        stdout_out, stderr_out = proc.communicate()
        total_time_ms = (time.perf_counter() - t0) * 1000.0

        full_output = stdout_out + "\n" + stderr_out

        # Parse timing metrics from llama output
        prompt_tokens = 0
        eval_tokens = 0
        tps = 0.0
        prompt_tps = 0.0

        m_prompt = re.search(r"prompt eval time\s*=\s*[\d\.]+\s*ms\s*/\s*(\d+)\s*tokens.*?([\d\.]+)\s*tokens per second", full_output)
        if m_prompt:
            prompt_tokens = int(m_prompt.group(1))
            prompt_tps = float(m_prompt.group(2))

        m_eval = re.search(r"eval time\s*=\s*[\d\.]+\s*ms\s*/\s*(\d+)\s*runs.*?([\d\.]+)\s*tokens per second", full_output)
        if m_eval:
            eval_tokens = int(m_eval.group(1))
            tps = float(m_eval.group(2))

        clean_text = stdout_out.strip()

        return ExecutionResult(
            text=clean_text,
            backend_used=plan.backend.upper(),
            model_name=os.path.basename(resolved_model),
            prompt_tokens=prompt_tokens,
            eval_tokens=eval_tokens,
            tokens_per_second=tps,
            prompt_tokens_per_second=prompt_tps,
            total_time_ms=total_time_ms,
            command=cmd,
            rationale=plan.rationale,
            return_code=proc.returncode,
        )


# --------------------------------------------------------------------------
# Backward-compatibility classes and helpers (VulkanContext)
# --------------------------------------------------------------------------
class VulkanContext:
    """Legacy VulkanContext for backward compatibility with ameva-vulkan-runtime."""
    def __init__(self, runtime: AmevaRuntime | None = None) -> None:
        self.runtime = runtime or AmevaRuntime.get_instance()
        self.profile = self.runtime.profile
        self.is_valid = (self.profile.recommended_backend == "vulkan")
        self.device_name = self.profile.gpu_family

    def bind(self, module_name: str, engine: Any = None) -> BindingResult:
        return self.runtime.bind_engine(module_name, engine)


def get_runtime() -> AmevaRuntime:
    """Convenience accessor for the global AmevaRuntime."""
    return AmevaRuntime.get_instance()


def create_context() -> VulkanContext:
    return VulkanContext()


def get_or_create_context() -> VulkanContext:
    return VulkanContext()


def run(
    model: str,
    prompt: str = "Hello! Who are you?",
    max_tokens: int = 64,
    temperature: float = 0.7,
    backend: str | None = None,
) -> ExecutionResult:
    """Top-level 1-liner to execute inference on optimal hardware."""
    return get_runtime().execute(
        model_path=model,
        prompt=prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        backend=backend,
    )


def plan(model: str = "", backend: str | None = None) -> ExecutionPlan:
    """Top-level helper to preview hardware execution plan."""
    return get_runtime().plan_execution(model_name=model, requested_backend=backend)

