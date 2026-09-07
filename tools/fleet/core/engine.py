"""
AMEVA Vulkan Runtime - Execution Engine Wrapper
"""
import re
import shlex
from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple

@dataclass
class RunResult:
    success: bool
    return_code: int
    duration_sec: float
    stdout: str
    stderr: str
    output_path: str
    step_time_ms: Optional[float] = None
    sample_time_ms: Optional[float] = None
    vram_peak_mb: Optional[float] = None
    error_message: Optional[str] = None

class DiffusionEngine:
    def __init__(self, binary_path: str, shim_path: Optional[str] = None, lib_dir: Optional[str] = None):
        self.binary_path = binary_path
        self.shim_path = shim_path
        self.lib_dir = lib_dir

    def build_command(
        self,
        model_path: str,
        prompt: str,
        output_path: str,
        steps: int = 1,
        cfg_scale: float = 1.0,
        guidance: Optional[float] = None,
        sample_method: str = "euler_a",
        seed: int = 42,
        width: int = 512,
        height: int = 512,
        threads: int = 4,
        vae_tiling: bool = False,
        clip_skip: int = 1,
        negative_prompt: Optional[str] = None,
        backend: Optional[str] = None,
        extra_env: Optional[Dict[str, str]] = None
    ) -> str:
        parts = []
        if extra_env:
            for k, v in extra_env.items():
                parts.append(f"{k}={shlex.quote(str(v))}")
        if self.lib_dir:
            parts.append(f"LD_LIBRARY_PATH={shlex.quote(self.lib_dir)}:$LD_LIBRARY_PATH")
        if self.shim_path:
            parts.append(f"LD_PRELOAD={shlex.quote(self.shim_path)}")
        
        parts.append(shlex.quote(self.binary_path))
        parts.extend(["-m", shlex.quote(model_path)])
        parts.extend(["-p", shlex.quote(prompt)])
        parts.extend(["-o", shlex.quote(output_path)])
        parts.extend(["--steps", str(steps)])
        parts.extend(["--cfg-scale", str(cfg_scale)])
        if guidance is not None:
            parts.extend(["--guidance", str(guidance)])
        parts.extend(["--sampling-method", str(sample_method)])
        parts.extend(["--seed", str(seed)])
        parts.extend(["-W", str(width)])
        parts.extend(["-H", str(height)])
        parts.extend(["-t", str(threads)])
        
        if backend:
            parts.extend(["--backend", shlex.quote(backend)])
        if vae_tiling:
            parts.append("--vae-tiling")
        if clip_skip > 1:
            parts.extend(["--clip-skip", str(clip_skip)])
        if negative_prompt:
            parts.extend(["-n", shlex.quote(negative_prompt)])
            
        return " ".join(parts)

    @staticmethod
    def parse_log(stdout: str, stderr: str) -> Dict[str, Any]:
        full_text = stdout + "\n" + stderr
        stats: Dict[str, Any] = {
            "total_duration_sec": 0.0,
            "sample_time_ms": None,
            "vram_peak_mb": None,
            "errors": []
        }
        
        # Parse total generation time
        m_time = re.search(r"total time:\s*([\d\.]+)\s*s", full_text, re.IGNORECASE)
        if m_time:
            stats["total_duration_sec"] = float(m_time.group(1))
            
        # Parse sampling time
        m_sample = re.search(r"sampling took\s*([\d\.]+)\s*ms", full_text, re.IGNORECASE)
        if m_sample:
            stats["sample_time_ms"] = float(m_sample.group(1))
            
        # Parse VRAM allocation
        m_vram = re.search(r"ggml_vulkan:\s*allocating ([\d\.]+)\s*MB", full_text, re.IGNORECASE)
        if m_vram:
            stats["vram_peak_mb"] = float(m_vram.group(1))
            
        # Check for error patterns
        error_keywords = [
            "VK_ERROR_OUT_OF_DEVICE_MEMORY",
            "Segmentation fault",
            "Bus error",
            "Aborted",
            "SIGSEGV",
            "Floating point exception"
        ]
        for kw in error_keywords:
            if kw in full_text:
                stats["errors"].append(kw)
                
        return stats
