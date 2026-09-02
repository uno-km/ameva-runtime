"""
Core Hardware Acceleration Context & Execution Configuration Management
"""
from __future__ import annotations

import os
import sys
from typing import Optional
from .doctor import Doctor
from .exceptions import PlatformNotSupportedError, BufferAllocationError

class VulkanContext:
    """RAII Hardware Acceleration Context with verified device capability binding."""

    def __init__(self, device_mode: str = "auto", memory_limit_mb: int = 1024):
        self.device_mode = str(device_mode or "auto").strip().lower()
        self.memory_limit_mb = memory_limit_mb
        self.doctor = Doctor()
        self.device_name = "ARM64 CPU"
        self.backend_type = "cpu_neon"
        self.loader_path = ""
        self.driver_version = ""
        self.vendor_id = 0
        self._is_active = False
        self.execution_flags = {}
        self.profile_quirks = {}
        self._bound_adapters: list = []

        self._initialize()

    def _initialize(self):
        """Initializes the backend following strict Fail-Fast or Auto-Recovery rules."""
        if self.device_mode in ("vulkan", "gpu"):
            is_supported = self.doctor.quick_probe()
            if not is_supported:
                raise PlatformNotSupportedError(
                    f"Explicit GPU backend requested ('device=\"{self.device_mode}\"'), but target hardware "
                    "failed Vulkan validation. Silent CPU fallback is disabled in explicit mode."
                )
            self.backend_type = "vulkan"
            dev_name = self.doctor.quick_probe_device()
            self.device_name = dev_name if dev_name else "Generic Vulkan GPU Accelerator"
            self.profile_quirks = self.doctor.load_hardware_profile(self.device_name, self.vendor_id)
            self.execution_flags = {
                "use_gpu": True,
                "gpu_layers": 99,
                "n_gpu_layers": 33,
                "backend": "vulkan",
                "threads": os.cpu_count() or 4
            }
            self._is_active = True

        elif self.device_mode == "cpu":
            # Zero-overhead bypass: Skip Vulkan loader and initialize CPU NEON context
            self.backend_type = "cpu_neon"
            self.device_name = "ARM64 NEON Vector CPU Engine"
            self.profile_quirks = {}
            self.execution_flags = {
                "use_gpu": False,
                "gpu_layers": 0,
                "n_gpu_layers": 0,
                "backend": "cpu_neon",
                "threads": os.cpu_count() or 4
            }
            self._is_active = True

        else:  # "auto" or fallback
            is_supported = self.doctor.quick_probe()
            if is_supported:
                self.backend_type = "vulkan"
                dev_name = self.doctor.quick_probe_device()
                self.device_name = dev_name if dev_name else "Generic Vulkan GPU Accelerator"
                self.execution_flags = {
                    "use_gpu": True,
                    "gpu_layers": 99,
                    "n_gpu_layers": 33,
                    "backend": "vulkan",
                    "threads": os.cpu_count() or 4
                }
            else:
                # Transparent recovery to high-performance NEON
                self.backend_type = "cpu_neon"
                self.device_name = "ARM64 NEON Vector CPU Engine (Auto-Recovered)"
                self.execution_flags = {
                    "use_gpu": False,
                    "gpu_layers": 0,
                    "n_gpu_layers": 0,
                    "backend": "cpu_neon",
                    "threads": os.cpu_count() or 4
                }
            self._is_active = True

    @property
    def is_gpu(self) -> bool:
        return self.backend_type == "vulkan"

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def bind_adapter(self, adapter_cls, engine: Any = None):
        """Binds an adapter to the target engine and registers it for lifecycle unbinding."""
        report = self.doctor.run_self_test(verbose=False)
        result = adapter_cls.bind(engine, report)
        self._bound_adapters.append((adapter_cls, engine))
        return result

    def unbind_all(self) -> None:
        """Unbinds all registered adapters and resets bound engine acceleration states."""
        for adapter_cls, engine in self._bound_adapters:
            try:
                if hasattr(adapter_cls, "unbind"):
                    adapter_cls.unbind(engine)
            except Exception:
                pass
        self._bound_adapters.clear()

    def close(self):
        """Release context resources, unbind all registered adapters, and deactivate status."""
        self.unbind_all()
        self._is_active = False

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def is_vulkan(self) -> bool:
        return self.backend_type == "vulkan"

    def to_engine_flags(self, engine_name: str = "default") -> dict:
        """Helper to extract engine-specific accelerator execution flags."""
        name = (engine_name or "").lower()
        base_threads = os.cpu_count() or 4

        if name in ("whisper", "stt"):
            return {
                "use_gpu": self.is_gpu,
                "gpu_layers": 33 if self.is_gpu else 0,
                "threads": max(1, base_threads // 2) if not self.is_gpu else 2,
                "backend": self.backend_type,
                "device": "vulkan" if self.is_gpu else "cpu",
            }
        elif name in ("bitnet", "llm", "llama", "llamacpp"):
            return {
                "n_gpu_layers": 33 if self.is_gpu else 0,
                "threads": base_threads,
                "backend": self.backend_type,
                "device": "vulkan" if self.is_gpu else "cpu",
            }
        elif name in ("diffusion", "sd"):
            return {
                "device": "vulkan" if self.is_gpu else "cpu",
                "use_vulkan": self.is_gpu,
                "backend": self.backend_type,
                "threads": base_threads,
            }
        elif name in ("tts",):
            return {
                "device": "vulkan" if self.is_gpu else "cpu",
                "backend": self.backend_type,
                "threads": base_threads,
            }
        elif name in ("vision",):
            return {
                "device": "vulkan" if self.is_gpu else "cpu",
                "backend": self.backend_type,
                "use_gpu": self.is_gpu,
            }
        return dict(self.execution_flags)

    def allocate_buffer(self, size_bytes: int) -> int:
        """Memory Budget Validator: Validates requested buffer size against memory budget limit threshold."""
        if size_bytes > self.memory_limit_mb * 1024 * 1024:
            raise BufferAllocationError(
                f"Requested buffer size ({size_bytes / (1024*1024):.2f} MB) exceeds configured memory limit ({self.memory_limit_mb} MB)."
            )
        return size_bytes


def create_context(device: str = "auto", memory_limit_mb: int = 1024) -> VulkanContext:
    """Factory helper to instantiate a verified VulkanContext."""
    return VulkanContext(device_mode=device, memory_limit_mb=memory_limit_mb)


def get_or_create_context(
    device: Optional[object] = "auto",
    memory_limit_mb: int = 1024
) -> VulkanContext:
    """
    Universal Entrypoint for all modality packages.
    - If device is already a VulkanContext, returns it as-is.
    - If device is None, defaults to 'auto'.
    - If device is string ("auto", "gpu", "vulkan", "cpu"), creates a corresponding VulkanContext.
    """
    if isinstance(device, VulkanContext):
        return device
    device_mode = str(device or "auto").lower()
    return VulkanContext(device_mode=device_mode, memory_limit_mb=memory_limit_mb)
