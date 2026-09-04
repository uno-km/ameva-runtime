"""
Core Hardware Acceleration Context & Execution Configuration Management
"""
from __future__ import annotations

import os
import sys
from typing import Optional, Any
from .doctor import Doctor
from .platform import detect_soc_environment, SoCInfo
from .exceptions import PlatformNotSupportedError, BufferAllocationError

class VulkanContext:
    """RAII Hardware Acceleration Context with verified device capability binding."""

    def __init__(self, device_mode: str = "auto", memory_limit_mb: int = 1024):
        self.device_mode = str(device_mode or "auto").strip().lower()
        self.memory_limit_mb = memory_limit_mb
        self.soc_info: SoCInfo = detect_soc_environment()
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
        """Initializes the backend following strict Zero-Silent-Fallback and SoC detection rules."""
        soc = self.soc_info

        if self.device_mode in ("vulkan", "gpu"):
            # 1. Check if platform/SoC cannot support headless Vulkan in Termux CLI
            if not soc.can_direct_vulkan_cli:
                raise PlatformNotSupportedError(
                    f"Explicit GPU backend requested ('device=\"{self.device_mode}\"'), but {soc.diagnosis_reason} "
                    "Silent CPU fallback is prohibited under Zero-Silent-Fallback engineering protocol."
                )

            is_supported = self.doctor.quick_probe()
            if not is_supported:
                raise PlatformNotSupportedError(
                    f"Explicit GPU backend requested ('device=\"{self.device_mode}\"'), but target hardware "
                    "failed Vulkan validation (0 physical devices or driver initialization failure). "
                    "Silent CPU fallback is prohibited under Zero-Silent-Fallback engineering protocol."
                )
            self.backend_type = "vulkan"
            dev_name = self.doctor.quick_probe_device()
            self.device_name = dev_name if dev_name else "Generic Vulkan GPU Accelerator"
            self.profile_quirks = self.doctor.load_hardware_profile(self.device_name, self.vendor_id)
            self.execution_flags = {
                "use_gpu": True,
                "backend": "vulkan",
                "threads": soc.cpu_cores,
                "soc_vendor": soc.vendor,
                "gpu_family": soc.gpu_family,
            }
            self._is_active = True

        elif self.device_mode == "cpu":
            # Zero-overhead bypass: Skip Vulkan loader and initialize CPU NEON context
            self.backend_type = "cpu_neon"
            self.device_name = f"{soc.cpu_model} NEON Vector CPU Engine"
            self.profile_quirks = {}
            self.execution_flags = {
                "use_gpu": False,
                "backend": "cpu_neon",
                "threads": soc.cpu_cores,
                "soc_vendor": soc.vendor,
                "gpu_family": soc.gpu_family,
            }
            self._is_active = True

        else:  # "auto" mode
            if not soc.can_direct_vulkan_cli:
                # Direct route to pure NEON CPU without wasting cycles or spitting noisy Vulkan errors
                self.backend_type = "cpu_neon"
                self.device_name = f"{soc.cpu_model} NEON Vector CPU Engine ({soc.vendor.title()} CLI Direct)"
                self.execution_flags = {
                    "use_gpu": False,
                    "backend": "cpu_neon",
                    "threads": soc.cpu_cores,
                    "soc_vendor": soc.vendor,
                    "gpu_family": soc.gpu_family,
                }
                self._is_active = True
                return

            is_supported = self.doctor.quick_probe()
            if is_supported:
                self.backend_type = "vulkan"
                dev_name = self.doctor.quick_probe_device()
                self.device_name = dev_name if dev_name else "Generic Vulkan GPU Accelerator"
                self.execution_flags = {
                    "use_gpu": True,
                    "backend": "vulkan",
                    "threads": soc.cpu_cores,
                    "soc_vendor": soc.vendor,
                    "gpu_family": soc.gpu_family,
                }
            else:
                # Transparent recovery to high-performance NEON
                self.backend_type = "cpu_neon"
                self.device_name = f"{soc.cpu_model} NEON Vector CPU Engine (Auto-Recovered)"
                self.execution_flags = {
                    "use_gpu": False,
                    "backend": "cpu_neon",
                    "threads": soc.cpu_cores,
                    "soc_vendor": soc.vendor,
                    "gpu_family": soc.gpu_family,
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
        if not self.is_gpu:
            from .doctor import DiagnosticReport
            report = DiagnosticReport(
                overall_success=False,
                device_name=self.device_name,
                driver_version="N/A",
                loader_path="",
                vendor_id=self.vendor_id,
                passed_stages=0,
                total_stages=12,
                total_elapsed_ms=0.0,
                recommended_backend="cpu_neon",
                stages=[],
                profile_quirks={},
            )
        else:
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
            except Exception as e:
                import logging
                logging.getLogger("ameva-vulkan-runtime").debug("[VulkanContext] unbind_all exception: %s", e)
        self._bound_adapters.clear()

    def close(self):
        """Release context resources, unbind all registered adapters, and deactivate status."""
        self.unbind_all()
        self._is_active = False

    def __del__(self):
        try:
            self.close()
        except Exception as e:
            import logging
            logging.getLogger("ameva-vulkan-runtime").debug("[VulkanContext] __del__ exception: %s", e)

    def is_vulkan(self) -> bool:
        return self.backend_type == "vulkan"

    def to_engine_flags(self, engine_name: str = "default", engine: Any = None) -> dict:
        """Helper to extract engine-specific accelerator execution flags with dynamic layer calculation."""
        name = (engine_name or "").lower()
        base_threads = os.cpu_count() or 4

        if name in ("whisper", "stt"):
            from .adapters.stt import _calculate_whisper_layers
            layers = _calculate_whisper_layers(engine) if self.is_gpu else 0
            return {
                "use_gpu": self.is_gpu,
                "gpu_layers": layers,
                "threads": max(1, base_threads // 2) if not self.is_gpu else 2,
                "backend": self.backend_type,
                "device": "vulkan" if self.is_gpu else "cpu",
            }
        elif name in ("bitnet", "llm"):
            from .adapters.bitnet import _calculate_bitnet_layers
            layers = _calculate_bitnet_layers(engine) if self.is_gpu else 0
            return {
                "n_gpu_layers": layers,
                "threads": base_threads,
                "backend": self.backend_type,
                "device": "vulkan" if self.is_gpu else "cpu",
            }
        elif name in ("llama", "llamacpp"):
            from .adapters.llamacpp import _calculate_llama_layers
            layers = _calculate_llama_layers(engine) if self.is_gpu else 0
            return {
                "n_gpu_layers": layers,
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

    def validate_buffer_budget(self, size_bytes: int) -> bool:
        """Validates requested buffer size against configured memory budget threshold."""
        if size_bytes > self.memory_limit_mb * 1024 * 1024:
            raise BufferAllocationError(
                f"Requested buffer size ({size_bytes / (1024*1024):.2f} MB) exceeds configured memory limit ({self.memory_limit_mb} MB)."
            )
        return True

    def allocate_buffer(self, size_bytes: int) -> bytearray:
        """Allocates a real host memory buffer backed by budget validation."""
        self.validate_buffer_budget(size_bytes)
        return bytearray(size_bytes)


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
