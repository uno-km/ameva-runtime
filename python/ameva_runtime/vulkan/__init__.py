"""
AMEVA Unified Vulkan Acceleration Runtime & SDK
"""
from ameva_runtime._version import __version__
__author__ = "Eunho Kim <contact@uno-km.com>"


from .core import VulkanContext, create_context, get_or_create_context
from .doctor import Doctor, DiagnosticReport, StageReport
from .bindings import AmevaVulkanLib, DiagnosticResultStruct, load_native_lib
from .protocol import IVulkanConsumer, BindingResult
from .platform import is_termux, is_android, is_proot, get_termux_prefix, get_termux_home
from .exceptions import (
    AmevaRuntimeError,
    PlatformNotSupportedError,
    DriverQuirkViolationError,
    BufferAllocationError,
    PipelineCreationError,
)
from .adapters import (
    SttAdapter,
    DiffusionAdapter,
    BitnetAdapter,
    LlamaCppAdapter,
    TtsAdapter,
    VisionAdapter,
)
# Re-export modern v2.0.0 Hardware Orchestrator & SmartRouter for unified single-import convenience
from ameva_runtime.router import SmartRouter, ExecutionPlan, get_router
from ameva_runtime.detector import HardwareProfile, detect_hardware

def is_available() -> bool:
    """현재 하드웨어에서 Vulkan 가속이 지원되는지 확인합니다."""
    return Doctor().quick_probe()

def get_device_name() -> str | None:
    """Returns physical Vulkan GPU device name via official doctor probe."""
    return Doctor().quick_probe_device()

__all__ = [
    "get_device_name",
    "VulkanContext",
    "create_context",
    "get_or_create_context",
    "Doctor",
    "DiagnosticReport",
    "StageReport",
    "AmevaVulkanLib",
    "DiagnosticResultStruct",
    "load_native_lib",
    "IVulkanConsumer",
    "BindingResult",
    # Platform utilities (SSOT for uno-km ecosystem)
    "is_termux",
    "is_android",
    "is_proot",
    "get_termux_prefix",
    "get_termux_home",
    # Exceptions
    "AmevaRuntimeError",
    "PlatformNotSupportedError",
    "DriverQuirkViolationError",
    "BufferAllocationError",
    "PipelineCreationError",
    # Adapters
    "SttAdapter",
    "DiffusionAdapter",
    "BitnetAdapter",
    "LlamaCppAdapter",
    "TtsAdapter",
    "VisionAdapter",
    "is_available",
    # v2.0.0 Modern Orchestrator
    "SmartRouter",
    "ExecutionPlan",
    "get_router",
    "HardwareProfile",
    "detect_hardware",
]
