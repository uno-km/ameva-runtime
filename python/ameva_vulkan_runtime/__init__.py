"""
AMEVA Unified Vulkan Acceleration Runtime & SDK
"""
__version__ = "1.2.0"
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

def is_available() -> bool:
    """현재 하드웨어에서 Vulkan 가속이 지원되는지 확인합니다."""
    return Doctor().quick_probe()

__all__ = [
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
]
