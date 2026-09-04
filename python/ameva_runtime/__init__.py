"""
AMEVA Runtime (Unified Next-Gen On-Device AI Runtime)
=====================================================
Multi-backend hardware orchestration (Vulkan, OpenCL, NPU, CPU-NEON)
for mobile & edge devices.
"""
from __future__ import annotations

from .core import (
    AmevaRuntime,
    get_runtime,
    VulkanContext,
    create_context,
    get_or_create_context,
    run,
    plan,
    ExecutionResult,
)
from .detector import detect_hardware, HardwareProfile
from .router import SmartRouter, ExecutionPlan, get_router
from .doctor import Doctor, DiagnosticReport, diagnose
from .protocol import IRuntimeConsumer, IVulkanConsumer, BindingResult
from .exceptions import (
    AmevaRuntimeError,
    HardwareDetectionError,
    SubsystemInitError,
    DriverLockupError,
    DeviceOOMError,
    ArchitectureUnsupportedError,
    InvalidAffinityError,
)

__version__ = "1.0.0"

__all__ = [
    "__version__",
    "AmevaRuntime",
    "get_runtime",
    "run",
    "plan",
    "ExecutionResult",
    "VulkanContext",
    "create_context",
    "get_or_create_context",
    "detect_hardware",
    "HardwareProfile",
    "SmartRouter",
    "ExecutionPlan",
    "get_router",
    "Doctor",
    "DiagnosticReport",
    "diagnose",
    "IRuntimeConsumer",
    "IVulkanConsumer",
    "BindingResult",
    "AmevaRuntimeError",
    "HardwareDetectionError",
    "SubsystemInitError",
    "DriverLockupError",
    "DeviceOOMError",
    "ArchitectureUnsupportedError",
    "InvalidAffinityError",
]

