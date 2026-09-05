"""
AMEVA Vulkan Runtime Domain Exceptions
"""

class AmevaRuntimeError(Exception):
    """Base exception for all AMEVA Vulkan runtime errors."""
    pass


class AmevaVulkanError(AmevaRuntimeError):
    """Raised when native Vulkan FFI or driver execution fails."""
    pass


class PlatformNotSupportedError(AmevaRuntimeError):
    """Raised when the target hardware or driver does not support Vulkan acceleration."""
    pass


class DriverQuirkViolationError(AmevaRuntimeError):
    """Raised when hardware alignment or subgroup bounds are violated."""
    pass


class BufferAllocationError(AmevaRuntimeError):
    """Raised when VRAM or host-coherent memory allocation fails."""
    pass


class PipelineCreationError(AmevaRuntimeError):
    """Raised when SPIR-V compute pipeline compilation fails."""
    pass
