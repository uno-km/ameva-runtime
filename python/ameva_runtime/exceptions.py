"""
ameva_runtime.exceptions
========================
AMEVA Runtime Exception Hierarchy.
Strict Zero-Silent-Fallback: All unexpected boundary conditions, driver locks,
and hardware constraints must raise explicit, structured errors.
"""
from __future__ import annotations


class AmevaRuntimeError(RuntimeError):
    """Base exception for all AMEVA Runtime errors."""
    def __init__(self, message: str, error_code: str = "ERR_AMEVA_UNKNOWN", cause: str | None = None):
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.cause = cause

    def __str__(self) -> str:
        base = f"[{self.error_code}] {self.message}"
        if self.cause:
            base += f" (Cause: {self.cause})"
        return base


class PlatformNotSupportedError(AmevaRuntimeError):
    """Raised when the host platform or kernel lacks minimum AI runtime prerequisites."""
    def __init__(self, message: str, cause: str | None = None):
        super().__init__(message, error_code="ERR_PLATFORM_NOT_SUPPORTED", cause=cause)


class HardwareDefectHazardError(AmevaRuntimeError):
    """Raised when a known hardware/driver hazard (e.g. Mali Vulkan fence lockup) is detected in strict mode."""
    def __init__(self, message: str, cause: str | None = None):
        super().__init__(message, error_code="ERR_HARDWARE_DEFECT_HAZARD", cause=cause)


class DriverQuirkViolationError(AmevaRuntimeError):
    """Raised when vendor GPU/NPU driver quirks invalidate compute invariants."""
    def __init__(self, message: str, cause: str | None = None):
        super().__init__(message, error_code="ERR_DRIVER_QUIRK_VIOLATION", cause=cause)


class BufferAllocationError(AmevaRuntimeError):
    """Raised when VRAM or host memory mapping fails."""
    def __init__(self, message: str, cause: str | None = None):
        super().__init__(message, error_code="ERR_BUFFER_ALLOCATION", cause=cause)


class PipelineCreationError(AmevaRuntimeError):
    """Raised when compute pipeline or shader compilation fails."""
    def __init__(self, message: str, cause: str | None = None):
        super().__init__(message, error_code="ERR_PIPELINE_CREATION", cause=cause)


class HardwareDetectionError(AmevaRuntimeError):
    """Raised when hardware detection probing fails unexpectedly."""
    def __init__(self, message: str, cause: str | None = None):
        super().__init__(message, error_code="ERR_HARDWARE_DETECTION", cause=cause)


class SubsystemInitError(AmevaRuntimeError):
    """Raised when a specific compute subsystem fails to initialize."""
    def __init__(self, message: str, cause: str | None = None):
        super().__init__(message, error_code="ERR_SUBSYSTEM_INIT", cause=cause)


class DriverLockupError(AmevaRuntimeError):
    """Raised when a kernel GPU driver fence lockup is detected or anticipated."""
    def __init__(self, message: str, cause: str | None = None):
        super().__init__(message, error_code="ERR_DRIVER_LOCKUP", cause=cause)


class DeviceOOMError(AmevaRuntimeError):
    """Raised when GPU VRAM or host memory budget is exhausted."""
    def __init__(self, message: str, cause: str | None = None):
        super().__init__(message, error_code="ERR_DEVICE_OOM", cause=cause)


class ArchitectureUnsupportedError(AmevaRuntimeError):
    """Raised when CPU/GPU ISA architecture is unsupported."""
    def __init__(self, message: str, cause: str | None = None):
        super().__init__(message, error_code="ERR_ARCH_UNSUPPORTED", cause=cause)


class InvalidAffinityError(AmevaRuntimeError):
    """Raised when CPU affinity cannot be bound to target cores due to cgroup restrictions."""
    def __init__(self, message: str, cause: str | None = None):
        super().__init__(message, error_code="ERR_INVALID_AFFINITY", cause=cause)


class ModelNotFoundError(AmevaRuntimeError):
    """Raised when the specified model cannot be resolved in any standard path."""
    def __init__(self, model_arg: str, searched_dirs: list[str] | None = None):
        msg = f"Model '{model_arg}' not found in candidate paths or search directories."
        if searched_dirs:
            msg += f" Searched directories: {searched_dirs}"
        super().__init__(msg, error_code="ERR_MODEL_NOT_FOUND", cause=f"Model identifier '{model_arg}' does not exist on disk.")
        self.model_arg = model_arg
        self.searched_dirs = searched_dirs or []


class AmbiguousModelMatchError(AmevaRuntimeError):
    """Raised when a fuzzy model pattern matches multiple candidates, preventing unsafe implicit selection."""
    def __init__(self, model_arg: str, candidates: list[str]):
        msg = (
            f"Fuzzy pattern '{model_arg}' matched {len(candidates)} candidate models. "
            f"Implicit coercion is prohibited under Zero-Silent-Fallback policy. "
            f"Please specify exact model filename or path: {candidates}"
        )
        super().__init__(
            msg,
            error_code="ERR_AMBIGUOUS_MODEL_MATCH",
            cause=f"Multiple model candidates matched pattern: {candidates}"
        )
        self.model_arg = model_arg
        self.candidates = candidates


# ==============================================================================
# 8 Standard AMEVA-LLAMA Error Hierarchy (E001 - E008)
# ==============================================================================
class AmevaLlamaAssetMissingError(AmevaRuntimeError):
    """AMEVA-LLAMA-E001: Managed llama.cpp asset missing in canonical location."""
    def __init__(self, message: str = "AMEVA-managed llama.cpp binary is not installed.", cause: str | None = None):
        super().__init__(message, error_code="AMEVA-LLAMA-E001", cause=cause)


class AmevaLlamaVerificationError(AmevaRuntimeError):
    """AMEVA-LLAMA-E002: Asset manifest or SHA-256 cryptographic verification failed."""
    def __init__(self, message: str, cause: str | None = None):
        super().__init__(message, error_code="AMEVA-LLAMA-E002", cause=cause)


class AmevaLlamaVulkanBlockedError(AmevaRuntimeError):
    """AMEVA-LLAMA-E003: Vulkan LLM acceleration blocked on this device profile."""
    def __init__(self, message: str, cause: str | None = None):
        super().__init__(message, error_code="AMEVA-LLAMA-E003", cause=cause)


class AmevaLlamaAbiError(AmevaRuntimeError):
    """AMEVA-LLAMA-E004: ELF ABI or DT_NEEDED dependency verification failed."""
    def __init__(self, message: str, cause: str | None = None):
        super().__init__(message, error_code="AMEVA-LLAMA-E004", cause=cause)


class AmevaLlamaSmokeTestError(AmevaRuntimeError):
    """AMEVA-LLAMA-E005: Staged binary execution smoke test failed."""
    def __init__(self, message: str, cause: str | None = None):
        super().__init__(message, error_code="AMEVA-LLAMA-E005", cause=cause)


class AmevaLlamaLinkActivationError(AmevaRuntimeError):
    """AMEVA-LLAMA-E006: Current release atomic symlink activation failed."""
    def __init__(self, message: str, cause: str | None = None):
        super().__init__(message, error_code="AMEVA-LLAMA-E006", cause=cause)


class AmevaLlamaLockTimeoutError(AmevaRuntimeError):
    """AMEVA-LLAMA-E007: Asset provisioning lock acquisition failed or timed out."""
    def __init__(self, message: str = "Timed out waiting for asset provisioning lock.", cause: str | None = None):
        super().__init__(message, error_code="AMEVA-LLAMA-E007", cause=cause)


class AmevaLlamaUnsafeArchiveError(AmevaRuntimeError):
    """AMEVA-LLAMA-E008: Unsafe archive member or path traversal detected."""
    def __init__(self, message: str, cause: str | None = None):
        super().__init__(message, error_code="AMEVA-LLAMA-E008", cause=cause)

