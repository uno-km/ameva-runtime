"""
ameva_runtime.protocol
======================
AMEVA Universal Consumer Protocol Definitions.
Defines typed contracts for all modality adapters and execution consumers.
"""
from __future__ import annotations

from typing import Any, Dict, Protocol, runtime_checkable


@runtime_checkable
class IRuntimeConsumer(Protocol):
    """Universal AI runtime consumer interface.
    Each modality adapter implements this interface.
    """

    @property
    def module_name(self) -> str:
        """Consumer package identifier (e.g. 'termux-llamacpp')."""
        ...

    def bind(self, engine: Any, profile: Any) -> "BindingResult":
        """Binds acceleration and execution configuration to the target engine."""
        ...

    def unbind(self) -> None:
        """Releases acceleration resources."""
        ...


# Backward-compatible alias
IVulkanConsumer = IRuntimeConsumer


class BindingResult:
    """Immutable value object holding binding configuration and active backend details."""

    __slots__ = (
        "_module",
        "_backend",
        "_is_vulkan",
        "_device_name",
        "_vendor_id",
        "_config",
        "_status",
        "_diagnosis",
    )

    def __init__(
        self,
        module: str,
        backend: str,
        is_vulkan: bool,
        device_name: str,
        vendor_id: int,
        config: Dict[str, Any],
        status: str = "BOUND",
        diagnosis: str = "",
    ) -> None:
        self._module = module
        self._backend = backend
        self._is_vulkan = is_vulkan
        self._device_name = device_name
        self._vendor_id = vendor_id
        self._config = dict(config)
        self._status = status
        self._diagnosis = diagnosis

    @property
    def module(self) -> str:
        return self._module

    @property
    def backend(self) -> str:
        return self._backend

    @property
    def is_vulkan(self) -> bool:
        return self._is_vulkan

    @property
    def device_name(self) -> str:
        return self._device_name

    @property
    def vendor_id(self) -> int:
        return self._vendor_id

    @property
    def config(self) -> Dict[str, Any]:
        return dict(self._config)

    @property
    def status(self) -> str:
        return self._status

    @property
    def diagnosis(self) -> str:
        return self._diagnosis

    @property
    def is_accelerated(self) -> bool:
        return self._is_vulkan or self._backend in ("vulkan", "opencl", "npu")

    @property
    def device_id(self) -> int:
        return self._vendor_id

    @property
    def target_modality(self) -> str:
        return self._module.replace("termux-", "")

    @property
    def diagnosis_reason(self) -> str:
        return self._diagnosis or str(self._config.get("diagnosis", ""))

    def to_dict(self) -> Dict[str, Any]:
        res = dict(self._config)
        res.update({
            "module": self._module,
            "backend": self._backend,
            "is_vulkan": self._is_vulkan,
            "device_name": self._device_name,
            "vendor_id": self._vendor_id,
            "status": self._status,
            "diagnosis": self._diagnosis,
            "diagnosis_reason": self.diagnosis_reason,
        })
        return res

    def __repr__(self) -> str:
        return (
            f"BindingResult(module={self._module!r}, backend={self._backend!r}, "
            f"vulkan={self._is_vulkan}, device={self._device_name!r}, status={self._status!r})"
        )
