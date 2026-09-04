"""
ameva_runtime.adapters.base
===========================
Base Utilities and Shared Helper Functions for AMEVA Modality Adapters.
"""
from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Any, Optional, Dict

from ..protocol import BindingResult
from ..detector import detect_hardware, HardwareProfile
from ..router import get_router, SmartRouter

logger = logging.getLogger("ameva_runtime.adapters")

_ADRENO_VENDOR_ID = 0x5143
_MALI_VENDOR_ID = 0x13B5


def _is_vulkan_viable(profile: HardwareProfile) -> bool:
    """Returns True if Vulkan execution is genuinely safe without driver lockup."""
    return profile.recommended_backend == "vulkan" and not profile.hardware_hazard


def _make_adaptive_binding(
    module: str,
    profile: HardwareProfile,
    config: Dict[str, Any],
    backend: str = "cpu_neon",
    status: str = "BOUND_ADAPTIVE",
) -> BindingResult:
    """Creates a transparent BindingResult adhering to Zero-Silent-Fallback."""
    logger.info(
        "[ameva-runtime:%s] Active Backend: %s (Reason: %s)",
        module, backend, profile.diagnosis_reason
    )
    config["backend"] = backend
    config["diagnosis"] = profile.diagnosis_reason
    is_vk = (backend == "vulkan")
    return BindingResult(
        module=module,
        backend=backend,
        is_vulkan=is_vk,
        device_name=profile.gpu_family,
        vendor_id=_MALI_VENDOR_ID if profile.gpu_family == "mali" else _ADRENO_VENDOR_ID,
        config=config,
        status=status,
        diagnosis=profile.diagnosis_reason,
    )


def find_system_vulkan_driver_dir() -> Optional[str]:
    """Dynamically finds the directory containing the active libvulkan.so."""
    is_64bit = sys.maxsize > 2**32
    candidates = [
        Path("/system/lib64" if is_64bit else "/system/lib"),
        Path("/vendor/lib64" if is_64bit else "/vendor/lib"),
        Path("/apex/com.android.runtime/lib64" if is_64bit else "/apex/com.android.runtime/lib"),
    ]
    for c in candidates:
        if (c / "libvulkan.so").exists():
            return str(c.resolve())
    return None
