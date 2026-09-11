"""
ameva_runtime.platform
======================
Android / Termux Platform Detection Utilities & Hardware Environment Prober.
Single Source of Truth (SSOT) for the uno-km ecosystem.
"""
from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass
from typing import Set

from .detector import detect_hardware, HardwareProfile

__all__ = [
    "is_android",
    "is_termux",
    "is_proot",
    "get_termux_prefix",
    "get_termux_home",
    "SoCInfo",
    "detect_soc_environment",
]


@dataclass
class SoCInfo:
    """Hardware SoC & GPU subsystem identification profile (SSOT compatible)."""
    vendor: str
    chipname: str
    gpu_family: str
    kgsl_accessible: bool
    mali_node_accessible: bool
    can_direct_vulkan_cli: bool
    recommended_backend: str
    cpu_model: str
    cpu_cores: int
    diagnosis_reason: str


def is_android() -> bool:
    if os.path.exists("/system/build.prop"):
        return True
    return bool(os.environ.get("ANDROID_ROOT") or os.environ.get("ANDROID_DATA"))


def is_termux() -> bool:
    if os.environ.get("TERMUX_VERSION") or os.environ.get("TERMUX_APP_PID"):
        return True
    prefix = os.environ.get("PREFIX", "")
    if "com.termux" in prefix or os.path.exists("/data/data/com.termux"):
        return True
    return False


def is_proot() -> bool:
    if "PROOT_TMP_DIR" in os.environ:
        return True
    return os.path.exists("/proc/sys/fs/binfmt_misc/proot")


def get_termux_prefix() -> str:
    return os.environ.get("PREFIX", "/data/data/com.termux/files/usr")


def get_termux_home() -> str:
    return os.environ.get("HOME", "/data/data/com.termux/files/home")


def detect_soc_environment() -> SoCInfo:
    """Detects SoC and hardware environment using the core detector."""
    profile: HardwareProfile = detect_hardware()
    can_vulkan = (profile.recommended_backend == "vulkan" and not profile.hardware_hazard)

    return SoCInfo(
        vendor=profile.vendor,
        chipname=profile.soc_model,
        gpu_family=profile.gpu_family,
        kgsl_accessible=profile.has_kgsl_node,
        mali_node_accessible=profile.has_mali_node,
        can_direct_vulkan_cli=can_vulkan,
        recommended_backend=profile.recommended_backend,
        cpu_model=f"{profile.total_cpu_cores}-core ARM",
        cpu_cores=profile.total_cpu_cores,
        diagnosis_reason=profile.diagnosis_reason,
    )
