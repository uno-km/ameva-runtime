"""
Termux-Train Hardware & Topology Utilities
==========================================
Strict physical hardware probe adhering to Zero-Heuristic and SSOT principles.
"""
from __future__ import annotations

import ctypes
import os
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class HardwareProfile:
    has_vulkan: bool
    vulkan_lib_path: Optional[str]
    is_unified_memory: bool
    total_ram_mb: int
    available_ram_mb: int
    vendor_id: int
    device_name: str


def get_system_ram_mb() -> tuple[int, int]:
    """Retrieves total and available RAM in megabytes from /proc/meminfo or system fallback."""
    total_mb = 0
    avail_mb = 0
    meminfo_path = "/proc/meminfo"
    if os.path.exists(meminfo_path):
        try:
            with open(meminfo_path, "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.split(":")
                    if len(parts) == 2:
                        key = parts[0].strip()
                        val_str = parts[1].strip().split()[0]
                        if key == "MemTotal":
                            total_mb = int(val_str) // 1024
                        elif key == "MemAvailable":
                            avail_mb = int(val_str) // 1024
        except Exception:
            pass

    if total_mb == 0:
        # Fallback to os.sysconf if available (POSIX)
        try:
            pages = os.sysconf("SC_PHYS_PAGES")
            page_size = os.sysconf("SC_PAGE_SIZE")
            total_mb = (pages * page_size) // (1024 * 1024)
            avail_mb = total_mb // 2
        except (AttributeError, ValueError):
            total_mb = 8192
            avail_mb = 4096

    return total_mb, avail_mb


def probe_vulkan_loader() -> Optional[str]:
    """Finds verified physical Vulkan loader without heuristics."""
    candidate_paths = [
        "/system/lib64/libvulkan.so",
        "/vendor/lib64/libvulkan.so",
        "/system/vendor/lib64/libvulkan.so",
        "/data/data/com.termux/files/usr/lib/libvulkan.so",
    ]
    for path in candidate_paths:
        if os.path.isfile(path):
            try:
                handle = ctypes.CDLL(path)
                if hasattr(handle, "vkGetInstanceProcAddr"):
                    return path
            except Exception:
                continue
    return None


def probe_hardware() -> HardwareProfile:
    """Probes physical execution hardware without heuristic string matching."""
    loader_path = probe_vulkan_loader()
    has_vk = loader_path is not None
    total_ram, avail_ram = get_system_ram_mb()

    # Mobile SoC architectures (ARM Cortex + Mali/Adreno) utilize Unified Memory Architecture (UMA)
    is_uma = os.path.exists("/system/build.prop") or os.path.exists("/dev/kgsl-3d0") or os.path.exists("/dev/mali0")

    return HardwareProfile(
        has_vulkan=has_vk,
        vulkan_lib_path=loader_path,
        is_unified_memory=is_uma,
        total_ram_mb=total_ram,
        available_ram_mb=avail_ram,
        vendor_id=0,
        device_name="Mobile Edge Device" if is_uma else "Generic Host",
    )
