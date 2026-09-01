"""
C ABI FFI Bindings for libameva_vulkan.so
"""
import ctypes
import os
import sys
from pathlib import Path

class DiagnosticResultStruct(ctypes.Structure):
    _fields_ = [
        ("overall_success", ctypes.c_bool),
        ("passed_stages", ctypes.c_int),
        ("total_stages", ctypes.c_int),
        ("total_elapsed_ms", ctypes.c_double),
        ("device_name", ctypes.c_char * 128),
        ("driver_version", ctypes.c_char * 64),
        ("loader_path", ctypes.c_char * 256),
        ("recommended_backend", ctypes.c_char * 32),
    ]


def load_native_lib():
    """Loads libameva_vulkan.so if compiled, or returns None for pure Python runtime."""
    candidates = [
        Path(__file__).parent / "libameva_vulkan.so",
        Path(__file__).parent / "lib" / "libameva_vulkan.so",
        Path("/system/lib64/libvulkan.so"),
    ]
    for p in candidates:
        if p.exists():
            try:
                return ctypes.CDLL(str(p))
            except Exception:
                pass
    return None
