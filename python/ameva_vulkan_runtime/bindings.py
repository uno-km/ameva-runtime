"""
C ABI FFI Bindings for libameva_vulkan.so / ameva_vulkan.dll
"""
from __future__ import annotations

import ctypes
import os
import sys
from pathlib import Path
from typing import Optional


import logging

logger = logging.getLogger(__name__)


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


def load_native_lib() -> Optional[ctypes.CDLL]:
    """Loads libameva_vulkan.so / ameva_vulkan.dll if compiled, or returns None for pure Python runtime."""
    lib_names = [
        "libameva_vulkan.so",
        "ameva_vulkan.dll",
        "libameva_vulkan.dylib",
    ]
    search_dirs = [
        Path(__file__).parent,
        Path(__file__).parent / "lib",
        Path(__file__).parent.parent.parent / "build",
        Path("/data/data/com.termux/files/usr/lib"),
        Path.home() / ".local" / "lib",
    ]
    for d in search_dirs:
        for name in lib_names:
            p = d / name
            if p.is_file():
                try:
                    return ctypes.CDLL(str(p))
                except Exception as e:
                    logger.debug("Failed to load %s: %s", p, e)
    return None


class AmevaVulkanLib:
    """Unified C ABI FFI Wrapper for native Vulkan compute operations."""

    def __init__(self, lib_path: Optional[str] = None):
        if lib_path and Path(lib_path).is_file():
            try:
                self._lib = ctypes.CDLL(str(lib_path))
            except Exception as e:
                logger.debug("Failed to load specified library '%s': %s", lib_path, e)
                self._lib = None
        else:
            self._lib = load_native_lib()

        self._setup_signatures()

    def is_loaded(self) -> bool:
        """Returns True if the native libameva_vulkan.so / ameva_vulkan.dll shared object is loaded."""
        return self._lib is not None

    def _setup_signatures(self) -> None:
        if not self._lib:
            return
        try:
            if hasattr(self._lib, "ameva_matmul_f32"):
                self._lib.ameva_matmul_f32.argtypes = [
                    ctypes.POINTER(ctypes.c_float),
                    ctypes.POINTER(ctypes.c_float),
                    ctypes.POINTER(ctypes.c_float),
                    ctypes.c_int32,
                    ctypes.c_int32,
                    ctypes.c_int32,
                ]
                self._lib.ameva_matmul_f32.restype = ctypes.c_int32
        except Exception as e:
            logger.debug("Failed to register FFI signatures: %s", e)

    def call_matmul_f32(self, a, b, c, m: int, k: int, n: int) -> int:
        """Invokes native Vulkan SGEMM kernel with contiguous bounds verification."""
        if not self.is_loaded() or not hasattr(self._lib, "ameva_matmul_f32"):
            logger.debug("AmevaVulkanLib is not loaded or ameva_matmul_f32 symbol missing.")
            return -1
        if m <= 0 or k <= 0 or n <= 0:
            logger.error("Matrix dimensions must be strictly positive (m=%d, k=%d, n=%d).", m, k, n)
            return -1
        try:
            import numpy as np
            a_contig = np.ascontiguousarray(a, dtype=np.float32)
            b_contig = np.ascontiguousarray(b, dtype=np.float32)
            if not isinstance(c, np.ndarray) or c.dtype != np.float32 or not c.flags['C_CONTIGUOUS']:
                logger.error("Destination matrix 'c' must be contiguous float32 ndarray.")
                return -1

            # Strict buffer bounds check: prevent buffer overflow (SIGSEGV)
            if a_contig.size < m * k:
                logger.error("Matrix 'a' size (%d) is smaller than required (%d).", a_contig.size, m * k)
                return -1
            if b_contig.size < k * n:
                logger.error("Matrix 'b' size (%d) is smaller than required (%d).", b_contig.size, k * n)
                return -1
            if c.size < m * n:
                logger.error("Matrix 'c' size (%d) is smaller than required (%d).", c.size, m * n)
                return -1

            a_ptr = a_contig.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
            b_ptr = b_contig.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
            c_ptr = c.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
            return int(self._lib.ameva_matmul_f32(a_ptr, b_ptr, c_ptr, m, k, n))
        except Exception as exc:
            logger.error("Exception in call_matmul_f32 FFI execution: %s", exc)
            return -1
