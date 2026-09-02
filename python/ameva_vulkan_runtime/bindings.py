"""
C ABI FFI Bindings for libameva_vulkan.so / ameva_vulkan.dll
"""
from __future__ import annotations

import ctypes
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
                    logger.warning("[ameva-vulkan-runtime] 네이티브 라이브러리 후보 '%s' dlopen 실패: %s", p, e)
    return None


class AmevaVulkanLib:
    """Unified C ABI FFI Wrapper for native Vulkan compute operations."""

    def __init__(self, lib_path: Optional[str] = None):
        if lib_path and Path(lib_path).is_file():
            try:
                self._lib = ctypes.CDLL(str(lib_path))
            except Exception as e:
                logger.error("[ameva-vulkan-runtime] 지정된 네이티브 라이브러리 '%s' 로드 실패: %s", lib_path, e)
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
            logger.warning("[ameva-vulkan-runtime] FFI 함수 시그니처 등록 실패: %s", e)

    def call_matmul_f32(self, a, b, c, m: int, k: int, n: int) -> int:
        """Invokes native Vulkan SGEMM kernel with contiguous bounds verification."""
        if not self.is_loaded() or not hasattr(self._lib, "ameva_matmul_f32"):
            logger.warning("[ameva-vulkan-runtime] AmevaVulkanLib 미로드 상태: ameva_matmul_f32 FFI 심볼을 호출할 수 없습니다.")
            return -1

        if m <= 0 or k <= 0 or n <= 0:
            raise ValueError(f"[ameva-vulkan-runtime] 행렬 차원은 0보다 커야 합니다 (m={m}, k={k}, n={n}).")

        import numpy as np
        a_contig = np.ascontiguousarray(a, dtype=np.float32)
        b_contig = np.ascontiguousarray(b, dtype=np.float32)
        if not isinstance(c, np.ndarray) or c.dtype != np.float32 or not c.flags['C_CONTIGUOUS']:
            raise TypeError("[ameva-vulkan-runtime] 목적지 행렬 'c'는 C-contiguous float32 numpy.ndarray 여야 합니다.")

        # Strict buffer bounds check: prevent buffer overflow (SIGSEGV)
        if a_contig.size < m * k:
            raise BufferError(f"[ameva-vulkan-runtime] 행렬 'a' 버퍼 크기 부족: {a_contig.size} < {m * k}")
        if b_contig.size < k * n:
            raise BufferError(f"[ameva-vulkan-runtime] 행렬 'b' 버퍼 크기 부족: {b_contig.size} < {k * n}")
        if c.size < m * n:
            raise BufferError(f"[ameva-vulkan-runtime] 행렬 'c' 목적지 버퍼 크기 부족: {c.size} < {m * n}")

        try:
            a_ptr = a_contig.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
            b_ptr = b_contig.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
            c_ptr = c.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
            return int(self._lib.ameva_matmul_f32(a_ptr, b_ptr, c_ptr, m, k, n))
        except Exception as exc:
            logger.error("[ameva-vulkan-runtime] C FFI ameva_matmul_f32 실행 중 치명적 예외: %s", exc)
            raise RuntimeError(f"[ameva-vulkan-runtime] Native SGEMM 실행 실패: {exc}") from exc
