"""
Base Utilities & Common Logic for Ameva Modality Adapters
"""
from __future__ import annotations

import logging
from typing import Any, Optional

try:
    from ..doctor import DiagnosticReport
except ImportError:
    DiagnosticReport = Any

try:
    from ..protocol import BindingResult
except ImportError:
    from dataclasses import dataclass
    @dataclass
    class BindingResult:
        module: str
        backend: str
        is_vulkan: bool
        device_name: str
        vendor_id: int
        config: dict
        status: str

logger = logging.getLogger("ameva_vulkan_runtime.adapters")

_ADRENO_VENDOR_ID = 0x5143
_MALI_VENDOR_ID = 0x13B5


def _is_vulkan_report(report: Optional[DiagnosticReport]) -> bool:
    if report is None:
        return False
    return bool(
        report.overall_success
        or report.recommended_backend in ("vulkan", "vulkan_driver_only")
        or report.passed_stages >= 7
    )


def _make_cpu_fallback(module: str, report: DiagnosticReport, config: dict) -> BindingResult:
    """Creates a standardized CPU NEON fallback BindingResult."""
    logger.info(
        "[ameva-vulkan-runtime:%s] Vulkan 가속 미지원 — CPU NEON 모드로 전환합니다. "
        "원인: %s", module, report.device_name or "Vulkan 불가"
    )
    config["backend"] = "cpu_neon"
    return BindingResult(
        module=module, backend="cpu_neon", is_vulkan=False,
        device_name=report.device_name, vendor_id=report.vendor_id,
        config=config, status="BOUND_CPU",
    )


def _get_optimal_threads() -> int:
    """Returns optimal threads count for big/performance cores."""
    import os
    cpu_count = os.cpu_count() or 8
    return max(1, min(4, cpu_count // 2 if cpu_count > 4 else cpu_count))


def find_system_vulkan_driver_dir() -> Optional[str]:
    """시스템 내에 실제로 설치된 유효한 libvulkan.so 드라이버 위치를 동적으로 프로빙하여 반환합니다."""
    import ctypes
    import ctypes.util
    import os
    from pathlib import Path

    # 1. 표준 find_library 및 직접 dlopen 검사
    lib_name = ctypes.util.find_library("vulkan")
    if lib_name:
        try:
            handle = ctypes.CDLL(lib_name)
            if hasattr(handle, "vkCreateInstance"):
                p = os.path.dirname(os.path.abspath(lib_name))
                if p: return p
        except Exception:
            pass

    # 2. Android HAL 및 시스템 드라이버 영역 동적 스캔
    probe_roots = [
        Path("/system/lib64"),
        Path("/vendor/lib64"),
        Path("/system/lib"),
        Path("/vendor/lib"),
        Path("/apex/com.android.runtime/lib64"),
        Path("/system/vendor/lib64"),
    ]

    for root in probe_roots:
        so_path = root / "libvulkan.so"
        if so_path.exists():
            try:
                handle = ctypes.CDLL(str(so_path))
                if hasattr(handle, "vkCreateInstance"):
                    return str(root)
            except Exception:
                # dlopen 실패 시에도 파일이 존재하면 후보로 반환
                return str(root)

    return None


def get_vulkan_env(base_env: Optional[dict[str, str]] = None) -> dict[str, str]:
    """
    AMEVA Vulkan 런타임 우선권 기반 환경 변수 맵을 반환합니다.
    동적으로 시스템 Vulkan 드라이버를 탐색하여 LD_LIBRARY_PATH 최상단(우선순위 1위)에 자동 주입합니다.
    """
    import os
    from pathlib import Path

    env = dict(base_env or os.environ)
    current_ld = env.get("LD_LIBRARY_PATH", "")

    # 동적 프로빙으로 실제 Vulkan 드라이버 디렉토리 우선 획득
    discovered_driver_dir = find_system_vulkan_driver_dir()
    
    # 우선순위 리스트 구성 (동적 드라이버 위치가 1순위)
    priority_dirs = []
    if discovered_driver_dir and discovered_driver_dir not in priority_dirs:
        priority_dirs.append(discovered_driver_dir)

    # 런타임 자체 번들 라이브러리 (2순위)
    llama_lib = str(Path.home() / ".termux-llama/current/lib")
    if os.path.exists(llama_lib) and llama_lib not in priority_dirs:
        priority_dirs.append(llama_lib)

    # 기존 경로 병합 (중복 방지 및 우선순위 유지)
    existing_parts = [p for p in current_ld.split(":") if p]
    final_dirs = priority_dirs + [p for p in existing_parts if p not in priority_dirs]

    env["LD_LIBRARY_PATH"] = ":".join(final_dirs)
    return env
