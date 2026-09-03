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


def get_vulkan_env(base_env: Optional[dict[str, str]] = None) -> dict[str, str]:
    """Android Termux 환경에서 시스템 Vulkan 드라이버 및 바이너리 라이브러리 경로를 안전하게 보장하는 환경 변수 맵을 반환합니다."""
    import os
    from pathlib import Path

    env = dict(base_env or os.environ)
    current_ld = env.get("LD_LIBRARY_PATH", "")

    search_dirs = [
        "/system/lib64",
        "/vendor/lib64",
        "/system/lib",
        "/vendor/lib",
        str(Path.home() / ".termux-llama/current/lib"),
        "/data/data/com.termux/files/usr/lib",
    ]

    valid_dirs = [d for d in search_dirs if os.path.exists(d)]
    existing_parts = [p for p in current_ld.split(":") if p]
    
    # Merge preserving order without duplicates
    merged = []
    for d in valid_dirs + existing_parts:
        if d not in merged:
            merged.append(d)

    env["LD_LIBRARY_PATH"] = ":".join(merged)
    return env
