"""
Base Utilities & Common Logic for Ameva Modality Adapters
"""
from __future__ import annotations

import logging
from typing import Any, Optional

from ..doctor import DiagnosticReport
from ..protocol import BindingResult

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
