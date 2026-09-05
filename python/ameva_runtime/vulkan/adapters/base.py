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
    if not report.device_name or report.device_name in ("Unknown", "None", ""):
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
    """
    시스템 내에 실제로 설치된 유효한 libvulkan.so 드라이버 디렉토리의 절대 경로를 동적으로 프로빙하여 반환합니다.
    1순위: /proc/self/maps 리눅스 커널 가상 메모리 매핑 직접 역추적 (Zero-Hardcoding / Zero-Dependency / Root 0%)
    2순위: 32-bit vs 64-bit ABI 격리 표준 Android HAL 경로 보조 검증
    """
    import sys
    import os
    import ctypes
    from pathlib import Path

    # 1순위: Linux 커널 프로세스 가상 메모리 매핑 직접 역추적 (가장 완벽한 Ground Truth)
    try:
        handle = ctypes.CDLL("libvulkan.so")
        if hasattr(handle, "vkCreateInstance"):
            maps_path = Path("/proc/self/maps")
            if maps_path.is_file():
                with open(maps_path, "r", encoding="utf-8", errors="ignore") as f:
                    for line in f:
                        if "libvulkan.so" in line:
                            parts = line.strip().split()
                            if len(parts) >= 6:
                                real_path = parts[-1]
                                if os.path.isabs(real_path) and os.path.exists(real_path):
                                    return str(Path(real_path).parent.resolve())
    except Exception:
        pass

    # 2순위: 64비트 vs 32비트 ABI 격리 표준 디렉토리 배열 순회 (보조 폴백)
    is_64bit = sys.maxsize > 2**32

    if is_64bit:
        candidate_dirs = [
            Path("/system/lib64"),
            Path("/vendor/lib64"),
            Path("/apex/com.android.runtime/lib64"),
            Path("/system/vendor/lib64"),
        ]
    else:
        candidate_dirs = [
            Path("/system/lib"),
            Path("/vendor/lib"),
            Path("/apex/com.android.runtime/lib"),
            Path("/system/vendor/lib"),
        ]

    for root in candidate_dirs:
        so_path = root / "libvulkan.so"
        if so_path.is_file():
            try:
                h = ctypes.CDLL(str(so_path))
                if hasattr(h, "vkCreateInstance"):
                    return str(root.resolve())
            except Exception:
                continue

    # 3순위 폴백: 단순 파일 존재 검사
    for root in candidate_dirs:
        if (root / "libvulkan.so").is_file():
            return str(root.resolve())

    return None


def get_vulkan_env(base_env: Optional[dict[str, str]] = None) -> dict[str, str]:
    """
    AMEVA Vulkan 런타임 환경 변수 맵을 반환합니다.
    Golden Link Order 적용:
    1순위: Termux usr/lib (Termux 자체 libc++, libomp, libz 보호)
    2순위: 시스템 Vulkan 드라이버 디렉토리 (/system/lib64)
    3순위: 번들 런타임 라이브러리 (~/.termux-llama/current/lib)
    4순위: 기존 시스템 및 유저 환경 경로
    """
    import os
    from pathlib import Path

    env = dict(base_env or os.environ)
    current_ld = env.get("LD_LIBRARY_PATH", "")

    # 동적 프로빙으로 실제 유효한 Vulkan 드라이버 디렉토리 획득
    discovered_driver_dir = find_system_vulkan_driver_dir()

    ordered_dirs = []

    # 1순위: 실제 스마트폰 시스템 Vulkan 드라이버 디렉토리 (/system/lib64)
    # Android 15 Bionic libunwindstack 심볼 충돌 방지: system liblzma가 Termux liblzma보다 먼저 바인딩되어야 함
    if discovered_driver_dir and discovered_driver_dir not in ordered_dirs:
        ordered_dirs.append(discovered_driver_dir)
    elif os.path.isdir("/system/lib64") and "/system/lib64" not in ordered_dirs:
        ordered_dirs.append("/system/lib64")

    # 2순위: Termux 네이티브 C++ 런타임
    termux_usr_lib = "/data/data/com.termux/files/usr/lib"
    if os.path.isdir(termux_usr_lib) and termux_usr_lib not in ordered_dirs:
        ordered_dirs.append(termux_usr_lib)

    # 3순위: 번들 llama/runtime 라이브러리
    llama_lib = str(Path.home() / ".termux-llama/current/lib")
    if os.path.isdir(llama_lib) and llama_lib not in ordered_dirs:
        ordered_dirs.append(llama_lib)

    # 4순위: 기존 경로 중복 없이 병합
    existing_parts = [p for p in current_ld.split(":") if p]
    env["LD_LIBRARY_PATH"] = ":".join(final_dirs)

    # Mali GPU 쿼크: ARM Mali Valhall 아키텍처에서 양자화 GEMM 무한 루프/데드락 방지
    try:
        from ..platform import detect_soc_environment
        soc = detect_soc_environment()
        if "mali" in str(soc.gpu_family).lower() or soc.vendor == "samsung":
            env.setdefault("GGML_VK_FORCE_MEDIUM_MATMUL", "1")
            env.setdefault("GGML_VK_DISABLE_F16", "1")
    except Exception:
        pass

    return env
