"""
ameva_vulkan_runtime.platform
==============================
Android / Termux 플랫폼 감지 유틸리티 모음.

이 모듈은 uno-km 생태계(termux-stt, termux-diffusion, termux-bitnet,
termux-train, termux-vision, termux-llamacpp)의 공통 플랫폼 감지 코드를
단일 신뢰 출처(Single Source of Truth)로 제공합니다.

마이그레이션 경로:
    # 각 패키지에서 기존 인라인 구현 대신 이 모듈을 사용합니다.
    from ameva_vulkan_runtime.platform import is_termux, is_android, get_termux_prefix

설계 원칙:
- 순수 함수 (Pure Functions): 부작용 없음, import 시 부작용 없음.
- stdlib 전용: 추가 의존성 없음.
- 모든 실패는 False/None 반환 — 절대 예외를 raise 하지 않음.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
import subprocess

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
    """Hardware SoC & GPU subsystem identification profile."""
    vendor: str                  # "qualcomm", "samsung_exynos", "mediatek", "generic"
    chipname: str                # e.g., "exynos1380", "sm8650", "generic"
    gpu_family: str              # "adreno", "mali", "powervr", "generic"
    kgsl_accessible: bool        # True if /dev/kgsl-3d0 exists and is read/write accessible
    mali_node_accessible: bool   # True if /dev/mali0 exists
    can_direct_vulkan_cli: bool  # True if Termux CLI can dispatch headless Vulkan safely
    recommended_backend: str     # "vulkan" or "cpu_neon"
    cpu_model: str               # e.g., "ARM Cortex-A78"
    cpu_cores: int               # Logical CPU count
    diagnosis_reason: str        # Ground truth explanation



def is_android() -> bool:
    """현재 런타임이 Android OS 위에서 실행 중인지 확인합니다.

    Returns:
        True  — /system/build.prop 존재 또는 ANDROID_ROOT / ANDROID_DATA 환경변수 설정.
        False — 그 외 모든 환경.
    """
    if os.path.exists("/system/build.prop"):
        return True
    if "ANDROID_ROOT" in os.environ or "ANDROID_DATA" in os.environ:
        return True
    return False


def is_termux() -> bool:
    """현재 런타임이 Termux 네이티브 환경인지 확인합니다.

    termux-diffusion `is_android_termux()`, termux-stt `is_termux()`,
    termux-train `is_termux()` 의 통합 구현입니다.

    Returns:
        True  — TERMUX_VERSION / TERMUX_APP_PID 환경변수 설정,
                또는 PREFIX 에 "com.termux" 포함,
                또는 /data/data/com.termux 디렉터리 존재.
        False — 그 외 모든 환경.
    """
    # TERMUX_VERSION 은 Termux 앱이 설정하는 공식 환경변수입니다.
    if os.environ.get("TERMUX_VERSION") or os.environ.get("TERMUX_APP_PID"):
        return True
    # PREFIX 는 Termux shell 에서 /data/data/com.termux/files/usr 로 설정됩니다.
    prefix = os.environ.get("PREFIX", "")
    if "com.termux" in prefix:
        return True
    # PRoot-distro 등 환경에서 PREFIX 가 설정되지 않을 수 있으므로 경로도 확인합니다.
    if os.path.exists("/data/data/com.termux"):
        return True
    return False


def is_proot() -> bool:
    """현재 런타임이 PRoot 가상화 환경인지 확인합니다.

    Returns:
        True  — PROOT_TMP_DIR 환경변수 설정 또는 binfmt_misc/proot 존재.
        False — 그 외 모든 환경.
    """
    if "PROOT_TMP_DIR" in os.environ:
        return True
    if os.path.exists("/proc/sys/fs/binfmt_misc/proot"):
        return True
    return False


def get_termux_prefix() -> str:
    """Termux PREFIX 디렉터리 경로를 반환합니다.

    Returns:
        PREFIX 환경변수 값 또는 기본값 "/data/data/com.termux/files/usr".
    """
    return os.environ.get("PREFIX", "/data/data/com.termux/files/usr")


def get_termux_home() -> str:
    """Termux HOME 디렉터리 경로를 반환합니다.

    Returns:
        HOME 환경변수 값 또는 기본값 "/data/data/com.termux/files/home".
    """
    return os.environ.get("HOME", "/data/data/com.termux/files/home")


_ARM_PART_MAP = {
    "0xd03": "Cortex-A53",
    "0xd05": "Cortex-A55",
    "0xd0b": "Cortex-A76",
    "0xd0d": "Cortex-A77",
    "0xd41": "Cortex-A78",
    "0xd44": "Cortex-X1",
    "0xd46": "Cortex-A510",
    "0xd47": "Cortex-A710",
    "0xd48": "Cortex-X2",
    "0xd4e": "Cortex-A715",
    "0xd4f": "Cortex-A720",
    "0xd4c": "Cortex-X3",
    "0xd49": "Cortex-X4",
}


def detect_soc_environment() -> SoCInfo:
    """하드웨어 SoC, GPU 계열 및 Termux CLI 런타임 제약 조건을 실시간 자동 감지합니다.

    Ground Truth 무결성 원칙:
    - Qualcomm Adreno (/dev/kgsl-3d0 rw 권한) -> Vulkan/Turnip CLI 직접 구동 가능.
    - Samsung Exynos / Mali (/dev/mali0) -> Termux CLI 환경에서 Surface 컨텍스트 부재로
      vkEnumeratePhysicalDevices 실패가 확정적이므로, 가짜 Vulkan 에러 뿜기 없이
      즉각 'ARM NEON 4-Thread FP16 CPU 모드'로 라우팅합니다.
    """
    cpu_cores = os.cpu_count() or 4
    cpu_info_text = ""
    vendor = "generic"
    chipname = "unknown"
    gpu_family = "generic"
    detected_parts = set()

    # 1. /proc/cpuinfo 파싱
    if os.path.exists("/proc/cpuinfo"):
        try:
            with open("/proc/cpuinfo", "r", encoding="utf-8", errors="replace") as f:
                cpu_info_text = f.read()
        except OSError:
            pass

    cpu_info_lower = cpu_info_text.lower()

    for line in cpu_info_text.splitlines():
        line_clean = line.strip()
        if line_clean.lower().startswith("cpu part"):
            parts = line_clean.split(":")
            if len(parts) == 2:
                detected_parts.add(parts[1].strip().lower())
        elif line_clean.lower().startswith("hardware"):
            parts = line_clean.split(":")
            if len(parts) == 2:
                chipname = parts[1].strip()

    # CPU 모델 결정
    cortex_models = [_ARM_PART_MAP[p] for p in detected_parts if p in _ARM_PART_MAP]
    if cortex_models:
        cpu_model = " + ".join(sorted(set(cortex_models)))
    else:
        cpu_model = "ARM64 Vector CPU"

    # 2. Android getprop 조회 (가능한 경우)
    prop_soc = ""
    prop_platform = ""
    if is_android() or is_termux():
        for prop_name in ["ro.soc.model", "ro.board.platform", "ro.hardware", "ro.chipname"]:
            try:
                val = subprocess.check_output(["getprop", prop_name], text=True, stderr=subprocess.DEVNULL).strip()
                if val:
                    prop_soc += " " + val.lower()
                    if chipname == "unknown":
                        chipname = val
            except Exception:
                pass

    combined_info = (cpu_info_lower + " " + prop_soc).strip()

    # 3. 벤더 및 GPU 계열 판별
    if any(q in combined_info for q in ["qualcomm", "qcom", "snapdragon", "sm8", "sm7", "sm6", "adreno"]):
        vendor = "qualcomm"
        gpu_family = "adreno"
    elif any(e in combined_info for e in ["exynos", "s5e", "universal", "samsung"]):
        vendor = "samsung_exynos"
        gpu_family = "mali"
    elif any(m in combined_info for m in ["mediatek", "mt6", "mt8", "dimensity"]):
        vendor = "mediatek"
        gpu_family = "mali"

    # 4. 장치 노드 실체 검사
    kgsl_path = "/dev/kgsl-3d0"
    mali_path = "/dev/mali0"

    kgsl_exists = os.path.exists(kgsl_path)
    kgsl_accessible = kgsl_exists and os.access(kgsl_path, os.R_OK | os.W_OK)
    mali_exists = os.path.exists(mali_path)

    if kgsl_exists and gpu_family == "generic":
        vendor = "qualcomm"
        gpu_family = "adreno"
    elif mali_exists and gpu_family == "generic":
        vendor = "samsung_exynos"
        gpu_family = "mali"

    # 5. Termux CLI 런타임 제약 및 최종 라우팅 결정 (Zero-Guesswork)
    termux_env = is_termux() or is_android()

    if gpu_family == "adreno" and (kgsl_accessible or kgsl_exists):
        can_direct_vulkan_cli = True
        recommended_backend = "vulkan"
        diagnosis_reason = (
            "Qualcomm Adreno GPU detected with accessible /dev/kgsl-3d0 node. "
            "Direct KGSL/Turnip Vulkan CLI hardware acceleration enabled."
        )
    elif gpu_family == "mali" and termux_env:
        # ARM Mali는 CLI 세션에서 Surface/Window 컨텍스트가 없으므로 vkEnumeratePhysicalDevices 실패 확정
        can_direct_vulkan_cli = False
        recommended_backend = "cpu_neon"
        diagnosis_reason = (
            f"{vendor.upper()} Mali GPU detected in Termux CLI. "
            "Headless Vulkan instance creation is restricted by driver/OS. "
            f"Routing directly to ARM NEON pure CPU engine ({cpu_model}) (Zero-Silent-Fallback)."
        )
    elif termux_env:
        can_direct_vulkan_cli = False
        recommended_backend = "cpu_neon"
        diagnosis_reason = (
            f"Generic Android SoC ({vendor}) in Termux CLI. "
            "Conservative routing to ARM NEON pure CPU engine."
        )
    else:
        # Non-Android (Host Linux / Windows / macOS)
        can_direct_vulkan_cli = True
        recommended_backend = "vulkan"
        diagnosis_reason = "Standard host platform with desktop GPU driver."

    return SoCInfo(
        vendor=vendor,
        chipname=chipname,
        gpu_family=gpu_family,
        kgsl_accessible=kgsl_accessible,
        mali_node_accessible=mali_exists,
        can_direct_vulkan_cli=can_direct_vulkan_cli,
        recommended_backend=recommended_backend,
        cpu_model=cpu_model,
        cpu_cores=cpu_cores,
        diagnosis_reason=diagnosis_reason,
    )

