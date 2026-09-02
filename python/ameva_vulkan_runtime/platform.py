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
from typing import Optional

__all__ = [
    "is_android",
    "is_termux",
    "is_proot",
    "get_termux_prefix",
    "get_termux_home",
]


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
