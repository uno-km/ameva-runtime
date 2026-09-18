"""
AMEVA Vulkan Runtime — Consumer Protocol 정의.

각 모달리티 패키지(termux-stt, termux-diffusion 등)가 반드시 구현해야 하는
표준 인터페이스 계약입니다.

설계 원칙:
- Layer 1 (ameva-vulkan-runtime) 은 Consumer 의 존재를 알지 못합니다.
- Consumer 가 VulkanContext 를 주입받아 자신의 엔진에 바인딩합니다.
- 역방향 의존성이 없으므로 순환 import 가 발생하지 않습니다.
"""
from __future__ import annotations

from typing import Any, Dict, Protocol, runtime_checkable

from .doctor import DiagnosticReport


from ameva_runtime.protocol import IVulkanConsumer, BindingResult, IRuntimeConsumer

__all__ = ["IVulkanConsumer", "IRuntimeConsumer", "BindingResult"]

