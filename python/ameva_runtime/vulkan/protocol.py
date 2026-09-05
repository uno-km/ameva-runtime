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


@runtime_checkable
class IVulkanConsumer(Protocol):
    """Vulkan 가속 바인딩 소비자 인터페이스.

    각 모달리티 어댑터는 이 Protocol 을 구현합니다.
    `runtime_checkable` 이므로 `isinstance(obj, IVulkanConsumer)` 로 확인 가능합니다.
    """

    @property
    def module_name(self) -> str:
        """소비자 패키지 식별자 (예: 'termux-stt')."""
        ...

    def bind(self, engine: Any, report: DiagnosticReport) -> "BindingResult":
        """엔진 인스턴스에 Vulkan 가속을 바인딩합니다.

        Args:
            engine: 실제 엔진 인스턴스 (WhisperEngine, BitNetEngine 등).
                    None 이 전달되면 구성 정보만 반환합니다.
            report: ameva Doctor 가 계측한 DiagnosticReport.
                    vendorID, device_name, recommended_backend 등을 참조합니다.

        Returns:
            BindingResult — 바인딩 상태 및 활성화된 설정.

        Raises:
            AmevaRuntimeError: 바인딩이 실패하여 실행을 계속할 수 없는 경우.
        """
        ...

    def unbind(self) -> None:
        """Vulkan 리소스를 해제하고 바인딩을 해제합니다.

        RAII 원칙에 따라 명시적 해제가 요청될 때 호출됩니다.
        예외를 절대 raise 하지 않아야 합니다 — 로그만 기록하고 리턴합니다.
        """
        ...


class BindingResult:
    """Vulkan 바인딩 결과 값 객체.

    불변(immutable) 데이터 컨테이너입니다.
    모든 필드는 생성 후 변경할 수 없습니다.
    """

    __slots__ = (
        "_module",
        "_backend",
        "_is_vulkan",
        "_device_name",
        "_vendor_id",
        "_config",
        "_status",
    )

    def __init__(
        self,
        module: str,
        backend: str,
        is_vulkan: bool,
        device_name: str,
        vendor_id: int,
        config: Dict[str, Any],
        status: str = "BOUND",
    ) -> None:
        self._module = module
        self._backend = backend
        self._is_vulkan = is_vulkan
        self._device_name = device_name
        self._vendor_id = vendor_id
        self._config = dict(config)  # 방어적 복사
        self._status = status

    @property
    def module(self) -> str:
        return self._module

    @property
    def backend(self) -> str:
        return self._backend

    @property
    def is_vulkan(self) -> bool:
        return self._is_vulkan

    @property
    def device_name(self) -> str:
        return self._device_name

    @property
    def vendor_id(self) -> int:
        return self._vendor_id

    @property
    def config(self) -> Dict[str, Any]:
        return dict(self._config)  # 방어적 복사 반환

    @property
    def status(self) -> str:
        return self._status

    def __repr__(self) -> str:
        return (
            f"BindingResult(module={self._module!r}, backend={self._backend!r}, "
            f"vulkan={self._is_vulkan}, device={self._device_name!r}, status={self._status!r})"
        )
