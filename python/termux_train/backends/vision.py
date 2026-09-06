"""
Termux-Train Vision Backend (YOLO-nano / ViT Head) — Planned Stage 5
===================================================================
"""
from __future__ import annotations

from .base import BaseTrainer, TrainingConfig, TrainingResult
from ..exceptions import TermuxTrainError


class VisionTrainer(BaseTrainer):
    """YOLO-nano / ViT Projection Head fine-tuning backend (Roadmap Stage 5)."""

    def resolve_binary_path(self) -> str:
        raise TermuxTrainError(
            "Vision (YOLO/ViT Head) 트레이너는 로드맵 제5단계 예정 항목입니다.",
            error_code="STAGE_NOT_YET_IMPLEMENTED",
        )

    def build_command(self) -> list[str]:
        raise NotImplementedError

    def train(self) -> TrainingResult:
        raise TermuxTrainError(
            "Vision 학습 파이프라인은 제4단계(Diffusion) 완료 후 활성화됩니다.",
            error_code="STAGE_NOT_YET_IMPLEMENTED",
        )
