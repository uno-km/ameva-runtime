"""
Termux-Train Core Session Manager
=================================
High-level session orchestrator binding training modalities with strict hardware telemetry.
"""
from __future__ import annotations

import logging
from typing import Optional

from .backends.base import BaseTrainer, TrainingConfig, TrainingResult
from .backends.diff import DiffusionTrainer
from .backends.llm import LlamaTrainer
from .backends.stt import WhisperTrainer
from .backends.tts import TtsTrainer
from .backends.vision import VisionTrainer
from .exceptions import TermuxTrainError
from .utils.hardware import HardwareProfile, probe_hardware

logger = logging.getLogger("termux_train.core")


class TrainingSession:
    """Orchestrates an edge-native on-device training run."""

    MODALITY_MAP = {
        "llm": LlamaTrainer,
        "stt": WhisperTrainer,
        "tts": TtsTrainer,
        "diff": DiffusionTrainer,
        "diffusion": DiffusionTrainer,
        "vision": VisionTrainer,
    }

    def __init__(self, modality: str, config: TrainingConfig):
        self.modality = modality.lower()
        self.config = config
        self.hardware: HardwareProfile = probe_hardware()

        if self.modality not in self.MODALITY_MAP:
            valid_modalities = ", ".join(self.MODALITY_MAP.keys())
            raise TermuxTrainError(
                f"알 수 없는 모달리티 '{modality}'. 지원 목록: {valid_modalities}",
                error_code="INVALID_MODALITY",
            )

        trainer_cls = self.MODALITY_MAP[self.modality]
        self.trainer: BaseTrainer = trainer_cls(config)

    def run(self) -> TrainingResult:
        """Executes the training session under zero-silent-fallback rules."""
        logger.info(
            "[TrainingSession] Starting training session for '%s' (Strict GPU: %s)",
            self.modality,
            self.config.strict_gpu,
        )
        return self.trainer.train()
