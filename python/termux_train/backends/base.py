"""
Termux-Train Base Trainer Interface
===================================
Abstract base trainer enforcing Zero-Silent-Fallback and Fail-Fast on-device standards.
"""
from __future__ import annotations

import abc
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ..exceptions import ExecutionEnvironmentError, GpuOperatorNotSupportedError
from ..utils.hardware import HardwareProfile, probe_hardware
from ..utils.monitor import ResourceMonitor

logger = logging.getLogger("termux_train.backends.base")


@dataclass
class TrainingConfig:
    """Standardized training parameters for edge-native adaptation."""
    model_path: str
    dataset_path: str
    output_dir: str
    lora_r: int = 8
    lora_alpha: int = 16
    batch_size: int = 1
    gradient_accumulation_steps: int = 16
    epochs: int = 3
    learning_rate: float = 1e-4
    save_every_steps: int = 50
    strict_gpu: bool = True
    context_length: int = 256
    threads: int = 4
    extra_args: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CheckpointInfo:
    step: int
    epoch: int
    loss: float
    checkpoint_path: str
    timestamp: float


@dataclass
class TrainingResult:
    success: bool
    total_steps: int
    final_loss: float
    duration_seconds: float
    output_artifact_path: str
    checkpoints: List[CheckpointInfo] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class BaseTrainer(abc.ABC):
    """Abstract edge trainer enforcing pure GPU offload and strict hardware boundary compliance."""

    def __init__(self, config: TrainingConfig):
        self.config = config
        self.hardware: HardwareProfile = probe_hardware()
        self.monitor = ResourceMonitor()
        self.checkpoints: List[CheckpointInfo] = []

    def validate_environment(self) -> None:
        """Validates physical hardware and binary prerequisites without silent fallback."""
        if not os.path.exists(self.config.model_path):
            raise ExecutionEnvironmentError(
                missing_component=f"Model weights file '{self.config.model_path}'",
                path_searched=os.path.abspath(self.config.model_path),
            )
        if not os.path.exists(self.config.dataset_path):
            raise ExecutionEnvironmentError(
                missing_component=f"Training dataset '{self.config.dataset_path}'",
                path_searched=os.path.abspath(self.config.dataset_path),
            )

        if self.config.strict_gpu:
            if not self.hardware.has_vulkan:
                raise ExecutionEnvironmentError(
                    missing_component="Native Vulkan Driver (/system/lib64/libvulkan.so)",
                    path_searched="/system/lib64, /vendor/lib64",
                )

        os.makedirs(self.config.output_dir, exist_ok=True)

    @abc.abstractmethod
    def resolve_binary_path(self) -> str:
        """Locates the native training binary on the host system."""
        raise NotImplementedError

    @abc.abstractmethod
    def build_command(self) -> List[str]:
        """Assembles the exact CLI invocation for the native training binary."""
        raise NotImplementedError

    @abc.abstractmethod
    def train(self) -> TrainingResult:
        """Executes the training pipeline with active hardware boundary monitoring."""
        raise NotImplementedError
