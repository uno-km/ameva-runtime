"""
Unit Tests for Termux-Train (Edge-Native Training Framework)
===========================================================
Validates Zero-Silent-Fallback, Fail-Fast mechanics, and GPU training pipeline standards.
"""
from __future__ import annotations

import os
import tempfile
import pytest

from termux_train.backends.base import TrainingConfig
from termux_train.backends.llm import LlamaTrainer
from termux_train.core import TrainingSession
from termux_train.exceptions import (
    ExecutionEnvironmentError,
    GpuOperatorNotSupportedError,
    TermuxTrainError,
    ThermalThrottledError,
    TrainingOutOfMemoryError,
)
from termux_train.utils.hardware import HardwareProfile, get_system_ram_mb, probe_hardware
from termux_train.utils.monitor import ResourceMonitor


def test_exception_hierarchy():
    err = GpuOperatorNotSupportedError(
        operator_name="OPT_STEP_ADAMW",
        backend="Vulkan",
        unsupported_reasons=["Op code 42 not supported in vulkan-shaders/matmul.comp"],
    )
    assert err.error_code == "GPU_OPERATOR_UNSUPPORTED"
    assert "OPT_STEP_ADAMW" in str(err)
    assert "침묵 폴백이 금지" in str(err)


def test_hardware_probe():
    total_ram, avail_ram = get_system_ram_mb()
    assert total_ram > 0
    assert avail_ram >= 0

    hw = probe_hardware()
    assert isinstance(hw, HardwareProfile)
    assert hw.total_ram_mb > 0


def test_resource_monitor_limits():
    # Force memory limit violation
    monitor = ResourceMonitor(min_safe_ram_mb=9999999)
    with pytest.raises(TrainingOutOfMemoryError) as exc_info:
        monitor.check_safety_limits(stage="unit_test")
    assert exc_info.value.error_code == "TRAIN_OOM_PREVENTED"

    # Force thermal violation
    monitor_thermal = ResourceMonitor(max_safe_temp_c=-10.0)
    # Only if temperature can be read on host
    current_temp = monitor_thermal.read_battery_temperature_c()
    if current_temp is not None:
        with pytest.raises(ThermalThrottledError) as exc_info:
            monitor_thermal.check_safety_limits(stage="unit_test")
        assert exc_info.value.error_code == "THERMAL_LIMIT_EXCEEDED"


def test_llama_trainer_command_building():
    with tempfile.NamedTemporaryFile(suffix=".gguf") as f_model, \
         tempfile.NamedTemporaryFile(suffix=".txt") as f_data:

        config = TrainingConfig(
            model_path=f_model.name,
            dataset_path=f_data.name,
            output_dir="./test_out",
            lora_r=16,
            lora_alpha=32,
            batch_size=1,
            gradient_accumulation_steps=8,
            epochs=2,
            threads=6,
            context_length=512,
            strict_gpu=True,
        )

        trainer = LlamaTrainer(config)

        # Mock binary resolution for unit test environment
        trainer.resolve_binary_path = lambda: "/usr/bin/llama-finetune"

        cmd = trainer.build_command()
        assert "/usr/bin/llama-finetune" in cmd
        assert "--model-base" in cmd
        assert f_model.name in cmd
        assert "--lora-r" in cmd
        assert "16" in cmd
        assert "--grad-accum" in cmd
        assert "8" in cmd
        assert "-ngl" in cmd
        assert "99" in cmd


def test_strict_gpu_validation_fails_fast_when_vulkan_missing():
    with tempfile.NamedTemporaryFile(suffix=".gguf") as f_model, \
         tempfile.NamedTemporaryFile(suffix=".txt") as f_data:

        config = TrainingConfig(
            model_path=f_model.name,
            dataset_path=f_data.name,
            output_dir="./test_out",
            strict_gpu=True,
        )

        trainer = LlamaTrainer(config)
        # Force hardware profile to show no vulkan
        trainer.hardware = HardwareProfile(
            has_vulkan=False,
            vulkan_lib_path=None,
            is_unified_memory=True,
            total_ram_mb=8192,
            available_ram_mb=4096,
            vendor_id=0,
            device_name="MockDevice",
        )

        with pytest.raises(ExecutionEnvironmentError) as exc_info:
            trainer.validate_environment()
        assert exc_info.value.error_code == "ENV_COMPONENT_MISSING"
        assert "libvulkan.so" in str(exc_info.value)


def test_roadmap_stubs_fail_fast():
    with tempfile.NamedTemporaryFile(suffix=".gguf") as f_model, \
         tempfile.NamedTemporaryFile(suffix=".txt") as f_data:

        config = TrainingConfig(
            model_path=f_model.name,
            dataset_path=f_data.name,
            output_dir="./test_out",
        )

        # Check STT, TTS, Diff, Vision fail-fast stubs
        for modality in ["stt", "tts", "diff", "vision"]:
            session = TrainingSession(modality=modality, config=config)
            with pytest.raises(TermuxTrainError) as exc_info:
                session.run()
            assert exc_info.value.error_code == "STAGE_NOT_YET_IMPLEMENTED"
