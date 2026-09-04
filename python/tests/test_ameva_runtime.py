"""
Comprehensive Unit Tests for ameva_runtime (v2.0.0)
===================================================
Tests hardware detector, SmartRouter, multi-modal adapters,
Zero-Silent-Fallback enforcement, and backward compatibility.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest

from ameva_runtime.detector import detect_hardware, HardwareProfile
from ameva_runtime.router import SmartRouter, ExecutionPlan, get_router
from ameva_runtime.core import AmevaRuntime, get_runtime, VulkanContext, create_context
from ameva_runtime.doctor import Doctor, DiagnosticReport
from ameva_runtime.exceptions import AmevaRuntimeError, PlatformNotSupportedError
from ameva_runtime.cli import build_parser


def test_hardware_detector_profile():
    """Verify hardware profile contains valid topology information."""
    profile = detect_hardware()
    assert isinstance(profile, HardwareProfile)
    assert profile.cpu_cores >= 1
    assert len(profile.allowed_cpus) >= 1
    assert profile.total_ram_mb > 0
    assert profile.recommended_backend in ("vulkan", "opencl", "cpu", "npu")
    assert profile.recommended_threads >= 1


def test_smart_router_mali_routing():
    """Verify SmartRouter routes Mali GPU safely to CPU-NEON to prevent lockup."""
    mali_profile = HardwareProfile(
        vendor="Samsung",
        arch="arm64-v8a",
        soc_model="Exynos 1380",
        gpu_family="ARM Mali-G68",
        driver_version="vulkan-1.3",
        has_vulkan_loader=True,
        has_opencl=True,
        has_npu=False,
        total_cpu_cores=8,
        allowed_cpus=[0, 1, 2],
        is_cgroup_restrained=True,
        total_ram_mb=8192,
        available_ram_mb=4096,
        recommended_backend="cpu",
        recommended_threads=3,
    )
    router = SmartRouter(mali_profile)
    plan = router.route_for_llm("qwen2.5-0.5b")

    assert plan.backend in ("cpu", "cpu_neon")
    assert plan.ngl == 0
    assert plan.threads == 3
    assert plan.affinity_cpus == [0, 1, 2]
    assert "Mali" in plan.rationale or "driver" in plan.rationale.lower()


def test_smart_router_adreno_routing():
    """Verify SmartRouter routes Adreno GPU to Vulkan acceleration."""
    adreno_profile = HardwareProfile(
        vendor="Qualcomm",
        arch="arm64-v8a",
        soc_model="Snapdragon 8 Gen 2",
        gpu_family="Qualcomm Adreno 740",
        driver_version="vulkan-1.3",
        has_vulkan_loader=True,
        has_opencl=True,
        has_npu=True,
        total_cpu_cores=8,
        allowed_cpus=list(range(8)),
        is_cgroup_restrained=False,
        total_ram_mb=12288,
        available_ram_mb=8192,
        recommended_backend="vulkan",
        recommended_threads=4,
    )
    router = SmartRouter(adreno_profile)
    plan = router.route_for_llm("qwen2.5-7b")

    assert plan.backend == "vulkan"
    assert plan.ngl >= 30
    assert plan.threads == 4
    assert "Vulkan" in plan.rationale


def test_runtime_binding_all_modalities():
    """Verify AmevaRuntime binds all 6 modalities properly."""
    runtime = get_runtime()

    modalities = [
        "termux-llamacpp",
        "termux-vision",
        "termux-diffusion",
        "termux-stt",
        "termux-tts",
        "termux-bitnet",
    ]
    for mod in modalities:
        binding = runtime.bind_engine(mod)
        assert binding is not None
        assert binding.backend in ("vulkan", "cpu_neon", "opencl", "npu", "cpu")
        assert binding.target_modality != ""


def test_runtime_unknown_module_fail_fast():
    """Verify unknown module fails immediately with AmevaRuntimeError."""
    runtime = get_runtime()
    with pytest.raises(AmevaRuntimeError) as exc_info:
        runtime.bind_engine("unknown_engine_xyz")
    assert "Unknown module" in str(exc_info.value)


def test_doctor_diagnostic():
    """Verify Doctor completes diagnostic self-test with valid report."""
    doc = Doctor()
    report = doc.run_self_test(verbose=False)
    assert isinstance(report, DiagnosticReport)
    assert report.total_stages == 12
    assert report.passed_stages >= 0
    assert len(report.stages) == 12


def test_backward_compatibility_wrapper():
    """Verify legacy VulkanContext functions seamlessly."""
    ctx = create_context()
    assert isinstance(ctx, VulkanContext)
    res = ctx.bind("termux-llamacpp")
    assert res is not None
    assert res.target_modality == "llamacpp"


def test_cli_parser():
    """Verify CLI parser configuration."""
    parser = build_parser()
    args_doctor = parser.parse_args(["doctor"])
    assert args_doctor.command == "doctor"

    args_profile = parser.parse_args(["profile"])
    assert args_profile.command == "profile"

    args_plan = parser.parse_args(["plan", "-m", "qwen2.5-0.5b", "-b", "auto"])
    assert args_plan.command == "plan"
    assert args_plan.model == "qwen2.5-0.5b"
    assert args_plan.backend == "auto"

    args_exec = parser.parse_args(["exec", "-m", "model.gguf", "-p", "hi", "-n", "32"])
    assert args_exec.command == "exec"
    assert args_exec.model == "model.gguf"
    assert args_exec.prompt == "hi"
    assert args_exec.max_tokens == 32


def test_top_level_plan_and_run_api():
    """Verify top-level plan() and run() functions."""
    import ameva_runtime as ameva

    plan_res = ameva.plan("qwen2.5-0.5b")
    assert plan_res.backend in ("vulkan", "cpu_neon", "cpu", "opencl")
    assert plan_res.threads >= 1

    # When binary/model is absent, execute() should raise AmevaRuntimeError fail-fast
    with pytest.raises(AmevaRuntimeError):
        ameva.run(model="non_existent_model_xyz.gguf", prompt="test")

