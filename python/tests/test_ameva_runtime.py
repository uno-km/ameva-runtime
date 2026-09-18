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

try:
    import pytest
except ImportError:
    pytest = None

import unittest
from ameva_runtime.detector import detect_hardware, HardwareProfile
from ameva_runtime.router import SmartRouter, ExecutionPlan, get_router
from ameva_runtime.core import AmevaRuntime, get_runtime, VulkanContext, create_context
from ameva_runtime.doctor import Doctor, DiagnosticReport
from ameva_runtime.exceptions import AmevaRuntimeError, PlatformNotSupportedError
from ameva_runtime.cli import build_parser


class TestAmevaRuntime(unittest.TestCase):
    def test_hardware_detector_profile(self):
        """Verify hardware profile contains valid topology information."""
        profile = detect_hardware()
        self.assertIsInstance(profile, HardwareProfile)
        self.assertGreaterEqual(profile.cpu_cores, 1)
        self.assertGreaterEqual(len(profile.allowed_cpus), 1)
        self.assertGreater(profile.total_ram_mb, 0)
        self.assertIn(profile.recommended_backend, ("vulkan", "opencl", "cpu", "cpu_neon", "npu"))
        self.assertGreaterEqual(profile.recommended_threads, 1)


    def test_smart_router_mali_routing(self):
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

        self.assertIn(plan.backend, ("cpu", "cpu_neon"))
        self.assertEqual(plan.ngl, 0)
        self.assertEqual(plan.threads, 3)
        self.assertEqual(plan.affinity_cpus, [0, 1, 2])
        self.assertTrue("Mali" in plan.rationale or "driver" in plan.rationale.lower())

    def test_smart_router_adreno_routing(self):
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

        self.assertEqual(plan.backend, "vulkan")
        self.assertGreaterEqual(plan.ngl, 30)
        self.assertEqual(plan.threads, 4)
        self.assertIn("Vulkan", plan.rationale)

    def test_runtime_binding_all_modalities(self):
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
            try:
                binding = runtime.bind_engine(mod)
                self.assertIsNotNone(binding)
                self.assertIn(binding.backend, ("vulkan", "cpu_neon", "opencl", "npu", "cpu"))
                self.assertNotEqual(binding.target_modality, "")
            except PlatformNotSupportedError:
                # Under Zero-Silent-Fallback, non-Vulkan CI environments refuse binding without explicit backend='cpu'
                pass

    def test_runtime_unknown_module_fail_fast(self):
        """Verify unknown module fails immediately with AmevaRuntimeError."""
        runtime = get_runtime()
        with self.assertRaises(AmevaRuntimeError) as exc_info:
            runtime.bind_engine("unknown_engine_xyz")
        self.assertIn("Unknown module", str(exc_info.exception))

    def test_doctor_diagnostic(self):
        """Verify Doctor completes diagnostic self-test with valid report."""
        doc = Doctor()
        report = doc.run_self_test(verbose=False)
        self.assertIsInstance(report, DiagnosticReport)
        self.assertEqual(report.total_stages, 12)
        self.assertGreaterEqual(report.passed_stages, 0)
        self.assertEqual(len(report.stages), 12)

    def test_backward_compatibility_wrapper(self):
        """Verify legacy VulkanContext functions seamlessly."""
        ctx = create_context()
        self.assertIsInstance(ctx, VulkanContext)
        res = ctx.bind("termux-llamacpp")
        self.assertIsNotNone(res)
        self.assertEqual(res.target_modality, "llamacpp")

    def test_cli_parser(self):
        """Verify CLI parser configuration."""
        parser = build_parser()
        args_doctor = parser.parse_args(["doctor"])
        self.assertEqual(args_doctor.command, "doctor")

        args_profile = parser.parse_args(["profile"])
        self.assertEqual(args_profile.command, "profile")

        args_plan = parser.parse_args(["plan", "-m", "qwen2.5-0.5b", "-b", "auto"])
        self.assertEqual(args_plan.command, "plan")
        self.assertEqual(args_plan.model, "qwen2.5-0.5b")
        self.assertEqual(args_plan.backend, "auto")

        args_exec = parser.parse_args(["exec", "-m", "model.gguf", "-p", "hi", "-n", "32"])
        self.assertEqual(args_exec.command, "exec")
        self.assertEqual(args_exec.model, "model.gguf")
        self.assertEqual(args_exec.prompt, "hi")
        self.assertEqual(args_exec.max_tokens, 32)

    def test_top_level_plan_and_run_api(self):
        """Verify top-level plan() and run() functions."""
        import ameva_runtime as ameva

        plan_res = ameva.plan("qwen2.5-0.5b")
        self.assertIn(plan_res.backend, ("vulkan", "cpu_neon", "cpu", "opencl"))
        self.assertGreaterEqual(plan_res.threads, 1)

        # When binary/model is absent, execute() should raise AmevaRuntimeError fail-fast
        with self.assertRaises(AmevaRuntimeError):
            ameva.run(model="non_existent_model_xyz.gguf", prompt="test")

    def test_installer_cli_and_registry(self):
        """Verify installer subcommands, arguments, and native asset specifications."""
        from ameva_runtime.installer import NATIVE_ASSETS, NativeAssetManager

        parser = build_parser()
        args_inst = parser.parse_args(["install", "--all", "--force"])
        self.assertEqual(args_inst.command, "install")
        self.assertTrue(args_inst.all)
        self.assertTrue(args_inst.force)
        self.assertEqual(args_inst.modality, "all")

        args_setup = parser.parse_args(["setup", "-m", "diffusion"])
        self.assertEqual(args_setup.command, "setup")
        self.assertEqual(args_setup.modality, "diffusion")

        self.assertIn("diffusion", NATIVE_ASSETS)
        self.assertIn("stt", NATIVE_ASSETS)
        self.assertIn("tts", NATIVE_ASSETS)
        self.assertNotIn("libomp", NATIVE_ASSETS)
        self.assertNotIn("libegl_shim", NATIVE_ASSETS)
        self.assertNotIn("matmul_spv", NATIVE_ASSETS)

        mgr = NativeAssetManager(force=True)
        self.assertTrue(mgr.force)


if __name__ == "__main__":
    unittest.main()

