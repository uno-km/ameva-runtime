"""
Unit tests for LlamaCppAdapter adhering to Zero-Silent-Fallback and Fail-Fast standards.
"""
import unittest

from ameva_runtime.vulkan.doctor import DiagnosticReport
from ameva_runtime.vulkan.protocol import BindingResult
from ameva_runtime.vulkan.adapters.llamacpp import (
    LlamaCppAdapter,
    verify_vulkan_llm_output,
    _calculate_llama_layers,
    DEFAULT_GPU_LAYERS,
)
from ameva_runtime.exceptions import AmevaRuntimeError, PlatformNotSupportedError


class TestLlamaCppAdapterRefactored(unittest.TestCase):
    def setUp(self):
        self.vulkan_report = DiagnosticReport(
            overall_success=True,
            device_name="Mali-G78",
            driver_version="1.3.219",
            loader_path="/system/lib64/libvulkan.so",
            vendor_id=0x13B5,
            passed_stages=11,
            total_stages=11,
            total_elapsed_ms=4.5,
            recommended_backend="vulkan",
        )
        self.cpu_report = DiagnosticReport(
            overall_success=False,
            device_name="Cortex-A78",
            driver_version="",
            loader_path="",
            vendor_id=0x0,
            passed_stages=0,
            total_stages=11,
            total_elapsed_ms=0.1,
            recommended_backend="cpu_neon",
        )

    def test_p0_1_vulkan_skip_checks_not_injected(self):
        """P0-1 Fix: GGML_VULKAN_SKIP_CHECKS must never be injected by default."""
        env = LlamaCppAdapter.get_execution_environment()
        self.assertNotIn("GGML_VULKAN_SKIP_CHECKS", env)

    def test_p0_2_unsupported_backend_raises_error(self):
        """P0-2 Fix: Unsupported backend string must raise AmevaRuntimeError."""
        with self.assertRaises(AmevaRuntimeError) as ctx:
            LlamaCppAdapter.bind(report=self.vulkan_report, requested_backend="metal")
        self.assertIn("Unsupported backend", str(ctx.exception))

    def test_p0_2_explicit_vulkan_without_hardware_raises_platform_error(self):
        """P0-2 Fix: Explicit vulkan request without hardware MUST fail-fast without silent fallback."""
        with self.assertRaises(PlatformNotSupportedError):
            LlamaCppAdapter.bind(report=self.cpu_report, requested_backend="vulkan")

        with self.assertRaises(PlatformNotSupportedError):
            LlamaCppAdapter.bind(report=self.cpu_report, requested_backend="gpu")

    def test_p0_2_auto_backend_adaptive_routing_on_cpu(self):
        """P0-2: auto routes adaptively to cpu_neon when Vulkan is unavailable."""
        res = LlamaCppAdapter.bind(report=self.cpu_report, requested_backend="auto")
        self.assertEqual(res.backend, "cpu_neon")
        self.assertIn("CPU", res.status)

    def test_p0_3_conflicting_ngl_raises_error(self):
        """P0-3 / P0-6 Fix: Requesting Vulkan with requested_ngl <= 0 must fail fast."""
        with self.assertRaises(AmevaRuntimeError) as ctx:
            LlamaCppAdapter.bind(
                report=self.vulkan_report,
                requested_backend="vulkan",
                requested_ngl=0,
            )
        self.assertIn("Conflicting configuration", str(ctx.exception))

        with self.assertRaises(AmevaRuntimeError) as ctx:
            LlamaCppAdapter.bind(
                report=self.vulkan_report,
                requested_backend="vulkan",
                requested_ngl=-5,
            )
        self.assertIn("Conflicting configuration", str(ctx.exception))

    def test_p0_4_atomic_mutation_and_precedence(self):
        """P0-4 Fix: Dict engine mutation must set values deterministically."""
        engine_dict = {"ngl": 0, "device": "none"}
        res = LlamaCppAdapter.bind(engine=engine_dict, report=self.vulkan_report, requested_ngl=28)
        self.assertEqual(res.status, "BOUND_VULKAN")
        self.assertEqual(engine_dict["ngl"], 28)
        self.assertEqual(engine_dict["device"], "vulkan")

    def test_p1_4_status_is_bound_vulkan(self):
        """P1-4 Fix: Status must accurately reflect BOUND_VULKAN."""
        res = LlamaCppAdapter.bind(report=self.vulkan_report)
        self.assertEqual(res.status, "BOUND_VULKAN")
        self.assertTrue(res.is_vulkan)

    def test_p1_7_verify_vulkan_llm_output_with_returncode(self):
        """P1-7 Fix: verify_vulkan_llm_output must inspect returncode and stdout/stderr."""
        with self.assertRaises(AmevaRuntimeError):
            verify_vulkan_llm_output("any output", returncode=139)

        with self.assertRaises(PlatformNotSupportedError):
            verify_vulkan_llm_output("llama_model_load: falling back to cpu", returncode=0)

        with self.assertRaises(PlatformNotSupportedError):
            verify_vulkan_llm_output("Normal text without positive vulkan init", returncode=0)

        # Positive output
        verify_vulkan_llm_output("ggml_vulkan: using device Mali-G78\nGeneration successful", returncode=0)

    def test_build_cli_args_vulkan_and_fail_fast(self):
        """CLI argument assembly validation."""
        args = LlamaCppAdapter.build_cli_args(
            executable="llama-cli",
            model_path="/path/to/model.gguf",
            target_backend="vulkan",
            ngl_override=32,
            device_name="Mali-G78",
        )
        self.assertIn("-ngl", args)
        self.assertEqual(args[args.index("-ngl") + 1], "32")
        self.assertIn("--device", args)
        self.assertEqual(args[args.index("--device") + 1], "Mali-G78")

        with self.assertRaises(AmevaRuntimeError):
            LlamaCppAdapter.build_cli_args(
                executable="llama-cli",
                model_path="/path/to/model.gguf",
                target_backend="vulkan",
                ngl_override=0,
            )

        with self.assertRaises(AmevaRuntimeError):
            LlamaCppAdapter.build_cli_args(
                executable="llama-cli",
                model_path="/path/to/model.gguf",
                target_backend="invalid_backend",
            )


if __name__ == "__main__":
    unittest.main()