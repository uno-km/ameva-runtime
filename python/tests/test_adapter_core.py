"""
AMEVA Runtime Core Adapter Framework Test Suite (test_adapter_core.py)
Validates 2-tier mutation, non-destructive stateless snapshot restore, CLI argument slicing,
stdout isolation against false positives, and complete elimination of GGML_VULKAN_SKIP_CHECKS.
"""
import os
import unittest
from types import SimpleNamespace
from typing import Any

from ameva_runtime.adapters.base import (
    BaseAdapter,
    _make_cpu_binding,
    _set_engine_property,
    _restore_engine_properties,
    DEFAULT_CPU_FALLBACK_PATTERNS,
)
from ameva_runtime.adapters.llamacpp import LlamaCppAdapter
from ameva_runtime.adapters.vision import VisionAdapter
from ameva_runtime.adapters.bitnet import BitnetAdapter
from ameva_runtime.adapters.diffusion import DiffusionAdapter
from ameva_runtime.adapters.tts import TtsAdapter
from ameva_runtime.adapters.stt import SttAdapter
from ameva_runtime.doctor import DiagnosticReport
from ameva_runtime.protocol import BindingResult
from ameva_runtime.exceptions import AmevaRuntimeError, PlatformNotSupportedError


class DummySubAdapter(BaseAdapter):
    """Concrete test adapter subclassing BaseAdapter."""
    module_name = "test-adapter"
    device_signatures = ("vulkan", "radv", "turnip", "mali")


def _create_mock_vulkan_report() -> DiagnosticReport:
    return DiagnosticReport(
        device_name="Mali-G68 MC4",
        vendor_id=0x13B5,
        overall_success=True,
        recommended_backend="vulkan",
        passed_stages=11,
        total_stages=11,
        loader_path="/system/lib64/libvulkan.so",
    )


def _create_mock_cpu_report() -> DiagnosticReport:
    return DiagnosticReport(
        device_name="Generic ARM64 CPU",
        vendor_id=0,
        overall_success=False,
        recommended_backend="cpu_neon",
        passed_stages=0,
        total_stages=11,
        loader_path="",
    )


class TestAdapterCoreFramework(unittest.TestCase):
    """Comprehensive test case for BaseAdapter architectural invariants."""

    def setUp(self):
        self.vk_report = _create_mock_vulkan_report()
        self.cpu_report = _create_mock_cpu_report()

    def test_complete_elimination_of_skip_checks_across_all_six_adapters(self):
        """P0: Ensure GGML_VULKAN_SKIP_CHECKS is NEVER injected in get_execution_environment across all 6 adapters."""
        adapters = [
            LlamaCppAdapter,
            VisionAdapter,
            BitnetAdapter,
            DiffusionAdapter,
            TtsAdapter,
            SttAdapter,
            DummySubAdapter,
        ]
        for adapter in adapters:
            # Test with clean base_env
            env = adapter.get_execution_environment({})
            self.assertNotIn(
                "GGML_VULKAN_SKIP_CHECKS",
                env,
                f"Adapter {adapter.__name__} injected forbidden GGML_VULKAN_SKIP_CHECKS",
            )
            # Test with pre-polluted env
            polluted_env = {"GGML_VULKAN_SKIP_CHECKS": "999999999"}
            cleaned = adapter.get_execution_environment(polluted_env)
            self.assertNotIn(
                "GGML_VULKAN_SKIP_CHECKS",
                cleaned,
                f"Adapter {adapter.__name__} failed to cleanse GGML_VULKAN_SKIP_CHECKS",
            )

    def test_non_llm_adapters_do_not_suffer_ngl_pollution(self):
        """P1: Ensure non-LLM modalities (Vision, Diffusion, TTS) do not inject 'ngl' or 'n_gpu_layers'."""
        engine_vision = SimpleNamespace(device="cpu", use_gpu=False)
        res_v = VisionAdapter.bind(engine_vision, self.vk_report)
        self.assertFalse(hasattr(engine_vision, "ngl"))
        self.assertFalse(hasattr(engine_vision, "n_gpu_layers"))
        self.assertNotIn("ngl", res_v.config)
        self.assertNotIn("n_gpu_layers", res_v.config)

        engine_tts = SimpleNamespace(device="cpu")
        res_t = TtsAdapter.bind(engine_tts, self.vk_report)
        self.assertFalse(hasattr(engine_tts, "ngl"))
        self.assertFalse(hasattr(engine_tts, "n_gpu_layers"))
        self.assertNotIn("ngl", res_t.config)
        self.assertNotIn("n_gpu_layers", res_t.config)

        engine_diff = SimpleNamespace(hw_profile=SimpleNamespace(vulkan_available=False, vulkan_driver=None))
        res_d = DiffusionAdapter.bind(engine_diff, self.vk_report)
        self.assertFalse(hasattr(engine_diff, "ngl"))
        self.assertNotIn("ngl", res_d.config)

    def test_stateless_snapshot_and_non_destructive_restore_for_dict(self):
        """P2: Validate stateless snapshot capture and exact rollback for dict engines."""
        original_dict = {
            "device": "cpu",
            "threads": 2,
            "ngl": 0,
            "custom_metadata": "keep_intact",
            "env": {"LD_LIBRARY_PATH": "/custom/path"},
        }
        engine = dict(original_dict)
        res = LlamaCppAdapter.bind(engine, self.vk_report)
        self.assertEqual(res.status, "CONFIGURED_VULKAN")
        self.assertEqual(engine["device"], "vulkan")
        self.assertGreater(engine["ngl"], 0)
        self.assertNotIn("LD_LIBRARY_PATH", engine.get("env", {}))

        # Perform unbind with token
        LlamaCppAdapter.unbind(engine, res)
        self.assertEqual(engine["device"], "cpu")
        self.assertEqual(engine["threads"], 2)
        self.assertEqual(engine["ngl"], 0)
        self.assertEqual(engine["custom_metadata"], "keep_intact")

    def test_stateless_snapshot_and_non_destructive_restore_for_objects(self):
        """P2: Validate stateless snapshot capture and exact rollback for class/config objects."""
        class MockConfig:
            def __init__(self):
                self.device = "cpu"
                self.ngl = 0
                self.flash_attn = False
                self.threads = 2

        class MockEngine:
            def __init__(self):
                self.config = MockConfig()
                self.session_id = "sess_001"

        engine = MockEngine()
        res = LlamaCppAdapter.bind(engine, self.vk_report)
        self.assertEqual(engine.config.device, "vulkan")
        self.assertGreater(engine.config.ngl, 0)
        self.assertTrue(engine.config.flash_attn)

        # Unbind
        LlamaCppAdapter.unbind(engine, res)
        self.assertEqual(engine.config.device, "cpu")
        self.assertEqual(engine.config.ngl, 0)
        self.assertFalse(engine.config.flash_attn)
        self.assertEqual(engine.session_id, "sess_001")

    def test_cli_argument_list_slice_snapshot_and_non_destructive_restore(self):
        """P2: Validate CLI arguments list slice recovery without clearing original user arguments."""
        initial_cmd = ["termux-llama", "-m", "/data/model.gguf", "-p", "hello world"]
        original_len = len(initial_cmd)
        cmd_copy = list(initial_cmd)

        res = LlamaCppAdapter.bind(cmd_copy, self.vk_report)
        self.assertGreater(len(cmd_copy), original_len)
        self.assertIn("-ngl", cmd_copy)
        self.assertIn("--device", cmd_copy)

        # Unbind must slice back exactly to original length
        LlamaCppAdapter.unbind(cmd_copy, res)
        self.assertEqual(cmd_copy, initial_cmd)
        self.assertEqual(len(cmd_copy), original_len)

    def test_stdout_isolation_prevents_false_positive(self):
        """P3: Validates that stdout text mentioning fallback or error keywords does NOT trigger false alarm."""
        # Simulated stdout containing technical user dialogue regarding fallback
        chat_stdout = (
            "User: How does silent cpu fallback work in llama.cpp?\n"
            "Assistant: Silent fallback to cpu occurs when vulkan initialization failed.\n"
        )
        clean_stderr = (
            "ggml_vulkan: Found 1 Vulkan device(s)\n"
            "ggml_vulkan: Device 0: Mali-G68 MC4 | vendor 0x13b5 | compute\n"
            "llm_load_tensors: offloaded 33/33 layers to Vulkan\n"
        )

        # Must pass cleanly without raising PlatformNotSupportedError
        try:
            DummySubAdapter.verify_vulkan_compute_output(
                stdout=chat_stdout,
                stderr=clean_stderr,
                returncode=0,
            )
        except Exception as exc:
            self.fail(f"verify_vulkan_compute_output raised unexpected exception on benign stdout: {exc}")

    def test_stderr_actual_fallback_triggers_immediate_fail_fast(self):
        """P3: Validates that true fallback keywords in stderr trigger PlatformNotSupportedError immediately."""
        stdout = "Some text"
        err_with_fallback = (
            "ggml_vulkan: failed to allocate memory\n"
            "llama_model_load: falling back to cpu\n"
        )

        with self.assertRaises(PlatformNotSupportedError) as ctx:
            DummySubAdapter.verify_vulkan_compute_output(
                stdout=stdout,
                stderr=err_with_fallback,
                returncode=0,
            )
        self.assertIn("Silent CPU fallback detected", str(ctx.exception))

    def test_non_zero_exit_code_triggers_ameva_runtime_error(self):
        """P3: Validates that process crash (returncode != 0) triggers AmevaRuntimeError before pattern checks."""
        with self.assertRaises(AmevaRuntimeError) as ctx:
            DummySubAdapter.verify_vulkan_compute_output(
                stdout="Output before segfault",
                stderr="Segmentation fault (SIGSEGV)",
                returncode=139,
            )
        self.assertIn("failed with exit code 139", str(ctx.exception))

    def test_absence_of_vulkan_signature_in_stderr_triggers_fail_fast(self):
        """P3: Validates that absence of positive Vulkan signature in stderr triggers PlatformNotSupportedError."""
        stderr_silent = "llama_model_load: loaded 33 tensors into memory\n"
        with self.assertRaises(PlatformNotSupportedError) as ctx:
            DummySubAdapter.verify_vulkan_compute_output(
                stdout="",
                stderr=stderr_silent,
                returncode=0,
                expected_signatures=("vulkan", "mali"),
            )
        self.assertIn("No positive Vulkan device signature logged", str(ctx.exception))

    def test_multithread_safe_stateless_tokens(self):
        """P4: Validates that multiple independent bindings do not cross-contaminate state."""
        dict_1 = {"device": "cpu", "ngl": 0, "id": 1}
        dict_2 = {"device": "cpu", "ngl": 10, "id": 2}

        res_1 = LlamaCppAdapter.bind(dict_1, self.vk_report)
        res_2 = LlamaCppAdapter.bind(dict_2, self.vk_report)

        self.assertEqual(dict_1["device"], "vulkan")
        self.assertEqual(dict_2["device"], "vulkan")

        # Unbind dict_1 only
        LlamaCppAdapter.unbind(dict_1, res_1)
        self.assertEqual(dict_1["device"], "cpu")
        self.assertEqual(dict_1["ngl"], 0)

        # dict_2 should remain vulkan until unbind
        self.assertEqual(dict_2["device"], "vulkan")
        LlamaCppAdapter.unbind(dict_2, res_2)
        self.assertEqual(dict_2["device"], "cpu")
        self.assertEqual(dict_2["ngl"], 10)


if __name__ == "__main__":
    unittest.main()
