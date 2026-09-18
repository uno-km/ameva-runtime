"""
Unit tests verifying strict Zero-Silent-Fallback model resolution and LLM adapter invariants.
"""
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from ameva_runtime.core import resolve_model_path, plan, AmevaRuntime
from ameva_runtime.exceptions import (
    ModelNotFoundError,
    AmbiguousModelMatchError,
    PlatformNotSupportedError,
)
from ameva_runtime.adapters.llamacpp import (
    LlamaCppAdapter,
    verify_vulkan_llm_output,
    _calculate_llama_layers,
)
from ameva_runtime.doctor import DiagnosticReport


class TestZeroSilentFallbackModelResolution(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.models_dir = Path(self.temp_dir.name) / "models"
        self.models_dir.mkdir(parents=True, exist_ok=True)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_empty_model_name_raises_model_not_found(self):
        with self.assertRaises(ModelNotFoundError):
            resolve_model_path("")
        with self.assertRaises(ModelNotFoundError):
            resolve_model_path("   ")

    def test_nonexistent_model_raises_model_not_found(self):
        with patch("os.path.expanduser", side_effect=lambda p: str(self.models_dir) if "models" in p else p):
            with self.assertRaises(ModelNotFoundError) as cm:
                resolve_model_path("non_existent_model_xyz")
            self.assertEqual(cm.exception.error_code, "ERR_MODEL_NOT_FOUND")
            self.assertEqual(cm.exception.model_arg, "non_existent_model_xyz")

    def test_single_model_match_succeeds(self):
        target_model = self.models_dir / "qwen2.5-0.5b-instruct-q4_k_m.gguf"
        target_model.touch()

        with patch("os.path.expanduser", side_effect=lambda p: str(self.models_dir) if "models" in p else p):
            resolved = resolve_model_path("qwen2.5-0.5b")
            self.assertEqual(resolved, os.path.realpath(str(target_model)))

    def test_ambiguous_model_match_raises_ambiguous_error(self):
        """Zero-Silent-Fallback: When multiple models match a fuzzy pattern, implicit selection is forbidden."""
        m1 = self.models_dir / "qwen2.5-0.5b-instruct-fp16.gguf"
        m2 = self.models_dir / "qwen2.5-0.5b-instruct-q4_k_m.gguf"
        m1.touch()
        m2.touch()

        with patch("os.path.expanduser", side_effect=lambda p: str(self.models_dir) if "models" in p else p):
            with self.assertRaises(AmbiguousModelMatchError) as cm:
                resolve_model_path("qwen2.5-0.5b")
            self.assertEqual(cm.exception.error_code, "ERR_AMBIGUOUS_MODEL_MATCH")
            self.assertEqual(len(cm.exception.candidates), 2)
            self.assertIn(os.path.realpath(str(m1)), cm.exception.candidates)
            self.assertIn(os.path.realpath(str(m2)), cm.exception.candidates)

    def test_exact_path_bypasses_ambiguity(self):
        """Specifying exact filename or path must always succeed unambiguously."""
        m1 = self.models_dir / "qwen2.5-0.5b-instruct-fp16.gguf"
        m2 = self.models_dir / "qwen2.5-0.5b-instruct-q4_k_m.gguf"
        m1.touch()
        m2.touch()

        resolved = resolve_model_path(str(m2))
        self.assertEqual(resolved, os.path.abspath(str(m2)))


class TestZeroStringHeuristicsLayersAndEnforcement(unittest.TestCase):
    def test_requested_ngl_enforced_unconditionally(self):
        """Caller requested_ngl must take absolute precedence over model name or defaults."""
        self.assertEqual(_calculate_llama_layers(engine=None, requested_ngl=12), 12)
        self.assertEqual(_calculate_llama_layers(engine={"ngl": 40}, requested_ngl=8), 8)

    def test_default_ngl_is_999(self):
        """Zero-String-Heuristics: In absence of explicit ngl, upstream standard full offload (999) is used."""
        self.assertEqual(_calculate_llama_layers(engine=None, requested_ngl=None), 999)
        self.assertEqual(_calculate_llama_layers(engine={}, requested_ngl=None), 999)

    def test_plan_passes_requested_ngl(self):
        plan_res = plan(backend="vulkan", ngl=42)
        self.assertEqual(plan_res.ngl, 42)
        self.assertIn("-ngl", plan_res.cli_flags)
        idx = plan_res.cli_flags.index("-ngl")
        self.assertEqual(plan_res.cli_flags[idx + 1], "42")


class TestVulkanVerificationAndFailFast(unittest.TestCase):
    def test_verify_vulkan_output_detects_cpu_fallback(self):
        cpu_fallback_logs = [
            "llama_init: no gpu found, falling back to cpu",
            "ggml_vulkan: failed to initialize vulkan device",
            "warning: backend unavailable, using cpu backend",
        ]
        for log in cpu_fallback_logs:
            with self.assertRaises(PlatformNotSupportedError):
                verify_vulkan_llm_output(log)

    def test_verify_vulkan_output_detects_lack_of_vulkan_init(self):
        silent_output = "llama_init: running on cpu with 4 threads. Hello world!"
        with self.assertRaises(PlatformNotSupportedError):
            verify_vulkan_llm_output(silent_output)

    def test_verify_vulkan_output_accepts_clean_vulkan_run(self):
        clean_output = "ggml_vulkan: found 1 vulkan device: Adreno (TM) 730\nggml_vulkan: using device\nsample output: Hello!"
        # Should not raise any exception
        verify_vulkan_llm_output(clean_output)

    def test_llamacpp_adapter_refuses_cpu_fallback_when_vulkan_forced(self):
        unsupported_report = DiagnosticReport(
            device_name="Generic Device",
            vendor_id=0,
            overall_success=False,
            recommended_backend="cpu_neon",
            passed_stages=0,
            total_stages=12,
            loader_path="",
        )
        with self.assertRaises(PlatformNotSupportedError):
            LlamaCppAdapter.bind(
                engine={},
                report=unsupported_report,
                requested_backend="vulkan",
            )


if __name__ == "__main__":
    unittest.main()
