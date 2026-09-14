"""
Unit tests for TtsAdapter MeloTTS GPU acceleration routing (Plan 1 NCNN Sliced & Plan 2 MNN Vulkan).
Adheres strictly to AOSF-ENG-STD-2026 and Zero-Silent-Fallback [AMEVA-TTS-E001].
"""
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock

from ameva_runtime.adapters.tts import TtsAdapter
from ameva_runtime.adapters.base import BindingResult, DiagnosticReport


class TestTtsAdapterMelo(unittest.TestCase):
    def setUp(self):
        self.mock_report = DiagnosticReport(
            device_name="GeForce GTX 1070 Ti",
            vendor_id=0x10DE,
            overall_success=True,
            recommended_backend="vulkan",
            passed_stages=12,
            total_stages=12,
            loader_path="/system/lib64/libvulkan.so",
        )

    def test_tts_adapter_resolve_model_dir_plan1_priority(self, tmp_path=None):
        """Test resolve_model_dir prioritizes Plan 1 NCNN sliced decoder over Plan 2 MNN."""
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            p = Path(td)
            (p / "melo_decoder.ncnn.param").write_text("7767517")
            (p / "melo_decoder.ncnn.bin").write_bytes(b"BIN")
            (p / "melo.mnn").write_bytes(b"MNN")

            with patch.object(TtsAdapter, "STANDARD_MODEL_DIRS", [p]):
                resolved = TtsAdapter.resolve_model_dir(model_type="melo")
                self.assertEqual(resolved, str(p.resolve()))

    def test_tts_adapter_resolve_model_dir_plan2_fallback(self):
        """Test resolve_model_dir selects Plan 2 MNN when Plan 1 is absent."""
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            p = Path(td)
            (p / "melo.mnn").write_bytes(b"MNN")

            with patch.object(TtsAdapter, "STANDARD_MODEL_DIRS", [p]):
                resolved = TtsAdapter.resolve_model_dir(model_type="melo")
                self.assertEqual(resolved, str(p.resolve()))

    def test_tts_adapter_bind_melo_vulkan(self):
        """Test TtsAdapter.bind successfully binds MeloTTS on Vulkan device."""
        mock_engine = MagicMock()
        result = TtsAdapter.bind(
            engine=mock_engine,
            report=self.mock_report,
            model_type="melo",
            tier="high",
        )
        self.assertIsInstance(result, BindingResult)
        self.assertEqual(result.backend, "vulkan")
        self.assertTrue(result.is_vulkan)
        self.assertEqual(result.status, "BOUND_VULKAN")
        self.assertIn(result.config["model_type"], ("melo_vulkan", "melo_mnn"))
        self.assertEqual(mock_engine.device, "vulkan")

    def test_tts_adapter_bind_melo_cpu_offload(self):
        """Test TtsAdapter.bind offloads to CPU when requested_backend='cpu'."""
        mock_engine = MagicMock()
        result = TtsAdapter.bind(
            engine=mock_engine,
            report=self.mock_report,
            requested_backend="cpu",
            model_type="melo",
        )
        self.assertIsInstance(result, BindingResult)
        self.assertEqual(result.backend, "cpu_neon")
        self.assertFalse(result.is_vulkan)
        self.assertEqual(mock_engine.device, "cpu")


if __name__ == "__main__":
    unittest.main()
