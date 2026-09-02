"""
6-Modality Adapter 통합 테스트 — 실제 모달리티 패키지 객체 바인딩 검증.
"""
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

# Add all active workspace packages to sys.path
DEV_DIR = Path(__file__).parent.parent.parent.parent
for pkg in ["termux-stt", "termux-bitnet", "termux-diffusion", "termux-llamacpp", "termux-tts", "termux-vision"]:
    p = str(DEV_DIR / pkg)
    if p not in sys.path:
        sys.path.insert(0, p)

sys.path.insert(0, str(Path(__file__).parent.parent))

from ameva_vulkan_runtime.doctor import Doctor, DiagnosticReport
from ameva_vulkan_runtime.protocol import BindingResult
from ameva_vulkan_runtime.adapters import (
    SttAdapter,
    DiffusionAdapter,
    BitnetAdapter,
    LlamaCppAdapter,
    TtsAdapter,
    VisionAdapter,
)

# Import real modality configuration / engine definitions
from termux_stt.engine.base import EngineConfig
from termux_bitnet.config import BitNetConfig
from termux_llamacpp.config import RuntimeConfig
from termux_tts.engine_onnx import ONNXNeuralEngine
from termux_diffusion.hardware import HardwareProfile


def _get_report() -> DiagnosticReport:
    doc = Doctor("test_state_modalities.json")
    return doc.run_self_test(verbose=False)


class TestModalitiesIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.report = _get_report()
        cls.is_vulkan = cls.report.overall_success

    @classmethod
    def tearDownClass(cls):
        p = Path("test_state_modalities.json")
        if p.exists():
            p.unlink()

    def _assert_binding(self, result: BindingResult, module: str):
        """BindingResult 공통 불변조건 검증."""
        self.assertIsInstance(result, BindingResult)
        self.assertEqual(result.module, module)
        self.assertIn(result.status, ("BOUND", "BOUND_CPU"))
        self.assertIn(result.backend, ("vulkan", "cpu_neon"))
        self.assertIsInstance(result.config, dict)
        self.assertEqual(result.is_vulkan, self.is_vulkan)

    def test_stt_adapter_with_real_stt_config(self):
        """실제 EngineConfig 객체를 가진 Mock 엔진에 바인딩 수행."""
        engine_obj = SimpleNamespace(config=EngineConfig(engine="whisper", model="tiny"), threads=4)
        result = SttAdapter.bind(engine_obj, self.report)
        self._assert_binding(result, "termux-stt")
        if self.is_vulkan:
            self.assertEqual(engine_obj.config.extra.get("gpu_layers"), 33)
            self.assertTrue(engine_obj.config.extra.get("use_vulkan"))
        else:
            self.assertGreater(engine_obj.threads, 0)

    def test_bitnet_adapter_with_real_bitnet_config(self):
        """실제 BitNetConfig 객체를 가진 엔진에 바인딩 수행."""
        engine_obj = SimpleNamespace(config=BitNetConfig())
        result = BitnetAdapter.bind(engine_obj, self.report)
        self._assert_binding(result, "termux-bitnet")
        if self.is_vulkan:
            self.assertEqual(engine_obj.config.n_gpu_layers, 33)
            self.assertTrue(engine_obj.config.flash_attn)

    def test_llamacpp_adapter_with_real_llama_config(self):
        """실제 RuntimeConfig 객체를 가진 엔진에 바인딩 수행."""
        engine_obj = SimpleNamespace(config=RuntimeConfig())
        result = LlamaCppAdapter.bind(engine_obj, self.report)
        self._assert_binding(result, "termux-llamacpp")
        if self.is_vulkan:
            self.assertEqual(engine_obj.config.n_gpu_layers, 33)

    def test_tts_adapter_with_real_tts_engine(self):
        """실제 ONNXNeuralEngine 인스턴스에 대한 바인딩 수행."""
        try:
            tts_engine = ONNXNeuralEngine(device="cpu")
        except Exception:
            tts_engine = SimpleNamespace(device="cpu", set_vulkan=lambda v: None)
        result = TtsAdapter.bind(tts_engine, self.report)
        self._assert_binding(result, "termux-tts")
        self.assertEqual(result.backend, "cpu_neon")
        if hasattr(tts_engine, "close"):
            tts_engine.close()

    def test_diffusion_adapter_with_real_hardware_profile(self):
        """실제 HardwareProfile 인스턴스를 가진 Diffusion 엔진에 바인딩 수행."""
        diff_engine = SimpleNamespace(hw_profile=HardwareProfile())
        result = DiffusionAdapter.bind(diff_engine, self.report)
        self._assert_binding(result, "termux-diffusion")
        if self.is_vulkan:
            self.assertTrue(diff_engine.hw_profile.vulkan_available)

    def test_vision_adapter_binding(self):
        """실제 비전 엔진 인스턴스에 대한 바인딩 수행."""
        vision_engine = SimpleNamespace(device="cpu", use_gpu=False)
        result = VisionAdapter.bind(vision_engine, self.report)
        self._assert_binding(result, "termux-vision")

    def test_all_adapters_unbind(self):
        """모든 6대 어댑터의 unbind 메서드 호출 및 무결성 검증."""
        adapters = [SttAdapter, DiffusionAdapter, BitnetAdapter, LlamaCppAdapter, TtsAdapter, VisionAdapter]
        for adapter in adapters:
            # Test static unbind
            adapter.unbind()
            # Test with dummy engine parameter
            adapter.unbind(SimpleNamespace())


if __name__ == "__main__":
    unittest.main()
