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

from ameva_runtime.vulkan.doctor import Doctor, DiagnosticReport
from ameva_runtime.vulkan.protocol import BindingResult
from ameva_runtime.vulkan.adapters import (
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
        self.assertIn(result.status, ("BOUND", "BOUND_CPU", "BOUND_VULKAN", "BOUND_CPU_NEON"))
        self.assertIn(result.backend, ("vulkan", "cpu_neon"))
        self.assertIsInstance(result.config, dict)
        self.assertEqual(result.is_vulkan, self.is_vulkan)

    def test_stt_adapter_with_real_stt_config(self):
        """실제 EngineConfig 객체를 가진 Mock 엔진에 바인딩 수행."""
        engine_obj = SimpleNamespace(config=EngineConfig(engine="whisper", model="tiny"), threads=4)
        result = SttAdapter.bind(engine_obj, self.report)
        self._assert_binding(result, "termux-stt")
        if self.is_vulkan:
            self.assertEqual(engine_obj.config.extra.get("gpu_layers"), 4)
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

    def test_llamacpp_adapter_mali_system_icd(self):
        """Mali GPU 진단 보고서에 대한 Android System ICD 및 브릿지 우선순위 검증."""
        mali_report = DiagnosticReport(
            overall_success=True,
            device_name="Mali-G68",
            driver_version="1.3.219",
            loader_path="/system/lib64/libvulkan.so",
            vendor_id=0x13B5,
            passed_stages=11,
            total_stages=11,
            total_elapsed_ms=4.88,
            recommended_backend="vulkan"
        )
        engine_dict = {"ngl": 0}
        result = LlamaCppAdapter.bind(engine_dict, mali_report)
        self.assertEqual(result.backend, "vulkan")
        self.assertTrue(result.config.get("mali_align"))
        self.assertTrue(result.config.get("system_icd_prioritized"))
        self.assertTrue(result.config.get("bridge_active"))
        self.assertTrue(engine_dict["env"]["LD_LIBRARY_PATH"].startswith("/system/lib64"))

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

    def test_adapters_fail_fast_when_vulkan_requested_but_unavailable(self):
        """Verify strict Fail-Fast: All 6 adapters MUST raise PlatformNotSupportedError when Vulkan is requested on unsupported hardware."""
        from ameva_runtime.exceptions import PlatformNotSupportedError
        unsupported_report = DiagnosticReport(
            overall_success=False,
            device_name="Generic Non-Vulkan Device",
            driver_version="0.0.0",
            loader_path="",
            vendor_id=0,
            passed_stages=0,
            total_stages=12,
            total_elapsed_ms=0.0,
            recommended_backend="cpu_neon",
        )
        adapters = [SttAdapter, DiffusionAdapter, BitnetAdapter, LlamaCppAdapter, TtsAdapter, VisionAdapter]
        for adapter in adapters:
            with self.subTest(adapter=adapter.module_name):
                with self.assertRaises(PlatformNotSupportedError) as ctx:
                    adapter.bind(
                        engine=SimpleNamespace(),
                        report=unsupported_report,
                        requested_backend="vulkan",
                    )
                self.assertIn("Vulkan acceleration backend explicitly requested", str(ctx.exception))

    def test_context_lifecycle_unbind_all(self):
        """Verify VulkanContext context manager unbinds all registered adapters on exit."""
        from ameva_runtime.vulkan.core import VulkanContext
        eng_llama = SimpleNamespace(config=RuntimeConfig())
        eng_vis = SimpleNamespace(device="vulkan", use_gpu=True)

        with VulkanContext(device_mode="auto") as ctx:
            ctx.bind_adapter(LlamaCppAdapter, eng_llama)
            ctx.bind_adapter(VisionAdapter, eng_vis)
            self.assertEqual(len(ctx._bound_adapters), 2)

        # Context closed -> unbind_all automatically executed
        self.assertEqual(len(ctx._bound_adapters), 0)
        self.assertEqual(eng_vis.device, "cpu")
        self.assertFalse(eng_vis.use_gpu)


if __name__ == "__main__":
    unittest.main()

