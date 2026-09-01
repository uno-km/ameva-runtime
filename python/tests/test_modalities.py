"""
6-Modality Adapter 통합 테스트 — bind() API 기반.

API 변경 사항:
- attach(engine, ctx) → bind(engine, report) 로 변경됨.
- engine=None 전달 시 BindingResult 를 반환하고 AmevaRuntimeError 를 raise 하지 않음.
"""
import sys
import unittest
from pathlib import Path

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


def _get_report() -> DiagnosticReport:
    """테스트용 DiagnosticReport 를 반환합니다.

    실제 Vulkan ICD 가 있으면 실제 진단 결과를 사용하고,
    없으면 CPU NEON 모드 결과를 반환합니다.
    """
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
        self.assertIn(result.status, ("BOUND", "BOUND_CPU"),
                      f"{module}: 알 수 없는 status: {result.status!r}")
        self.assertIn(result.backend, ("vulkan", "cpu_neon"),
                      f"{module}: 알 수 없는 backend: {result.backend!r}")
        self.assertIsInstance(result.config, dict)
        # Vulkan 모드면 is_vulkan=True, CPU 모드면 is_vulkan=False
        self.assertEqual(result.is_vulkan, self.is_vulkan,
                         f"{module}: is_vulkan 불일치 (report={self.is_vulkan}, result={result.is_vulkan})")

    def test_stt_adapter(self):
        result = SttAdapter.bind(None, self.report)
        self._assert_binding(result, "termux-stt")
        if self.is_vulkan:
            self.assertEqual(result.config.get("gpu_layers"), 33)
            self.assertTrue(result.config.get("encoder_fp16"))

    def test_diffusion_adapter(self):
        result = DiffusionAdapter.bind(None, self.report)
        self._assert_binding(result, "termux-diffusion")
        if self.is_vulkan:
            self.assertIsNotNone(result.config.get("vulkan_lib_path"))
            self.assertTrue(result.config.get("sd_vulkan_flag"))

    def test_bitnet_adapter(self):
        result = BitnetAdapter.bind(None, self.report)
        self._assert_binding(result, "termux-bitnet")
        if self.is_vulkan:
            self.assertEqual(result.config.get("n_gpu_layers"), 33)
            self.assertEqual(result.config.get("kernel"), "ggml_vk_mul_mat_i2_s")

    def test_llamacpp_adapter(self):
        result = LlamaCppAdapter.bind(None, self.report)
        self._assert_binding(result, "termux-llamacpp")
        if self.is_vulkan:
            self.assertEqual(result.config.get("ngl"), 33)
            self.assertEqual(result.config.get("device_flag"), "vulkan")

    def test_tts_adapter(self):
        result = TtsAdapter.bind(None, self.report)
        self._assert_binding(result, "termux-tts")
        if self.is_vulkan:
            self.assertTrue(result.config.get("transposed_conv_vulkan"))
            self.assertAlmostEqual(result.config.get("latency_ms_target"), 38.5)

    def test_vision_adapter(self):
        result = VisionAdapter.bind(None, self.report)
        self._assert_binding(result, "termux-vision")
        if self.is_vulkan:
            self.assertTrue(result.config.get("vit_acceleration"))

    def test_binding_result_is_immutable(self):
        """BindingResult 불변성 검증 — config 방어적 복사 확인."""
        result = SttAdapter.bind(None, self.report)
        config_copy = result.config
        config_copy["injected_key"] = "SHOULD_NOT_PROPAGATE"
        # 원본 config 는 변경되지 않아야 함
        self.assertNotIn("injected_key", result.config,
                         "BindingResult.config 가 외부 변경에 노출되었습니다 (방어적 복사 실패).")


if __name__ == "__main__":
    unittest.main()
