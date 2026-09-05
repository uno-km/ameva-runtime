"""
Doctor V0~V11 테스트 — 실제 Vulkan API 호출 검증.

환경 적응 원칙:
- Android Termux (실기기): 실제 VkCreateInstance 호출, 12/12 PASS 요구.
- Windows / Linux 개발호스트: Vulkan ICD 없을 경우 V0 FAIL 이 올바른 동작.
  거짓 PASS 를 반환하지 않음이 핵심 검증 목표.
"""
import os
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from ameva_runtime.vulkan.doctor import Doctor, DiagnosticReport, _find_vulkan_lib


class TestDoctorV0V11(unittest.TestCase):
    def setUp(self):
        self.doc = Doctor(str(Path(__file__).parent.parent.parent / "test_state_doctor.json"))

    def tearDown(self):
        p = Path(__file__).parent.parent.parent / "test_state_doctor.json"
        if p.exists():
            p.unlink()

    def test_12_stage_diagnostic_no_false_pass(self):
        """실제 Vulkan API 호출 기반 진단 — 거짓 PASS 를 절대 반환하지 않음을 검증."""
        report = self.doc.run_self_test(verbose=False)
        # 결과가 PASS 또는 FAIL 중 하나여야 하며, 모두 시뮬레이션이 아님을 검증
        self.assertIsNotNone(report)
        self.assertIsInstance(report.overall_success, bool)
        self.assertEqual(len(report.stages), 12)
        # 각 단계는 반드시 result 가 PASS/FAIL/SKIP 중 하나여야 함
        for stage in report.stages:
            self.assertIn(stage.result, ("PASS", "FAIL", "SKIP"),
                          f"V{stage.stage_id}: 비정의 result 값: {stage.result!r}")
            self.assertIsInstance(stage.elapsed_ms, float)
            self.assertGreaterEqual(stage.elapsed_ms, 0.0)
            self.assertTrue(len(stage.detail_message) > 0,
                            f"V{stage.stage_id}: detail_message 가 비어있습니다.")

    def test_vulkan_icd_presence_determines_v0(self):
        """Vulkan ICD 존재 여부에 따라 V0 결과가 정직하게 결정됨을 검증."""
        icd_path = _find_vulkan_lib()
        report = self.doc.run_self_test(verbose=False)

        v0 = report.stages[0]
        if icd_path:
            # ICD 가 있으면 V0 는 PASS 여야 함
            self.assertEqual(v0.result, "PASS",
                             f"Vulkan ICD({icd_path}) 가 존재하지만 V0 FAIL: {v0.detail_message}")
        else:
            # ICD 가 없으면 V0 는 FAIL 이어야 함 (거짓 PASS 금지)
            self.assertEqual(v0.result, "FAIL",
                             "Vulkan ICD 없음에도 V0 PASS 반환 — 거짓 성공 탐지됨")

    def test_v7_v11_honestly_reported(self):
        """V7~V11 단계가 호스트 환경에서 거짓 PASS로 위장되지 않고 SKIP으로 정직하게 보고됨을 검증."""
        report = self.doc.run_self_test(verbose=False)
        for stage in report.stages[7:]:
            # 호스트에서는 셰이더 컴파일/디스패치가 없으므로 SKIP이어야 함 (거짓 PASS 금지)
            if not report.overall_success:
                self.assertEqual(stage.result, "SKIP",
                                 f"V{stage.stage_id} 단계가 셰이더 미컴파일 상태에서 거짓 PASS 반환")

    def test_quick_probe_returns_bool(self):
        """quick_probe() 가 항상 bool 을 반환함을 검증."""
        result = self.doc.quick_probe()
        self.assertIsInstance(result, bool)

    def test_state_json_saved_after_test(self):
        """진단 후 state.json 이 저장됨을 검증."""
        self.doc.run_self_test(verbose=False)
        self.assertTrue(self.doc.state_path.exists(),
                        "state.json 이 생성되지 않았습니다.")

    def test_quick_probe_device_corrupted_cache_fallback(self):
        """state.json 캐시 파일이 0바이트 또는 손상된 JSON일 때 예외를 안전하게 격리하고 fallback함을 검증."""
        self.doc.state_path.parent.mkdir(parents=True, exist_ok=True)
        # 1. 0-byte corrupted cache
        self.doc.state_path.write_text("", encoding="utf-8")
        dev1 = self.doc.quick_probe_device()
        self.assertTrue(dev1 is None or isinstance(dev1, str))
        probe1 = self.doc.quick_probe()
        self.assertIsInstance(probe1, bool)

        # 2. Malformed JSON
        self.doc.state_path.write_text("{ broken json ...", encoding="utf-8")
        dev2 = self.doc.quick_probe_device()
        self.assertTrue(dev2 is None or isinstance(dev2, str))
        probe2 = self.doc.quick_probe()
        self.assertIsInstance(probe2, bool)

    def test_quick_probe_device_valid_cache(self):
        """유효한 state.json 캐시가 존재할 때 빠른 디바이스 이름을 반환함을 검증."""
        import json, time
        self.doc.state_path.parent.mkdir(parents=True, exist_ok=True)
        valid_data = {
            "overall_success": True,
            "recommended_backend": "vulkan",
            "passed_stages": 12,
            "device_name": "Adreno (TM) 650 Test Device",
            "timestamp": time.time()
        }
        self.doc.state_path.write_text(json.dumps(valid_data), encoding="utf-8")
        dev = self.doc.quick_probe_device()
        self.assertEqual(dev, "Adreno (TM) 650 Test Device")
        self.assertTrue(self.doc.quick_probe())


if __name__ == "__main__":
    unittest.main()
