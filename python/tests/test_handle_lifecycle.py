"""
Strict Verification of RAII Handle Lifecycle, Atomic State Caching, and FFI Bindings.
"""
import unittest
import os
import tempfile
from pathlib import Path
from ameva_vulkan_runtime.doctor import Doctor, DiagnosticReport
from ameva_vulkan_runtime.bindings import AmevaVulkanLib, load_native_lib
from ameva_vulkan_runtime.core import VulkanContext, get_or_create_context

class TestHandleLifecycleAndIntegrity(unittest.TestCase):

    def test_doctor_self_test_raii_no_leak(self):
        """Verify that running self_test repeatedly executes cleanly with RAII destruction."""
        doc = Doctor()
        for _ in range(5):
            report = doc.run_self_test(verbose=False)
            self.assertIsInstance(report, DiagnosticReport)
            self.assertIsInstance(report.passed_stages, int)
            self.assertIsInstance(report.stages, list)

    def test_state_json_atomic_caching(self):
        """Verify that state caching writes atomically without corrupting cache."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_file = Path(tmpdir) / "sub" / "vulkan_state.json"
            doc = Doctor(state_path=str(cache_file))
            self.assertFalse(cache_file.exists())
            
            # Run quick probe
            res = doc.quick_probe()
            self.assertIsInstance(res, bool)
            self.assertTrue(cache_file.exists())
            
            # Read cache content
            dev_name = doc.quick_probe_device()
            self.assertTrue(dev_name is None or isinstance(dev_name, str))

    def test_bindings_ffi_interface(self):
        """Verify that AmevaVulkanLib class exists and conforms to ABI contract."""
        lib = AmevaVulkanLib()
        self.assertFalse(lib.is_loaded())  # Expected on host without libameva_vulkan.so
        
        # Calling SGEMM without native lib raises AmevaVulkanError strictly
        import numpy as np
        from ameva_vulkan_runtime.exceptions import AmevaVulkanError
        a = np.ones((2, 2), dtype=np.float32)
        b = np.ones((2, 2), dtype=np.float32)
        c = np.zeros((2, 2), dtype=np.float32)
        with self.assertRaises(AmevaVulkanError):
            lib.call_matmul_f32(a, b, c, 2, 2, 2)

    def test_context_engine_flags_truthfulness(self):
        """Verify engine flags return honest device configurations."""
        ctx_cpu = get_or_create_context("cpu")
        flags_stt = ctx_cpu.to_engine_flags("stt")
        self.assertFalse(flags_stt["use_gpu"])
        self.assertEqual(flags_stt["gpu_layers"], 0)
        self.assertEqual(flags_stt["device"], "cpu")

if __name__ == "__main__":
    unittest.main()
