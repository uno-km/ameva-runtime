"""
Unit and Integration Tests: Vulkan Submodule Architecture (Pure ameva_runtime.vulkan)
=====================================================================================
Verifies that:
1. ameva_runtime.vulkan houses the complete Vulkan acceleration engine.
2. Direct imports and convenience access via ameva_runtime.vulkan work cleanly.
3. Hardware profile loading, Doctor, VulkanContext, and Adapters operate without legacy dependencies.
4. Single Source of Truth versioning (2.0.0) is consistently reported across all namespaces.
"""
import unittest
import ameva_runtime
from ameva_runtime import vulkan
import ameva_runtime.vulkan.doctor as vulkan_doctor
import ameva_runtime.vulkan.core as vulkan_core
import ameva_runtime.vulkan.bindings as vulkan_bindings
import ameva_runtime.vulkan.adapters as vulkan_adapters
import ameva_runtime.vulkan.adapters.stt as vulkan_stt
import ameva_runtime.vulkan.platform as vulkan_platform


class TestVulkanSubmoduleArchitecture(unittest.TestCase):
    def test_version_consistency(self):
        """Verify unified version across all namespaces."""
        from ameva_runtime._version import __version__ as expected_ver
        self.assertEqual(ameva_runtime.__version__, expected_ver)
        self.assertEqual(vulkan.__version__, expected_ver)

    def test_submodule_exports(self):
        """Verify that ameva_runtime.vulkan correctly exports all required acceleration classes."""
        self.assertTrue(hasattr(vulkan, "Doctor"))
        self.assertTrue(hasattr(vulkan, "VulkanContext"))
        self.assertTrue(hasattr(vulkan, "create_context"))
        self.assertTrue(hasattr(vulkan, "get_or_create_context"))
        self.assertTrue(hasattr(vulkan, "AmevaVulkanLib"))
        self.assertTrue(hasattr(vulkan, "SttAdapter"))
        self.assertTrue(hasattr(vulkan, "LlamaCppAdapter"))

    def test_hardware_profile_loading(self):
        """Verify that modern Vulkan Doctor correctly locates and loads validated hardware profiles."""
        doc = vulkan_doctor.Doctor()
        # Galaxy S25
        s25_prof = doc.load_hardware_profile("SM-S931N", 0x5143)
        self.assertEqual(s25_prof.get("model"), "SM-S931N")
        self.assertTrue(s25_prof.get("subgroup_control_bypass"))

        # Galaxy A35
        a35_prof = doc.load_hardware_profile("SM-A356N", 0x13B5)
        self.assertEqual(a35_prof.get("model"), "SM-A356N")
        self.assertTrue(a35_prof.get("enforce_medium_matmul"))

    def test_convenience_access_from_runtime(self):
        """Verify that ameva_runtime exposes .vulkan seamlessly."""
        self.assertTrue(hasattr(ameva_runtime, "vulkan"))
        self.assertIs(ameva_runtime.vulkan, vulkan)
        self.assertIs(vulkan.Doctor, vulkan_doctor.Doctor)
        self.assertIs(vulkan.VulkanContext, vulkan_core.VulkanContext)
        self.assertIs(vulkan.adapters.SttAdapter, vulkan_stt.SttAdapter)


if __name__ == "__main__":
    unittest.main()
