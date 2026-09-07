/**
 * @file libegl_shim.c
 * @brief AMEVA Unified Mobile Vulkan HAL Interceptor & Samsung GOS Bypass Shim
 * 
 * Provides unified hardware compatibility across Samsung Galaxy Exynos (Mali-G68/G78)
 * and Qualcomm Snapdragon (Adreno 650/830) devices under Android Termux environments:
 * 1. Samsung GOS Shim: Export `_ZN7android18egl_get_connectionEv` to satisfy Samsung Android HAL.
 * 2. Buffer Device Address (BDA) Sanitizer: Intercept `vkGetPhysicalDeviceFeatures2` to set
 *    `bufferDeviceAddress = VK_FALSE`, preventing null-pointer SIGSEGV when loader fails to
 *    resolve device-level function pointers dynamically.
 * 3. Cooperative Matrix Sanitizer: Neutralize experimental coopmat2 flags on mobile devices.
 * 4. Subgroup Size Sanitizer: Ensure Qualcomm / Mali subgroup properties stay within safe ranges.
 * 5. Float Controls Sanitizer: Disable unsupported FP16 float rounding modes on legacy drivers.
 */

#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <dlfcn.h>
#include <stdint.h>
#include <string.h>
#include <vulkan/vulkan.h>

#ifdef __cplusplus
extern "C" {
#endif

__attribute__((constructor))
void ameva_mobile_shim_init(void) {
    fprintf(stderr, "[AMEVA-SHIM] === AMEVA UNIFIED MOBILE VULKAN HAL SHIM LOADED ===\n");
}

// 1. Samsung Game Optimizing Service (GOS) symbol shim
void* _ZN7android18egl_get_connectionEv(void) {
    return NULL;
}

static void* g_vulkan_handle = NULL;
static PFN_vkGetPhysicalDeviceProperties2 g_real_props2 = NULL;
static PFN_vkGetPhysicalDeviceFeatures2 g_real_features2 = NULL;
static PFN_vkGetInstanceProcAddr g_real_gpa = NULL;
static PFN_vkGetDeviceProcAddr g_real_gdpa = NULL;

static void init_vulkan_ptrs(void) {
    if (!g_vulkan_handle) {
        // Prefer direct system driver to bypass broken termux loader boundaries
        g_vulkan_handle = dlopen("/system/lib64/libvulkan.so", RTLD_NOW | RTLD_GLOBAL);
        if (!g_vulkan_handle) {
            g_vulkan_handle = dlopen("libvulkan.so.1", RTLD_NOW | RTLD_GLOBAL);
        }
        if (!g_vulkan_handle) {
            g_vulkan_handle = dlopen("libvulkan.so", RTLD_NOW | RTLD_GLOBAL);
        }

        if (g_vulkan_handle) {
            g_real_props2 = (PFN_vkGetPhysicalDeviceProperties2)dlsym(g_vulkan_handle, "vkGetPhysicalDeviceProperties2");
            if (!g_real_props2) g_real_props2 = (PFN_vkGetPhysicalDeviceProperties2)dlsym(g_vulkan_handle, "vkGetPhysicalDeviceProperties2KHR");
            
            g_real_features2 = (PFN_vkGetPhysicalDeviceFeatures2)dlsym(g_vulkan_handle, "vkGetPhysicalDeviceFeatures2");
            if (!g_real_features2) g_real_features2 = (PFN_vkGetPhysicalDeviceFeatures2)dlsym(g_vulkan_handle, "vkGetPhysicalDeviceFeatures2KHR");

            g_real_gpa = (PFN_vkGetInstanceProcAddr)dlsym(g_vulkan_handle, "vkGetInstanceProcAddr");
            g_real_gdpa = (PFN_vkGetDeviceProcAddr)dlsym(g_vulkan_handle, "vkGetDeviceProcAddr");
        }
    }
}

static void sanitize_props_pnext(void* pNext) {
    VkBaseOutStructure* curr = (VkBaseOutStructure*)pNext;
    while (curr) {
        if (curr->sType == VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES) {
            VkPhysicalDeviceSubgroupProperties* sub = (VkPhysicalDeviceSubgroupProperties*)curr;
            if (sub->subgroupSize > 32) {
                sub->subgroupSize = 32;
            }
        } else if (curr->sType == VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_PROPERTIES) {
            VkPhysicalDeviceVulkan12Properties* p12 = (VkPhysicalDeviceVulkan12Properties*)curr;
            p12->shaderRoundingModeRTEFloat16 = VK_FALSE;
            p12->shaderDenormPreserveFloat16 = VK_FALSE;
            p12->shaderRoundingModeRTEFloat32 = VK_FALSE;
            p12->shaderDenormPreserveFloat32 = VK_FALSE;
        } else if (curr->sType == VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FLOAT_CONTROLS_PROPERTIES) {
            VkPhysicalDeviceFloatControlsProperties* pfc = (VkPhysicalDeviceFloatControlsProperties*)curr;
            pfc->shaderRoundingModeRTEFloat16 = VK_FALSE;
            pfc->shaderDenormPreserveFloat16 = VK_FALSE;
            pfc->shaderRoundingModeRTEFloat32 = VK_FALSE;
            pfc->shaderDenormPreserveFloat32 = VK_FALSE;
        }
        curr = curr->pNext;
    }
}

VKAPI_ATTR void VKAPI_CALL vkGetPhysicalDeviceProperties2(VkPhysicalDevice physicalDevice, VkPhysicalDeviceProperties2* pProperties) {
    init_vulkan_ptrs();
    if (g_real_props2) g_real_props2(physicalDevice, pProperties);
    if (pProperties) sanitize_props_pnext(pProperties->pNext);
}

VKAPI_ATTR void VKAPI_CALL vkGetPhysicalDeviceProperties2KHR(VkPhysicalDevice physicalDevice, VkPhysicalDeviceProperties2* pProperties) {
    vkGetPhysicalDeviceProperties2(physicalDevice, pProperties);
}

static void sanitize_features_pnext(void* pNext) {
    VkBaseOutStructure* curr = (VkBaseOutStructure*)pNext;
    while (curr) {
        if (curr->sType == VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES) {
            VkPhysicalDeviceVulkan12Features* f12 = (VkPhysicalDeviceVulkan12Features*)curr;
            f12->bufferDeviceAddress = VK_FALSE;
            f12->bufferDeviceAddressCaptureReplay = VK_FALSE;
            f12->bufferDeviceAddressMultiDevice = VK_FALSE;
        } else if (curr->sType == VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_BUFFER_DEVICE_ADDRESS_FEATURES) {
            VkPhysicalDeviceBufferDeviceAddressFeatures* fbda = (VkPhysicalDeviceBufferDeviceAddressFeatures*)curr;
            fbda->bufferDeviceAddress = VK_FALSE;
            fbda->bufferDeviceAddressCaptureReplay = VK_FALSE;
            fbda->bufferDeviceAddressMultiDevice = VK_FALSE;
        } else if (curr->sType == VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_FEATURES_KHR) {
            VkPhysicalDeviceCooperativeMatrixFeaturesKHR* fcoop = (VkPhysicalDeviceCooperativeMatrixFeaturesKHR*)curr;
            fcoop->cooperativeMatrix = VK_FALSE;
            fcoop->cooperativeMatrixRobustBufferAccess = VK_FALSE;
        }
        curr = curr->pNext;
    }
}

VKAPI_ATTR void VKAPI_CALL vkGetPhysicalDeviceFeatures2(VkPhysicalDevice physicalDevice, VkPhysicalDeviceFeatures2* pFeatures) {
    init_vulkan_ptrs();
    if (g_real_features2) {
        g_real_features2(physicalDevice, pFeatures);
    }
    if (pFeatures) {
        sanitize_features_pnext(pFeatures->pNext);
    }
}

VKAPI_ATTR void VKAPI_CALL vkGetPhysicalDeviceFeatures2KHR(VkPhysicalDevice physicalDevice, VkPhysicalDeviceFeatures2* pFeatures) {
    vkGetPhysicalDeviceFeatures2(physicalDevice, pFeatures);
}

VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL vkGetDeviceProcAddr(VkDevice device, const char* pName) {
    init_vulkan_ptrs();
    if (pName && g_real_gdpa) {
        PFN_vkVoidFunction ptr = g_real_gdpa(device, pName);
        if (ptr) return ptr;
    }
    if (pName && g_real_gpa) {
        PFN_vkVoidFunction ptr = g_real_gpa((VkInstance)device, pName);
        if (ptr) return ptr;
    }
    return (PFN_vkVoidFunction)dlsym(g_vulkan_handle, pName);
}

VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL vkGetInstanceProcAddr(VkInstance instance, const char* pName) {
    init_vulkan_ptrs();
    if (pName) {
        if (strcmp(pName, "vkGetDeviceProcAddr") == 0) return (PFN_vkVoidFunction)vkGetDeviceProcAddr;
        if (strcmp(pName, "vkGetInstanceProcAddr") == 0) return (PFN_vkVoidFunction)vkGetInstanceProcAddr;
        if (strcmp(pName, "vkGetPhysicalDeviceProperties2") == 0 || strcmp(pName, "vkGetPhysicalDeviceProperties2KHR") == 0) return (PFN_vkVoidFunction)vkGetPhysicalDeviceProperties2;
        if (strcmp(pName, "vkGetPhysicalDeviceFeatures2") == 0 || strcmp(pName, "vkGetPhysicalDeviceFeatures2KHR") == 0) return (PFN_vkVoidFunction)vkGetPhysicalDeviceFeatures2;
    }
    if (g_real_gpa) {
        PFN_vkVoidFunction ptr = g_real_gpa(instance, pName);
        if (ptr) return ptr;
    }
    return (PFN_vkVoidFunction)dlsym(g_vulkan_handle, pName);
}

#ifdef __cplusplus
}
#endif
