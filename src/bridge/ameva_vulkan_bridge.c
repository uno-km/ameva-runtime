#include <vulkan/vulkan.h>
#include <dlfcn.h>
#include <stdio.h>
#include <string.h>

static void* get_sys_vulkan() {
    static void* handle = NULL;
    if (!handle) {
        handle = dlopen("/system/lib64/libvulkan.so", RTLD_NOW | RTLD_LOCAL);
        if (!handle) {
            fprintf(stderr, "[AmevaVulkanBridge] dlopen(/system/lib64/libvulkan.so) failed: %s\n", dlerror());
        } else {
            fprintf(stderr, "[AmevaVulkanBridge] Bound to /system/lib64/libvulkan.so (Hardware ICD Active)\n");
        }
    }
    return handle;
}

#define FORWARD_GLOBAL(ret, name, args, params) \
VKAPI_ATTR ret VKAPI_CALL name args { \
    void* lib = get_sys_vulkan(); \
    if (!lib) return (ret)0; \
    static PFN_##name real_fn = NULL; \
    if (!real_fn) real_fn = (PFN_##name)dlsym(lib, #name); \
    if (!real_fn) { \
        fprintf(stderr, "[AmevaVulkanBridge] Missing symbol: %s\n", #name); \
        return (ret)0; \
    } \
    return real_fn params; \
}

#define FORWARD_VOID(name, args, params) \
VKAPI_ATTR void VKAPI_CALL name args { \
    void* lib = get_sys_vulkan(); \
    if (!lib) return; \
    static PFN_##name real_fn = NULL; \
    if (!real_fn) real_fn = (PFN_##name)dlsym(lib, #name); \
    if (real_fn) real_fn params; \
}

// Core functions
FORWARD_GLOBAL(PFN_vkVoidFunction, vkGetInstanceProcAddr, (VkInstance instance, const char* pName), (instance, pName))
FORWARD_GLOBAL(PFN_vkVoidFunction, vkGetDeviceProcAddr, (VkDevice device, const char* pName), (device, pName))
FORWARD_GLOBAL(VkResult, vkCreateInstance, (const VkInstanceCreateInfo* pCreateInfo, const VkAllocationCallbacks* pAllocator, VkInstance* pInstance), (pCreateInfo, pAllocator, pInstance))
FORWARD_GLOBAL(VkResult, vkEnumeratePhysicalDevices, (VkInstance instance, uint32_t* pPhysicalDeviceCount, VkPhysicalDevice* pPhysicalDevices), (instance, pPhysicalDeviceCount, pPhysicalDevices))
FORWARD_GLOBAL(VkResult, vkCreateDevice, (VkPhysicalDevice physicalDevice, const VkDeviceCreateInfo* pCreateInfo, const VkAllocationCallbacks* pAllocator, VkDevice* pDevice), (physicalDevice, pCreateInfo, pAllocator, pDevice))

// Explicitly needed by ggml-vulkan
FORWARD_VOID(vkGetPhysicalDeviceFeatures2, (VkPhysicalDevice physicalDevice, VkPhysicalDeviceFeatures2* pFeatures), (physicalDevice, pFeatures))
FORWARD_VOID(vkCmdCopyBuffer, (VkCommandBuffer commandBuffer, VkBuffer srcBuffer, VkBuffer dstBuffer, uint32_t regionCount, const VkBufferCopy* pRegions), (commandBuffer, srcBuffer, dstBuffer, regionCount, pRegions))
