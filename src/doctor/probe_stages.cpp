#include "probe_stages.h"
#include "../core/vulkan_loader.h"
#include "../quirks/adreno_quirks.h"
#include "../quirks/mali_quirks.h"
#include "../shaders/matmul_spv.h"

#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <cmath>
#include <cstring>
#include <cstdlib>
#include <algorithm>
#include <ctime>

#if defined(_WIN32)
#include <windows.h>
#include <shlobj.h>
#else
#include <unistd.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <pwd.h>
#endif

// Strictly require official Vulkan headers without manual struct mocks
#if __has_include(<vulkan/vulkan.h>)
#include <vulkan/vulkan.h>
#else
#error "Vulkan header <vulkan/vulkan.h> is strictly required for compiling ameva-runtime doctor. Manual struct mock fallbacks are prohibited by anti-deception protocol."
#endif

namespace ameva {
namespace doctor {

static std::string GetStandardCachePath() {
#if defined(_WIN32)
    char userProfile[MAX_PATH];
    if (GetEnvironmentVariableA("USERPROFILE", userProfile, MAX_PATH) > 0) {
        std::string dir = std::string(userProfile) + "\\.cache\\ameva";
        CreateDirectoryA((std::string(userProfile) + "\\.cache").c_str(), NULL);
        CreateDirectoryA(dir.c_str(), NULL);
        return dir + "\\vulkan_state.json";
    }
    return "vulkan_state.json";
#else
    const char* home = getenv("HOME");
    if (!home) {
        struct passwd* pw = getpwuid(getuid());
        if (pw) home = pw->pw_dir;
    }
    if (home) {
        std::string dir = std::string(home) + "/.cache/ameva";
        mkdir((std::string(home) + "/.cache").c_str(), 0755);
        mkdir(dir.c_str(), 0755);
        return dir + "/vulkan_state.json";
    }
    return "/tmp/ameva_vulkan_state.json";
#endif
}

ProbeSuite::ProbeSuite() = default;
ProbeSuite::~ProbeSuite() = default;

static std::string FormatStageName(int id, const char* name) {
    std::stringstream ss;
    ss << "V" << id << ": " << name;
    return ss.str();
}

DiagnosticReport ProbeSuite::RunFullDiagnostic(bool verbose) {
    DiagnosticReport report;
    report.overall_success = false;
    report.passed_stages = 0;
    report.total_stages = 12;
    report.recommended_backend = "cpu_neon";
    report.device_name = "Unknown";
    report.driver_version = "Unknown";
    report.vendor_id = 0;

    auto total_start = std::chrono::high_resolution_clock::now();

    if (verbose) {
        std::cout << "\n============================================================" << std::endl;
        std::cout << "  AMEVA-Vulkan-Runtime: 12-Stage Diagnostic Suite (V0-V11)  " << std::endl;
        std::cout << "============================================================" << std::endl;
    }

    core::VulkanLoader loader;
    VkInstance vk_instance = VK_NULL_HANDLE;
    VkDevice vk_device = VK_NULL_HANDLE;
    VkPhysicalDevice physical_device = VK_NULL_HANDLE;
    uint32_t compute_queue_family_index = 0;
    VkQueue compute_queue = VK_NULL_HANDLE;

    // Core dynamic function pointers resolved via loader
    PFN_vkEnumerateInstanceVersion pfn_vkEnumerateInstanceVersion = nullptr;
    PFN_vkCreateInstance pfn_vkCreateInstance = nullptr;
    PFN_vkDestroyInstance pfn_vkDestroyInstance = nullptr;
    PFN_vkEnumeratePhysicalDevices pfn_vkEnumeratePhysicalDevices = nullptr;
    PFN_vkGetPhysicalDeviceProperties pfn_vkGetPhysicalDeviceProperties = nullptr;
    PFN_vkGetPhysicalDeviceQueueFamilyProperties pfn_vkGetPhysicalDeviceQueueFamilyProperties = nullptr;
    PFN_vkCreateDevice pfn_vkCreateDevice = nullptr;
    PFN_vkDestroyDevice pfn_vkDestroyDevice = nullptr;
    PFN_vkGetDeviceQueue pfn_vkGetDeviceQueue = nullptr;
    PFN_vkGetPhysicalDeviceMemoryProperties pfn_vkGetPhysicalDeviceMemoryProperties = nullptr;
    PFN_vkCreateBuffer pfn_vkCreateBuffer = nullptr;
    PFN_vkDestroyBuffer pfn_vkDestroyBuffer = nullptr;
    PFN_vkGetBufferMemoryRequirements pfn_vkGetBufferMemoryRequirements = nullptr;
    PFN_vkAllocateMemory pfn_vkAllocateMemory = nullptr;
    PFN_vkFreeMemory pfn_vkFreeMemory = nullptr;
    PFN_vkBindBufferMemory pfn_vkBindBufferMemory = nullptr;
    PFN_vkMapMemory pfn_vkMapMemory = nullptr;
    PFN_vkUnmapMemory pfn_vkUnmapMemory = nullptr;
    PFN_vkFlushMappedMemoryRanges pfn_vkFlushMappedMemoryRanges = nullptr;
    PFN_vkInvalidateMappedMemoryRanges pfn_vkInvalidateMappedMemoryRanges = nullptr;
    PFN_vkCreateShaderModule pfn_vkCreateShaderModule = nullptr;
    PFN_vkDestroyShaderModule pfn_vkDestroyShaderModule = nullptr;
    PFN_vkCreateDescriptorSetLayout pfn_vkCreateDescriptorSetLayout = nullptr;
    PFN_vkDestroyDescriptorSetLayout pfn_vkDestroyDescriptorSetLayout = nullptr;
    PFN_vkCreatePipelineLayout pfn_vkCreatePipelineLayout = nullptr;
    PFN_vkDestroyPipelineLayout pfn_vkDestroyPipelineLayout = nullptr;
    PFN_vkCreateComputePipelines pfn_vkCreateComputePipelines = nullptr;
    PFN_vkDestroyPipeline pfn_vkDestroyPipeline = nullptr;
    PFN_vkCreateDescriptorPool pfn_vkCreateDescriptorPool = nullptr;
    PFN_vkDestroyDescriptorPool pfn_vkDestroyDescriptorPool = nullptr;
    PFN_vkAllocateDescriptorSets pfn_vkAllocateDescriptorSets = nullptr;
    PFN_vkUpdateDescriptorSets pfn_vkUpdateDescriptorSets = nullptr;
    PFN_vkCreateCommandPool pfn_vkCreateCommandPool = nullptr;
    PFN_vkDestroyCommandPool pfn_vkDestroyCommandPool = nullptr;
    PFN_vkAllocateCommandBuffers pfn_vkAllocateCommandBuffers = nullptr;
    PFN_vkBeginCommandBuffer pfn_vkBeginCommandBuffer = nullptr;
    PFN_vkEndCommandBuffer pfn_vkEndCommandBuffer = nullptr;
    PFN_vkCmdBindPipeline pfn_vkCmdBindPipeline = nullptr;
    PFN_vkCmdBindDescriptorSets pfn_vkCmdBindDescriptorSets = nullptr;
    PFN_vkCmdPushConstants pfn_vkCmdPushConstants = nullptr;
    PFN_vkCmdDispatch pfn_vkCmdDispatch = nullptr;
    PFN_vkQueueSubmit pfn_vkQueueSubmit = nullptr;
    PFN_vkQueueWaitIdle pfn_vkQueueWaitIdle = nullptr;

    bool stage_success = true;

    // V0: Loader Open
    {
        auto t0 = std::chrono::high_resolution_clock::now();
        bool ok = loader.Load();
        auto t1 = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

        StageReport s;
        s.stage_id = 0;
        s.stage_name = FormatStageName(0, "Vulkan Loader Open");
        s.elapsed_ms = ms;
        s.allocated_bytes = 0;

        if (ok) {
            report.loader_path = loader.GetLoadedPath();
            s.result = StageResult::PASS;
            s.detail_message = "Bound to: " + report.loader_path;
            report.passed_stages++;

            pfn_vkEnumerateInstanceVersion = (PFN_vkEnumerateInstanceVersion)loader.GetProcAddr("vkEnumerateInstanceVersion");
            pfn_vkCreateInstance = (PFN_vkCreateInstance)loader.GetProcAddr("vkCreateInstance");
            pfn_vkDestroyInstance = (PFN_vkDestroyInstance)loader.GetProcAddr("vkDestroyInstance");
            pfn_vkEnumeratePhysicalDevices = (PFN_vkEnumeratePhysicalDevices)loader.GetProcAddr("vkEnumeratePhysicalDevices");
            pfn_vkGetPhysicalDeviceProperties = (PFN_vkGetPhysicalDeviceProperties)loader.GetProcAddr("vkGetPhysicalDeviceProperties");
            pfn_vkGetPhysicalDeviceQueueFamilyProperties = (PFN_vkGetPhysicalDeviceQueueFamilyProperties)loader.GetProcAddr("vkGetPhysicalDeviceQueueFamilyProperties");
            pfn_vkCreateDevice = (PFN_vkCreateDevice)loader.GetProcAddr("vkCreateDevice");
            pfn_vkDestroyDevice = (PFN_vkDestroyDevice)loader.GetProcAddr("vkDestroyDevice");
            pfn_vkGetDeviceQueue = (PFN_vkGetDeviceQueue)loader.GetProcAddr("vkGetDeviceQueue");
            pfn_vkGetPhysicalDeviceMemoryProperties = (PFN_vkGetPhysicalDeviceMemoryProperties)loader.GetProcAddr("vkGetPhysicalDeviceMemoryProperties");
            pfn_vkCreateBuffer = (PFN_vkCreateBuffer)loader.GetProcAddr("vkCreateBuffer");
            pfn_vkDestroyBuffer = (PFN_vkDestroyBuffer)loader.GetProcAddr("vkDestroyBuffer");
            pfn_vkGetBufferMemoryRequirements = (PFN_vkGetBufferMemoryRequirements)loader.GetProcAddr("vkGetBufferMemoryRequirements");
            pfn_vkAllocateMemory = (PFN_vkAllocateMemory)loader.GetProcAddr("vkAllocateMemory");
            pfn_vkFreeMemory = (PFN_vkFreeMemory)loader.GetProcAddr("vkFreeMemory");
            pfn_vkBindBufferMemory = (PFN_vkBindBufferMemory)loader.GetProcAddr("vkBindBufferMemory");
            pfn_vkMapMemory = (PFN_vkMapMemory)loader.GetProcAddr("vkMapMemory");
            pfn_vkUnmapMemory = (PFN_vkUnmapMemory)loader.GetProcAddr("vkUnmapMemory");
            pfn_vkFlushMappedMemoryRanges = (PFN_vkFlushMappedMemoryRanges)loader.GetProcAddr("vkFlushMappedMemoryRanges");
            pfn_vkInvalidateMappedMemoryRanges = (PFN_vkInvalidateMappedMemoryRanges)loader.GetProcAddr("vkInvalidateMappedMemoryRanges");
            pfn_vkCreateShaderModule = (PFN_vkCreateShaderModule)loader.GetProcAddr("vkCreateShaderModule");
            pfn_vkDestroyShaderModule = (PFN_vkDestroyShaderModule)loader.GetProcAddr("vkDestroyShaderModule");
            pfn_vkCreateDescriptorSetLayout = (PFN_vkCreateDescriptorSetLayout)loader.GetProcAddr("vkCreateDescriptorSetLayout");
            pfn_vkDestroyDescriptorSetLayout = (PFN_vkDestroyDescriptorSetLayout)loader.GetProcAddr("vkDestroyDescriptorSetLayout");
            pfn_vkCreatePipelineLayout = (PFN_vkCreatePipelineLayout)loader.GetProcAddr("vkCreatePipelineLayout");
            pfn_vkDestroyPipelineLayout = (PFN_vkDestroyPipelineLayout)loader.GetProcAddr("vkDestroyPipelineLayout");
            pfn_vkCreateComputePipelines = (PFN_vkCreateComputePipelines)loader.GetProcAddr("vkCreateComputePipelines");
            pfn_vkDestroyPipeline = (PFN_vkDestroyPipeline)loader.GetProcAddr("vkDestroyPipeline");
            pfn_vkCreateDescriptorPool = (PFN_vkCreateDescriptorPool)loader.GetProcAddr("vkCreateDescriptorPool");
            pfn_vkDestroyDescriptorPool = (PFN_vkDestroyDescriptorPool)loader.GetProcAddr("vkDestroyDescriptorPool");
            pfn_vkAllocateDescriptorSets = (PFN_vkAllocateDescriptorSets)loader.GetProcAddr("vkAllocateDescriptorSets");
            pfn_vkUpdateDescriptorSets = (PFN_vkUpdateDescriptorSets)loader.GetProcAddr("vkUpdateDescriptorSets");
            pfn_vkCreateCommandPool = (PFN_vkCreateCommandPool)loader.GetProcAddr("vkCreateCommandPool");
            pfn_vkDestroyCommandPool = (PFN_vkDestroyCommandPool)loader.GetProcAddr("vkDestroyCommandPool");
            pfn_vkAllocateCommandBuffers = (PFN_vkAllocateCommandBuffers)loader.GetProcAddr("vkAllocateCommandBuffers");
            pfn_vkBeginCommandBuffer = (PFN_vkBeginCommandBuffer)loader.GetProcAddr("vkBeginCommandBuffer");
            pfn_vkEndCommandBuffer = (PFN_vkEndCommandBuffer)loader.GetProcAddr("vkEndCommandBuffer");
            pfn_vkCmdBindPipeline = (PFN_vkCmdBindPipeline)loader.GetProcAddr("vkCmdBindPipeline");
            pfn_vkCmdBindDescriptorSets = (PFN_vkCmdBindDescriptorSets)loader.GetProcAddr("vkCmdBindDescriptorSets");
            pfn_vkCmdPushConstants = (PFN_vkCmdPushConstants)loader.GetProcAddr("vkCmdPushConstants");
            pfn_vkCmdDispatch = (PFN_vkCmdDispatch)loader.GetProcAddr("vkCmdDispatch");
            pfn_vkQueueSubmit = (PFN_vkQueueSubmit)loader.GetProcAddr("vkQueueSubmit");
            pfn_vkQueueWaitIdle = (PFN_vkQueueWaitIdle)loader.GetProcAddr("vkQueueWaitIdle");
        } else {
            s.result = StageResult::FAIL;
            s.detail_message = "Failed to dynamically open libvulkan.so or resolve symbols";
            stage_success = false;
        }
        report.stages.push_back(s);
        if (verbose) {
            std::cout << "  [" << (s.result == StageResult::PASS ? "PASS" : "FAIL") << "] " << std::left << std::setw(32) << s.stage_name
                      << " (" << std::fixed << std::setprecision(2) << s.elapsed_ms << " ms) - " << s.detail_message << std::endl;
        }
    }

    // V1: Instance Creation
    if (stage_success && pfn_vkCreateInstance && pfn_vkDestroyInstance) {
        auto t0 = std::chrono::high_resolution_clock::now();
        uint32_t apiVersion = VK_API_VERSION_1_1;
        if (pfn_vkEnumerateInstanceVersion) {
            uint32_t queriedVersion = 0;
            if (pfn_vkEnumerateInstanceVersion(&queriedVersion) == VK_SUCCESS && queriedVersion > 0) {
                apiVersion = queriedVersion;
            }
        }

        VkApplicationInfo appInfo{};
        appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
        appInfo.pApplicationName = "AmevaVulkanDoctor";
        appInfo.applicationVersion = VK_MAKE_VERSION(1, 2, 0);
        appInfo.pEngineName = "AmevaRuntime";
        appInfo.engineVersion = VK_MAKE_VERSION(1, 2, 0);
        appInfo.apiVersion = apiVersion;

        VkInstanceCreateInfo createInfo{};
        createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
        createInfo.pApplicationInfo = &appInfo;

        int res = pfn_vkCreateInstance(&createInfo, nullptr, &vk_instance);
        auto t1 = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

        StageReport s;
        s.stage_id = 1;
        s.stage_name = FormatStageName(1, "Instance Creation");
        s.elapsed_ms = ms;
        s.allocated_bytes = 0;

        if (res == VK_SUCCESS && vk_instance) {
            s.result = StageResult::PASS;
            s.detail_message = "vkCreateInstance() SUCCESS (API " +
                               std::to_string(VK_VERSION_MAJOR(apiVersion)) + "." +
                               std::to_string(VK_VERSION_MINOR(apiVersion)) + ")";
            report.passed_stages++;
        } else {
            s.result = StageResult::FAIL;
            s.detail_message = "vkCreateInstance() failed with error code " + std::to_string(res);
            stage_success = false;
        }
        report.stages.push_back(s);
        if (verbose) {
            std::cout << "  [" << (s.result == StageResult::PASS ? "PASS" : "FAIL") << "] " << std::left << std::setw(32) << s.stage_name
                      << " (" << std::fixed << std::setprecision(2) << s.elapsed_ms << " ms) - " << s.detail_message << std::endl;
        }
    }

    // V2: Physical Device Enumeration
    if (stage_success && pfn_vkEnumeratePhysicalDevices) {
        auto t0 = std::chrono::high_resolution_clock::now();
        uint32_t count = 0;
        int res = pfn_vkEnumeratePhysicalDevices(vk_instance, &count, nullptr);
        
        StageReport s;
        s.stage_id = 2;
        s.stage_name = FormatStageName(2, "Physical Device Enumeration");
        s.allocated_bytes = 0;

        if (res == VK_SUCCESS && count > 0) {
            std::vector<VkPhysicalDevice> devices(count);
            pfn_vkEnumeratePhysicalDevices(vk_instance, &count, devices.data());
            physical_device = devices[0];
            s.result = StageResult::PASS;
            s.detail_message = "Found " + std::to_string(count) + " Vulkan physical device(s)";
            report.passed_stages++;
        } else {
            s.result = StageResult::FAIL;
            s.detail_message = "No physical Vulkan devices enumerated (count=" + std::to_string(count) + ")";
            stage_success = false;
        }
        auto t1 = std::chrono::high_resolution_clock::now();
        s.elapsed_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        report.stages.push_back(s);
        if (verbose) {
            std::cout << "  [" << (s.result == StageResult::PASS ? "PASS" : "FAIL") << "] " << std::left << std::setw(32) << s.stage_name
                      << " (" << std::fixed << std::setprecision(2) << s.elapsed_ms << " ms) - " << s.detail_message << std::endl;
        }
    }

    // V3: Hardware GPU Selection & Property Query
    if (stage_success && pfn_vkGetPhysicalDeviceProperties) {
        auto t0 = std::chrono::high_resolution_clock::now();
        VkPhysicalDeviceProperties props{};
        pfn_vkGetPhysicalDeviceProperties(physical_device, &props);

        report.device_name = props.deviceName;
        report.vendor_id = props.vendorID;
        report.device_id = props.deviceID;
        report.api_version = props.apiVersion;
        
        uint32_t major = (props.driverVersion >> 22) & 0x3FF;
        uint32_t minor = (props.driverVersion >> 12) & 0x3FF;
        uint32_t patch = props.driverVersion & 0xFFF;
        std::stringstream ss;
        ss << major << "." << minor << "." << patch;
        report.driver_version = ss.str();

        StageReport s;
        s.stage_id = 3;
        s.stage_name = FormatStageName(3, "Hardware GPU Selection");
        s.result = StageResult::PASS;
        s.detail_message = "Selected: " + report.device_name + " (Vendor: 0x" + [&props]() {
            std::stringstream h;
            h << std::hex << props.vendorID;
            return h.str();
        }() + ")";
        report.passed_stages++;

        auto t1 = std::chrono::high_resolution_clock::now();
        s.elapsed_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        report.stages.push_back(s);
        if (verbose) {
            std::cout << "  [PASS] " << std::left << std::setw(32) << s.stage_name
                      << " (" << std::fixed << std::setprecision(2) << s.elapsed_ms << " ms) - " << s.detail_message << std::endl;
        }
    }

    // V4: Compute Queue Family Probe
    if (stage_success && pfn_vkGetPhysicalDeviceQueueFamilyProperties) {
        auto t0 = std::chrono::high_resolution_clock::now();
        uint32_t qCount = 0;
        pfn_vkGetPhysicalDeviceQueueFamilyProperties(physical_device, &qCount, nullptr);
        std::vector<VkQueueFamilyProperties> qProps(qCount);
        pfn_vkGetPhysicalDeviceQueueFamilyProperties(physical_device, &qCount, qProps.data());

        bool found_compute = false;
        for (uint32_t qi = 0; qi < qCount; ++qi) {
            if (qProps[qi].queueFlags & VK_QUEUE_COMPUTE_BIT) {
                compute_queue_family_index = qi;
                found_compute = true;
                break;
            }
        }

        StageReport s;
        s.stage_id = 4;
        s.stage_name = FormatStageName(4, "Compute Queue Family Probe");
        s.allocated_bytes = 0;

        if (found_compute) {
            s.result = StageResult::PASS;
            s.detail_message = "Compute Queue Family Index: " + std::to_string(compute_queue_family_index);
            report.passed_stages++;
        } else {
            s.result = StageResult::FAIL;
            s.detail_message = "No compute queue family found on physical device";
            stage_success = false;
        }
        auto t1 = std::chrono::high_resolution_clock::now();
        s.elapsed_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        report.stages.push_back(s);
        if (verbose) {
            std::cout << "  [" << (s.result == StageResult::PASS ? "PASS" : "FAIL") << "] " << std::left << std::setw(32) << s.stage_name
                      << " (" << std::fixed << std::setprecision(2) << s.elapsed_ms << " ms) - " << s.detail_message << std::endl;
        }
    }

    // V5: Logical Device Creation
    if (stage_success && pfn_vkCreateDevice && pfn_vkDestroyDevice) {
        auto t0 = std::chrono::high_resolution_clock::now();
        float queuePriority = 1.0f;
        VkDeviceQueueCreateInfo qCreateInfo{};
        qCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
        qCreateInfo.queueFamilyIndex = compute_queue_family_index;
        qCreateInfo.queueCount = 1;
        qCreateInfo.pQueuePriorities = &queuePriority;

        VkDeviceCreateInfo devCreateInfo{};
        devCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
        devCreateInfo.queueCreateInfoCount = 1;
        devCreateInfo.pQueueCreateInfos = &qCreateInfo;

        int res = pfn_vkCreateDevice(physical_device, &devCreateInfo, nullptr, &vk_device);

        StageReport s;
        s.stage_id = 5;
        s.stage_name = FormatStageName(5, "Logical Device Creation");
        s.allocated_bytes = 0;

        if (res == VK_SUCCESS && vk_device) {
            s.result = StageResult::PASS;
            s.detail_message = "Logical Device Created successfully";
            report.passed_stages++;
            if (pfn_vkGetDeviceQueue) {
                pfn_vkGetDeviceQueue(vk_device, compute_queue_family_index, 0, &compute_queue);
            }
        } else {
            s.result = StageResult::FAIL;
            s.detail_message = "vkCreateDevice() failed with error code " + std::to_string(res);
            stage_success = false;
        }
        auto t1 = std::chrono::high_resolution_clock::now();
        s.elapsed_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        report.stages.push_back(s);
        if (verbose) {
            std::cout << "  [" << (s.result == StageResult::PASS ? "PASS" : "FAIL") << "] " << std::left << std::setw(32) << s.stage_name
                      << " (" << std::fixed << std::setprecision(2) << s.elapsed_ms << " ms) - " << s.detail_message << std::endl;
        }
    }

    // Shared handles across V6, V7, V8, V9
    VkDeviceMemory v6_memory = VK_NULL_HANDLE;
    VkShaderModule shader_module = VK_NULL_HANDLE;
    VkDescriptorSetLayout descriptor_set_layout = VK_NULL_HANDLE;
    VkPipelineLayout pipeline_layout = VK_NULL_HANDLE;
    VkPipeline compute_pipeline = VK_NULL_HANDLE;

    VkBuffer bufA = VK_NULL_HANDLE, bufB = VK_NULL_HANDLE, bufC = VK_NULL_HANDLE;
    VkDeviceMemory memA = VK_NULL_HANDLE, memB = VK_NULL_HANDLE, memC = VK_NULL_HANDLE;
    VkDescriptorPool desc_pool = VK_NULL_HANDLE;
    VkDescriptorSet desc_set = VK_NULL_HANDLE;
    VkCommandPool cmd_pool = VK_NULL_HANDLE;
    VkCommandBuffer cmd_buffer = VK_NULL_HANDLE;

    // Helper to find memory type with memoryTypeBits filter
    auto find_memory_type = [&](uint32_t type_bits, VkMemoryPropertyFlags req_props, bool* out_is_coherent, VkMemoryPropertyFlags* out_flags) -> int {
        VkPhysicalDeviceMemoryProperties memProps{};
        pfn_vkGetPhysicalDeviceMemoryProperties(physical_device, &memProps);

        // 1st priority: Exact match with HOST_COHERENT
        for (uint32_t i = 0; i < memProps.memoryTypeCount; ++i) {
            if ((type_bits & (1u << i)) &&
                ((memProps.memoryTypes[i].propertyFlags & req_props) == req_props) &&
                (memProps.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)) {
                if (out_is_coherent) *out_is_coherent = true;
                if (out_flags) *out_flags = memProps.memoryTypes[i].propertyFlags;
                return (int)i;
            }
        }
        // 2nd priority: Fallback without HOST_COHERENT
        for (uint32_t i = 0; i < memProps.memoryTypeCount; ++i) {
            if ((type_bits & (1u << i)) && ((memProps.memoryTypes[i].propertyFlags & req_props) == req_props)) {
                if (out_is_coherent) *out_is_coherent = ((memProps.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) != 0);
                if (out_flags) *out_flags = memProps.memoryTypes[i].propertyFlags;
                return (int)i;
            }
        }
        return -1;
    };

    // V6: Buffer Allocation, Binding & Mapping Probe
    if (stage_success && pfn_vkGetPhysicalDeviceMemoryProperties && pfn_vkCreateBuffer &&
        pfn_vkGetBufferMemoryRequirements && pfn_vkAllocateMemory && pfn_vkBindBufferMemory &&
        pfn_vkMapMemory && pfn_vkUnmapMemory && pfn_vkDestroyBuffer && pfn_vkFreeMemory) {

        auto t0 = std::chrono::high_resolution_clock::now();

        VkPhysicalDeviceProperties devProps{};
        pfn_vkGetPhysicalDeviceProperties(physical_device, &devProps);
        VkDeviceSize atomSize = devProps.limits.nonCoherentAtomSize;
        if (atomSize == 0) atomSize = 1;

        VkDeviceSize testAllocSize = 4 * 1024 * 1024; // 4MB probe
        VkBufferCreateInfo bufferInfo{};
        bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        bufferInfo.size = testAllocSize;
        bufferInfo.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
        bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

        VkBuffer test_buffer = VK_NULL_HANDLE;
        int res_buf = pfn_vkCreateBuffer(vk_device, &bufferInfo, nullptr, &test_buffer);

        StageReport s;
        s.stage_id = 6;
        s.stage_name = FormatStageName(6, "Buffer Allocation & Mapping");

        if (res_buf == VK_SUCCESS && test_buffer) {
            VkMemoryRequirements memReqs{};
            pfn_vkGetBufferMemoryRequirements(vk_device, test_buffer, &memReqs);

            bool is_coherent = false;
            VkMemoryPropertyFlags foundFlags = 0;
            int memTypeIndex = find_memory_type(memReqs.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT, &is_coherent, &foundFlags);

            if (memTypeIndex >= 0) {
                VkMemoryAllocateInfo allocInfo{};
                allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
                allocInfo.allocationSize = memReqs.size; // Strict requirement: use memReqs.size!
                allocInfo.memoryTypeIndex = (uint32_t)memTypeIndex;

                int res_mem = pfn_vkAllocateMemory(vk_device, &allocInfo, nullptr, &v6_memory);
                if (res_mem == VK_SUCCESS && v6_memory) {
                    pfn_vkBindBufferMemory(vk_device, test_buffer, v6_memory, 0);

                    void* mapped_ptr = nullptr;
                    int res_map = pfn_vkMapMemory(vk_device, v6_memory, 0, memReqs.size, 0, &mapped_ptr);
                    if (res_map == VK_SUCCESS && mapped_ptr) {
                        // Test pattern write
                        memset(mapped_ptr, 0xAB, (size_t)std::min(memReqs.size, (VkDeviceSize)1024));

                        // If non-coherent, flush host write
                        if (!is_coherent && pfn_vkFlushMappedMemoryRanges) {
                            VkMappedMemoryRange range{};
                            range.sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE;
                            range.memory = v6_memory;
                            range.offset = 0;
                            range.size = VK_WHOLE_SIZE;
                            pfn_vkFlushMappedMemoryRanges(vk_device, 1, &range);
                        }

                        // Invalidate read
                        if (!is_coherent && pfn_vkInvalidateMappedMemoryRanges) {
                            VkMappedMemoryRange range{};
                            range.sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE;
                            range.memory = v6_memory;
                            range.offset = 0;
                            range.size = VK_WHOLE_SIZE;
                            pfn_vkInvalidateMappedMemoryRanges(vk_device, 1, &range);
                        }

                        uint8_t read_val = ((uint8_t*)mapped_ptr)[0];
                        pfn_vkUnmapMemory(vk_device, v6_memory);

                        if (read_val == 0xAB) {
                            s.result = StageResult::PASS;
                            s.allocated_bytes = memReqs.size;
                            std::stringstream ss;
                            ss << "Allocated & Mapped " << (memReqs.size / 1024) << "KB (type=" << memTypeIndex
                               << ", bits=0x" << std::hex << memReqs.memoryTypeBits << std::dec
                               << ", flags=0x" << std::hex << foundFlags << std::dec
                               << ", coherent=" << (is_coherent ? "true" : "false")
                               << ", atom=" << atomSize << ")";
                            s.detail_message = ss.str();
                            report.passed_stages++;
                        } else {
                            s.result = StageResult::FAIL;
                            s.detail_message = "Mapped buffer verification readback mismatch";
                            stage_success = false;
                        }
                    } else {
                        s.result = StageResult::FAIL;
                        s.detail_message = "vkMapMemory() failed with error code " + std::to_string(res_map);
                        stage_success = false;
                    }
                } else {
                    s.result = StageResult::FAIL;
                    s.detail_message = "vkAllocateMemory() failed with error code " + std::to_string(res_mem);
                    stage_success = false;
                }
            } else {
                s.result = StageResult::FAIL;
                s.detail_message = "No Host-Visible memory type matching memoryTypeBits 0x" + [&memReqs]() {
                    std::stringstream ss; ss << std::hex << memReqs.memoryTypeBits; return ss.str();
                }();
                stage_success = false;
            }

            pfn_vkDestroyBuffer(vk_device, test_buffer, nullptr);
            if (v6_memory) {
                pfn_vkFreeMemory(vk_device, v6_memory, nullptr);
                v6_memory = VK_NULL_HANDLE;
            }
        } else {
            s.result = StageResult::FAIL;
            s.detail_message = "vkCreateBuffer() failed with error code " + std::to_string(res_buf);
            stage_success = false;
        }

        auto t1 = std::chrono::high_resolution_clock::now();
        s.elapsed_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        report.stages.push_back(s);
        if (verbose) {
            std::cout << "  [" << (s.result == StageResult::PASS ? "PASS" : "FAIL") << "] " << std::left << std::setw(32) << s.stage_name
                      << " (" << std::fixed << std::setprecision(2) << s.elapsed_ms << " ms) - " << s.detail_message << std::endl;
        }
    }

    // V7: SPIR-V Shader Module & Compute Pipeline Creation
    if (stage_success && pfn_vkCreateShaderModule && pfn_vkCreateDescriptorSetLayout &&
        pfn_vkCreatePipelineLayout && pfn_vkCreateComputePipelines) {

        auto t0 = std::chrono::high_resolution_clock::now();

        // 1. Create Shader Module
        VkShaderModuleCreateInfo smci{};
        smci.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
        smci.codeSize = shaders::kMatmulComputeShaderSpvSize;
        smci.pCode = shaders::kMatmulComputeShaderSpv;

        int res_sm = pfn_vkCreateShaderModule(vk_device, &smci, nullptr, &shader_module);

        // 2. Create Descriptor Set Layout (Bindings 0, 1, 2)
        VkDescriptorSetLayoutBinding bindings[3]{};
        for (int i = 0; i < 3; ++i) {
            bindings[i].binding = i;
            bindings[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            bindings[i].descriptorCount = 1;
            bindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        }
        VkDescriptorSetLayoutCreateInfo dslci{};
        dslci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        dslci.bindingCount = 3;
        dslci.pBindings = bindings;
        int res_dsl = pfn_vkCreateDescriptorSetLayout(vk_device, &dslci, nullptr, &descriptor_set_layout);

        // 3. Create Pipeline Layout with Push Constants (12 bytes: uint32 M, K, N)
        VkPushConstantRange pcr{};
        pcr.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        pcr.offset = 0;
        pcr.size = 12;

        VkPipelineLayoutCreateInfo plci{};
        plci.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        plci.setLayoutCount = 1;
        plci.pSetLayouts = &descriptor_set_layout;
        plci.pushConstantRangeCount = 1;
        plci.pPushConstantRanges = &pcr;
        int res_pl = pfn_vkCreatePipelineLayout(vk_device, &plci, nullptr, &pipeline_layout);

        // 4. Create Compute Pipeline
        VkComputePipelineCreateInfo cpci{};
        cpci.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
        cpci.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        cpci.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        cpci.stage.module = shader_module;
        cpci.stage.pName = "main";
        cpci.layout = pipeline_layout;
        int res_pipe = pfn_vkCreateComputePipelines(vk_device, VK_NULL_HANDLE, 1, &cpci, nullptr, &compute_pipeline);

        auto t1 = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

        StageReport s;
        s.stage_id = 7;
        s.stage_name = FormatStageName(7, "SPIR-V Pipeline Compilation");
        s.elapsed_ms = ms;
        s.allocated_bytes = shaders::kMatmulComputeShaderSpvSize;

        if (res_sm == VK_SUCCESS && res_dsl == VK_SUCCESS && res_pl == VK_SUCCESS && res_pipe == VK_SUCCESS && compute_pipeline) {
            s.result = StageResult::PASS;
            s.detail_message = "16x16 Tiled SGEMM Compute Pipeline Compiled (SPIR-V " +
                               std::to_string(shaders::kMatmulComputeShaderSpvSize) + " bytes)";
            report.passed_stages++;
        } else {
            s.result = StageResult::FAIL;
            s.detail_message = "Pipeline creation failed (sm=" + std::to_string(res_sm) +
                               ", pipe=" + std::to_string(res_pipe) + ")";
            stage_success = false;
        }
        report.stages.push_back(s);
        if (verbose) {
            std::cout << "  [" << (s.result == StageResult::PASS ? "PASS" : "FAIL") << "] " << std::left << std::setw(32) << s.stage_name
                      << " (" << std::fixed << std::setprecision(2) << ms << " ms) - " << s.detail_message << std::endl;
        }
    }

    // Helper to create & map buffer
    auto create_storage_buffer = [&](VkDeviceSize size, VkBuffer& out_buf, VkDeviceMemory& out_mem, bool& out_coherent) -> bool {
        VkBufferCreateInfo bi{};
        bi.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        bi.size = size;
        bi.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
        bi.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        if (pfn_vkCreateBuffer(vk_device, &bi, nullptr, &out_buf) != VK_SUCCESS) return false;

        VkMemoryRequirements reqs{};
        pfn_vkGetBufferMemoryRequirements(vk_device, out_buf, &reqs);

        int midx = find_memory_type(reqs.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT, &out_coherent, nullptr);
        if (midx < 0) return false;

        VkMemoryAllocateInfo ai{};
        ai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        ai.allocationSize = reqs.size;
        ai.memoryTypeIndex = (uint32_t)midx;
        if (pfn_vkAllocateMemory(vk_device, &ai, nullptr, &out_mem) != VK_SUCCESS) return false;
        if (pfn_vkBindBufferMemory(vk_device, out_buf, out_mem, 0) != VK_SUCCESS) return false;
        return true;
    };

    bool coherentA = false, coherentB = false, coherentC = false;
    const uint32_t kDim = 16;
    const size_t kMatrixElements = kDim * kDim; // 256 floats
    const VkDeviceSize kMatrixBytes = kMatrixElements * sizeof(float); // 1024 bytes

    // V8: Compute Shader Dispatch (16x16 Tiled SGEMM on Real Hardware)
    if (stage_success && compute_pipeline && pfn_vkCreateDescriptorPool &&
        pfn_vkAllocateDescriptorSets && pfn_vkUpdateDescriptorSets &&
        pfn_vkCreateCommandPool && pfn_vkAllocateCommandBuffers &&
        pfn_vkBeginCommandBuffer && pfn_vkCmdBindPipeline &&
        pfn_vkCmdBindDescriptorSets && pfn_vkCmdPushConstants &&
        pfn_vkCmdDispatch && pfn_vkEndCommandBuffer &&
        pfn_vkQueueSubmit && pfn_vkQueueWaitIdle && compute_queue) {

        auto t0 = std::chrono::high_resolution_clock::now();

        // 1. Create buffers for A, B, C
        bool okA = create_storage_buffer(kMatrixBytes, bufA, memA, coherentA);
        bool okB = create_storage_buffer(kMatrixBytes, bufB, memB, coherentB);
        bool okC = create_storage_buffer(kMatrixBytes, bufC, memC, coherentC);

        if (okA && okB && okC) {
            // Fill A with sequential floats: A[i*16 + j] = i + j + 1
            void* ptrA = nullptr;
            pfn_vkMapMemory(vk_device, memA, 0, kMatrixBytes, 0, &ptrA);
            float* fA = (float*)ptrA;
            for (uint32_t i = 0; i < kMatrixElements; ++i) {
                fA[i] = (float)(i + 1);
            }
            if (!coherentA && pfn_vkFlushMappedMemoryRanges) {
                VkMappedMemoryRange r{ VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE, nullptr, memA, 0, VK_WHOLE_SIZE };
                pfn_vkFlushMappedMemoryRanges(vk_device, 1, &r);
            }
            pfn_vkUnmapMemory(vk_device, memA);

            // Fill B with Identity Matrix (16x16): B[i*16 + j] = (i == j ? 1.0 : 0.0)
            void* ptrB = nullptr;
            pfn_vkMapMemory(vk_device, memB, 0, kMatrixBytes, 0, &ptrB);
            float* fB = (float*)ptrB;
            for (uint32_t r = 0; r < kDim; ++r) {
                for (uint32_t c = 0; c < kDim; ++c) {
                    fB[r * kDim + c] = (r == c) ? 1.0f : 0.0f;
                }
            }
            if (!coherentB && pfn_vkFlushMappedMemoryRanges) {
                VkMappedMemoryRange r{ VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE, nullptr, memB, 0, VK_WHOLE_SIZE };
                pfn_vkFlushMappedMemoryRanges(vk_device, 1, &r);
            }
            pfn_vkUnmapMemory(vk_device, memB);

            // Zero out C
            void* ptrC = nullptr;
            pfn_vkMapMemory(vk_device, memC, 0, kMatrixBytes, 0, &ptrC);
            memset(ptrC, 0, (size_t)kMatrixBytes);
            if (!coherentC && pfn_vkFlushMappedMemoryRanges) {
                VkMappedMemoryRange r{ VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE, nullptr, memC, 0, VK_WHOLE_SIZE };
                pfn_vkFlushMappedMemoryRanges(vk_device, 1, &r);
            }
            pfn_vkUnmapMemory(vk_device, memC);

            // 2. Create Descriptor Pool & Allocate Descriptor Set
            VkDescriptorPoolSize poolSize{};
            poolSize.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            poolSize.descriptorCount = 3;

            VkDescriptorPoolCreateInfo dpci{};
            dpci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
            dpci.maxSets = 1;
            dpci.poolSizeCount = 1;
            dpci.pPoolSizes = &poolSize;
            pfn_vkCreateDescriptorPool(vk_device, &dpci, nullptr, &desc_pool);

            VkDescriptorSetAllocateInfo dsai{};
            dsai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
            dsai.descriptorPool = desc_pool;
            dsai.descriptorSetCount = 1;
            dsai.pSetLayouts = &descriptor_set_layout;
            pfn_vkAllocateDescriptorSets(vk_device, &dsai, &desc_set);

            // 3. Update Descriptor Sets
            VkDescriptorBufferInfo dbi[3]{};
            dbi[0] = { bufA, 0, kMatrixBytes };
            dbi[1] = { bufB, 0, kMatrixBytes };
            dbi[2] = { bufC, 0, kMatrixBytes };

            VkWriteDescriptorSet wds[3]{};
            for (int i = 0; i < 3; ++i) {
                wds[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                wds[i].dstSet = desc_set;
                wds[i].dstBinding = i;
                wds[i].descriptorCount = 1;
                wds[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
                wds[i].pBufferInfo = &dbi[i];
            }
            pfn_vkUpdateDescriptorSets(vk_device, 3, wds, 0, nullptr);

            // 4. Create Command Pool & Command Buffer
            VkCommandPoolCreateInfo cpci{};
            cpci.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
            cpci.queueFamilyIndex = compute_queue_family_index;
            pfn_vkCreateCommandPool(vk_device, &cpci, nullptr, &cmd_pool);

            VkCommandBufferAllocateInfo cbai{};
            cbai.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
            cbai.commandPool = cmd_pool;
            cbai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
            cbai.commandBufferCount = 1;
            pfn_vkAllocateCommandBuffers(vk_device, &cbai, &cmd_buffer);

            // 5. Record Dispatch Commands
            VkCommandBufferBeginInfo beginInfo{};
            beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
            pfn_vkBeginCommandBuffer(cmd_buffer, &beginInfo);

            pfn_vkCmdBindPipeline(cmd_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, compute_pipeline);
            pfn_vkCmdBindDescriptorSets(cmd_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_layout, 0, 1, &desc_set, 0, nullptr);

            uint32_t pushConstants[3] = { kDim, kDim, kDim }; // M=16, K=16, N=16
            pfn_vkCmdPushConstants(cmd_buffer, pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pushConstants), pushConstants);

            // Local size is 16x16, so 1 group in X and Y covers all 16x16 elements
            pfn_vkCmdDispatch(cmd_buffer, 1, 1, 1);
            pfn_vkEndCommandBuffer(cmd_buffer);

            // 6. Submit to Queue & Synchronize
            VkSubmitInfo submitInfo{};
            submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
            submitInfo.commandBufferCount = 1;
            submitInfo.pCommandBuffers = &cmd_buffer;

            int res_submit = pfn_vkQueueSubmit(compute_queue, 1, &submitInfo, VK_NULL_HANDLE);
            int res_wait = pfn_vkQueueWaitIdle(compute_queue);

            auto t1 = std::chrono::high_resolution_clock::now();
            double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

            StageReport s;
            s.stage_id = 8;
            s.stage_name = FormatStageName(8, "Compute Shader Dispatch");
            s.elapsed_ms = ms;
            s.allocated_bytes = kMatrixBytes * 3;

            if (res_submit == VK_SUCCESS && res_wait == VK_SUCCESS) {
                s.result = StageResult::PASS;
                s.detail_message = "16x16 SGEMM GPU Dispatch completed (Submit=" + std::to_string(res_submit) +
                                   ", WaitIdle=" + std::to_string(res_wait) + ")";
                report.passed_stages++;
            } else {
                s.result = StageResult::FAIL;
                s.detail_message = "Queue execution failed (submit=" + std::to_string(res_submit) +
                                   ", wait=" + std::to_string(res_wait) + ")";
                stage_success = false;
            }
            report.stages.push_back(s);
            if (verbose) {
                std::cout << "  [" << (s.result == StageResult::PASS ? "PASS" : "FAIL") << "] " << std::left << std::setw(32) << s.stage_name
                          << " (" << std::fixed << std::setprecision(2) << ms << " ms) - " << s.detail_message << std::endl;
            }
        } else {
            StageReport s;
            s.stage_id = 8;
            s.stage_name = FormatStageName(8, "Compute Shader Dispatch");
            s.elapsed_ms = 0.0;
            s.result = StageResult::FAIL;
            s.detail_message = "Failed to allocate GPU buffers for SGEMM dispatch";
            stage_success = false;
            report.stages.push_back(s);
        }
    }

    // V9: Result Checksum & Full Output Verification
    if (stage_success && bufC && memC) {
        auto t0 = std::chrono::high_resolution_clock::now();

        void* mappedC = nullptr;
        pfn_vkMapMemory(vk_device, memC, 0, kMatrixBytes, 0, &mappedC);

        if (!coherentC && pfn_vkInvalidateMappedMemoryRanges) {
            VkMappedMemoryRange r{ VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE, nullptr, memC, 0, VK_WHOLE_SIZE };
            pfn_vkInvalidateMappedMemoryRanges(vk_device, 1, &r);
        }

        const float* fC = (const float*)mappedC;
        size_t mismatches = 0;
        double checksum = 0.0;

        // Since B is Identity matrix, C should exactly equal A: C[i] == (float)(i + 1)
        for (uint32_t i = 0; i < kMatrixElements; ++i) {
            float expected = (float)(i + 1);
            float actual = fC[i];
            checksum += actual;
            if (std::abs(actual - expected) > 1e-4f) {
                mismatches++;
            }
        }
        pfn_vkUnmapMemory(vk_device, memC);

        auto t1 = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

        StageReport s;
        s.stage_id = 9;
        s.stage_name = FormatStageName(9, "Result Checksum Validation");
        s.elapsed_ms = ms;
        s.allocated_bytes = 0;

        if (mismatches == 0) {
            s.result = StageResult::PASS;
            std::stringstream ss;
            ss << "Full 256/256 elements matched CPU reference (Checksum: " << std::fixed << std::setprecision(1) << checksum << ", 0 errors)";
            s.detail_message = ss.str();
            report.passed_stages++;
        } else {
            s.result = StageResult::FAIL;
            s.detail_message = "Numerical mismatch in " + std::to_string(mismatches) + "/256 elements (Checksum: " + std::to_string(checksum) + ")";
            stage_success = false;
        }
        report.stages.push_back(s);
        if (verbose) {
            std::cout << "  [" << (s.result == StageResult::PASS ? "PASS" : "FAIL") << "] " << std::left << std::setw(32) << s.stage_name
                      << " (" << std::fixed << std::setprecision(2) << ms << " ms) - " << s.detail_message << std::endl;
        }
    }

    // V10 and V11: Honest reporting of Model Inference Stages (Advanced Certification)
    const char* v10_v11_titles[2] = {
        "GGML MatMul Tensor Ops",
        "End-to-End Model Inference"
    };

    for (int i = 10; i <= 11; ++i) {
        StageReport s;
        s.stage_id = i;
        s.stage_name = FormatStageName(i, v10_v11_titles[i - 10]);
        s.elapsed_ms = 0.0;
        s.allocated_bytes = 0;
        s.result = StageResult::SKIPPED;

        if (stage_success) {
            if (i == 10) s.detail_message = "Vulkan Compute certified (V0~V9 passed); GGML MatMul engine deferred to model runtime";
            else if (i == 11) s.detail_message = "End-to-End graph inference deferred to model pipeline";
        } else {
            s.detail_message = "Skipped due to preceding stage failure";
        }
        report.stages.push_back(s);
        if (verbose) {
            std::cout << "  [SKIP] " << std::left << std::setw(32) << s.stage_name
                      << " (" << std::fixed << std::setprecision(2) << s.elapsed_ms << " ms) - " << s.detail_message << std::endl;
        }
    }

    // STRICT RAII CLEANUP OF ALL ALLOCATED HANDLES
    if (cmd_pool && pfn_vkDestroyCommandPool && vk_device) {
        pfn_vkDestroyCommandPool(vk_device, cmd_pool, nullptr);
    }
    if (desc_pool && pfn_vkDestroyDescriptorPool && vk_device) {
        pfn_vkDestroyDescriptorPool(vk_device, desc_pool, nullptr);
    }
    if (bufA && pfn_vkDestroyBuffer && vk_device) pfn_vkDestroyBuffer(vk_device, bufA, nullptr);
    if (memA && pfn_vkFreeMemory && vk_device) pfn_vkFreeMemory(vk_device, memA, nullptr);
    if (bufB && pfn_vkDestroyBuffer && vk_device) pfn_vkDestroyBuffer(vk_device, bufB, nullptr);
    if (memB && pfn_vkFreeMemory && vk_device) pfn_vkFreeMemory(vk_device, memB, nullptr);
    if (bufC && pfn_vkDestroyBuffer && vk_device) pfn_vkDestroyBuffer(vk_device, bufC, nullptr);
    if (memC && pfn_vkFreeMemory && vk_device) pfn_vkFreeMemory(vk_device, memC, nullptr);

    if (compute_pipeline && pfn_vkDestroyPipeline && vk_device) {
        pfn_vkDestroyPipeline(vk_device, compute_pipeline, nullptr);
    }
    if (pipeline_layout && pfn_vkDestroyPipelineLayout && vk_device) {
        pfn_vkDestroyPipelineLayout(vk_device, pipeline_layout, nullptr);
    }
    if (descriptor_set_layout && pfn_vkDestroyDescriptorSetLayout && vk_device) {
        pfn_vkDestroyDescriptorSetLayout(vk_device, descriptor_set_layout, nullptr);
    }
    if (shader_module && pfn_vkDestroyShaderModule && vk_device) {
        pfn_vkDestroyShaderModule(vk_device, shader_module, nullptr);
    }
    if (vk_device && pfn_vkDestroyDevice) {
        pfn_vkDestroyDevice(vk_device, nullptr);
    }
    if (vk_instance && pfn_vkDestroyInstance) {
        pfn_vkDestroyInstance(vk_instance, nullptr);
    }

    auto total_end = std::chrono::high_resolution_clock::now();
    report.total_elapsed_ms = std::chrono::duration<double, std::milli>(total_end - total_start).count();

    // Strict certification rule: V0 through V9 MUST each individually PASS!
    bool computeCertified = (
        report.stages.size() >= 10 &&
        report.stages[0].result == StageResult::PASS &&
        report.stages[1].result == StageResult::PASS &&
        report.stages[2].result == StageResult::PASS &&
        report.stages[3].result == StageResult::PASS &&
        report.stages[4].result == StageResult::PASS &&
        report.stages[5].result == StageResult::PASS &&
        report.stages[6].result == StageResult::PASS &&
        report.stages[7].result == StageResult::PASS &&
        report.stages[8].result == StageResult::PASS &&
        report.stages[9].result == StageResult::PASS
    );

    report.compute_certified = computeCertified;
    report.model_certified = false;
    report.overall_success = computeCertified;
    report.recommended_backend = computeCertified ? "vulkan" : "cpu_neon";

    if (verbose) {
        std::cout << "------------------------------------------------------------" << std::endl;
        std::cout << "  Scorecard: " << report.passed_stages << "/" << report.total_stages << " Stages Passed"
                  << " | Compute Certified: " << (computeCertified ? "TRUE" : "FALSE")
                  << " | Model Certified: FALSE (V10/V11 deferred)" << std::endl;
        std::cout << "  Total Time: " << std::fixed << std::setprecision(2) << report.total_elapsed_ms << " ms"
                  << " | Target: " << report.recommended_backend << std::endl;
        std::cout << "============================================================\n" << std::endl;
    }

    return report;
}

bool ProbeSuite::QuickProbe(std::string* out_device_name) {
    DiagnosticReport report;
    std::string state_path = GetStandardCachePath();
    if (LoadState(&report, state_path)) {
        // Strict certification: must be compute-certified and valid device
        bool is_certified = report.compute_certified || (report.overall_success && report.passed_stages >= 10);
        if (report.overall_success && report.passed_stages >= 10 && is_certified && !report.device_name.empty()) {
            if (out_device_name) *out_device_name = report.device_name;
            return true;
        }
    }
    return false;
}

bool ProbeSuite::SaveState(const DiagnosticReport& report, const std::string& state_file_path) {
    std::string path = state_file_path.empty() ? GetStandardCachePath() : state_file_path;
    std::ofstream ofs(path);
    if (!ofs.is_open()) return false;

    // Generate ISO-8601 UTC timestamp
    std::time_t now = std::time(nullptr);
    std::tm tm_buf;
#if defined(_WIN32)
    gmtime_s(&tm_buf, &now);
#else
    gmtime_r(&now, &tm_buf);
#endif
    char time_str[64];
    std::strftime(time_str, sizeof(time_str), "%Y-%m-%dT%H:%M:%SZ", &tm_buf);

    ofs << "{\n";
    ofs << "  \"schemaVersion\": 2,\n";
    ofs << "  \"diagnosticScope\": \"compute\",\n";
    ofs << "  \"scopeSuccess\": " << (report.overall_success ? "true" : "false") << ",\n";
    ofs << "  \"overallSuccess\": " << (report.overall_success ? "true" : "false") << ",\n";
    ofs << "  \"computeCertified\": " << (report.compute_certified ? "true" : "false") << ",\n";
    ofs << "  \"modelCertified\": " << (report.model_certified ? "true" : "false") << ",\n";
    ofs << "  \"verifiedAt\": \"" << time_str << "\",\n";
    ofs << "  \"passedStages\": " << report.passed_stages << ",\n";
    ofs << "  \"totalStages\": " << report.total_stages << ",\n";
    ofs << "  \"totalElapsedMs\": " << report.total_elapsed_ms << ",\n";
    ofs << "  \"verificationSource\": \"native_c_hal\",\n";
    ofs << "  \"deviceFingerprint\": {\n";
    ofs << "    \"deviceName\": \"" << report.device_name << "\",\n";
    ofs << "    \"vendorId\": " << report.vendor_id << ",\n";
    ofs << "    \"deviceId\": " << report.device_id << ",\n";
    ofs << "    \"apiVersion\": " << report.api_version << ",\n";
    ofs << "    \"driverVersion\": \"" << report.driver_version << "\",\n";
    ofs << "    \"loaderPath\": \"" << report.loader_path << "\"\n";
    ofs << "  },\n";
    ofs << "  \"recommendedBackend\": \"" << report.recommended_backend << "\"\n";
    ofs << "}\n";
    return true;
}

bool ProbeSuite::LoadState(DiagnosticReport* out_report, const std::string& state_file_path) {
    if (!out_report) return false;
    std::string path = state_file_path.empty() ? GetStandardCachePath() : state_file_path;
    std::ifstream ifs(path);
    if (!ifs.is_open()) return false;

    // Minimal JSON parse
    std::string line;
    while (std::getline(ifs, line)) {
        if (line.find("\"overallSuccess\": true") != std::string::npos) {
            out_report->overall_success = true;
        } else if (line.find("\"computeCertified\": true") != std::string::npos) {
            out_report->compute_certified = true;
        } else if (line.find("\"modelCertified\": true") != std::string::npos) {
            out_report->model_certified = true;
        } else if (line.find("\"passedStages\":") != std::string::npos) {
            size_t pos = line.find(":");
            if (pos != std::string::npos) {
                out_report->passed_stages = std::atoi(line.c_str() + pos + 1);
            }
        } else if (line.find("\"deviceName\": \"") != std::string::npos) {
            size_t start = line.find("\"deviceName\": \"") + 15;
            size_t end = line.find("\"", start);
            if (start != std::string::npos && end != std::string::npos) {
                out_report->device_name = line.substr(start, end - start);
            }
        }
    }
    return true;
}

} // namespace doctor
} // namespace ameva
