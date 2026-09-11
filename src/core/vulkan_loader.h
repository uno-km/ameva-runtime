#pragma once
#include <string>
#include <cstdint>
#include <functional>

#if defined(__ANDROID__) || defined(__linux__)
#include <vulkan/vulkan.h>
#else
// Windows/Non-Vulkan Host Mock Typedefs for Compilation Compatibility
typedef void* VkInstance;
typedef void* VkPhysicalDevice;
typedef void* VkDevice;
typedef void* VkQueue;
typedef void* VkCommandBuffer;
typedef uint32_t VkResult;
#define VK_SUCCESS 0
#define VK_ERROR_INITIALIZATION_FAILED -3
#endif

namespace ameva {
namespace core {

/**
 * @brief Vulkan System ICD Dynamic Loader & Single Loader Chain Enforcer
 * 
 * Prevents symbol collisions between Termux Mesa user-space loader and
 * native Android Bionic system vendor ICD (/system/lib64/libvulkan.so).
 */
class VulkanLoader {
public:
    VulkanLoader();
    ~VulkanLoader();

    // Disable copy
    VulkanLoader(const VulkanLoader&) = delete;
    VulkanLoader& operator=(const VulkanLoader&) = delete;

    /**
     * @brief Loads the system Vulkan library dynamically.
     * @param explicit_path Optional override path (defaults to /system/lib64/libvulkan.so on Android).
     * @return true if successfully loaded and symbols resolved.
     */
    bool Load(const std::string& explicit_path = "");

    /**
     * @brief Unloads the library and resets all function pointers.
     */
    void Unload();

    bool IsLoaded() const { return library_handle_ != nullptr; }
    const std::string& GetLoadedPath() const { return loaded_path_; }

    // Core Vulkan Function Pointers
    void* GetProcAddr(const char* name) const;

private:
    void* library_handle_{nullptr};
    std::string loaded_path_;
};

} // namespace core
} // namespace ameva
