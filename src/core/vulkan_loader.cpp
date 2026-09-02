#include "vulkan_loader.h"
#include <iostream>
#include <vector>

#if defined(_WIN32)
#include <windows.h>
#define DLOPEN(path) LoadLibraryA(path)
#define DLSYM(handle, name) GetProcAddress((HMODULE)handle, name)
#define DLCLOSE(handle) FreeLibrary((HMODULE)handle)
#else
#include <dlfcn.h>
#define DLOPEN(path) dlopen(path, RTLD_LAZY | RTLD_LOCAL)
#define DLSYM(handle, name) dlsym(handle, name)
#define DLCLOSE(handle) dlclose(handle)
#endif

namespace ameva {
namespace core {

VulkanLoader::VulkanLoader() = default;

VulkanLoader::~VulkanLoader() {
    Unload();
}

bool VulkanLoader::Load(const std::string& explicit_path) {
    if (library_handle_ != nullptr) {
        return true;
    }

    std::vector<std::string> search_candidates;
    if (!explicit_path.empty()) {
        search_candidates.push_back(explicit_path);
    }

#if defined(__ANDROID__)
    // Android Bionic System Vendor Driver Priority (Single Loader Chain)
    search_candidates.push_back("/system/lib64/libvulkan.so");
    search_candidates.push_back("/vendor/lib64/hw/vulkan.adreno.so");
    search_candidates.push_back("libvulkan.so");
#elif defined(_WIN32)
    search_candidates.push_back("vulkan-1.dll");
#else
    search_candidates.push_back("libvulkan.so.1");
    search_candidates.push_back("libvulkan.so");
#endif

    for (const auto& path : search_candidates) {
        library_handle_ = DLOPEN(path.c_str());
        if (library_handle_ != nullptr) {
            loaded_path_ = path;
            break;
        }
    }

    return library_handle_ != nullptr;
}

void VulkanLoader::Unload() {
    if (library_handle_ != nullptr) {
        DLCLOSE(library_handle_);
        library_handle_ = nullptr;
        loaded_path_.clear();
    }
}

void* VulkanLoader::GetProcAddr(const char* name) const {
    if (library_handle_ == nullptr || name == nullptr) {
        return nullptr;
    }
    return (void*)DLSYM(library_handle_, name);
}

} // namespace core
} // namespace ameva
