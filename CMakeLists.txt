cmake_minimum_required(VERSION 3.18)
project(ameva_vulkan_runtime VERSION 1.0.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Source Files
set(CORE_SOURCES
    src/core/vulkan_loader.cpp
    src/quirks/adreno_quirks.cpp
    src/quirks/mali_quirks.cpp
    src/doctor/probe_stages.cpp
    src/c_api/ameva_vulkan_c_api.cpp
)

# Shared Library (libameva_vulkan.so / ameva_vulkan.dll)
add_library(ameva_vulkan SHARED ${CORE_SOURCES})

target_include_directories(ameva_vulkan PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>
    $<INSTALL_INTERFACE:include>
)

if (ANDROID)
    target_link_libraries(ameva_vulkan PRIVATE dl log)
elseif (UNIX)
    target_link_libraries(ameva_vulkan PRIVATE dl pthread)
endif()

# Standalone CLI Executable (ameva-gpu)
add_executable(ameva_gpu_native src/c_api/ameva_vulkan_c_api.cpp ${CORE_SOURCES})
set_target_properties(ameva_gpu_native PROPERTIES OUTPUT_NAME "ameva-gpu-native")
