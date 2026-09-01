#pragma once
#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

#if defined(_WIN32)
#define AMEVA_API __declspec(dllexport)
#else
#define AMEVA_API __attribute__((visibility("default")))
#endif

typedef struct {
    bool overall_success;
    int passed_stages;
    int total_stages;
    double total_elapsed_ms;
    char device_name[128];
    char driver_version[64];
    char loader_path[256];
    char recommended_backend[32];
} AmevaDiagnosticResult;

/**
 * @brief Runs 12-stage hardware diagnostic (V0 to V11).
 */
AMEVA_API int ameva_run_diagnostic(bool verbose, AmevaDiagnosticResult* out_result);

/**
 * @brief Performs 0ms quick capability probe via state.json cache.
 */
AMEVA_API bool ameva_quick_probe(char* out_device_name, int max_len);

/**
 * @brief Checks if Vulkan acceleration is available and supported.
 */
AMEVA_API bool ameva_is_vulkan_available(void);

/**
 * @brief Validates if tensor dimensions satisfy Mali 128-byte strict alignment.
 */
AMEVA_API bool ameva_is_tensor_aligned(uint32_t ne01, uint32_t ne11);

/**
 * @brief Returns the version string of ameva-vulkan-runtime.
 */
AMEVA_API const char* ameva_get_version(void);

#ifdef __cplusplus
}
#endif
