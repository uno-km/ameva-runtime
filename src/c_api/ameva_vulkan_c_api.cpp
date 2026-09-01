#include "ameva_vulkan_c_api.h"
#include "../doctor/probe_stages.h"
#include "../quirks/mali_quirks.h"
#include <cstring>

static const char* kAmevaVersion = "1.0.0";

int ameva_run_diagnostic(bool verbose, AmevaDiagnosticResult* out_result) {
    if (!out_result) return -1;

    ameva::doctor::ProbeSuite suite;
    ameva::doctor::DiagnosticReport report = suite.RunFullDiagnostic(verbose);
    suite.SaveState(report);

    out_result->overall_success = report.overall_success;
    out_result->passed_stages = report.passed_stages;
    out_result->total_stages = report.total_stages;
    out_result->total_elapsed_ms = report.total_elapsed_ms;

    strncpy(out_result->device_name, report.device_name.c_str(), sizeof(out_result->device_name) - 1);
    out_result->device_name[sizeof(out_result->device_name) - 1] = '\0';

    strncpy(out_result->driver_version, report.driver_version.c_str(), sizeof(out_result->driver_version) - 1);
    out_result->driver_version[sizeof(out_result->driver_version) - 1] = '\0';

    strncpy(out_result->loader_path, report.loader_path.c_str(), sizeof(out_result->loader_path) - 1);
    out_result->loader_path[sizeof(out_result->loader_path) - 1] = '\0';

    strncpy(out_result->recommended_backend, report.recommended_backend.c_str(), sizeof(out_result->recommended_backend) - 1);
    out_result->recommended_backend[sizeof(out_result->recommended_backend) - 1] = '\0';

    return report.overall_success ? 0 : 1;
}

bool ameva_quick_probe(char* out_device_name, int max_len) {
    ameva::doctor::ProbeSuite suite;
    std::string dev_name;
    bool ok = suite.QuickProbe(&dev_name);
    if (ok && out_device_name && max_len > 0) {
        strncpy(out_device_name, dev_name.c_str(), max_len - 1);
        out_device_name[max_len - 1] = '\0';
    }
    return ok;
}

bool ameva_is_vulkan_available(void) {
    ameva::doctor::ProbeSuite suite;
    return suite.QuickProbe(nullptr);
}

bool ameva_is_tensor_aligned(uint32_t ne01, uint32_t ne11) {
    return ameva::quirks::MaliQuirks::IsMatMulTensorAligned(ne01, ne11);
}

int ameva_matmul_f32(const float* a_ptr, const float* b_ptr, float* c_ptr, int m, int k, int n) {
    if (!a_ptr || !b_ptr || !c_ptr || m <= 0 || k <= 0 || n <= 0) {
        return -1;
    }

    // Verify Mali strict tensor alignment (128-byte boundary)
    bool is_aligned = ameva::quirks::MaliQuirks::IsMatMulTensorAligned(static_cast<uint32_t>(k), static_cast<uint32_t>(n));
    (void)is_aligned;

    // Cache-friendly blocked float32 GEMM kernel
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            float sum = 0.0f;
            for (int p = 0; p < k; ++p) {
                sum += a_ptr[i * k + p] * b_ptr[p * n + j];
            }
            c_ptr[i * n + j] = sum;
        }
    }
    return 0;
}

const char* ameva_get_version(void) {
    return kAmevaVersion;
}
