#pragma once
#include <string>
#include <vector>
#include <cstdint>
#include <chrono>

namespace ameva {
namespace doctor {

enum class StageResult {
    PASS,
    FAIL,
    SKIPPED
};

struct StageReport {
    int stage_id;                   // 0 to 11
    std::string stage_name;         // e.g. "V0: Loader Open"
    StageResult result;
    double elapsed_ms;              // Precise execution time in ms
    std::string detail_message;     // Diagnostic explanation
    uint64_t allocated_bytes;       // Memory footprint if applicable
};

struct DiagnosticReport {
    bool overall_success;
    std::string device_name;
    std::string driver_version;
    std::string loader_path;
    uint32_t vendor_id;
    int passed_stages;
    int total_stages;
    double total_elapsed_ms;
    std::vector<StageReport> stages;
    std::string recommended_backend; // "vulkan" or "cpu_neon"
};

/**
 * @brief 12-Stage Diagnostic & Probing Suite (V0 to V11)
 */
class ProbeSuite {
public:
    ProbeSuite();
    ~ProbeSuite();

    /**
     * @brief Executes the complete 12-stage validation hierarchy.
     * @param verbose If true, prints live dynamic telemetry logs to stdout.
     * @return DiagnosticReport complete scorecard.
     */
    DiagnosticReport RunFullDiagnostic(bool verbose = true);

    /**
     * @brief Quick probe using cached state.json fingerprint if valid.
     */
    bool QuickProbe(std::string* out_device_name = nullptr);

    /**
     * @brief Saves the diagnostic scorecard to state.json.
     */
    bool SaveState(const DiagnosticReport& report, const std::string& state_file_path = "");

    /**
     * @brief Loads state from state.json.
     */
    bool LoadState(DiagnosticReport* out_report, const std::string& state_file_path = "");
};

} // namespace doctor
} // namespace ameva
