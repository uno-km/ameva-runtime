#pragma once
#include <cstdint>
#include <string>

namespace ameva {
namespace quirks {

/**
 * @brief Qualcomm Adreno 830 / 700 / 600 Hardware Quirks & Shader Patch Engine
 */
class AdrenoQuirks {
public:
    static constexpr uint32_t QUALCOMM_VENDOR_ID = 0x5143;

    /**
     * @brief Checks if the physical device belongs to Qualcomm Adreno series.
     */
    static bool IsAdreno(uint32_t vendor_id, const std::string& device_name);

    /**
     * @brief Determines whether Subgroup Size Control structure should be passed.
     * Adreno drivers fail if requiredSubgroupSize is equal to native subgroupSize.
     */
    static bool ShouldAttachSubgroupControl(
        bool subgroup_size_control_supported,
        uint32_t required_subgroup_size,
        uint32_t native_subgroup_size,
        uint32_t min_subgroup_size,
        uint32_t max_subgroup_size
    );

    /**
     * @brief Clamps compute workgroup invocations to prevent Adreno driver crashes.
     */
    static uint32_t ClampWorkGroupSize(uint32_t target_size, uint32_t max_invocations);

    /**
     * @brief Returns safe reduction mode (Forces SHMEM when FP16 subgroup arithmetic is buggy).
     */
    static bool IsSafeForSubgroupArithmetic(uint32_t vendor_id, const std::string& device_name);
};

} // namespace quirks
} // namespace ameva
