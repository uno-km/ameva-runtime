#include "adreno_quirks.h"
#include <algorithm>

namespace ameva {
namespace quirks {

bool AdrenoQuirks::IsAdreno(uint32_t vendor_id, const std::string& device_name) {
    if (vendor_id == QUALCOMM_VENDOR_ID) {
        return true;
    }
    return device_name.find("Adreno") != std::string::npos;
}

bool AdrenoQuirks::ShouldAttachSubgroupControl(
    bool subgroup_size_control_supported,
    uint32_t required_subgroup_size,
    uint32_t native_subgroup_size,
    uint32_t min_subgroup_size,
    uint32_t max_subgroup_size
) {
    if (!subgroup_size_control_supported || required_subgroup_size == 0) {
        return false;
    }
    // Bug avoidance: If required size equals native hardware size, DO NOT pass pNext struct
    if (required_subgroup_size == native_subgroup_size) {
        return false;
    }
    return (min_subgroup_size <= required_subgroup_size && required_subgroup_size <= max_subgroup_size);
}

uint32_t AdrenoQuirks::ClampWorkGroupSize(uint32_t target_size, uint32_t max_invocations) {
    if (target_size * 6 > max_invocations) {
        return max_invocations / 6;
    }
    return target_size;
}

bool AdrenoQuirks::IsSafeForSubgroupArithmetic(uint32_t vendor_id, const std::string& device_name) {
    // Adreno driver fp16 subgroup arithmetic requires fallback to shared memory reduction
    return !IsAdreno(vendor_id, device_name);
}

} // namespace quirks
} // namespace ameva
