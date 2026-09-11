#include "mali_quirks.h"

namespace ameva {
namespace quirks {

bool MaliQuirks::IsMali(uint32_t vendor_id, const std::string& device_name) {
    if (vendor_id == ARM_VENDOR_ID) {
        return true;
    }
    return device_name.find("Mali") != std::string::npos;
}

bool MaliQuirks::IsMatMulTensorAligned(uint32_t ne01, uint32_t ne11) {
    // S21 Node 1055 & A35 Mali OOB Prevention:
    // Requires exact modulo 128 boundary match
    return (ne01 % REQUIRED_STRICT_ALIGNMENT == 0) && (ne11 % REQUIRED_STRICT_ALIGNMENT == 0);
}

uint32_t MaliQuirks::AlignSize(uint32_t size, uint32_t align_bytes) {
    if (align_bytes == 0) return size;
    return ((size + align_bytes - 1) / align_bytes) * align_bytes;
}

bool MaliQuirks::ShouldEnforceMediumMatMulKernel(uint32_t subgroup_size, uint32_t m, uint32_t n) {
    // When subgroup size is < 32 (e.g. 16 on ARM Mali Valhall/Bifrost):
    // In unaligned Small quantized GEMM shaders (BK=32, LOAD_VEC_B=1):
    // loadstride_b = gl_WorkGroupSize.x * LOAD_VEC_B / BK = 16 * 1 / 32 = 0 (integer division).
    // This produces an infinite GPU loop: for (uint l = 0; l < BN; l += loadstride_b).
    // Routing to Medium pipeline (workgroup=128, loadstride_b=4 > 0) prevents GPU hang.
    if (subgroup_size < 32) {
        return true;
    }
    (void)m;
    (void)n;
    return false;
}

uint32_t MaliQuirks::GetSafeWorkgroupDenom(uint32_t subgroup_size) {
    // Return 64 or 128 to ensure workgroup dimension is divisible and positive stride
    return (subgroup_size < 32) ? 64 : 32;
}

} // namespace quirks
} // namespace ameva

