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

} // namespace quirks
} // namespace ameva
