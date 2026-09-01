#pragma once
#include <cstdint>
#include <string>

namespace ameva {
namespace quirks {

/**
 * @brief ARM Mali-G78 / Mali-G68 / Valhall Memory Alignment & OOB Defense Engine
 */
class MaliQuirks {
public:
    static constexpr uint32_t ARM_VENDOR_ID = 0x13b5;
    static constexpr uint32_t REQUIRED_STRICT_ALIGNMENT = 128;

    /**
     * @brief Checks if device is ARM Mali series.
     */
    static bool IsMali(uint32_t vendor_id, const std::string& device_name);

    /**
     * @brief Validates if tensor dimensions satisfy strict alignment to prevent Node 1055 buffer overflows.
     * @param ne01 Dimension 1 of tensor 0 (e.g. sequence length or batch)
     * @param ne11 Dimension 1 of tensor 1
     * @return true if 100% safe to dispatch aligned MatMul shader, false to route to unaligned kernel.
     */
    static bool IsMatMulTensorAligned(uint32_t ne01, uint32_t ne11);

    /**
     * @brief Computes aligned row padding size.
     */
    static uint32_t AlignSize(uint32_t size, uint32_t align_bytes);
};

} // namespace quirks
} // namespace ameva
