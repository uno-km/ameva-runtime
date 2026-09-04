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

    /**
     * @brief Determines whether to enforce the Medium (_m) MatMul compute pipeline.
     * On ARM Mali GPUs (Valhall, e.g. Mali-G68) with subgroup_size < 32 (typically 16),
     * the Small (_s) unaligned quantized GEMM shader encounters integer truncation:
     * loadstride_b = gl_WorkGroupSize.x * LOAD_VEC_B / BK = 16 * 1 / 32 = 0,
     * producing an infinite GPU loop: `for (uint l = 0; l < BN; l += 0)`.
     * Enforcing the Medium (_m) pipeline with workgroup_size = 128 guarantees loadstride_b = 4 > 0,
     * completely eliminating GPU TDR resets and VK_ERROR_DEVICE_LOST.
     * @param subgroup_size Vulkan subgroup size (e.g. 16 on Mali)
     * @param m Tensor M dimension
     * @param n Tensor N dimension
     * @return true to enforce Medium kernel, false to permit Small kernel.
     */
    static bool ShouldEnforceMediumMatMulKernel(uint32_t subgroup_size, uint32_t m, uint32_t n);

    /**
     * @brief Returns safe workgroup denomination for quantized matmul on Mali.
     */
    static uint32_t GetSafeWorkgroupDenom(uint32_t subgroup_size);
};

} // namespace quirks
} // namespace ameva

