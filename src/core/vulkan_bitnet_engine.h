#pragma once

#include "vulkan_loader.h"
#include "../quirks/gpu_quirks.h"
#include <string>
#include <vector>
#include <stdexcept>
#include <cstdint>

#if defined(__ANDROID__) || defined(__linux__)
#include <vulkan/vulkan.h>
#else
// Standalone minimal Vulkan definitions for compilation without SDK
typedef uint32_t VkResult;
typedef void* VkInstance;
typedef void* VkPhysicalDevice;
typedef void* VkDevice;
typedef void* VkQueue;
typedef void* VkBuffer;
typedef void* VkDeviceMemory;
typedef void* VkDescriptorSetLayout;
typedef void* VkDescriptorPool;
typedef void* VkDescriptorSet;
typedef void* VkPipelineLayout;
typedef void* VkPipeline;
typedef void* VkCommandPool;
typedef void* VkCommandBuffer;
typedef void* VkFence;
typedef void* VkShaderModule;
#define VK_SUCCESS 0
#endif

namespace ameva {
namespace core {

/**
 * @brief Fail-Fast Vulkan Execution Exception.
 * Contains exact failure location, failing API step, and VkResult error code.
 * Silent fallback is strictly forbidden by the Anti-Deception protocol.
 */
class AmevaVulkanExecutionError : public std::runtime_error {
public:
    AmevaVulkanExecutionError(const std::string& step, int vk_result, const char* file, int line);

    const std::string& GetStep() const { return step_; }
    int GetVkResult() const { return vk_result_; }
    const char* GetFile() const { return file_; }
    int GetLine() const { return line_; }

private:
    std::string step_;
    int vk_result_;
    const char* file_;
    int line_;
};

const char* VkResultToString(int res);

struct BitNetPushConstants {
    uint32_t in_dim;             // Multiple of 128
    uint32_t out_dim;            // M rows
    float dequant_scale;         // Activation dequant scale factor
    uint32_t weight_offset_bytes;// Byte offset into raw_weights buffer (multiple of 4)
    uint32_t output_offset_words;// Element offset into output_vec buffer (in floats)
};

struct LayerGpuOffsets {
    uint32_t offset_attn_norm{0xFFFFFFFF};
    uint32_t offset_wq{0};
    uint32_t offset_wk{0};
    uint32_t offset_wv{0};
    uint32_t offset_attn_sub_norm{0xFFFFFFFF};
    uint32_t offset_wo{0};
    uint32_t offset_ffn_norm{0xFFFFFFFF};
    uint32_t offset_w_gate{0};
    uint32_t offset_w_up{0};
    uint32_t offset_ffn_sub_norm{0xFFFFFFFF};
    uint32_t offset_w_down{0};
};

/**
 * @brief Pure Native Vulkan Compute GPU Engine for BitNet 1.58-bit (i2_s) GEMV.
 * Hardware Targets: ARM Mali (Bifrost/Valhall) & Qualcomm Adreno (6xx/7xx/8xx).
 */
class VulkanBitNetEngine {
public:
    VulkanBitNetEngine();
    ~VulkanBitNetEngine();

    VulkanBitNetEngine(const VulkanBitNetEngine&) = delete;
    VulkanBitNetEngine& operator=(const VulkanBitNetEngine&) = delete;

    /**
     * @brief Initializes Vulkan logical device, compute queue, and shader pipeline.
     * @param explicit_driver_path Optional override driver path.
     * @throws AmevaVulkanExecutionError on any failure (Zero Silent Fallback).
     */
    void Initialize(const std::string& explicit_driver_path = "");

    /**
     * @brief Preallocates GPU unified memory buffers for weights, activations, and output.
     * @param dim_m Output dimension (rows)
     * @param dim_k Input dimension (columns, multiple of 128)
     */
    void AllocateBuffers(uint32_t dim_m, uint32_t dim_k);

    /**
     * @brief Llama.cpp-style permanent model weight residency: pre-allocates unified GPU memory for entire model.
     * @param total_weight_bytes Total size of all offloaded weight matrices
     * @param max_m Maximum row dimension (e.g. n_ffn)
     * @param max_k Maximum column dimension (e.g. n_ffn)
     */
    void AllocateModelBuffers(size_t total_weight_bytes, uint32_t max_m, uint32_t max_k, uint32_t n_layers = 30, uint32_t n_ctx = 2048);

    /**
     * @brief Uploads a weight tensor into the persistent GPU buffer at a specific offset.
     */
    void UploadWeightAtOffset(const void* raw_weights, size_t byte_size, uint32_t offset_bytes);

    /**
     * @brief Executes zero-copy GPU GEMV using pre-loaded resident weights.
     * ZERO weights are copied during this call. Only the activation vector x is uploaded.
     */
    void DispatchPreloadedGemv(uint32_t weight_offset_bytes, const float* x, float* y, uint32_t dim_m, uint32_t dim_k, float scale = 1.0f);

    /**
     * @brief Batched Attention GEMV dispatch: Wq, Wk, Wv dispatched into a SINGLE command buffer.
     * Submits to compute queue ONCE and waits on fence ONCE (eliminates 2 driver submissions per layer).
     */
    void DispatchBatchQKV(uint32_t offset_wq, uint32_t offset_wk, uint32_t offset_wv,
                          const float* x, float* q, float* k, float* v,
                          uint32_t q_dim, uint32_t kv_dim, uint32_t dim_k, float scale = 1.0f);

    /**
     * @brief Batched SwiGLU GEMV dispatch: W_gate, W_up dispatched into a SINGLE command buffer.
     * Submits to compute queue ONCE and waits on fence ONCE (eliminates 1 driver submission per layer).
     */
    void DispatchBatchGateUp(uint32_t offset_gate, uint32_t offset_up,
                             const float* x, float* gate, float* up,
                             uint32_t dim_m, uint32_t dim_k, float scale = 1.0f);

    /**
     * @brief Batched SwiGLU GEMV + In-place SiLU fusion dispatch:
     * Dispatches W_gate GEMV, W_up GEMV, memory barrier, and in-place SwiGLU compute kernel
     * in a SINGLE command buffer.
     * Output vector contains SiLU(gate) * up of size dim_m.
     */
    void DispatchBatchGateUpSwiGLU(uint32_t offset_gate, uint32_t offset_up,
                                   const float* x, float* swiglu_out,
                                   uint32_t dim_m, uint32_t dim_k, float scale = 1.0f);

    /**
     * @brief Fully fused FFN dispatch:
     * Records W_gate GEMV, W_up GEMV, barrier, SwiGLU SiLU kernel, barrier,
     * (optional) RMSNorm kernel, barrier, DMA buffer copy to activation_buffer,
     * barrier, and W_down GEMV in a SINGLE command buffer.
     * Entire FFN executes on GPU with ZERO CPU roundtrips or intermediate copies.
     */
    void DispatchFullFFN(uint32_t offset_gate, uint32_t offset_up, uint32_t offset_down,
                         uint32_t offset_sub_norm_gamma,
                         const float* x, float* out,
                         uint32_t dim_m, uint32_t dim_k,
                         float eps = 1e-5f, float scale = 1.0f);

    /**
     * @brief Combined Attention Block dispatch:
     * Dispatches Wq, Wk, Wv GEMV, in-place RoPE on Q & K, GPU KV Cache store,
     * decode Multi-Head Attention, optional attention sub-norm, DMA copy,
     * and Wo output projection GEMV in a SINGLE command buffer.
     */
    void DispatchAttentionBlock(uint32_t offset_wq, uint32_t offset_wk, uint32_t offset_wv, uint32_t offset_wo,
                                uint32_t offset_attn_sub_norm,
                                uint32_t layer_idx, uint32_t current_pos,
                                const float* x, float* wo_out,
                                uint32_t q_dim, uint32_t kv_dim, uint32_t dim_k,
                                float rope_theta = 10000.0f,
                                float eps = 1e-5f, float scale = 1.0f);

    /**
     * @brief Full-Pipeline On-Chain Token Execution:
     * Records all N offloaded transformer layers end-to-end into a SINGLE command buffer:
     * [attn_norm -> QKV GEMV -> RoPE -> KV store -> Attention Decode -> Sub-Norm -> Wo GEMV -> Residual Add
     *  -> ffn_norm -> Gate/Up GEMV -> SwiGLU -> Sub-Norm -> W_down GEMV -> Residual Add] x N layers!
     * Executes in 1 single driver submission and 1 fence wait per generated token!
     */
    void DispatchFullTokenChain(uint32_t current_pos, const float* x_in, float* x_out,
                                const LayerGpuOffsets* layers, uint32_t n_layers,
                                uint32_t q_dim, uint32_t kv_dim, uint32_t dim_k, uint32_t dim_m,
                                float rope_theta = 10000.0f, float eps = 1e-5f, float scale = 1.0f);

    /**
     * @brief Uploads packed BitNet 1.58-bit (i2_s) weights to GPU buffer.
     */
    void UploadWeights(const void* raw_weights, size_t byte_size);

    /**
     * @brief Executes GPU GEMV compute dispatch on Mali/Adreno GPU.
     */
    void DispatchGemv(const float* x, float* y, uint32_t dim_m, uint32_t dim_k, float scale = 1.0f);

    /**
     * @brief High-level single-call GEMV: uploads weights and activations, dispatches GPU compute shader, and retrieves output vector.
     */
    void ComputeGemvDynamic(const void* raw_weights, const float* x, float* y, uint32_t dim_m, uint32_t dim_k, float scale = 1.0f);

    /**
     * @brief Releases all Vulkan device resources and pipelines.
     */
    void Shutdown();

    /**
     * @brief Allocates dedicated GPU unified memory for the FP16 LM Head weight matrix.
     */
    void AllocateLMHeadBuffer(size_t byte_size, uint32_t n_vocab, uint32_t n_embd);

    /**
     * @brief Uploads LM Head FP16 weights into GPU buffer.
     */
    void UploadLMHeadWeights(const void* raw_weights, size_t byte_size);

    /**
     * @brief Dispatches the FP16 LM Head GEMV compute shader.
     */
    void DispatchLMHead(const float* x_norm, float* logits, uint32_t n_vocab, uint32_t n_embd);

    bool HasLMHead() const { return lm_head_buffer_ != nullptr; }

    bool IsInitialized() const { return initialized_; }
    const quirks::GpuDeviceInfo& GetDeviceInfo() const { return device_info_; }

private:
    uint32_t FindMemoryType(uint32_t type_filter, uint32_t properties);
    void CreateBuffer(size_t size, uint32_t usage, uint32_t properties, VkBuffer& buffer, VkDeviceMemory& memory);

    VulkanLoader loader_;
    bool initialized_{false};
    quirks::GpuDeviceInfo device_info_;

    VkInstance instance_{nullptr};
    VkPhysicalDevice physical_device_{nullptr};
    VkDevice device_{nullptr};
    VkQueue compute_queue_{nullptr};
    uint32_t compute_queue_family_index_{0};

    VkDescriptorSetLayout descriptor_set_layout_{nullptr};
    VkDescriptorPool descriptor_pool_{nullptr};
    VkDescriptorSet descriptor_set_{nullptr};
    VkPipelineLayout pipeline_layout_{nullptr};
    VkPipeline compute_pipeline_{nullptr};
    VkShaderModule shader_module_{nullptr};
    VkPipeline swiglu_pipeline_{nullptr};
    VkShaderModule swiglu_shader_module_{nullptr};
    VkPipeline rmsnorm_pipeline_{nullptr};
    VkShaderModule rmsnorm_shader_module_{nullptr};
    VkPipeline rope_pipeline_{nullptr};
    VkShaderModule rope_shader_module_{nullptr};
    VkPipeline attention_decode_pipeline_{nullptr};
    VkShaderModule attention_decode_shader_module_{nullptr};
    VkPipeline rmsnorm_norm_pipeline_{nullptr};
    VkShaderModule rmsnorm_norm_shader_module_{nullptr};
    VkPipeline residual_add_pipeline_{nullptr};
    VkShaderModule residual_add_shader_module_{nullptr};
    VkPipeline gemv_f16_pipeline_{nullptr};
    VkShaderModule gemv_f16_shader_module_{nullptr};

    VkCommandPool command_pool_{nullptr};
    VkCommandBuffer command_buffer_{nullptr};
    VkFence execution_fence_{nullptr};

    // GPU Tensor Buffers
    VkBuffer weight_buffer_{nullptr};
    VkDeviceMemory weight_memory_{nullptr};
    void* weight_mapped_ptr_{nullptr};
    size_t weight_buffer_size_{0};

    VkBuffer activation_buffer_{nullptr};
    VkDeviceMemory activation_memory_{nullptr};
    void* activation_mapped_ptr_{nullptr};
    size_t activation_buffer_size_{0};

    VkBuffer output_buffer_{nullptr};
    VkDeviceMemory output_memory_{nullptr};
    void* output_mapped_ptr_{nullptr};
    size_t output_buffer_size_{0};

    VkBuffer kv_cache_buffer_{nullptr};
    VkDeviceMemory kv_cache_memory_{nullptr};
    void* kv_cache_mapped_ptr_{nullptr};
    size_t kv_cache_buffer_size_{0};

    VkBuffer residual_buffer_{nullptr};
    VkDeviceMemory residual_memory_{nullptr};
    void* residual_mapped_ptr_{nullptr};
    size_t residual_buffer_size_{0};

    // LM Head Dedicated GPU Buffers
    VkBuffer lm_head_buffer_{nullptr};
    VkDeviceMemory lm_head_memory_{nullptr};
    void* lm_head_mapped_ptr_{nullptr};
    size_t lm_head_buffer_size_{0};
    VkDescriptorSet lm_head_descriptor_set_{nullptr};
    uint32_t lm_head_vocab_{0};
    uint32_t lm_head_embd_{0};

    uint32_t allocated_dim_m_{0};
    uint32_t allocated_dim_k_{0};
    uint32_t n_ctx_{2048};
    uint32_t n_layers_{30};
};

} // namespace core
} // namespace ameva
