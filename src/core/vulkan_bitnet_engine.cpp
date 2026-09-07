#include "vulkan_bitnet_engine.h"
#include "../shaders/bitnet_gemv_i2_s_spv.h"
#include "../shaders/swiglu_silu_spv.h"
#include "../shaders/rmsnorm_spv.h"
#include "../shaders/rope_spv.h"
#include "../shaders/attention_decode_spv.h"
#include "../shaders/rmsnorm_norm_spv.h"
#include "../shaders/residual_add_spv.h"
#include "../shaders/bitnet_gemv_f16_spv.h"

#include <iostream>
#include <sstream>
#include <cstring>
#include <cstdlib>

#define VK_CHECK_ENGINE(call, step_name) \
    do { \
        VkResult _res = (call); \
        if (_res != VK_SUCCESS) { \
            std::cerr << "\n[ZERO-SILENT-FALLBACK FATAL ERROR]\n" \
                      << "  Operation: " << (step_name) << "\n" \
                      << "  VkResult:  " << _res << " (" << ameva::core::VkResultToString(_res) << ")\n" \
                      << "  Location:  " << __FILE__ << ":" << __LINE__ << " (" << __func__ << ")\n" \
                      << "  Action:    Failing immediately without silent fallback.\n" << std::endl; \
            throw ameva::core::AmevaVulkanExecutionError(step_name, (int)_res, __FILE__, __LINE__); \
        } \
    } while (0)

namespace ameva {
namespace core {

const char* VkResultToString(int res) {
    switch (res) {
        case 0: return "VK_SUCCESS";
        case 1: return "VK_NOT_READY";
        case 2: return "VK_TIMEOUT";
        case 3: return "VK_EVENT_SET";
        case 4: return "VK_EVENT_RESET";
        case 5: return "VK_INCOMPLETE";
        case -1: return "VK_ERROR_OUT_OF_HOST_MEMORY";
        case -2: return "VK_ERROR_OUT_OF_DEVICE_MEMORY";
        case -3: return "VK_ERROR_INITIALIZATION_FAILED";
        case -4: return "VK_ERROR_DEVICE_LOST";
        case -5: return "VK_ERROR_MEMORY_MAP_FAILED";
        case -6: return "VK_ERROR_LAYER_NOT_PRESENT";
        case -7: return "VK_ERROR_EXTENSION_NOT_PRESENT";
        case -8: return "VK_ERROR_FEATURE_NOT_PRESENT";
        case -9: return "VK_ERROR_INCOMPATIBLE_DRIVER";
        case -10: return "VK_ERROR_TOO_MANY_OBJECTS";
        case -11: return "VK_ERROR_FORMAT_NOT_SUPPORTED";
        case -12: return "VK_ERROR_FRAGMENTED_POOL";
        case -13: return "VK_ERROR_UNKNOWN";
        default: return "VK_ERROR_UNRECOGNIZED";
    }
}

AmevaVulkanExecutionError::AmevaVulkanExecutionError(const std::string& step, int vk_result, const char* file, int line)
    : std::runtime_error("AmevaVulkanExecutionError in " + step + ": " + VkResultToString(vk_result)),
      step_(step), vk_result_(vk_result), file_(file), line_(line) {}

VulkanBitNetEngine::VulkanBitNetEngine() = default;

VulkanBitNetEngine::~VulkanBitNetEngine() {
    Shutdown();
}

void VulkanBitNetEngine::Initialize(const std::string& explicit_driver_path) {
    if (initialized_) return;

    if (!loader_.Load(explicit_driver_path)) {
        std::cerr << "[ZERO-SILENT-FALLBACK FATAL] Vulkan driver loading failed from candidates." << std::endl;
        throw AmevaVulkanExecutionError("VulkanLoader::Load", -3, __FILE__, __LINE__);
    }

    // 1. Create Instance
    VkApplicationInfo appInfo{};
    appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    appInfo.pApplicationName = "AmevaBitNetVulkanEngine";
    appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
    appInfo.pEngineName = "AmevaRuntime";
    appInfo.engineVersion = VK_MAKE_VERSION(1, 0, 0);
    appInfo.apiVersion = VK_API_VERSION_1_1;

    VkInstanceCreateInfo createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    createInfo.pApplicationInfo = &appInfo;

    VK_CHECK_ENGINE(vkCreateInstance(&createInfo, nullptr, &instance_), "vkCreateInstance");

    // 2. Select Physical Device
    uint32_t deviceCount = 0;
    VK_CHECK_ENGINE(vkEnumeratePhysicalDevices(instance_, &deviceCount, nullptr), "vkEnumeratePhysicalDevices (count)");
    if (deviceCount == 0) {
        throw AmevaVulkanExecutionError("vkEnumeratePhysicalDevices (zero devices found)", -8, __FILE__, __LINE__);
    }

    std::vector<VkPhysicalDevice> devices(deviceCount);
    VK_CHECK_ENGINE(vkEnumeratePhysicalDevices(instance_, &deviceCount, devices.data()), "vkEnumeratePhysicalDevices (list)");
    physical_device_ = devices[0];

    VkPhysicalDeviceProperties props;
    vkGetPhysicalDeviceProperties(physical_device_, &props);
    device_info_ = quirks::GpuQuirksEngine::AnalyzeDevice(props.vendorID, props.deviceID, props.deviceName, props.apiVersion, props.driverVersion);

    std::cout << "[VulkanBitNetEngine] Selected GPU: " << device_info_.device_name
              << " (Vendor: 0x" << std::hex << device_info_.vendor_id << std::dec << ")"
              << " [Mali: " << (device_info_.is_mali ? "YES" : "NO")
              << ", Adreno: " << (device_info_.is_adreno ? "YES" : "NO") << "]" << std::endl;

    // 3. Find Compute Queue Family
    uint32_t queueFamilyCount = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(physical_device_, &queueFamilyCount, nullptr);
    std::vector<VkQueueFamilyProperties> queueFamilies(queueFamilyCount);
    vkGetPhysicalDeviceQueueFamilyProperties(physical_device_, &queueFamilyCount, queueFamilies.data());

    bool found_compute = false;
    for (uint32_t i = 0; i < queueFamilyCount; ++i) {
        if (queueFamilies[i].queueFlags & VK_QUEUE_COMPUTE_BIT) {
            compute_queue_family_index_ = i;
            found_compute = true;
            break;
        }
    }
    if (!found_compute) {
        throw AmevaVulkanExecutionError("vkGetPhysicalDeviceQueueFamilyProperties (no compute queue)", -8, __FILE__, __LINE__);
    }

    // 4. Create Logical Device & Queue
    float queuePriority = 1.0f;
    VkDeviceQueueCreateInfo queueCreateInfo{};
    queueCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    queueCreateInfo.queueFamilyIndex = compute_queue_family_index_;
    queueCreateInfo.queueCount = 1;
    queueCreateInfo.pQueuePriorities = &queuePriority;

    VkDeviceCreateInfo deviceCreateInfo{};
    deviceCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    deviceCreateInfo.queueCreateInfoCount = 1;
    deviceCreateInfo.pQueueCreateInfos = &queueCreateInfo;

    VK_CHECK_ENGINE(vkCreateDevice(physical_device_, &deviceCreateInfo, nullptr, &device_), "vkCreateDevice");
    vkGetDeviceQueue(device_, compute_queue_family_index_, 0, &compute_queue_);

    // 5. Create SPIR-V Shader Module from Embedded Header
    VkShaderModuleCreateInfo smci{};
    smci.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    smci.codeSize = shaders::kBitnetGemvI2SSpvByteSize;
    smci.pCode = shaders::kBitnetGemvI2SSpv;
    VK_CHECK_ENGINE(vkCreateShaderModule(device_, &smci, nullptr, &shader_module_), "vkCreateShaderModule (i2_s GEMV)");

    // 6. Create Descriptor Set Layout (5 Storage Buffers: Weights, Activations, Output, KV Cache, Residual)
    VkDescriptorSetLayoutBinding bindings[5]{};
    for (int i = 0; i < 5; ++i) {
        bindings[i].binding = i;
        bindings[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        bindings[i].descriptorCount = 1;
        bindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    }

    VkDescriptorSetLayoutCreateInfo layoutInfo{};
    layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    layoutInfo.bindingCount = 5;
    layoutInfo.pBindings = bindings;
    VK_CHECK_ENGINE(vkCreateDescriptorSetLayout(device_, &layoutInfo, nullptr, &descriptor_set_layout_), "vkCreateDescriptorSetLayout");

    // 7. Create Pipeline Layout with Push Constants
    VkPushConstantRange pushConstantRange{};
    pushConstantRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    pushConstantRange.offset = 0;
    pushConstantRange.size = sizeof(BitNetPushConstants);

    VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
    pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pipelineLayoutInfo.setLayoutCount = 1;
    pipelineLayoutInfo.pSetLayouts = &descriptor_set_layout_;
    pipelineLayoutInfo.pushConstantRangeCount = 1;
    pipelineLayoutInfo.pPushConstantRanges = &pushConstantRange;
    VK_CHECK_ENGINE(vkCreatePipelineLayout(device_, &pipelineLayoutInfo, nullptr, &pipeline_layout_), "vkCreatePipelineLayout");

    // 8. Create Compute Pipeline
    VkComputePipelineCreateInfo computePipelineInfo{};
    computePipelineInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    computePipelineInfo.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    computePipelineInfo.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    computePipelineInfo.stage.module = shader_module_;
    computePipelineInfo.stage.pName = "main";
    computePipelineInfo.layout = pipeline_layout_;
    VK_CHECK_ENGINE(vkCreateComputePipelines(device_, nullptr, 1, &computePipelineInfo, nullptr, &compute_pipeline_), "vkCreateComputePipelines");

    // 8b. Create SwiGLU Compute Pipeline
    VkShaderModuleCreateInfo swiglu_smci{};
    swiglu_smci.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    swiglu_smci.codeSize = shaders::kSwigluSiluSpvByteSize;
    swiglu_smci.pCode = shaders::kSwigluSiluSpv;
    VK_CHECK_ENGINE(vkCreateShaderModule(device_, &swiglu_smci, nullptr, &swiglu_shader_module_), "vkCreateShaderModule (SwiGLU SiLU)");

    VkComputePipelineCreateInfo swigluPipelineInfo{};
    swigluPipelineInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    swigluPipelineInfo.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    swigluPipelineInfo.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    swigluPipelineInfo.stage.module = swiglu_shader_module_;
    swigluPipelineInfo.stage.pName = "main";
    swigluPipelineInfo.layout = pipeline_layout_;
    VK_CHECK_ENGINE(vkCreateComputePipelines(device_, nullptr, 1, &swigluPipelineInfo, nullptr, &swiglu_pipeline_), "vkCreateComputePipelines (SwiGLU SiLU)");

    // 8c. Create RMSNorm Compute Pipeline (in-place)
    VkShaderModuleCreateInfo rmsnorm_smci{};
    rmsnorm_smci.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    rmsnorm_smci.codeSize = shaders::kRmsNormSpvByteSize;
    rmsnorm_smci.pCode = shaders::kRmsNormSpv;
    VK_CHECK_ENGINE(vkCreateShaderModule(device_, &rmsnorm_smci, nullptr, &rmsnorm_shader_module_), "vkCreateShaderModule (RMSNorm)");

    VkComputePipelineCreateInfo rmsnormPipelineInfo{};
    rmsnormPipelineInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    rmsnormPipelineInfo.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    rmsnormPipelineInfo.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    rmsnormPipelineInfo.stage.module = rmsnorm_shader_module_;
    rmsnormPipelineInfo.stage.pName = "main";
    rmsnormPipelineInfo.layout = pipeline_layout_;
    VK_CHECK_ENGINE(vkCreateComputePipelines(device_, nullptr, 1, &rmsnormPipelineInfo, nullptr, &rmsnorm_pipeline_), "vkCreateComputePipelines (RMSNorm)");

    // 8d. Create RoPE Compute Pipeline
    VkShaderModuleCreateInfo rope_smci{};
    rope_smci.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    rope_smci.codeSize = shaders::kRopeSpvByteSize;
    rope_smci.pCode = shaders::kRopeSpv;
    VK_CHECK_ENGINE(vkCreateShaderModule(device_, &rope_smci, nullptr, &rope_shader_module_), "vkCreateShaderModule (RoPE)");

    VkComputePipelineCreateInfo ropePipelineInfo{};
    ropePipelineInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    ropePipelineInfo.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    ropePipelineInfo.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    ropePipelineInfo.stage.module = rope_shader_module_;
    ropePipelineInfo.stage.pName = "main";
    ropePipelineInfo.layout = pipeline_layout_;
    VK_CHECK_ENGINE(vkCreateComputePipelines(device_, nullptr, 1, &ropePipelineInfo, nullptr, &rope_pipeline_), "vkCreateComputePipelines (RoPE)");

    // 8e. Create Attention Decode Compute Pipeline
    VkShaderModuleCreateInfo attn_smci{};
    attn_smci.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    attn_smci.codeSize = shaders::kAttentionDecodeSpvByteSize;
    attn_smci.pCode = shaders::kAttentionDecodeSpv;
    VK_CHECK_ENGINE(vkCreateShaderModule(device_, &attn_smci, nullptr, &attention_decode_shader_module_), "vkCreateShaderModule (Attention Decode)");

    VkComputePipelineCreateInfo attnPipelineInfo{};
    attnPipelineInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    attnPipelineInfo.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    attnPipelineInfo.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    attnPipelineInfo.stage.module = attention_decode_shader_module_;
    attnPipelineInfo.stage.pName = "main";
    attnPipelineInfo.layout = pipeline_layout_;
    VK_CHECK_ENGINE(vkCreateComputePipelines(device_, nullptr, 1, &attnPipelineInfo, nullptr, &attention_decode_pipeline_), "vkCreateComputePipelines (Attention Decode)");

    // 8f. Create RMSNorm Norm Compute Pipeline (residual_buffer -> activation_buffer)
    VkShaderModuleCreateInfo norm_smci{};
    norm_smci.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    norm_smci.codeSize = shaders::kRmsnormNormSpvByteSize;
    norm_smci.pCode = shaders::kRmsnormNormSpv;
    VK_CHECK_ENGINE(vkCreateShaderModule(device_, &norm_smci, nullptr, &rmsnorm_norm_shader_module_), "vkCreateShaderModule (RMSNorm Norm)");

    VkComputePipelineCreateInfo normPipelineInfo{};
    normPipelineInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    normPipelineInfo.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    normPipelineInfo.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    normPipelineInfo.stage.module = rmsnorm_norm_shader_module_;
    normPipelineInfo.stage.pName = "main";
    normPipelineInfo.layout = pipeline_layout_;
    VK_CHECK_ENGINE(vkCreateComputePipelines(device_, nullptr, 1, &normPipelineInfo, nullptr, &rmsnorm_norm_pipeline_), "vkCreateComputePipelines (RMSNorm Norm)");

    // 8g. Create Residual Add Compute Pipeline (residual_buffer += output_buffer)
    VkShaderModuleCreateInfo add_smci{};
    add_smci.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    add_smci.codeSize = shaders::kResidualAddSpvByteSize;
    add_smci.pCode = shaders::kResidualAddSpv;
    VK_CHECK_ENGINE(vkCreateShaderModule(device_, &add_smci, nullptr, &residual_add_shader_module_), "vkCreateShaderModule (Residual Add)");

    VkComputePipelineCreateInfo addPipelineInfo{};
    addPipelineInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    addPipelineInfo.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    addPipelineInfo.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    addPipelineInfo.stage.module = residual_add_shader_module_;
    addPipelineInfo.stage.pName = "main";
    addPipelineInfo.layout = pipeline_layout_;
    VK_CHECK_ENGINE(vkCreateComputePipelines(device_, nullptr, 1, &addPipelineInfo, nullptr, &residual_add_pipeline_), "vkCreateComputePipelines (Residual Add)");

    // 8h. Create FP16 GEMV Compute Pipeline (LM Head)
    VkShaderModuleCreateInfo gemv_f16_smci{};
    gemv_f16_smci.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    gemv_f16_smci.codeSize = shaders::kBitnetGemvF16SpvByteSize;
    gemv_f16_smci.pCode = shaders::kBitnetGemvF16Spv;
    VK_CHECK_ENGINE(vkCreateShaderModule(device_, &gemv_f16_smci, nullptr, &gemv_f16_shader_module_), "vkCreateShaderModule (FP16 GEMV)");

    VkComputePipelineCreateInfo gemvF16PipelineInfo{};
    gemvF16PipelineInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    gemvF16PipelineInfo.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    gemvF16PipelineInfo.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    gemvF16PipelineInfo.stage.module = gemv_f16_shader_module_;
    gemvF16PipelineInfo.stage.pName = "main";
    gemvF16PipelineInfo.layout = pipeline_layout_;
    VK_CHECK_ENGINE(vkCreateComputePipelines(device_, nullptr, 1, &gemvF16PipelineInfo, nullptr, &gemv_f16_pipeline_), "vkCreateComputePipelines (FP16 GEMV)");

    // 9. Create Command Pool & Allocate Command Buffer
    VkCommandPoolCreateInfo poolInfo{};
    poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    poolInfo.queueFamilyIndex = compute_queue_family_index_;
    poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    VK_CHECK_ENGINE(vkCreateCommandPool(device_, &poolInfo, nullptr, &command_pool_), "vkCreateCommandPool");

    VkCommandBufferAllocateInfo cmdAllocInfo{};
    cmdAllocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    cmdAllocInfo.commandPool = command_pool_;
    cmdAllocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cmdAllocInfo.commandBufferCount = 1;
    VK_CHECK_ENGINE(vkAllocateCommandBuffers(device_, &cmdAllocInfo, &command_buffer_), "vkAllocateCommandBuffers");

    // 10. Create Fence
    VkFenceCreateInfo fenceInfo{};
    fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    fenceInfo.flags = 0;
    VK_CHECK_ENGINE(vkCreateFence(device_, &fenceInfo, nullptr, &execution_fence_), "vkCreateFence");

    // 11. Create Descriptor Pool & Allocate Descriptor Set (5 Storage Buffers)
    VkDescriptorPoolSize poolSize{};
    poolSize.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    poolSize.descriptorCount = 20;

    VkDescriptorPoolCreateInfo descPoolInfo{};
    descPoolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    descPoolInfo.poolSizeCount = 1;
    descPoolInfo.pPoolSizes = &poolSize;
    descPoolInfo.maxSets = 4;
    VK_CHECK_ENGINE(vkCreateDescriptorPool(device_, &descPoolInfo, nullptr, &descriptor_pool_), "vkCreateDescriptorPool");

    VkDescriptorSetAllocateInfo descAllocInfo{};
    descAllocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    descAllocInfo.descriptorPool = descriptor_pool_;
    descAllocInfo.descriptorSetCount = 1;
    descAllocInfo.pSetLayouts = &descriptor_set_layout_;
    VK_CHECK_ENGINE(vkAllocateDescriptorSets(device_, &descAllocInfo, &descriptor_set_), "vkAllocateDescriptorSets");

    initialized_ = true;
    std::cout << "[VulkanBitNetEngine] Initialized successfully with Zero Silent Fallback." << std::endl;
}

uint32_t VulkanBitNetEngine::FindMemoryType(uint32_t type_filter, uint32_t properties) {
    VkPhysicalDeviceMemoryProperties memProps;
    vkGetPhysicalDeviceMemoryProperties(physical_device_, &memProps);
    for (uint32_t i = 0; i < memProps.memoryTypeCount; ++i) {
        if ((type_filter & (1 << i)) && (memProps.memoryTypes[i].propertyFlags & properties) == properties) {
            return i;
        }
    }
    throw AmevaVulkanExecutionError("FindMemoryType (no matching memory type)", -8, __FILE__, __LINE__);
}

void VulkanBitNetEngine::CreateBuffer(size_t size, uint32_t usage, uint32_t properties, VkBuffer& buffer, VkDeviceMemory& memory) {
    VkBufferCreateInfo bufferInfo{};
    bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufferInfo.size = size;
    bufferInfo.usage = usage;
    bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    VK_CHECK_ENGINE(vkCreateBuffer(device_, &bufferInfo, nullptr, &buffer), "vkCreateBuffer");

    VkMemoryRequirements memReqs;
    vkGetBufferMemoryRequirements(device_, buffer, &memReqs);

    VkMemoryAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocInfo.allocationSize = memReqs.size;
    allocInfo.memoryTypeIndex = FindMemoryType(memReqs.memoryTypeBits, properties);
    VK_CHECK_ENGINE(vkAllocateMemory(device_, &allocInfo, nullptr, &memory), "vkAllocateMemory");
    VK_CHECK_ENGINE(vkBindBufferMemory(device_, buffer, memory, 0), "vkBindBufferMemory");
}

void VulkanBitNetEngine::AllocateBuffers(uint32_t dim_m, uint32_t dim_k) {
    if (!initialized_) {
        throw AmevaVulkanExecutionError("AllocateBuffers called before Initialize()", -3, __FILE__, __LINE__);
    }

    if (!quirks::GpuQuirksEngine::ValidateTensorAlignment(dim_k, dim_m)) {
        throw AmevaVulkanExecutionError("AllocateBuffers: dim_k must be multiple of 128", -8, __FILE__, __LINE__);
    }

    allocated_dim_m_ = dim_m;
    allocated_dim_k_ = dim_k;

    // Weight buffer: 32 bytes per 128 weights
    size_t num_blocks = dim_k / 128;
    weight_buffer_size_ = quirks::GpuQuirksEngine::AlignSize(dim_m * num_blocks * 32, device_info_.required_alignment);
    activation_buffer_size_ = quirks::GpuQuirksEngine::AlignSize(dim_k * sizeof(float), device_info_.required_alignment);
    output_buffer_size_ = quirks::GpuQuirksEngine::AlignSize(dim_m * sizeof(float), device_info_.required_alignment);

    uint32_t mem_flags = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;

    CreateBuffer(weight_buffer_size_, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, mem_flags, weight_buffer_, weight_memory_);
    VK_CHECK_ENGINE(vkMapMemory(device_, weight_memory_, 0, weight_buffer_size_, 0, &weight_mapped_ptr_), "vkMapMemory (weights)");

    CreateBuffer(activation_buffer_size_, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, mem_flags, activation_buffer_, activation_memory_);
    VK_CHECK_ENGINE(vkMapMemory(device_, activation_memory_, 0, activation_buffer_size_, 0, &activation_mapped_ptr_), "vkMapMemory (activations)");

    CreateBuffer(output_buffer_size_, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, mem_flags, output_buffer_, output_memory_);
    VK_CHECK_ENGINE(vkMapMemory(device_, output_memory_, 0, output_buffer_size_, 0, &output_mapped_ptr_), "vkMapMemory (output)");

    // Update Descriptor Set to point to our buffers
    VkDescriptorBufferInfo bufferInfos[3]{};
    bufferInfos[0].buffer = weight_buffer_;
    bufferInfos[0].offset = 0;
    bufferInfos[0].range = weight_buffer_size_;

    bufferInfos[1].buffer = activation_buffer_;
    bufferInfos[1].offset = 0;
    bufferInfos[1].range = activation_buffer_size_;

    bufferInfos[2].buffer = output_buffer_;
    bufferInfos[2].offset = 0;
    bufferInfos[2].range = output_buffer_size_;

    VkWriteDescriptorSet descriptorWrites[3]{};
    for (int i = 0; i < 3; ++i) {
        descriptorWrites[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        descriptorWrites[i].dstSet = descriptor_set_;
        descriptorWrites[i].dstBinding = i;
        descriptorWrites[i].dstArrayElement = 0;
        descriptorWrites[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        descriptorWrites[i].descriptorCount = 1;
        descriptorWrites[i].pBufferInfo = &bufferInfos[i];
    }
    vkUpdateDescriptorSets(device_, 3, descriptorWrites, 0, nullptr);
}

void VulkanBitNetEngine::AllocateModelBuffers(size_t total_weight_bytes, uint32_t max_m, uint32_t max_k, uint32_t n_layers, uint32_t n_ctx) {
    if (!initialized_) {
        throw AmevaVulkanExecutionError("AllocateModelBuffers called before Initialize()", -3, __FILE__, __LINE__);
    }

    if (weight_buffer_ || activation_buffer_ || output_buffer_ || kv_cache_buffer_ || residual_buffer_) {
        vkDeviceWaitIdle(device_);
        if (weight_mapped_ptr_) { vkUnmapMemory(device_, weight_memory_); weight_mapped_ptr_ = nullptr; }
        if (weight_buffer_) { vkDestroyBuffer(device_, weight_buffer_, nullptr); weight_buffer_ = nullptr; }
        if (weight_memory_) { vkFreeMemory(device_, weight_memory_, nullptr); weight_memory_ = nullptr; }

        if (activation_mapped_ptr_) { vkUnmapMemory(device_, activation_memory_); activation_mapped_ptr_ = nullptr; }
        if (activation_buffer_) { vkDestroyBuffer(device_, activation_buffer_, nullptr); activation_buffer_ = nullptr; }
        if (activation_memory_) { vkFreeMemory(device_, activation_memory_, nullptr); activation_memory_ = nullptr; }

        if (output_mapped_ptr_) { vkUnmapMemory(device_, output_memory_); output_mapped_ptr_ = nullptr; }
        if (output_buffer_) { vkDestroyBuffer(device_, output_buffer_, nullptr); output_buffer_ = nullptr; }
        if (output_memory_) { vkFreeMemory(device_, output_memory_, nullptr); output_memory_ = nullptr; }

        if (kv_cache_mapped_ptr_) { vkUnmapMemory(device_, kv_cache_memory_); kv_cache_mapped_ptr_ = nullptr; }
        if (kv_cache_buffer_) { vkDestroyBuffer(device_, kv_cache_buffer_, nullptr); kv_cache_buffer_ = nullptr; }
        if (kv_cache_memory_) { vkFreeMemory(device_, kv_cache_memory_, nullptr); kv_cache_memory_ = nullptr; }

        if (residual_mapped_ptr_) { vkUnmapMemory(device_, residual_memory_); residual_mapped_ptr_ = nullptr; }
        if (residual_buffer_) { vkDestroyBuffer(device_, residual_buffer_, nullptr); residual_buffer_ = nullptr; }
        if (residual_memory_) { vkFreeMemory(device_, residual_memory_, nullptr); residual_memory_ = nullptr; }
    }

    allocated_dim_m_ = max_m;
    allocated_dim_k_ = max_k;
    n_layers_ = n_layers;
    n_ctx_ = n_ctx;

    weight_buffer_size_ = quirks::GpuQuirksEngine::AlignSize((uint32_t)total_weight_bytes, device_info_.required_alignment);
    activation_buffer_size_ = quirks::GpuQuirksEngine::AlignSize(max_k * sizeof(float), device_info_.required_alignment);
    // Allocate 3x max_m so output buffer can hold Wq + Wk + Wv simultaneously
    output_buffer_size_ = quirks::GpuQuirksEngine::AlignSize(3 * max_m * sizeof(float), device_info_.required_alignment);
    // n_layers * n_ctx * 5 kv_heads * 128 head_dim * 4 bytes * 2 (K and V)
    size_t kv_cache_bytes = (size_t)n_layers * n_ctx * 5 * 128 * sizeof(float) * 2;
    kv_cache_buffer_size_ = quirks::GpuQuirksEngine::AlignSize((uint32_t)kv_cache_bytes, device_info_.required_alignment);
    residual_buffer_size_ = quirks::GpuQuirksEngine::AlignSize(max_k * sizeof(float), device_info_.required_alignment);

    uint32_t mem_flags = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    uint32_t buf_usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;

    CreateBuffer(weight_buffer_size_, buf_usage, mem_flags, weight_buffer_, weight_memory_);
    VK_CHECK_ENGINE(vkMapMemory(device_, weight_memory_, 0, weight_buffer_size_, 0, &weight_mapped_ptr_), "vkMapMemory (persistent model weights)");

    CreateBuffer(activation_buffer_size_, buf_usage, mem_flags, activation_buffer_, activation_memory_);
    VK_CHECK_ENGINE(vkMapMemory(device_, activation_memory_, 0, activation_buffer_size_, 0, &activation_mapped_ptr_), "vkMapMemory (activations)");

    CreateBuffer(output_buffer_size_, buf_usage, mem_flags, output_buffer_, output_memory_);
    VK_CHECK_ENGINE(vkMapMemory(device_, output_memory_, 0, output_buffer_size_, 0, &output_mapped_ptr_), "vkMapMemory (output)");

    CreateBuffer(kv_cache_buffer_size_, buf_usage, mem_flags, kv_cache_buffer_, kv_cache_memory_);
    VK_CHECK_ENGINE(vkMapMemory(device_, kv_cache_memory_, 0, kv_cache_buffer_size_, 0, &kv_cache_mapped_ptr_), "vkMapMemory (kv_cache)");

    CreateBuffer(residual_buffer_size_, buf_usage, mem_flags, residual_buffer_, residual_memory_);
    VK_CHECK_ENGINE(vkMapMemory(device_, residual_memory_, 0, residual_buffer_size_, 0, &residual_mapped_ptr_), "vkMapMemory (residual)");

    // Update Descriptor Set to point to our 5 buffers
    VkDescriptorBufferInfo bufferInfos[5]{};
    bufferInfos[0].buffer = weight_buffer_;
    bufferInfos[0].offset = 0;
    bufferInfos[0].range = weight_buffer_size_;

    bufferInfos[1].buffer = activation_buffer_;
    bufferInfos[1].offset = 0;
    bufferInfos[1].range = activation_buffer_size_;

    bufferInfos[2].buffer = output_buffer_;
    bufferInfos[2].offset = 0;
    bufferInfos[2].range = output_buffer_size_;

    bufferInfos[3].buffer = kv_cache_buffer_;
    bufferInfos[3].offset = 0;
    bufferInfos[3].range = kv_cache_buffer_size_;

    bufferInfos[4].buffer = residual_buffer_;
    bufferInfos[4].offset = 0;
    bufferInfos[4].range = residual_buffer_size_;

    VkWriteDescriptorSet descriptorWrites[5]{};
    for (int i = 0; i < 5; ++i) {
        descriptorWrites[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        descriptorWrites[i].dstSet = descriptor_set_;
        descriptorWrites[i].dstBinding = i;
        descriptorWrites[i].dstArrayElement = 0;
        descriptorWrites[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        descriptorWrites[i].descriptorCount = 1;
        descriptorWrites[i].pBufferInfo = &bufferInfos[i];
    }
    vkUpdateDescriptorSets(device_, 5, descriptorWrites, 0, nullptr);
}

void VulkanBitNetEngine::UploadWeightAtOffset(const void* raw_weights, size_t byte_size, uint32_t offset_bytes) {
    if (!weight_mapped_ptr_) {
        throw AmevaVulkanExecutionError("UploadWeightAtOffset: buffer not allocated or mapped", -5, __FILE__, __LINE__);
    }
    if ((size_t)offset_bytes + byte_size > weight_buffer_size_) {
        throw AmevaVulkanExecutionError("UploadWeightAtOffset: offset + byte_size exceeds allocated weight buffer", -1, __FILE__, __LINE__);
    }
    std::memcpy(static_cast<char*>(weight_mapped_ptr_) + offset_bytes, raw_weights, byte_size);
}

void VulkanBitNetEngine::DispatchPreloadedGemv(uint32_t weight_offset_bytes, const float* x, float* y, uint32_t dim_m, uint32_t dim_k, float scale) {
    if (!initialized_ || !activation_mapped_ptr_ || !output_mapped_ptr_ || !weight_mapped_ptr_) {
        throw AmevaVulkanExecutionError("DispatchPreloadedGemv: engine or buffers not initialized", -3, __FILE__, __LINE__);
    }
    if (dim_k % 128 != 0) {
        throw AmevaVulkanExecutionError("DispatchPreloadedGemv: dim_k must be multiple of 128", -8, __FILE__, __LINE__);
    }
    if (dim_k * sizeof(float) > activation_buffer_size_) {
        throw AmevaVulkanExecutionError("DispatchPreloadedGemv: activation size exceeds buffer", -1, __FILE__, __LINE__);
    }
    if (dim_m * sizeof(float) > output_buffer_size_) {
        throw AmevaVulkanExecutionError("DispatchPreloadedGemv: output size exceeds buffer", -1, __FILE__, __LINE__);
    }

    // 1. Copy activation vector to GPU coherent memory (ZERO weights copied!)
    std::memcpy(activation_mapped_ptr_, x, dim_k * sizeof(float));

    // 2. Record Command Buffer
    VkCommandBufferBeginInfo beginInfo{};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    VK_CHECK_ENGINE(vkBeginCommandBuffer(command_buffer_, &beginInfo), "vkBeginCommandBuffer");

    vkCmdBindPipeline(command_buffer_, VK_PIPELINE_BIND_POINT_COMPUTE, compute_pipeline_);
    vkCmdBindDescriptorSets(command_buffer_, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_layout_, 0, 1, &descriptor_set_, 0, nullptr);

    BitNetPushConstants pushConstants{};
    pushConstants.in_dim = dim_k;
    pushConstants.out_dim = dim_m;
    pushConstants.dequant_scale = scale;
    pushConstants.weight_offset_bytes = weight_offset_bytes;
    pushConstants.output_offset_words = 0;
    vkCmdPushConstants(command_buffer_, pipeline_layout_, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(BitNetPushConstants), &pushConstants);

    uint32_t group_count_x = (dim_m + 3) / 4;
    vkCmdDispatch(command_buffer_, group_count_x, 1, 1);

    VK_CHECK_ENGINE(vkEndCommandBuffer(command_buffer_), "vkEndCommandBuffer");

    // 3. Submit Queue & Synchronize
    VK_CHECK_ENGINE(vkResetFences(device_, 1, &execution_fence_), "vkResetFences");

    VkSubmitInfo submitInfo{};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &command_buffer_;

    VK_CHECK_ENGINE(vkQueueSubmit(compute_queue_, 1, &submitInfo, execution_fence_), "vkQueueSubmit");

    VkResult waitRes = vkWaitForFences(device_, 1, &execution_fence_, VK_TRUE, 5000000000ULL);
    if (waitRes != VK_SUCCESS) {
        throw AmevaVulkanExecutionError("vkWaitForFences timeout/hang in DispatchPreloadedGemv", waitRes, __FILE__, __LINE__);
    }

    // 4. Retrieve output
    std::memcpy(y, output_mapped_ptr_, dim_m * sizeof(float));
}

void VulkanBitNetEngine::DispatchBatchQKV(uint32_t offset_wq, uint32_t offset_wk, uint32_t offset_wv,
                                          const float* x, float* q, float* k, float* v,
                                          uint32_t q_dim, uint32_t kv_dim, uint32_t dim_k, float scale) {
    if (!initialized_ || !activation_mapped_ptr_ || !output_mapped_ptr_ || !weight_mapped_ptr_) {
        throw AmevaVulkanExecutionError("DispatchBatchQKV: engine or buffers not initialized", -3, __FILE__, __LINE__);
    }
    if (dim_k % 128 != 0) {
        throw AmevaVulkanExecutionError("DispatchBatchQKV: dim_k must be multiple of 128", -8, __FILE__, __LINE__);
    }
    size_t total_out_elements = (size_t)q_dim + kv_dim + kv_dim;
    if (total_out_elements * sizeof(float) > output_buffer_size_) {
        throw AmevaVulkanExecutionError("DispatchBatchQKV: total output size exceeds buffer", -1, __FILE__, __LINE__);
    }

    // 1. Copy input vector ONCE
    std::memcpy(activation_mapped_ptr_, x, dim_k * sizeof(float));

    // 2. Record ALL 3 dispatches into a SINGLE Command Buffer
    VkCommandBufferBeginInfo beginInfo{};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    VK_CHECK_ENGINE(vkBeginCommandBuffer(command_buffer_, &beginInfo), "vkBeginCommandBuffer");

    vkCmdBindPipeline(command_buffer_, VK_PIPELINE_BIND_POINT_COMPUTE, compute_pipeline_);
    vkCmdBindDescriptorSets(command_buffer_, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_layout_, 0, 1, &descriptor_set_, 0, nullptr);

    // 2a. Wq
    BitNetPushConstants pushConstants{};
    pushConstants.in_dim = dim_k;
    pushConstants.out_dim = q_dim;
    pushConstants.dequant_scale = scale;
    pushConstants.weight_offset_bytes = offset_wq;
    pushConstants.output_offset_words = 0;
    vkCmdPushConstants(command_buffer_, pipeline_layout_, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(BitNetPushConstants), &pushConstants);
    vkCmdDispatch(command_buffer_, (q_dim + 3) / 4, 1, 1);

    // 2b. Wk
    pushConstants.out_dim = kv_dim;
    pushConstants.weight_offset_bytes = offset_wk;
    pushConstants.output_offset_words = q_dim;
    vkCmdPushConstants(command_buffer_, pipeline_layout_, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(BitNetPushConstants), &pushConstants);
    vkCmdDispatch(command_buffer_, (kv_dim + 3) / 4, 1, 1);

    // 2c. Wv
    pushConstants.out_dim = kv_dim;
    pushConstants.weight_offset_bytes = offset_wv;
    pushConstants.output_offset_words = q_dim + kv_dim;
    vkCmdPushConstants(command_buffer_, pipeline_layout_, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(BitNetPushConstants), &pushConstants);
    vkCmdDispatch(command_buffer_, (kv_dim + 3) / 4, 1, 1);

    VK_CHECK_ENGINE(vkEndCommandBuffer(command_buffer_), "vkEndCommandBuffer");

    // 3. Submit Queue & Synchronize ONCE for all 3!
    VK_CHECK_ENGINE(vkResetFences(device_, 1, &execution_fence_), "vkResetFences");

    VkSubmitInfo submitInfo{};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &command_buffer_;

    VK_CHECK_ENGINE(vkQueueSubmit(compute_queue_, 1, &submitInfo, execution_fence_), "vkQueueSubmit");

    VkResult waitRes = vkWaitForFences(device_, 1, &execution_fence_, VK_TRUE, 5000000000ULL);
    if (waitRes != VK_SUCCESS) {
        throw AmevaVulkanExecutionError("vkWaitForFences timeout/hang in DispatchBatchQKV", waitRes, __FILE__, __LINE__);
    }

    // 4. Retrieve results
    const float* out_base = static_cast<const float*>(output_mapped_ptr_);
    std::memcpy(q, out_base, q_dim * sizeof(float));
    std::memcpy(k, out_base + q_dim, kv_dim * sizeof(float));
    std::memcpy(v, out_base + q_dim + kv_dim, kv_dim * sizeof(float));
}

void VulkanBitNetEngine::DispatchBatchGateUp(uint32_t offset_gate, uint32_t offset_up,
                                             const float* x, float* gate, float* up,
                                             uint32_t dim_m, uint32_t dim_k, float scale) {
    if (!initialized_ || !activation_mapped_ptr_ || !output_mapped_ptr_ || !weight_mapped_ptr_) {
        throw AmevaVulkanExecutionError("DispatchBatchGateUp: engine or buffers not initialized", -3, __FILE__, __LINE__);
    }
    if (dim_k % 128 != 0) {
        throw AmevaVulkanExecutionError("DispatchBatchGateUp: dim_k must be multiple of 128", -8, __FILE__, __LINE__);
    }
    size_t total_out_elements = (size_t)dim_m + dim_m;
    if (total_out_elements * sizeof(float) > output_buffer_size_) {
        throw AmevaVulkanExecutionError("DispatchBatchGateUp: total output size exceeds buffer", -1, __FILE__, __LINE__);
    }

    // 1. Copy input vector ONCE
    std::memcpy(activation_mapped_ptr_, x, dim_k * sizeof(float));

    // 2. Record BOTH dispatches into a SINGLE Command Buffer
    VkCommandBufferBeginInfo beginInfo{};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    VK_CHECK_ENGINE(vkBeginCommandBuffer(command_buffer_, &beginInfo), "vkBeginCommandBuffer");

    vkCmdBindPipeline(command_buffer_, VK_PIPELINE_BIND_POINT_COMPUTE, compute_pipeline_);
    vkCmdBindDescriptorSets(command_buffer_, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_layout_, 0, 1, &descriptor_set_, 0, nullptr);

    // 2a. W_gate
    BitNetPushConstants pushConstants{};
    pushConstants.in_dim = dim_k;
    pushConstants.out_dim = dim_m;
    pushConstants.dequant_scale = scale;
    pushConstants.weight_offset_bytes = offset_gate;
    pushConstants.output_offset_words = 0;
    vkCmdPushConstants(command_buffer_, pipeline_layout_, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(BitNetPushConstants), &pushConstants);
    vkCmdDispatch(command_buffer_, (dim_m + 3) / 4, 1, 1);

    // 2b. W_up
    pushConstants.out_dim = dim_m;
    pushConstants.weight_offset_bytes = offset_up;
    pushConstants.output_offset_words = dim_m;
    vkCmdPushConstants(command_buffer_, pipeline_layout_, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(BitNetPushConstants), &pushConstants);
    vkCmdDispatch(command_buffer_, (dim_m + 3) / 4, 1, 1);

    VK_CHECK_ENGINE(vkEndCommandBuffer(command_buffer_), "vkEndCommandBuffer");

    // 3. Submit Queue & Synchronize ONCE
    VK_CHECK_ENGINE(vkResetFences(device_, 1, &execution_fence_), "vkResetFences");

    VkSubmitInfo submitInfo{};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &command_buffer_;

    VK_CHECK_ENGINE(vkQueueSubmit(compute_queue_, 1, &submitInfo, execution_fence_), "vkQueueSubmit");

    VkResult waitRes = vkWaitForFences(device_, 1, &execution_fence_, VK_TRUE, 5000000000ULL);
    if (waitRes != VK_SUCCESS) {
        throw AmevaVulkanExecutionError("vkWaitForFences timeout/hang in DispatchBatchGateUp", waitRes, __FILE__, __LINE__);
    }

    // 4. Retrieve results
    const float* out_base = static_cast<const float*>(output_mapped_ptr_);
    std::memcpy(gate, out_base, dim_m * sizeof(float));
    std::memcpy(up, out_base + dim_m, dim_m * sizeof(float));
}

void VulkanBitNetEngine::DispatchBatchGateUpSwiGLU(uint32_t offset_gate, uint32_t offset_up,
                                                   const float* x, float* swiglu_out,
                                                   uint32_t dim_m, uint32_t dim_k, float scale) {
    if (!initialized_ || !activation_mapped_ptr_ || !output_mapped_ptr_ || !weight_mapped_ptr_) {
        throw AmevaVulkanExecutionError("DispatchBatchGateUpSwiGLU: engine or buffers not initialized", -3, __FILE__, __LINE__);
    }
    if (dim_k % 128 != 0) {
        throw AmevaVulkanExecutionError("DispatchBatchGateUpSwiGLU: dim_k must be multiple of 128", -8, __FILE__, __LINE__);
    }
    size_t total_out_elements = (size_t)dim_m + dim_m;
    if (total_out_elements * sizeof(float) > output_buffer_size_) {
        throw AmevaVulkanExecutionError("DispatchBatchGateUpSwiGLU: total output size exceeds buffer", -1, __FILE__, __LINE__);
    }

    // 1. Copy input vector ONCE
    std::memcpy(activation_mapped_ptr_, x, dim_k * sizeof(float));

    // 2. Record W_gate, W_up, Barrier, and SwiGLU kernel into a SINGLE Command Buffer
    VkCommandBufferBeginInfo beginInfo{};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    VK_CHECK_ENGINE(vkBeginCommandBuffer(command_buffer_, &beginInfo), "vkBeginCommandBuffer");

    vkCmdBindPipeline(command_buffer_, VK_PIPELINE_BIND_POINT_COMPUTE, compute_pipeline_);
    vkCmdBindDescriptorSets(command_buffer_, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_layout_, 0, 1, &descriptor_set_, 0, nullptr);

    // 2a. W_gate GEMV -> writes to output_buffer_[0 .. dim_m-1]
    BitNetPushConstants pushConstants{};
    pushConstants.in_dim = dim_k;
    pushConstants.out_dim = dim_m;
    pushConstants.dequant_scale = scale;
    pushConstants.weight_offset_bytes = offset_gate;
    pushConstants.output_offset_words = 0;
    vkCmdPushConstants(command_buffer_, pipeline_layout_, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(BitNetPushConstants), &pushConstants);
    vkCmdDispatch(command_buffer_, (dim_m + 3) / 4, 1, 1);

    // 2b. W_up GEMV -> writes to output_buffer_[dim_m .. 2*dim_m-1]
    pushConstants.out_dim = dim_m;
    pushConstants.weight_offset_bytes = offset_up;
    pushConstants.output_offset_words = dim_m;
    vkCmdPushConstants(command_buffer_, pipeline_layout_, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(BitNetPushConstants), &pushConstants);
    vkCmdDispatch(command_buffer_, (dim_m + 3) / 4, 1, 1);

    // 2c. Pipeline barrier: Ensure GEMV writes are visible to SwiGLU compute shader
    VkBufferMemoryBarrier barrier{};
    barrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
    barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.buffer = output_buffer_;
    barrier.offset = 0;
    barrier.size = VK_WHOLE_SIZE;

    vkCmdPipelineBarrier(
        command_buffer_,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        0,
        0, nullptr,
        1, &barrier,
        0, nullptr
    );

    // 2d. Bind SwiGLU Pipeline & Dispatch
    vkCmdBindPipeline(command_buffer_, VK_PIPELINE_BIND_POINT_COMPUTE, swiglu_pipeline_);
    vkCmdBindDescriptorSets(command_buffer_, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_layout_, 0, 1, &descriptor_set_, 0, nullptr);

    pushConstants.out_dim = dim_m; // params.out_dim is used as n_ffn
    vkCmdPushConstants(command_buffer_, pipeline_layout_, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(BitNetPushConstants), &pushConstants);

    uint32_t swiglu_groups = (dim_m + 255) / 256;
    vkCmdDispatch(command_buffer_, swiglu_groups, 1, 1);

    VK_CHECK_ENGINE(vkEndCommandBuffer(command_buffer_), "vkEndCommandBuffer");

    // 3. Submit Queue & Synchronize ONCE
    VK_CHECK_ENGINE(vkResetFences(device_, 1, &execution_fence_), "vkResetFences");

    VkSubmitInfo submitInfo{};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &command_buffer_;

    VK_CHECK_ENGINE(vkQueueSubmit(compute_queue_, 1, &submitInfo, execution_fence_), "vkQueueSubmit");

    VkResult waitRes = vkWaitForFences(device_, 1, &execution_fence_, VK_TRUE, 5000000000ULL);
    if (waitRes != VK_SUCCESS) {
        throw AmevaVulkanExecutionError("vkWaitForFences timeout/hang in DispatchBatchGateUpSwiGLU", waitRes, __FILE__, __LINE__);
    }

    // 4. Retrieve fused SwiGLU result (dim_m floats)
    std::memcpy(swiglu_out, output_mapped_ptr_, dim_m * sizeof(float));
}

void VulkanBitNetEngine::DispatchFullFFN(uint32_t offset_gate, uint32_t offset_up, uint32_t offset_down,
                                         uint32_t offset_sub_norm_gamma,
                                         const float* x, float* out,
                                         uint32_t dim_m, uint32_t dim_k,
                                         float eps, float scale) {
    if (!initialized_ || !activation_mapped_ptr_ || !output_mapped_ptr_ || !weight_mapped_ptr_) {
        throw AmevaVulkanExecutionError("DispatchFullFFN: engine or buffers not initialized", -3, __FILE__, __LINE__);
    }
    if (dim_k % 128 != 0) {
        throw AmevaVulkanExecutionError("DispatchFullFFN: dim_k must be multiple of 128", -8, __FILE__, __LINE__);
    }
    if (dim_m % 128 != 0) {
        throw AmevaVulkanExecutionError("DispatchFullFFN: dim_m must be multiple of 128", -8, __FILE__, __LINE__);
    }
    size_t total_out_elements = (size_t)dim_m + dim_m;
    if (total_out_elements * sizeof(float) > output_buffer_size_) {
        throw AmevaVulkanExecutionError("DispatchFullFFN: total output size exceeds buffer", -1, __FILE__, __LINE__);
    }

    // 1. Copy input activation vector ONCE
    std::memcpy(activation_mapped_ptr_, x, dim_k * sizeof(float));

    // 2. Record ENTIRE FFN into a SINGLE Command Buffer
    VkCommandBufferBeginInfo beginInfo{};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    VK_CHECK_ENGINE(vkBeginCommandBuffer(command_buffer_, &beginInfo), "vkBeginCommandBuffer");

    vkCmdBindDescriptorSets(command_buffer_, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_layout_, 0, 1, &descriptor_set_, 0, nullptr);

    // 2a. W_gate GEMV -> writes to output_buffer_[0 .. dim_m - 1]
    vkCmdBindPipeline(command_buffer_, VK_PIPELINE_BIND_POINT_COMPUTE, compute_pipeline_);
    BitNetPushConstants pushConstants{};
    pushConstants.in_dim = dim_k;
    pushConstants.out_dim = dim_m;
    pushConstants.dequant_scale = scale;
    pushConstants.weight_offset_bytes = offset_gate;
    pushConstants.output_offset_words = 0;
    vkCmdPushConstants(command_buffer_, pipeline_layout_, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(BitNetPushConstants), &pushConstants);
    vkCmdDispatch(command_buffer_, (dim_m + 3) / 4, 1, 1);

    // 2b. W_up GEMV -> writes to output_buffer_[dim_m .. 2*dim_m - 1]
    pushConstants.out_dim = dim_m;
    pushConstants.weight_offset_bytes = offset_up;
    pushConstants.output_offset_words = dim_m;
    vkCmdPushConstants(command_buffer_, pipeline_layout_, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(BitNetPushConstants), &pushConstants);
    vkCmdDispatch(command_buffer_, (dim_m + 3) / 4, 1, 1);

    // 2c. Barrier: GEMV writes -> SwiGLU reads/writes on output_buffer_
    VkBufferMemoryBarrier barrier{};
    barrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
    barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.buffer = output_buffer_;
    barrier.offset = 0;
    barrier.size = VK_WHOLE_SIZE;

    vkCmdPipelineBarrier(
        command_buffer_,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        0,
        0, nullptr,
        1, &barrier,
        0, nullptr
    );

    // 2d. Bind SwiGLU Pipeline & Dispatch (in-place on output_buffer_[0 .. dim_m - 1])
    vkCmdBindPipeline(command_buffer_, VK_PIPELINE_BIND_POINT_COMPUTE, swiglu_pipeline_);
    pushConstants.out_dim = dim_m; // params.out_dim is used as n_ffn
    vkCmdPushConstants(command_buffer_, pipeline_layout_, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(BitNetPushConstants), &pushConstants);
    uint32_t swiglu_groups = (dim_m + 255) / 256;
    vkCmdDispatch(command_buffer_, swiglu_groups, 1, 1);

    // 2e. (Optional Sub-LayerNorm)
    if (offset_sub_norm_gamma != 0xFFFFFFFFu) {
        barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
        vkCmdPipelineBarrier(
            command_buffer_,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            0,
            0, nullptr,
            1, &barrier,
            0, nullptr
        );

        vkCmdBindPipeline(command_buffer_, VK_PIPELINE_BIND_POINT_COMPUTE, rmsnorm_pipeline_);
        pushConstants.out_dim = dim_m;
        pushConstants.dequant_scale = eps;
        pushConstants.weight_offset_bytes = offset_sub_norm_gamma;
        pushConstants.output_offset_words = 0;
        vkCmdPushConstants(command_buffer_, pipeline_layout_, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(BitNetPushConstants), &pushConstants);
        vkCmdDispatch(command_buffer_, 1, 1, 1);
    }

    // 2f. Barrier: Compute shader write -> Transfer read on output_buffer_
    barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    barrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
    vkCmdPipelineBarrier(
        command_buffer_,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        VK_PIPELINE_STAGE_TRANSFER_BIT,
        0,
        0, nullptr,
        1, &barrier,
        0, nullptr
    );

    // 2g. Fast GPU DMA copy: output_buffer_[0..dim_m-1] -> activation_buffer_[0..dim_m-1]
    VkBufferCopy copyRegion{};
    copyRegion.srcOffset = 0;
    copyRegion.dstOffset = 0;
    copyRegion.size = dim_m * sizeof(float);
    vkCmdCopyBuffer(command_buffer_, output_buffer_, activation_buffer_, 1, &copyRegion);

    // 2h. Barrier: Transfer write -> Compute shader read on activation_buffer_
    VkBufferMemoryBarrier actBarrier{};
    actBarrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
    actBarrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    actBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    actBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    actBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    actBarrier.buffer = activation_buffer_;
    actBarrier.offset = 0;
    actBarrier.size = VK_WHOLE_SIZE;

    vkCmdPipelineBarrier(
        command_buffer_,
        VK_PIPELINE_STAGE_TRANSFER_BIT,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        0,
        0, nullptr,
        1, &actBarrier,
        0, nullptr
    );

    // 2i. W_down GEMV: input from activation_buffer_ (dim_m), writes to output_buffer_[0 .. dim_k - 1]
    vkCmdBindPipeline(command_buffer_, VK_PIPELINE_BIND_POINT_COMPUTE, compute_pipeline_);
    pushConstants.in_dim = dim_m;
    pushConstants.out_dim = dim_k;
    pushConstants.dequant_scale = 1.0f;
    pushConstants.weight_offset_bytes = offset_down;
    pushConstants.output_offset_words = 0;
    vkCmdPushConstants(command_buffer_, pipeline_layout_, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(BitNetPushConstants), &pushConstants);
    vkCmdDispatch(command_buffer_, (dim_k + 3) / 4, 1, 1);

    VK_CHECK_ENGINE(vkEndCommandBuffer(command_buffer_), "vkEndCommandBuffer");

    // 3. Submit Queue & Synchronize ONCE for the entire FFN!
    VK_CHECK_ENGINE(vkResetFences(device_, 1, &execution_fence_), "vkResetFences");

    VkSubmitInfo submitInfo{};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &command_buffer_;

    VK_CHECK_ENGINE(vkQueueSubmit(compute_queue_, 1, &submitInfo, execution_fence_), "vkQueueSubmit");

    VkResult waitRes = vkWaitForFences(device_, 1, &execution_fence_, VK_TRUE, 5000000000ULL);
    if (waitRes != VK_SUCCESS) {
        throw AmevaVulkanExecutionError("vkWaitForFences timeout/hang in DispatchFullFFN", waitRes, __FILE__, __LINE__);
    }

    // 4. Retrieve final FFN output directly into out (size dim_k)
    std::memcpy(out, output_mapped_ptr_, dim_k * sizeof(float));
}

void VulkanBitNetEngine::DispatchAttentionBlock(
    uint32_t offset_wq, uint32_t offset_wk, uint32_t offset_wv, uint32_t offset_wo,
    uint32_t offset_attn_sub_norm,
    uint32_t layer_idx, uint32_t current_pos,
    const float* x, float* wo_out,
    uint32_t q_dim, uint32_t kv_dim, uint32_t dim_k,
    float rope_theta, float eps, float scale) {

    if (!initialized_ || !activation_mapped_ptr_ || !output_mapped_ptr_ || !weight_mapped_ptr_ || !kv_cache_mapped_ptr_) {
        throw AmevaVulkanExecutionError("DispatchAttentionBlock: engine or buffers not initialized", -3, __FILE__, __LINE__);
    }
    if (dim_k % 128 != 0 || q_dim % 128 != 0 || kv_dim % 128 != 0) {
        throw AmevaVulkanExecutionError("DispatchAttentionBlock: dimensions must be multiple of 128", -8, __FILE__, __LINE__);
    }
    size_t total_out_elements = (size_t)q_dim + kv_dim + kv_dim;
    if (total_out_elements * sizeof(float) > output_buffer_size_) {
        throw AmevaVulkanExecutionError("DispatchAttentionBlock: QKV output size exceeds output buffer", -1, __FILE__, __LINE__);
    }

    uint32_t n_heads = q_dim / 128;
    uint32_t n_kv_heads = kv_dim / 128;
    uint32_t pos_stride = kv_dim; // e.g. 640 floats
    uint32_t layer_stride = n_ctx_ * pos_stride;
    uint32_t layer_base = layer_idx * (2u * layer_stride);

    // 1. Copy input activation vector ONCE
    std::memcpy(activation_mapped_ptr_, x, dim_k * sizeof(float));

    // 2. Record ENTIRE Attention Block into a SINGLE Command Buffer
    VkCommandBufferBeginInfo beginInfo{};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    VK_CHECK_ENGINE(vkBeginCommandBuffer(command_buffer_, &beginInfo), "vkBeginCommandBuffer");

    vkCmdBindDescriptorSets(command_buffer_, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_layout_, 0, 1, &descriptor_set_, 0, nullptr);

    // 2a. W_q GEMV -> writes to output_buffer_[0 .. q_dim - 1]
    vkCmdBindPipeline(command_buffer_, VK_PIPELINE_BIND_POINT_COMPUTE, compute_pipeline_);
    BitNetPushConstants pushConstants{};
    pushConstants.in_dim = dim_k;
    pushConstants.out_dim = q_dim;
    pushConstants.dequant_scale = scale;
    pushConstants.weight_offset_bytes = offset_wq;
    pushConstants.output_offset_words = 0;
    vkCmdPushConstants(command_buffer_, pipeline_layout_, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(BitNetPushConstants), &pushConstants);
    vkCmdDispatch(command_buffer_, (q_dim + 3) / 4, 1, 1);

    // 2b. W_k GEMV -> writes to output_buffer_[q_dim .. q_dim + kv_dim - 1]
    pushConstants.out_dim = kv_dim;
    pushConstants.weight_offset_bytes = offset_wk;
    pushConstants.output_offset_words = q_dim;
    vkCmdPushConstants(command_buffer_, pipeline_layout_, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(BitNetPushConstants), &pushConstants);
    vkCmdDispatch(command_buffer_, (kv_dim + 3) / 4, 1, 1);

    // 2c. W_v GEMV -> writes to output_buffer_[q_dim + kv_dim .. q_dim + 2*kv_dim - 1]
    pushConstants.out_dim = kv_dim;
    pushConstants.weight_offset_bytes = offset_wv;
    pushConstants.output_offset_words = q_dim + kv_dim;
    vkCmdPushConstants(command_buffer_, pipeline_layout_, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(BitNetPushConstants), &pushConstants);
    vkCmdDispatch(command_buffer_, (kv_dim + 3) / 4, 1, 1);

    // 2d. Barrier: GEMV writes -> RoPE reads/writes on output_buffer_
    VkBufferMemoryBarrier outBarrier{};
    outBarrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
    outBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    outBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
    outBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    outBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    outBarrier.buffer = output_buffer_;
    outBarrier.offset = 0;
    outBarrier.size = VK_WHOLE_SIZE;

    vkCmdPipelineBarrier(command_buffer_, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         0, 0, nullptr, 1, &outBarrier, 0, nullptr);

    // 2e. RoPE on Q & K in-place
    vkCmdBindPipeline(command_buffer_, VK_PIPELINE_BIND_POINT_COMPUTE, rope_pipeline_);
    // RoPE for Q
    pushConstants.out_dim = n_heads;
    pushConstants.dequant_scale = rope_theta;
    pushConstants.weight_offset_bytes = current_pos;
    pushConstants.output_offset_words = 0;
    vkCmdPushConstants(command_buffer_, pipeline_layout_, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(BitNetPushConstants), &pushConstants);
    vkCmdDispatch(command_buffer_, n_heads, 1, 1);

    // RoPE for K
    pushConstants.out_dim = n_kv_heads;
    pushConstants.dequant_scale = rope_theta;
    pushConstants.weight_offset_bytes = current_pos;
    pushConstants.output_offset_words = q_dim;
    vkCmdPushConstants(command_buffer_, pipeline_layout_, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(BitNetPushConstants), &pushConstants);
    vkCmdDispatch(command_buffer_, n_kv_heads, 1, 1);

    // 2f. Barrier: RoPE writes -> Transfer read on output_buffer_
    outBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    outBarrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
    vkCmdPipelineBarrier(command_buffer_, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
                         0, 0, nullptr, 1, &outBarrier, 0, nullptr);

    // 2g. Store K & V to kv_cache_buffer_ via GPU DMA
    VkBufferCopy copyRegions[2]{};
    // K copy
    copyRegions[0].srcOffset = (VkDeviceSize)q_dim * sizeof(float);
    copyRegions[0].dstOffset = (VkDeviceSize)(layer_base + current_pos * pos_stride) * sizeof(float);
    copyRegions[0].size = (VkDeviceSize)kv_dim * sizeof(float);
    // V copy
    copyRegions[1].srcOffset = (VkDeviceSize)(q_dim + kv_dim) * sizeof(float);
    copyRegions[1].dstOffset = (VkDeviceSize)(layer_base + layer_stride + current_pos * pos_stride) * sizeof(float);
    copyRegions[1].size = (VkDeviceSize)kv_dim * sizeof(float);

    vkCmdCopyBuffer(command_buffer_, output_buffer_, kv_cache_buffer_, 2, copyRegions);

    // 2h. Barrier: Transfer write on kv_cache_buffer_ -> Compute read, and Transfer read on output_buffer_ -> Compute read/write
    VkBufferMemoryBarrier kvBarrier{};
    kvBarrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
    kvBarrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    kvBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    kvBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    kvBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    kvBarrier.buffer = kv_cache_buffer_;
    kvBarrier.offset = 0;
    kvBarrier.size = VK_WHOLE_SIZE;

    outBarrier.srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
    outBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;

    VkBufferMemoryBarrier midBarriers[2] = { kvBarrier, outBarrier };
    vkCmdPipelineBarrier(command_buffer_, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         0, 0, nullptr, 2, midBarriers, 0, nullptr);

    // 2i. Decode Multi-Head Attention (in-place on output_buffer_[0 .. q_dim - 1])
    vkCmdBindPipeline(command_buffer_, VK_PIPELINE_BIND_POINT_COMPUTE, attention_decode_pipeline_);
    pushConstants.in_dim = n_ctx_;
    pushConstants.out_dim = n_heads;
    pushConstants.dequant_scale = 1.0f / std::sqrt(128.0f);
    pushConstants.weight_offset_bytes = current_pos;
    pushConstants.output_offset_words = layer_idx;
    vkCmdPushConstants(command_buffer_, pipeline_layout_, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(BitNetPushConstants), &pushConstants);
    vkCmdDispatch(command_buffer_, n_heads, 1, 1);

    // 2j. (Optional Attention Sub-LayerNorm)
    if (offset_attn_sub_norm != 0xFFFFFFFFu) {
        outBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        outBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
        vkCmdPipelineBarrier(command_buffer_, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                             0, 0, nullptr, 1, &outBarrier, 0, nullptr);

        vkCmdBindPipeline(command_buffer_, VK_PIPELINE_BIND_POINT_COMPUTE, rmsnorm_pipeline_);
        pushConstants.out_dim = q_dim;
        pushConstants.dequant_scale = eps;
        pushConstants.weight_offset_bytes = offset_attn_sub_norm;
        pushConstants.output_offset_words = 0;
        vkCmdPushConstants(command_buffer_, pipeline_layout_, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(BitNetPushConstants), &pushConstants);
        vkCmdDispatch(command_buffer_, 1, 1, 1);
    }

    // 2k. Barrier: Compute shader write -> Transfer read on output_buffer_
    outBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    outBarrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
    vkCmdPipelineBarrier(command_buffer_, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
                         0, 0, nullptr, 1, &outBarrier, 0, nullptr);

    // 2l. Fast GPU DMA copy: output_buffer_[0 .. q_dim - 1] -> activation_buffer_[0 .. q_dim - 1]
    VkBufferCopy actCopy{};
    actCopy.srcOffset = 0;
    actCopy.dstOffset = 0;
    actCopy.size = (VkDeviceSize)q_dim * sizeof(float);
    vkCmdCopyBuffer(command_buffer_, output_buffer_, activation_buffer_, 1, &actCopy);

    // 2m. Barrier: Transfer write -> Compute shader read on activation_buffer_
    VkBufferMemoryBarrier actBarrier{};
    actBarrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
    actBarrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    actBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    actBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    actBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    actBarrier.buffer = activation_buffer_;
    actBarrier.offset = 0;
    actBarrier.size = VK_WHOLE_SIZE;

    vkCmdPipelineBarrier(command_buffer_, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         0, 0, nullptr, 1, &actBarrier, 0, nullptr);

    // 2n. W_o GEMV: reads from activation_buffer_ (q_dim), writes to output_buffer_[0 .. dim_k - 1]
    vkCmdBindPipeline(command_buffer_, VK_PIPELINE_BIND_POINT_COMPUTE, compute_pipeline_);
    pushConstants.in_dim = q_dim;
    pushConstants.out_dim = dim_k;
    pushConstants.dequant_scale = 1.0f;
    pushConstants.weight_offset_bytes = offset_wo;
    pushConstants.output_offset_words = 0;
    vkCmdPushConstants(command_buffer_, pipeline_layout_, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(BitNetPushConstants), &pushConstants);
    vkCmdDispatch(command_buffer_, (dim_k + 3) / 4, 1, 1);

    VK_CHECK_ENGINE(vkEndCommandBuffer(command_buffer_), "vkEndCommandBuffer");

    // 3. Submit Queue & Synchronize ONCE for the entire Attention Block!
    VK_CHECK_ENGINE(vkResetFences(device_, 1, &execution_fence_), "vkResetFences");

    VkSubmitInfo submitInfo{};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &command_buffer_;

    VK_CHECK_ENGINE(vkQueueSubmit(compute_queue_, 1, &submitInfo, execution_fence_), "vkQueueSubmit");

    VkResult waitRes = vkWaitForFences(device_, 1, &execution_fence_, VK_TRUE, 5000000000ULL);
    if (waitRes != VK_SUCCESS) {
        throw AmevaVulkanExecutionError("vkWaitForFences timeout/hang in DispatchAttentionBlock", waitRes, __FILE__, __LINE__);
    }

    // 4. Retrieve final Wo output directly into wo_out (size dim_k)
    std::memcpy(wo_out, output_mapped_ptr_, dim_k * sizeof(float));
}

void VulkanBitNetEngine::DispatchFullTokenChain(uint32_t current_pos, const float* x_in, float* x_out,
                                                const LayerGpuOffsets* layers, uint32_t n_layers,
                                                uint32_t q_dim, uint32_t kv_dim, uint32_t dim_k, uint32_t dim_m,
                                                float rope_theta, float eps, float scale) {
    if (!initialized_ || !activation_mapped_ptr_ || !output_mapped_ptr_ || !weight_mapped_ptr_ || !kv_cache_mapped_ptr_ || !residual_mapped_ptr_) {
        throw AmevaVulkanExecutionError("DispatchFullTokenChain: engine or buffers not initialized", -3, __FILE__, __LINE__);
    }
    if (layers == nullptr || n_layers == 0) {
        throw AmevaVulkanExecutionError("DispatchFullTokenChain: invalid layers pointer or zero layers", -8, __FILE__, __LINE__);
    }
    if (dim_k % 128 != 0 || q_dim % 128 != 0 || kv_dim % 128 != 0 || dim_m % 128 != 0) {
        throw AmevaVulkanExecutionError("DispatchFullTokenChain: dimensions must be multiple of 128", -8, __FILE__, __LINE__);
    }
    if (dim_k * sizeof(float) > residual_buffer_size_ || dim_k * sizeof(float) > activation_buffer_size_) {
        throw AmevaVulkanExecutionError("DispatchFullTokenChain: dim_k exceeds residual/activation buffer size", -1, __FILE__, __LINE__);
    }
    if (dim_m * sizeof(float) > activation_buffer_size_) {
        throw AmevaVulkanExecutionError("DispatchFullTokenChain: dim_m exceeds activation buffer size", -1, __FILE__, __LINE__);
    }
    if (3 * dim_m * sizeof(float) > output_buffer_size_) {
        throw AmevaVulkanExecutionError("DispatchFullTokenChain: 3*dim_m exceeds output buffer size", -1, __FILE__, __LINE__);
    }

    uint32_t n_heads = q_dim / 128;
    uint32_t n_kv_heads = kv_dim / 128;
    uint32_t pos_stride = kv_dim;
    uint32_t layer_stride = n_ctx_ * pos_stride;

    // 1. Copy initial token embedding vector to residual_buffer_ ONCE
    std::memcpy(residual_mapped_ptr_, x_in, dim_k * sizeof(float));

    // 2. Record ENTIRE Token Generation Chain across ALL offloaded layers into a SINGLE Command Buffer!
    VkCommandBufferBeginInfo beginInfo{};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    VK_CHECK_ENGINE(vkBeginCommandBuffer(command_buffer_, &beginInfo), "vkBeginCommandBuffer");

    vkCmdBindDescriptorSets(command_buffer_, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_layout_, 0, 1, &descriptor_set_, 0, nullptr);

    // Initial Host Write -> Compute Shader Read barrier on residual_buffer_
    VkBufferMemoryBarrier resBarrier{};
    resBarrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
    resBarrier.srcAccessMask = VK_ACCESS_HOST_WRITE_BIT;
    resBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    resBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    resBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    resBarrier.buffer = residual_buffer_;
    resBarrier.offset = 0;
    resBarrier.size = VK_WHOLE_SIZE;
    vkCmdPipelineBarrier(command_buffer_, VK_PIPELINE_STAGE_HOST_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         0, 0, nullptr, 1, &resBarrier, 0, nullptr);

    BitNetPushConstants pushConstants{};

    for (uint32_t l = 0; l < n_layers; ++l) {
        const auto& lay = layers[l];
        uint32_t layer_base = l * (2u * layer_stride);

        // ====================================================================
        // Substep 1: Attention RMSNorm
        // residual_buffer_ -> activation_buffer_ (via lay.offset_attn_norm)
        // ====================================================================
        vkCmdBindPipeline(command_buffer_, VK_PIPELINE_BIND_POINT_COMPUTE, rmsnorm_norm_pipeline_);
        pushConstants.out_dim = dim_k;
        pushConstants.dequant_scale = eps;
        pushConstants.weight_offset_bytes = lay.offset_attn_norm;
        pushConstants.output_offset_words = 0;
        vkCmdPushConstants(command_buffer_, pipeline_layout_, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(BitNetPushConstants), &pushConstants);
        vkCmdDispatch(command_buffer_, 1, 1, 1);

        // Barrier: activation_buffer_ Compute Shader Write -> Compute Shader Read
        VkBufferMemoryBarrier actBarrier{};
        actBarrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
        actBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        actBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        actBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        actBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        actBarrier.buffer = activation_buffer_;
        actBarrier.offset = 0;
        actBarrier.size = VK_WHOLE_SIZE;
        vkCmdPipelineBarrier(command_buffer_, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                             0, 0, nullptr, 1, &actBarrier, 0, nullptr);

        // ====================================================================
        // Substep 2: Q, K, V GEMVs
        // activation_buffer_ -> output_buffer_
        // ====================================================================
        vkCmdBindPipeline(command_buffer_, VK_PIPELINE_BIND_POINT_COMPUTE, compute_pipeline_);
        // Wq GEMV
        pushConstants.in_dim = dim_k;
        pushConstants.out_dim = q_dim;
        pushConstants.dequant_scale = scale;
        pushConstants.weight_offset_bytes = lay.offset_wq;
        pushConstants.output_offset_words = 0;
        vkCmdPushConstants(command_buffer_, pipeline_layout_, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(BitNetPushConstants), &pushConstants);
        vkCmdDispatch(command_buffer_, (q_dim + 3) / 4, 1, 1);

        // Wk GEMV
        pushConstants.out_dim = kv_dim;
        pushConstants.weight_offset_bytes = lay.offset_wk;
        pushConstants.output_offset_words = q_dim;
        vkCmdPushConstants(command_buffer_, pipeline_layout_, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(BitNetPushConstants), &pushConstants);
        vkCmdDispatch(command_buffer_, (kv_dim + 3) / 4, 1, 1);

        // Wv GEMV
        pushConstants.out_dim = kv_dim;
        pushConstants.weight_offset_bytes = lay.offset_wv;
        pushConstants.output_offset_words = q_dim + kv_dim;
        vkCmdPushConstants(command_buffer_, pipeline_layout_, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(BitNetPushConstants), &pushConstants);
        vkCmdDispatch(command_buffer_, (kv_dim + 3) / 4, 1, 1);

        // Barrier: output_buffer_ Compute Shader Write -> Compute Shader Read/Write
        VkBufferMemoryBarrier outBarrier{};
        outBarrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
        outBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        outBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
        outBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        outBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        outBarrier.buffer = output_buffer_;
        outBarrier.offset = 0;
        outBarrier.size = VK_WHOLE_SIZE;
        vkCmdPipelineBarrier(command_buffer_, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                             0, 0, nullptr, 1, &outBarrier, 0, nullptr);

        // ====================================================================
        // Substep 3: In-Place RoPE on Q & K
        // ====================================================================
        vkCmdBindPipeline(command_buffer_, VK_PIPELINE_BIND_POINT_COMPUTE, rope_pipeline_);
        // RoPE for Q
        pushConstants.out_dim = n_heads;
        pushConstants.dequant_scale = rope_theta;
        pushConstants.weight_offset_bytes = current_pos;
        pushConstants.output_offset_words = 0;
        vkCmdPushConstants(command_buffer_, pipeline_layout_, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(BitNetPushConstants), &pushConstants);
        vkCmdDispatch(command_buffer_, n_heads, 1, 1);

        // RoPE for K
        pushConstants.out_dim = n_kv_heads;
        pushConstants.dequant_scale = rope_theta;
        pushConstants.weight_offset_bytes = current_pos;
        pushConstants.output_offset_words = q_dim;
        vkCmdPushConstants(command_buffer_, pipeline_layout_, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(BitNetPushConstants), &pushConstants);
        vkCmdDispatch(command_buffer_, n_kv_heads, 1, 1);

        // Barrier: output_buffer_ Compute Shader Write -> Transfer Read
        outBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        outBarrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
        vkCmdPipelineBarrier(command_buffer_, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
                             0, 0, nullptr, 1, &outBarrier, 0, nullptr);

        // ====================================================================
        // Substep 4: Store K & V to kv_cache_buffer_ via GPU DMA
        // ====================================================================
        VkBufferCopy copyRegions[2]{};
        copyRegions[0].srcOffset = (VkDeviceSize)q_dim * sizeof(float);
        copyRegions[0].dstOffset = (VkDeviceSize)(layer_base + current_pos * pos_stride) * sizeof(float);
        copyRegions[0].size = (VkDeviceSize)kv_dim * sizeof(float);

        copyRegions[1].srcOffset = (VkDeviceSize)(q_dim + kv_dim) * sizeof(float);
        copyRegions[1].dstOffset = (VkDeviceSize)(layer_base + layer_stride + current_pos * pos_stride) * sizeof(float);
        copyRegions[1].size = (VkDeviceSize)kv_dim * sizeof(float);
        vkCmdCopyBuffer(command_buffer_, output_buffer_, kv_cache_buffer_, 2, copyRegions);

        // Barrier: kv_cache_buffer_ Transfer Write -> Shader Read
        // and output_buffer_ Transfer Read -> Shader Read/Write
        VkBufferMemoryBarrier kvBarrier{};
        kvBarrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
        kvBarrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        kvBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        kvBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        kvBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        kvBarrier.buffer = kv_cache_buffer_;
        kvBarrier.offset = 0;
        kvBarrier.size = VK_WHOLE_SIZE;

        outBarrier.srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
        outBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;

        VkBufferMemoryBarrier midBarriers[2] = { kvBarrier, outBarrier };
        vkCmdPipelineBarrier(command_buffer_, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                             0, 0, nullptr, 2, midBarriers, 0, nullptr);

        // ====================================================================
        // Substep 5: Multi-Head Self-Attention Decode
        // output_buffer_[0..q_dim-1] reads Q, writes Attention Output
        // ====================================================================
        vkCmdBindPipeline(command_buffer_, VK_PIPELINE_BIND_POINT_COMPUTE, attention_decode_pipeline_);
        pushConstants.in_dim = n_ctx_;
        pushConstants.out_dim = n_heads;
        pushConstants.dequant_scale = 1.0f / std::sqrt(128.0f);
        pushConstants.weight_offset_bytes = current_pos;
        pushConstants.output_offset_words = l;
        vkCmdPushConstants(command_buffer_, pipeline_layout_, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(BitNetPushConstants), &pushConstants);
        vkCmdDispatch(command_buffer_, n_heads, 1, 1);

        // ====================================================================
        // Substep 6: Optional Attention Sub-LayerNorm
        // ====================================================================
        if (lay.offset_attn_sub_norm != 0xFFFFFFFFu) {
            outBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
            outBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
            vkCmdPipelineBarrier(command_buffer_, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                 0, 0, nullptr, 1, &outBarrier, 0, nullptr);

            vkCmdBindPipeline(command_buffer_, VK_PIPELINE_BIND_POINT_COMPUTE, rmsnorm_pipeline_);
            pushConstants.out_dim = q_dim;
            pushConstants.dequant_scale = eps;
            pushConstants.weight_offset_bytes = lay.offset_attn_sub_norm;
            pushConstants.output_offset_words = 0;
            vkCmdPushConstants(command_buffer_, pipeline_layout_, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(BitNetPushConstants), &pushConstants);
            vkCmdDispatch(command_buffer_, 1, 1, 1);
        }

        // ====================================================================
        // Substep 7: Copy Attention Output to activation_buffer_ via GPU DMA
        // ====================================================================
        outBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        outBarrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
        vkCmdPipelineBarrier(command_buffer_, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
                             0, 0, nullptr, 1, &outBarrier, 0, nullptr);

        VkBufferCopy actCopy{};
        actCopy.srcOffset = 0;
        actCopy.dstOffset = 0;
        actCopy.size = (VkDeviceSize)q_dim * sizeof(float);
        vkCmdCopyBuffer(command_buffer_, output_buffer_, activation_buffer_, 1, &actCopy);

        actBarrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        actBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        vkCmdPipelineBarrier(command_buffer_, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                             0, 0, nullptr, 1, &actBarrier, 0, nullptr);

        // ====================================================================
        // Substep 8: Wo Output Projection GEMV
        // activation_buffer_ -> output_buffer_[0..dim_k-1]
        // ====================================================================
        vkCmdBindPipeline(command_buffer_, VK_PIPELINE_BIND_POINT_COMPUTE, compute_pipeline_);
        pushConstants.in_dim = q_dim;
        pushConstants.out_dim = dim_k;
        pushConstants.dequant_scale = 1.0f;
        pushConstants.weight_offset_bytes = lay.offset_wo;
        pushConstants.output_offset_words = 0;
        vkCmdPushConstants(command_buffer_, pipeline_layout_, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(BitNetPushConstants), &pushConstants);
        vkCmdDispatch(command_buffer_, (dim_k + 3) / 4, 1, 1);

        // ====================================================================
        // Substep 9: In-Place Residual Stream Accumulation (Attention)
        // residual_buffer_ += output_buffer_
        // ====================================================================
        outBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        outBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

        resBarrier.srcAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
        resBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;

        VkBufferMemoryBarrier resMidBarriers[2] = { outBarrier, resBarrier };
        vkCmdPipelineBarrier(command_buffer_, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                             0, 0, nullptr, 2, resMidBarriers, 0, nullptr);

        vkCmdBindPipeline(command_buffer_, VK_PIPELINE_BIND_POINT_COMPUTE, residual_add_pipeline_);
        pushConstants.in_dim = dim_k;
        vkCmdPushConstants(command_buffer_, pipeline_layout_, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(BitNetPushConstants), &pushConstants);
        vkCmdDispatch(command_buffer_, (dim_k + 255) / 256, 1, 1);

        // ====================================================================
        // Substep 10: FFN RMSNorm
        // residual_buffer_ -> activation_buffer_ (via lay.offset_ffn_norm)
        // ====================================================================
        resBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        resBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        vkCmdPipelineBarrier(command_buffer_, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                             0, 0, nullptr, 1, &resBarrier, 0, nullptr);

        vkCmdBindPipeline(command_buffer_, VK_PIPELINE_BIND_POINT_COMPUTE, rmsnorm_norm_pipeline_);
        pushConstants.out_dim = dim_k;
        pushConstants.dequant_scale = eps;
        pushConstants.weight_offset_bytes = lay.offset_ffn_norm;
        pushConstants.output_offset_words = 0;
        vkCmdPushConstants(command_buffer_, pipeline_layout_, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(BitNetPushConstants), &pushConstants);
        vkCmdDispatch(command_buffer_, 1, 1, 1);

        // Barrier: activation_buffer_ Compute Shader Write -> Compute Shader Read
        actBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        actBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        vkCmdPipelineBarrier(command_buffer_, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                             0, 0, nullptr, 1, &actBarrier, 0, nullptr);

        // ====================================================================
        // Substep 11: Gate & Up GEMVs
        // activation_buffer_ -> output_buffer_
        // ====================================================================
        vkCmdBindPipeline(command_buffer_, VK_PIPELINE_BIND_POINT_COMPUTE, compute_pipeline_);
        // W_gate GEMV
        pushConstants.in_dim = dim_k;
        pushConstants.out_dim = dim_m;
        pushConstants.dequant_scale = scale;
        pushConstants.weight_offset_bytes = lay.offset_w_gate;
        pushConstants.output_offset_words = 0;
        vkCmdPushConstants(command_buffer_, pipeline_layout_, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(BitNetPushConstants), &pushConstants);
        vkCmdDispatch(command_buffer_, (dim_m + 3) / 4, 1, 1);

        // W_up GEMV
        pushConstants.out_dim = dim_m;
        pushConstants.weight_offset_bytes = lay.offset_w_up;
        pushConstants.output_offset_words = dim_m;
        vkCmdPushConstants(command_buffer_, pipeline_layout_, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(BitNetPushConstants), &pushConstants);
        vkCmdDispatch(command_buffer_, (dim_m + 3) / 4, 1, 1);

        // Barrier: output_buffer_ Compute Shader Write -> Compute Shader Read/Write
        outBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        outBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
        vkCmdPipelineBarrier(command_buffer_, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                             0, 0, nullptr, 1, &outBarrier, 0, nullptr);

        // ====================================================================
        // Substep 12: In-Place SwiGLU SiLU
        // output_buffer_[0..dim_m-1] = SiLU(gate) * up
        // ====================================================================
        vkCmdBindPipeline(command_buffer_, VK_PIPELINE_BIND_POINT_COMPUTE, swiglu_pipeline_);
        pushConstants.out_dim = dim_m;
        vkCmdPushConstants(command_buffer_, pipeline_layout_, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(BitNetPushConstants), &pushConstants);
        vkCmdDispatch(command_buffer_, (dim_m + 255) / 256, 1, 1);

        // ====================================================================
        // Substep 13: Optional FFN Sub-LayerNorm
        // ====================================================================
        if (lay.offset_ffn_sub_norm != 0xFFFFFFFFu) {
            outBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
            outBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
            vkCmdPipelineBarrier(command_buffer_, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                 0, 0, nullptr, 1, &outBarrier, 0, nullptr);

            vkCmdBindPipeline(command_buffer_, VK_PIPELINE_BIND_POINT_COMPUTE, rmsnorm_pipeline_);
            pushConstants.out_dim = dim_m;
            pushConstants.dequant_scale = eps;
            pushConstants.weight_offset_bytes = lay.offset_ffn_sub_norm;
            pushConstants.output_offset_words = 0;
            vkCmdPushConstants(command_buffer_, pipeline_layout_, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(BitNetPushConstants), &pushConstants);
            vkCmdDispatch(command_buffer_, 1, 1, 1);
        }

        // ====================================================================
        // Substep 14: Copy FFN Output to activation_buffer_ via GPU DMA
        // ====================================================================
        outBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        outBarrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
        vkCmdPipelineBarrier(command_buffer_, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
                             0, 0, nullptr, 1, &outBarrier, 0, nullptr);

        VkBufferCopy ffnCopy{};
        ffnCopy.srcOffset = 0;
        ffnCopy.dstOffset = 0;
        ffnCopy.size = (VkDeviceSize)dim_m * sizeof(float);
        vkCmdCopyBuffer(command_buffer_, output_buffer_, activation_buffer_, 1, &ffnCopy);

        actBarrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        actBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        vkCmdPipelineBarrier(command_buffer_, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                             0, 0, nullptr, 1, &actBarrier, 0, nullptr);

        // ====================================================================
        // Substep 15: W_down Projection GEMV
        // activation_buffer_ -> output_buffer_[0..dim_k-1]
        // ====================================================================
        vkCmdBindPipeline(command_buffer_, VK_PIPELINE_BIND_POINT_COMPUTE, compute_pipeline_);
        pushConstants.in_dim = dim_m;
        pushConstants.out_dim = dim_k;
        pushConstants.dequant_scale = 1.0f;
        pushConstants.weight_offset_bytes = lay.offset_w_down;
        pushConstants.output_offset_words = 0;
        vkCmdPushConstants(command_buffer_, pipeline_layout_, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(BitNetPushConstants), &pushConstants);
        vkCmdDispatch(command_buffer_, (dim_k + 3) / 4, 1, 1);

        // ====================================================================
        // Substep 16: In-Place Residual Stream Accumulation (FFN)
        // residual_buffer_ += output_buffer_
        // ====================================================================
        outBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        outBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

        resBarrier.srcAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
        resBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;

        VkBufferMemoryBarrier resMidBarriers2[2] = { outBarrier, resBarrier };
        vkCmdPipelineBarrier(command_buffer_, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                             0, 0, nullptr, 2, resMidBarriers2, 0, nullptr);

        vkCmdBindPipeline(command_buffer_, VK_PIPELINE_BIND_POINT_COMPUTE, residual_add_pipeline_);
        pushConstants.in_dim = dim_k;
        vkCmdPushConstants(command_buffer_, pipeline_layout_, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(BitNetPushConstants), &pushConstants);
        vkCmdDispatch(command_buffer_, (dim_k + 255) / 256, 1, 1);

        // If this is the last layer in chain, synchronize for Host Read.
        // Otherwise, synchronize for next layer's RMSNorm Compute Shader Read.
        if (l == n_layers - 1) {
            resBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
            resBarrier.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
            vkCmdPipelineBarrier(command_buffer_, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_HOST_BIT,
                                 0, 0, nullptr, 1, &resBarrier, 0, nullptr);
        } else {
            resBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
            resBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
            vkCmdPipelineBarrier(command_buffer_, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                 0, 0, nullptr, 1, &resBarrier, 0, nullptr);
        }
    }

    VK_CHECK_ENGINE(vkEndCommandBuffer(command_buffer_), "vkEndCommandBuffer");

    // 3. Submit Queue & Synchronize ONCE for the ENTIRE Model!
    VK_CHECK_ENGINE(vkResetFences(device_, 1, &execution_fence_), "vkResetFences");

    VkSubmitInfo submitInfo{};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &command_buffer_;

    VK_CHECK_ENGINE(vkQueueSubmit(compute_queue_, 1, &submitInfo, execution_fence_), "vkQueueSubmit");

    VkResult waitRes = vkWaitForFences(device_, 1, &execution_fence_, VK_TRUE, 10000000000ULL);
    if (waitRes != VK_SUCCESS) {
        throw AmevaVulkanExecutionError("vkWaitForFences timeout/hang in DispatchFullTokenChain", waitRes, __FILE__, __LINE__);
    }

    // 4. Retrieve final output activation vector directly from residual_buffer_ into x_out
    std::memcpy(x_out, residual_mapped_ptr_, dim_k * sizeof(float));
}

void VulkanBitNetEngine::UploadWeights(const void* raw_weights, size_t byte_size) {
    if (!weight_mapped_ptr_) {
        throw AmevaVulkanExecutionError("UploadWeights: buffer not allocated or mapped", -5, __FILE__, __LINE__);
    }
    if (byte_size > weight_buffer_size_) {
        throw AmevaVulkanExecutionError("UploadWeights: byte_size exceeds allocated weight buffer", -1, __FILE__, __LINE__);
    }
    std::memcpy(weight_mapped_ptr_, raw_weights, byte_size);
}

void VulkanBitNetEngine::DispatchGemv(const float* x, float* y, uint32_t dim_m, uint32_t dim_k, float scale) {
    if (!initialized_ || !activation_mapped_ptr_ || !output_mapped_ptr_) {
        throw AmevaVulkanExecutionError("DispatchGemv: engine or buffers not initialized", -3, __FILE__, __LINE__);
    }
    if (dim_m != allocated_dim_m_ || dim_k != allocated_dim_k_) {
        throw AmevaVulkanExecutionError("DispatchGemv: dimensions mismatch allocated tensor", -8, __FILE__, __LINE__);
    }

    // 1. Copy activations to host-visible mapped GPU memory
    std::memcpy(activation_mapped_ptr_, x, dim_k * sizeof(float));

    // 2. Record Command Buffer
    VkCommandBufferBeginInfo beginInfo{};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    VK_CHECK_ENGINE(vkBeginCommandBuffer(command_buffer_, &beginInfo), "vkBeginCommandBuffer");

    vkCmdBindPipeline(command_buffer_, VK_PIPELINE_BIND_POINT_COMPUTE, compute_pipeline_);
    vkCmdBindDescriptorSets(command_buffer_, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_layout_, 0, 1, &descriptor_set_, 0, nullptr);

    BitNetPushConstants pushConstants{};
    pushConstants.in_dim = dim_k;
    pushConstants.out_dim = dim_m;
    pushConstants.dequant_scale = scale;
    pushConstants.weight_offset_bytes = 0;
    pushConstants.output_offset_words = 0;
    vkCmdPushConstants(command_buffer_, pipeline_layout_, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(BitNetPushConstants), &pushConstants);

    // Dispatch: Each workgroup tile computes 4 rows
    uint32_t group_count_x = (dim_m + 3) / 4;
    vkCmdDispatch(command_buffer_, group_count_x, 1, 1);

    VK_CHECK_ENGINE(vkEndCommandBuffer(command_buffer_), "vkEndCommandBuffer");

    // 3. Submit Queue & Wait on Fence
    VK_CHECK_ENGINE(vkResetFences(device_, 1, &execution_fence_), "vkResetFences");

    VkSubmitInfo submitInfo{};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &command_buffer_;

    VK_CHECK_ENGINE(vkQueueSubmit(compute_queue_, 1, &submitInfo, execution_fence_), "vkQueueSubmit");

    // Wait with 5-second hard timeout to detect and fail-fast on any GPU hang
    VkResult waitRes = vkWaitForFences(device_, 1, &execution_fence_, VK_TRUE, 5000000000ULL);
    if (waitRes != VK_SUCCESS) {
        throw AmevaVulkanExecutionError("vkWaitForFences timeout/hang", waitRes, __FILE__, __LINE__);
    }

    // 4. Retrieve result from output buffer
    std::memcpy(y, output_mapped_ptr_, dim_m * sizeof(float));
}

void VulkanBitNetEngine::ComputeGemvDynamic(const void* raw_weights, const float* x, float* y, uint32_t dim_m, uint32_t dim_k, float scale) {
    if (!initialized_ || !activation_mapped_ptr_ || !output_mapped_ptr_ || !weight_mapped_ptr_) {
        throw AmevaVulkanExecutionError("ComputeGemvDynamic: engine or buffers not initialized", -3, __FILE__, __LINE__);
    }
    if (dim_k % 128 != 0) {
        throw AmevaVulkanExecutionError("ComputeGemvDynamic: dim_k must be multiple of 128", -8, __FILE__, __LINE__);
    }

    size_t num_blocks = dim_k / 128;
    size_t weight_bytes = dim_m * num_blocks * 32;
    if (weight_bytes > weight_buffer_size_) {
        throw AmevaVulkanExecutionError("ComputeGemvDynamic: weight_bytes exceeds preallocated buffer", -1, __FILE__, __LINE__);
    }
    if (dim_k * sizeof(float) > activation_buffer_size_) {
        throw AmevaVulkanExecutionError("ComputeGemvDynamic: activation size exceeds buffer", -1, __FILE__, __LINE__);
    }
    if (dim_m * sizeof(float) > output_buffer_size_) {
        throw AmevaVulkanExecutionError("ComputeGemvDynamic: output size exceeds buffer", -1, __FILE__, __LINE__);
    }

    // 1. Copy weights & activations to mapped coherent GPU memory
    std::memcpy(weight_mapped_ptr_, raw_weights, weight_bytes);
    std::memcpy(activation_mapped_ptr_, x, dim_k * sizeof(float));

    // 2. Record Command Buffer
    VkCommandBufferBeginInfo beginInfo{};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    VK_CHECK_ENGINE(vkBeginCommandBuffer(command_buffer_, &beginInfo), "vkBeginCommandBuffer");

    vkCmdBindPipeline(command_buffer_, VK_PIPELINE_BIND_POINT_COMPUTE, compute_pipeline_);
    vkCmdBindDescriptorSets(command_buffer_, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_layout_, 0, 1, &descriptor_set_, 0, nullptr);

    BitNetPushConstants pushConstants{};
    pushConstants.in_dim = dim_k;
    pushConstants.out_dim = dim_m;
    pushConstants.dequant_scale = scale;
    pushConstants.weight_offset_bytes = 0;
    pushConstants.output_offset_words = 0;
    vkCmdPushConstants(command_buffer_, pipeline_layout_, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(BitNetPushConstants), &pushConstants);

    uint32_t group_count_x = (dim_m + 3) / 4;
    vkCmdDispatch(command_buffer_, group_count_x, 1, 1);

    VK_CHECK_ENGINE(vkEndCommandBuffer(command_buffer_), "vkEndCommandBuffer");

    // 3. Submit & Synchronize
    VK_CHECK_ENGINE(vkResetFences(device_, 1, &execution_fence_), "vkResetFences");

    VkSubmitInfo submitInfo{};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &command_buffer_;

    VK_CHECK_ENGINE(vkQueueSubmit(compute_queue_, 1, &submitInfo, execution_fence_), "vkQueueSubmit");

    VkResult waitRes = vkWaitForFences(device_, 1, &execution_fence_, VK_TRUE, 5000000000ULL);
    if (waitRes != VK_SUCCESS) {
        throw AmevaVulkanExecutionError("vkWaitForFences timeout/hang in ComputeGemvDynamic", waitRes, __FILE__, __LINE__);
    }

    // 4. Retrieve output
    std::memcpy(y, output_mapped_ptr_, dim_m * sizeof(float));
}

void VulkanBitNetEngine::AllocateLMHeadBuffer(size_t byte_size, uint32_t n_vocab, uint32_t n_embd) {
    if (!initialized_) {
        throw AmevaVulkanExecutionError("AllocateLMHeadBuffer called before Initialize()", -3, __FILE__, __LINE__);
    }

    if (lm_head_buffer_) {
        vkDeviceWaitIdle(device_);
        if (lm_head_mapped_ptr_) { vkUnmapMemory(device_, lm_head_memory_); lm_head_mapped_ptr_ = nullptr; }
        if (lm_head_buffer_) { vkDestroyBuffer(device_, lm_head_buffer_, nullptr); lm_head_buffer_ = nullptr; }
        if (lm_head_memory_) { vkFreeMemory(device_, lm_head_memory_, nullptr); lm_head_memory_ = nullptr; }
    }

    lm_head_vocab_ = n_vocab;
    lm_head_embd_ = n_embd;
    lm_head_buffer_size_ = quirks::GpuQuirksEngine::AlignSize((uint32_t)byte_size, device_info_.required_alignment);

    uint32_t mem_flags = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    uint32_t buf_usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;

    CreateBuffer(lm_head_buffer_size_, buf_usage, mem_flags, lm_head_buffer_, lm_head_memory_);
    VK_CHECK_ENGINE(vkMapMemory(device_, lm_head_memory_, 0, lm_head_buffer_size_, 0, &lm_head_mapped_ptr_), "vkMapMemory (lm_head_weights)");

    if (!lm_head_descriptor_set_) {
        VkDescriptorSetAllocateInfo descAllocInfo{};
        descAllocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        descAllocInfo.descriptorPool = descriptor_pool_;
        descAllocInfo.descriptorSetCount = 1;
        descAllocInfo.pSetLayouts = &descriptor_set_layout_;
        VK_CHECK_ENGINE(vkAllocateDescriptorSets(device_, &descAllocInfo, &lm_head_descriptor_set_), "vkAllocateDescriptorSets (lm_head)");
    }

    // Update LM Head Descriptor Set:
    // Binding 0: lm_head_buffer_ (FP16 weights)
    // Binding 1: activation_buffer_ (Input hidden states)
    // Binding 2: output_buffer_ (Output logits)
    // Binding 3: kv_cache_buffer_ (fallback)
    // Binding 4: residual_buffer_ (fallback)
    VkDescriptorBufferInfo bufferInfos[5]{};
    bufferInfos[0].buffer = lm_head_buffer_;
    bufferInfos[0].offset = 0;
    bufferInfos[0].range = lm_head_buffer_size_;

    bufferInfos[1].buffer = activation_buffer_ ? activation_buffer_ : lm_head_buffer_;
    bufferInfos[1].offset = 0;
    bufferInfos[1].range = activation_buffer_ ? activation_buffer_size_ : lm_head_buffer_size_;

    bufferInfos[2].buffer = output_buffer_ ? output_buffer_ : lm_head_buffer_;
    bufferInfos[2].offset = 0;
    bufferInfos[2].range = output_buffer_ ? output_buffer_size_ : lm_head_buffer_size_;

    bufferInfos[3].buffer = kv_cache_buffer_ ? kv_cache_buffer_ : lm_head_buffer_;
    bufferInfos[3].offset = 0;
    bufferInfos[3].range = kv_cache_buffer_ ? kv_cache_buffer_size_ : lm_head_buffer_size_;

    bufferInfos[4].buffer = residual_buffer_ ? residual_buffer_ : lm_head_buffer_;
    bufferInfos[4].offset = 0;
    bufferInfos[4].range = residual_buffer_ ? residual_buffer_size_ : lm_head_buffer_size_;

    VkWriteDescriptorSet descriptorWrites[5]{};
    for (int i = 0; i < 5; ++i) {
        descriptorWrites[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        descriptorWrites[i].dstSet = lm_head_descriptor_set_;
        descriptorWrites[i].dstBinding = i;
        descriptorWrites[i].dstArrayElement = 0;
        descriptorWrites[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        descriptorWrites[i].descriptorCount = 1;
        descriptorWrites[i].pBufferInfo = &bufferInfos[i];
    }
    vkUpdateDescriptorSets(device_, 5, descriptorWrites, 0, nullptr);
}

void VulkanBitNetEngine::UploadLMHeadWeights(const void* raw_weights, size_t byte_size) {
    if (!lm_head_mapped_ptr_) {
        throw AmevaVulkanExecutionError("UploadLMHeadWeights called before AllocateLMHeadBuffer", -4, __FILE__, __LINE__);
    }
    if (byte_size > lm_head_buffer_size_) {
        throw AmevaVulkanExecutionError("UploadLMHeadWeights: byte_size exceeds allocated buffer", -5, __FILE__, __LINE__);
    }
    std::memcpy(lm_head_mapped_ptr_, raw_weights, byte_size);
}

void VulkanBitNetEngine::DispatchLMHead(const float* x_norm, float* logits, uint32_t n_vocab, uint32_t n_embd) {
    if (!initialized_ || !lm_head_buffer_ || !gemv_f16_pipeline_) {
        throw AmevaVulkanExecutionError("DispatchLMHead called with uninitialized LM Head", -4, __FILE__, __LINE__);
    }
    if (n_embd * sizeof(float) > activation_buffer_size_) {
        throw AmevaVulkanExecutionError("DispatchLMHead: n_embd exceeds activation_buffer_size_", -5, __FILE__, __LINE__);
    }
    if (n_vocab * sizeof(float) > output_buffer_size_) {
        throw AmevaVulkanExecutionError("DispatchLMHead: n_vocab exceeds output_buffer_size_", -6, __FILE__, __LINE__);
    }

    // 1. Copy x_norm to activation_mapped_ptr_
    std::memcpy(activation_mapped_ptr_, x_norm, n_embd * sizeof(float));

    // 2. Record command buffer
    vkResetCommandBuffer(command_buffer_, 0);

    VkCommandBufferBeginInfo beginInfo{};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    VK_CHECK_ENGINE(vkBeginCommandBuffer(command_buffer_, &beginInfo), "vkBeginCommandBuffer (DispatchLMHead)");

    vkCmdBindPipeline(command_buffer_, VK_PIPELINE_BIND_POINT_COMPUTE, gemv_f16_pipeline_);
    vkCmdBindDescriptorSets(command_buffer_, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_layout_, 0, 1, &lm_head_descriptor_set_, 0, nullptr);

    BitNetPushConstants push_constants{};
    push_constants.in_dim = n_embd;
    push_constants.out_dim = n_vocab;
    push_constants.dequant_scale = 1.0f;
    push_constants.weight_offset_bytes = 0;
    push_constants.output_offset_words = 0;
    vkCmdPushConstants(command_buffer_, pipeline_layout_, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(BitNetPushConstants), &push_constants);

    // Each workgroup calculates 4 rows (local_size_x = 32, local_size_y = 4)
    uint32_t num_workgroups_x = (n_vocab + 3) / 4;
    vkCmdDispatch(command_buffer_, num_workgroups_x, 1, 1);

    VK_CHECK_ENGINE(vkEndCommandBuffer(command_buffer_), "vkEndCommandBuffer (DispatchLMHead)");

    // 3. Submit & Wait Fence
    VK_CHECK_ENGINE(vkResetFences(device_, 1, &execution_fence_), "vkResetFences (DispatchLMHead)");

    VkSubmitInfo submitInfo{};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &command_buffer_;

    VK_CHECK_ENGINE(vkQueueSubmit(compute_queue_, 1, &submitInfo, execution_fence_), "vkQueueSubmit (DispatchLMHead)");
    VkResult waitRes = vkWaitForFences(device_, 1, &execution_fence_, VK_TRUE, 5000000000ULL);
    if (waitRes != VK_SUCCESS) {
        throw AmevaVulkanExecutionError("vkWaitForFences timeout/hang in DispatchLMHead", waitRes, __FILE__, __LINE__);
    }

    // 4. Retrieve logits
    std::memcpy(logits, output_mapped_ptr_, n_vocab * sizeof(float));
}

void VulkanBitNetEngine::Shutdown() {
    if (device_) {
        vkDeviceWaitIdle(device_);

        if (lm_head_mapped_ptr_) { vkUnmapMemory(device_, lm_head_memory_); lm_head_mapped_ptr_ = nullptr; }
        if (lm_head_buffer_) { vkDestroyBuffer(device_, lm_head_buffer_, nullptr); lm_head_buffer_ = nullptr; }
        if (lm_head_memory_) { vkFreeMemory(device_, lm_head_memory_, nullptr); lm_head_memory_ = nullptr; }
        lm_head_descriptor_set_ = nullptr;

        if (weight_mapped_ptr_) { vkUnmapMemory(device_, weight_memory_); weight_mapped_ptr_ = nullptr; }
        if (weight_buffer_) { vkDestroyBuffer(device_, weight_buffer_, nullptr); weight_buffer_ = nullptr; }
        if (weight_memory_) { vkFreeMemory(device_, weight_memory_, nullptr); weight_memory_ = nullptr; }

        if (activation_mapped_ptr_) { vkUnmapMemory(device_, activation_memory_); activation_mapped_ptr_ = nullptr; }
        if (activation_buffer_) { vkDestroyBuffer(device_, activation_buffer_, nullptr); activation_buffer_ = nullptr; }
        if (activation_memory_) { vkFreeMemory(device_, activation_memory_, nullptr); activation_memory_ = nullptr; }

        if (output_mapped_ptr_) { vkUnmapMemory(device_, output_memory_); output_mapped_ptr_ = nullptr; }
        if (output_buffer_) { vkDestroyBuffer(device_, output_buffer_, nullptr); output_buffer_ = nullptr; }
        if (output_memory_) { vkFreeMemory(device_, output_memory_, nullptr); output_memory_ = nullptr; }

        if (kv_cache_mapped_ptr_) { vkUnmapMemory(device_, kv_cache_memory_); kv_cache_mapped_ptr_ = nullptr; }
        if (kv_cache_buffer_) { vkDestroyBuffer(device_, kv_cache_buffer_, nullptr); kv_cache_buffer_ = nullptr; }
        if (kv_cache_memory_) { vkFreeMemory(device_, kv_cache_memory_, nullptr); kv_cache_memory_ = nullptr; }

        if (residual_mapped_ptr_) { vkUnmapMemory(device_, residual_memory_); residual_mapped_ptr_ = nullptr; }
        if (residual_buffer_) { vkDestroyBuffer(device_, residual_buffer_, nullptr); residual_buffer_ = nullptr; }
        if (residual_memory_) { vkFreeMemory(device_, residual_memory_, nullptr); residual_memory_ = nullptr; }

        if (execution_fence_) { vkDestroyFence(device_, execution_fence_, nullptr); execution_fence_ = nullptr; }
        if (command_pool_) { vkDestroyCommandPool(device_, command_pool_, nullptr); command_pool_ = nullptr; }
        if (descriptor_pool_) { vkDestroyDescriptorPool(device_, descriptor_pool_, nullptr); descriptor_pool_ = nullptr; }
        if (compute_pipeline_) { vkDestroyPipeline(device_, compute_pipeline_, nullptr); compute_pipeline_ = nullptr; }
        if (swiglu_pipeline_) { vkDestroyPipeline(device_, swiglu_pipeline_, nullptr); swiglu_pipeline_ = nullptr; }
        if (rmsnorm_pipeline_) { vkDestroyPipeline(device_, rmsnorm_pipeline_, nullptr); rmsnorm_pipeline_ = nullptr; }
        if (rope_pipeline_) { vkDestroyPipeline(device_, rope_pipeline_, nullptr); rope_pipeline_ = nullptr; }
        if (attention_decode_pipeline_) { vkDestroyPipeline(device_, attention_decode_pipeline_, nullptr); attention_decode_pipeline_ = nullptr; }
        if (rmsnorm_norm_pipeline_) { vkDestroyPipeline(device_, rmsnorm_norm_pipeline_, nullptr); rmsnorm_norm_pipeline_ = nullptr; }
        if (residual_add_pipeline_) { vkDestroyPipeline(device_, residual_add_pipeline_, nullptr); residual_add_pipeline_ = nullptr; }
        if (gemv_f16_pipeline_) { vkDestroyPipeline(device_, gemv_f16_pipeline_, nullptr); gemv_f16_pipeline_ = nullptr; }
        if (pipeline_layout_) { vkDestroyPipelineLayout(device_, pipeline_layout_, nullptr); pipeline_layout_ = nullptr; }
        if (descriptor_set_layout_) { vkDestroyDescriptorSetLayout(device_, descriptor_set_layout_, nullptr); descriptor_set_layout_ = nullptr; }
        if (shader_module_) { vkDestroyShaderModule(device_, shader_module_, nullptr); shader_module_ = nullptr; }
        if (swiglu_shader_module_) { vkDestroyShaderModule(device_, swiglu_shader_module_, nullptr); swiglu_shader_module_ = nullptr; }
        if (rmsnorm_shader_module_) { vkDestroyShaderModule(device_, rmsnorm_shader_module_, nullptr); rmsnorm_shader_module_ = nullptr; }
        if (rope_shader_module_) { vkDestroyShaderModule(device_, rope_shader_module_, nullptr); rope_shader_module_ = nullptr; }
        if (attention_decode_shader_module_) { vkDestroyShaderModule(device_, attention_decode_shader_module_, nullptr); attention_decode_shader_module_ = nullptr; }
        if (rmsnorm_norm_shader_module_) { vkDestroyShaderModule(device_, rmsnorm_norm_shader_module_, nullptr); rmsnorm_norm_shader_module_ = nullptr; }
        if (residual_add_shader_module_) { vkDestroyShaderModule(device_, residual_add_shader_module_, nullptr); residual_add_shader_module_ = nullptr; }
        if (gemv_f16_shader_module_) { vkDestroyShaderModule(device_, gemv_f16_shader_module_, nullptr); gemv_f16_shader_module_ = nullptr; }

        vkDestroyDevice(device_, nullptr);
        device_ = nullptr;
    }

    if (instance_) {
        vkDestroyInstance(instance_, nullptr);
        instance_ = nullptr;
    }

    loader_.Unload();
    initialized_ = false;
}

} // namespace core
} // namespace ameva
