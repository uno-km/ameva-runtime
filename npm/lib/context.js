/**
 * AMEVA Vulkan Hardware Context for Node.js
 */
const { Doctor } = require('./doctor');
const { PlatformNotSupportedError } = require('./errors');
const os = require('os');

class VulkanContext {
  constructor(options = {}) {
    const rawMode = typeof options === 'string' ? options : (options.device || "auto");
    this.deviceMode = String(rawMode).trim().toLowerCase();
    this.memoryLimitMb = options.memoryLimitMb || 1024;
    this.doctor = new Doctor();
    this.deviceName = "CPU";
    this.backendType = "cpu_neon";
    this.vulkanVersion = "1.3.284";
    this.isActive = false;
    this.executionFlags = {};
    this.selectedBackend = "cpu_neon";
    this.selectionReason = "uninitialized";

    this._initialize();
  }

  _initialize() {
    if (this.deviceMode === "vulkan" || this.deviceMode === "gpu") {
      const isSupported = this.doctor.quickProbe();
      if (!isSupported) {
        throw new PlatformNotSupportedError(
          "Explicit GPU backend requested ('device=\"gpu\"' or 'device=\"vulkan\"'), but target hardware " +
          "or driver failed validation. Silent CPU fallback is disabled."
        );
      }
      this.backendType = "vulkan";
      this.selectedBackend = "vulkan";
      this.selectionReason = "explicit_gpu_request_verified";
      this.deviceName = this.doctor.quickProbeDevice() || "Qualcomm Adreno / ARM Mali Vulkan GPU";
      this.executionFlags = { useGpu: true, gpuLayers: 99, backend: "vulkan" };
      this.isActive = true;
    } else if (this.deviceMode === "cpu") {
      this.backendType = "cpu_neon";
      this.selectedBackend = "cpu_neon";
      this.selectionReason = "explicit_cpu_request";
      this.deviceName = "ARM64 NEON Vector CPU Engine";
      this.executionFlags = { useGpu: false, threads: (os.cpus() || []).length || 4, backend: "cpu_neon" };
      this.isActive = true;
    } else { // "auto"
      const isSupported = this.doctor.quickProbe();
      if (isSupported) {
        this.backendType = "vulkan";
        this.selectedBackend = "vulkan";
        this.selectionReason = "vulkan_certified_hardware";
        this.deviceName = this.doctor.quickProbeDevice() || "Qualcomm Adreno / ARM Mali Vulkan GPU";
        this.executionFlags = { useGpu: true, gpuLayers: 99, backend: "vulkan" };
      } else {
        this.backendType = "cpu_neon";
        this.selectedBackend = "cpu_neon";
        this.selectionReason = "vulkan_probe_unverified";
        this.deviceName = "ARM64 NEON Vector CPU Engine (Auto-Routed)";
        this.executionFlags = { useGpu: false, threads: (os.cpus() || []).length || 4, backend: "cpu_neon" };
      }
      this.isActive = true;
    }
  }

  get isGpu() {
    return this.backendType === "vulkan";
  }

  isVulkan() {
    return this.backendType === "vulkan";
  }

  toEngineFlags(engineName = "default") {
    const name = String(engineName || "").toLowerCase();
    if (name === "whisper" || name === "stt") {
      return {
        useGpu: this.isGpu,
        gpuLayers: this.isGpu ? 33 : 0,
        threads: this.isGpu ? 2 : 4,
        backend: this.backendType
      };
    } else if (name === "bitnet" || name === "llm" || name === "llama") {
      return {
        nGpuLayers: this.isGpu ? 33 : 0,
        threads: (os.cpus() || []).length || 4,
        backend: this.backendType
      };
    } else if (name === "diffusion" || name === "sd") {
      return {
        device: this.isGpu ? "vulkan" : "cpu",
        useVulkan: this.isGpu,
        backend: this.backendType
      };
    } else if (name === "tts") {
      return {
        device: this.isGpu ? "vulkan" : "cpu",
        backend: this.backendType,
        threads: (os.cpus() || []).length || 4
      };
    } else if (name === "vision") {
      return {
        device: this.isGpu ? "vulkan" : "cpu",
        backend: this.backendType,
        useGpu: this.isGpu
      };
    }
    return { ...this.executionFlags };
  }

  validateBufferBudget(sizeBytes) {
    if (sizeBytes > this.memoryLimitMb * 1024 * 1024) {
      throw new Error(
        `Requested buffer size (${(sizeBytes / (1024*1024)).toFixed(2)} MB) exceeds configured memory limit (${this.memoryLimitMb} MB).`
      );
    }
    return sizeBytes;
  }

  allocateBuffer(sizeBytes) {
    process.emitWarning(
      "allocateBuffer() does not allocate native memory and is deprecated. " +
      "Use validateBufferBudget(). This alias will be removed in v3.0.0.",
      {
        code: "AMEVA_DEPRECATED_FAKE_ALLOCATION",
        type: "DeprecationWarning"
      }
    );
    return this.validateBufferBudget(sizeBytes);
  }

  close() {
    this.isActive = false;
  }
}

function createContext(options = {}) {
  const ctx = new VulkanContext(options);
  return ctx;
}

function getOrCreateContext(options = {}) {
  if (options instanceof VulkanContext) {
    return options;
  }
  return new VulkanContext(options);
}

module.exports = { VulkanContext, createContext, getOrCreateContext, PlatformNotSupportedError };
