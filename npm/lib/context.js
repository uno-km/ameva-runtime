/**
 * AMEVA Vulkan Hardware Context for Node.js
 */
const { Doctor } = require('./doctor');
const os = require('os');

class PlatformNotSupportedError extends Error {
  constructor(message) {
    super(message);
    this.name = "PlatformNotSupportedError";
  }
}

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
      this.deviceName = "Qualcomm Adreno / ARM Mali Vulkan GPU";
      this.executionFlags = { useGpu: true, gpuLayers: 99, backend: "vulkan" };
      this.isActive = true;
    } else if (this.deviceMode === "cpu") {
      this.backendType = "cpu_neon";
      this.deviceName = "ARM64 NEON Vector CPU Engine";
      this.executionFlags = { useGpu: false, threads: (os.cpus() || []).length || 4, backend: "cpu_neon" };
      this.isActive = true;
    } else { // "auto"
      const isSupported = this.doctor.quickProbe();
      if (isSupported) {
        this.backendType = "vulkan";
        this.deviceName = "Qualcomm Adreno / ARM Mali Vulkan GPU";
        this.executionFlags = { useGpu: true, gpuLayers: 99, backend: "vulkan" };
      } else {
        this.backendType = "cpu_neon";
        this.deviceName = "ARM64 NEON Vector CPU Engine (Auto-Recovered)";
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

  allocateBuffer(sizeBytes) {
    if (sizeBytes > this.memoryLimitMb * 1024 * 1024) {
      throw new Error(
        `Requested buffer size (${(sizeBytes / (1024*1024)).toFixed(2)} MB) exceeds configured memory limit (${this.memoryLimitMb} MB).`
      );
    }
    return sizeBytes;
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
