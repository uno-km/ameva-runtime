/**
 * 6-Modality Acceleration Adapters for Node.js
 */
class SttAdapter {
  static attach(engine, ctx) {
    const isVk = Boolean(ctx && ctx.isVulkan());
    return {
      module: "termux-stt",
      backend: isVk ? "vulkan" : "cpu_neon",
      isVulkan: isVk,
      rtfTarget: isVk ? 0.28 : 0.80,
      status: isVk ? "BOUND" : "BOUND_CPU"
    };
  }
}

class DiffusionAdapter {
  static attach(engine, ctx) {
    const isVk = Boolean(ctx && ctx.isVulkan());
    return {
      module: "termux-diffusion",
      backend: isVk ? "vulkan" : "cpu_neon",
      isVulkan: isVk,
      unetTiling: isVk,
      status: isVk ? "BOUND" : "BOUND_CPU"
    };
  }
}

class BitnetAdapter {
  static attach(engine, ctx) {
    const isVk = Boolean(ctx && ctx.isVulkan());
    return {
      module: "termux-bitnet",
      backend: isVk ? "vulkan" : "cpu_neon",
      isVulkan: isVk,
      kernel: isVk ? "ggml_vk_mul_mat_i2_s" : "neon_dotprod",
      status: isVk ? "BOUND" : "BOUND_CPU"
    };
  }
}

class LlamaCppAdapter {
  static attach(engine, ctx) {
    const isVk = Boolean(ctx && ctx.isVulkan());
    return {
      module: "termux-llamacpp",
      backend: isVk ? "vulkan" : "cpu_neon",
      isVulkan: isVk,
      ngl: isVk ? 33 : 0,
      status: isVk ? "BOUND" : "BOUND_CPU"
    };
  }
}

class TtsAdapter {
  static attach(engine, ctx) {
    const isVk = Boolean(ctx && ctx.isVulkan());
    return {
      module: "termux-tts",
      backend: isVk ? "vulkan" : "cpu_neon",
      isVulkan: isVk,
      latencyMs: isVk ? 38.5 : 115.0,
      status: isVk ? "BOUND" : "BOUND_CPU"
    };
  }
}

class VisionAdapter {
  static attach(engine, ctx) {
    const isVk = Boolean(ctx && ctx.isVulkan());
    return {
      module: "termux-vision",
      backend: isVk ? "vulkan" : "cpu_neon",
      isVulkan: isVk,
      vitAcceleration: isVk,
      status: isVk ? "BOUND" : "BOUND_CPU"
    };
  }
}

module.exports = {
  SttAdapter,
  DiffusionAdapter,
  BitnetAdapter,
  LlamaCppAdapter,
  TtsAdapter,
  VisionAdapter
};
