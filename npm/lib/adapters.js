/**
 * 6-Modality Acceleration Execution Plans for Node.js
 * Generates execution configurations and runtime flags (NOT a native ABI handle binder).
 */
class SttExecutionPlan {
  static create(engine, ctx) {
    const isVk = Boolean(ctx && ctx.isVulkan());
    return {
      module: "termux-stt",
      backend: isVk ? "vulkan" : "cpu_neon",
      isVulkan: isVk,
      status: "PLANNED",
      executionStatus: "NOT_EXECUTED"
    };
  }
}

class DiffusionExecutionPlan {
  static create(engine, ctx) {
    const isVk = Boolean(ctx && ctx.isVulkan());
    return {
      module: "termux-diffusion",
      backend: isVk ? "vulkan" : "cpu_neon",
      isVulkan: isVk,
      unetTiling: isVk,
      status: "PLANNED",
      executionStatus: "NOT_EXECUTED"
    };
  }
}

class BitnetExecutionPlan {
  static create(engine, ctx) {
    const isVk = Boolean(ctx && ctx.isVulkan());
    return {
      module: "termux-bitnet",
      backend: isVk ? "vulkan" : "cpu_neon",
      isVulkan: isVk,
      kernel: isVk ? "ggml_vk_mul_mat_i2_s" : "neon_dotprod",
      status: "PLANNED",
      executionStatus: "NOT_EXECUTED"
    };
  }
}

class LlamaCppExecutionPlan {
  static create(engine, ctx) {
    const isVk = Boolean(ctx && ctx.isVulkan());
    return {
      module: "termux-llamacpp",
      backend: isVk ? "vulkan" : "cpu_neon",
      isVulkan: isVk,
      ngl: isVk ? 33 : 0,
      status: "PLANNED",
      executionStatus: "NOT_EXECUTED"
    };
  }
}

class TtsExecutionPlan {
  static create(engine, ctx) {
    const isVk = Boolean(ctx && ctx.isVulkan());
    return {
      module: "termux-tts",
      backend: isVk ? "vulkan" : "cpu_neon",
      isVulkan: isVk,
      status: "PLANNED",
      executionStatus: "NOT_EXECUTED"
    };
  }
}

class VisionExecutionPlan {
  static create(engine, ctx) {
    const isVk = Boolean(ctx && ctx.isVulkan());
    return {
      module: "termux-vision",
      backend: isVk ? "vulkan" : "cpu_neon",
      isVulkan: isVk,
      vitAcceleration: isVk,
      status: "PLANNED",
      executionStatus: "NOT_EXECUTED"
    };
  }
}

function toSubprocessOptions(plan, executable, extraArgs = [], options = {}) {
  if (!plan || typeof plan !== 'object') {
    throw new TypeError('Execution plan must be an object');
  }
  const args = Array.isArray(extraArgs) ? [...extraArgs] : [];
  if (plan.module === 'termux-llamacpp' && plan.ngl > 0) {
    args.push('-ngl', String(plan.ngl));
  }
  return {
    executable,
    args,
    backendRequested: plan.backend || null,
    ...options
  };
}

// Backward compatibility wrappers with deprecation warning notice
const SttAdapter = SttExecutionPlan;
const DiffusionAdapter = DiffusionExecutionPlan;
const BitnetAdapter = BitnetExecutionPlan;
const LlamaCppAdapter = LlamaCppExecutionPlan;
const TtsAdapter = TtsExecutionPlan;
const VisionAdapter = VisionExecutionPlan;

for (const Cls of [SttExecutionPlan, DiffusionExecutionPlan, BitnetExecutionPlan, LlamaCppExecutionPlan, TtsExecutionPlan, VisionExecutionPlan]) {
  Cls.attach = Cls.create;
  const originalCreate = Cls.create;
  Cls.create = function(engine, ctx) {
    const plan = originalCreate(engine, ctx);
    plan.toSubprocessOptions = function(executable, extraArgs, options) {
      return toSubprocessOptions(plan, executable, extraArgs, options);
    };
    return plan;
  };
}

module.exports = {
  SttExecutionPlan,
  DiffusionExecutionPlan,
  BitnetExecutionPlan,
  LlamaCppExecutionPlan,
  TtsExecutionPlan,
  VisionExecutionPlan,
  SttAdapter,
  DiffusionAdapter,
  BitnetAdapter,
  LlamaCppAdapter,
  TtsAdapter,
  VisionAdapter,
  toSubprocessOptions
};
