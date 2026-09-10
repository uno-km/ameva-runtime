const { Doctor } = require('./lib/doctor');
const { VulkanContext, createContext, getOrCreateContext, PlatformNotSupportedError } = require('./lib/context');
const {
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
} = require('./lib/adapters');
const { executeSubprocess, validateExecutable } = require('./lib/subprocess');

function isAvailable() {
  const doc = new Doctor();
  return doc.quickProbe();
}

module.exports = {
  Doctor,
  VulkanContext,
  createContext,
  getOrCreateContext,
  PlatformNotSupportedError,
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
  toSubprocessOptions,
  executeSubprocess,
  validateExecutable,
  isAvailable,
  nativeBridge: require('./lib/native_bridge')
};
