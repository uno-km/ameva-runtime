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
  VisionAdapter
} = require('./lib/adapters');

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
  isAvailable,
  nativeBridge: require('./lib/native_bridge')
};
