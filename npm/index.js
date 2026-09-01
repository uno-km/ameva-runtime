const { Doctor } = require('./lib/doctor');
const { VulkanContext, createContext } = require('./lib/context');
const {
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
  SttAdapter,
  DiffusionAdapter,
  BitnetAdapter,
  LlamaCppAdapter,
  TtsAdapter,
  VisionAdapter,
  isAvailable
};
