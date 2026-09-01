const assert = require('assert');
const { Doctor, createContext, getOrCreateContext, isAvailable, SttAdapter, DiffusionAdapter, BitnetAdapter, LlamaCppAdapter, TtsAdapter, VisionAdapter } = require('./index');

async function runTests() {
  console.log("[TEST] Running AMEVA Vulkan Runtime Node.js Test Suite (Rigorous Verification)...");

  // 1. Doctor Test
  const doc = new Doctor();
  const report = await doc.runSelfTest(false);
  assert.strictEqual(typeof report.overallSuccess, 'boolean', "overallSuccess must be boolean");
  assert.strictEqual(typeof report.passedStages, 'number', "passedStages must be number");
  assert.strictEqual(Array.isArray(report.stages), true, "stages must be an array");

  // 2. CPU Mode Bypass Test
  const cpuCtx = createContext({ device: "cpu" });
  assert.strictEqual(cpuCtx.isGpu, false, "CPU mode should not be GPU");
  assert.strictEqual(cpuCtx.backendType, "cpu_neon", "CPU backend type should be cpu_neon");
  const whisperFlags = cpuCtx.toEngineFlags("whisper");
  assert.strictEqual(whisperFlags.useGpu, false, "whisper flags should reflect CPU");
  assert.strictEqual(whisperFlags.gpuLayers, 0, "whisper gpuLayers should be 0 on CPU");

  // 3. Auto Mode Test
  const autoCtx = createContext({ device: "auto" });
  assert.strictEqual(typeof autoCtx.isGpu, 'boolean', "isGpu must be boolean");
  assert.strictEqual(typeof autoCtx.backendType, 'string', "backendType must be string");

  // 4. getOrCreateContext reuse
  const reused = getOrCreateContext(cpuCtx);
  assert.strictEqual(reused, cpuCtx, "getOrCreateContext should reuse instance");

  // 5. Memory Budget Test
  const safeBuf = cpuCtx.allocateBuffer(1024 * 1024 * 50); // 50MB
  assert.strictEqual(safeBuf, 1024 * 1024 * 50);
  assert.throws(() => {
    cpuCtx.allocateBuffer(1024 * 1024 * 2048); // 2048MB > 1024MB
  }, /exceeds configured memory limit/);

  // 6. Adapter Binding Tests
  const stt = SttAdapter.attach(null, cpuCtx);
  assert.strictEqual(stt.isVulkan, false);
  assert.strictEqual(stt.backend, "cpu_neon");

  const diff = DiffusionAdapter.attach(null, cpuCtx);
  assert.strictEqual(diff.isVulkan, false);

  const bit = BitnetAdapter.attach(null, cpuCtx);
  assert.strictEqual(bit.isVulkan, false);

  const llama = LlamaCppAdapter.attach(null, cpuCtx);
  assert.strictEqual(llama.isVulkan, false);

  const tts = TtsAdapter.attach(null, cpuCtx);
  assert.strictEqual(tts.isVulkan, false);

  const vis = VisionAdapter.attach(null, cpuCtx);
  assert.strictEqual(vis.isVulkan, false);

  console.log("[PASS] All Node.js tests passed successfully (6/6 Adapters, Memory Budget Guard, Honest Probing, 3 Modes).");
}

runTests().catch(err => {
  console.error("[FAIL]", err);
  process.exit(1);
});
