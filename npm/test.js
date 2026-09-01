const assert = require('assert');
const { Doctor, createContext, isAvailable, SttAdapter, DiffusionAdapter, BitnetAdapter, LlamaCppAdapter, TtsAdapter, VisionAdapter } = require('./index');

async function runTests() {
  console.log("[TEST] Running AMEVA Vulkan Runtime Node.js Test Suite...");

  // 1. Doctor Test
  const doc = new Doctor();
  const report = await doc.runSelfTest(false);
  assert.strictEqual(report.overallSuccess, true, "Doctor self-test should succeed");
  assert.strictEqual(report.passedStages, 12, "All 12 stages must pass");

  // 2. Context Test
  const ctx = await createContext({ device: "auto" });
  assert.strictEqual(ctx.isVulkan(), true, "Should initialize Vulkan backend");

  // 3. Adapter Binding Tests
  const stt = SttAdapter.attach(null, ctx);
  assert.strictEqual(stt.status, "BOUND");

  const diff = DiffusionAdapter.attach(null, ctx);
  assert.strictEqual(diff.status, "BOUND");

  const bit = BitnetAdapter.attach(null, ctx);
  assert.strictEqual(bit.status, "BOUND");

  const llama = LlamaCppAdapter.attach(null, ctx);
  assert.strictEqual(llama.status, "BOUND");

  const tts = TtsAdapter.attach(null, ctx);
  assert.strictEqual(tts.status, "BOUND");

  const vis = VisionAdapter.attach(null, ctx);
  assert.strictEqual(vis.status, "BOUND");

  console.log("[PASS] All Node.js tests passed successfully (6/6 Adapters, 12/12 Stages).");
}

runTests().catch(err => {
  console.error("[FAIL]", err);
  process.exit(1);
});
