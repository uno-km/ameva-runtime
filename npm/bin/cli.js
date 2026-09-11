#!/usr/bin/env node
const { Doctor, createContext } = require('../index');

async function main() {
  const cmd = process.argv[2] || 'doctor';
  const doc = new Doctor();

  if (cmd === 'doctor') {
    const report = await doc.runSelfTest(true);
    process.exit(report.overallSuccess ? 0 : 1);
  } else if (cmd === 'install') {
    console.log("\n[PROVISIONING] Initializing AMEVA Vulkan Acceleration Engine...");
    const report = await doc.runSelfTest(true);
    if (report.overallSuccess) {
      console.log(`[SUCCESS] AMEVA Vulkan Runtime ready on ${report.deviceName}.`);
      process.exit(0);
    } else {
      console.log(`[WARNING] Active fallback: ${report.recommendedBackend}`);
      process.exit(1);
    }
  } else if (cmd === 'benchmark') {
    console.log("\n============================================================");
    console.log("  AMEVA-Vulkan-Runtime (Node.js): Multi-Modal Execution Plan");
    console.log("============================================================");
    const ctx = await createContext({ device: "auto" });
    console.log(`  Active Device   : ${ctx.deviceName}`);
    console.log(`  Selected Backend: ${ctx.selectedBackend} (Reason: ${ctx.selectionReason})`);
    console.log("  • STT (Whisper)       -> PLANNED (Status: NOT_EXECUTED)");
    console.log("  • Diffusion (SDXS)    -> PLANNED (Status: NOT_EXECUTED)");
    console.log("  • LLM (BitNet/GGUF)   -> PLANNED (Status: NOT_EXECUTED)");
    console.log("  • TTS (Piper)         -> PLANNED (Status: NOT_EXECUTED)");
    console.log("  • Vision (LLaVA/YOLO) -> PLANNED (Status: NOT_EXECUTED)");
    console.log("============================================================\n");
  } else {
    console.log("Usage: ameva-gpu [doctor|install|benchmark]");
  }
}

main();
