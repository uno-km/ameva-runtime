export interface StageReport {
  stageId: number;
  stageName: string;
  result: "PASS" | "FAIL" | "SKIP";
  elapsedMs: number;
  detailMessage: string;
}

export interface DiagnosticReport {
  overallSuccess: bool;
  deviceName: string;
  driverVersion: string;
  loaderPath: string;
  passedStages: number;
  totalStages: number;
  totalElapsedMs: number;
  recommendedBackend: string;
  stages: StageReport[];
}

export class Doctor {
  constructor(statePath?: string);
  runSelfTest(verbose?: boolean): Promise<DiagnosticReport>;
  quickProbe(): boolean;
}

export class VulkanContext {
  deviceMode: string;
  memoryLimitMb: number;
  deviceName: string;
  backendType: string;
  vulkanVersion: string;
  isActive: boolean;
  isVulkan(): boolean;
  close(): void;
}

export function createContext(options?: { device?: string; memoryLimitMb?: number }): Promise<VulkanContext>;
export function isAvailable(): boolean;

export class SttAdapter { static attach(engine: any, ctx: VulkanContext): any; }
export class DiffusionAdapter { static attach(engine: any, ctx: VulkanContext): any; }
export class BitnetAdapter { static attach(engine: any, ctx: VulkanContext): any; }
export class LlamaCppAdapter { static attach(engine: any, ctx: VulkanContext): any; }
export class TtsAdapter { static attach(engine: any, ctx: VulkanContext): any; }
export class VisionAdapter { static attach(engine: any, ctx: VulkanContext): any; }
