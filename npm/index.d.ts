export interface StageReport {
  stageId: number;
  stageName: string;
  result: "PASS" | "FAIL" | "SKIP";
  elapsedMs: number;
  detailMessage: string;
}

export interface DiagnosticReport {
  /**
   * Indicates whether the requested diagnostic scope (typically 'compute') succeeded.
   * Not to be confused with full end-to-end model pipeline certification.
   */
  overallSuccess: boolean;
  /** Diagnostic evaluation scope, e.g. 'compute' (V0-V9) */
  diagnosticScope: "compute" | "model" | "full";
  /** True when all stages within the requested diagnosticScope passed */
  scopeSuccess: boolean;
  /** True strictly when stages V0 through V9 all individually evaluate to PASS */
  computeCertified: boolean;
  /** True strictly when stages V0 through V11 all individually evaluate to PASS */
  modelCertified: boolean;
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
  quickProbeDevice(): string | null;
  validateCertificateMetadata(data: any): boolean;
  validateFingerprint(data: any): boolean;
}

export class VulkanContext {
  deviceMode: string;
  memoryLimitMb: number;
  deviceName: string;
  backendType: string;
  selectedBackend: string;
  selectionReason: string;
  vulkanVersion: string;
  isActive: boolean;
  isGpu: boolean;
  executionFlags: Record<string, any>;
  isVulkan(): boolean;
  toEngineFlags(engineName?: string): Record<string, any>;
  validateBufferBudget(sizeBytes: number): number;
  /** @deprecated Use validateBufferBudget(). Will be removed in v3.0.0 */
  allocateBuffer(sizeBytes: number): number;
  close(): void;
}

export class PlatformNotSupportedError extends Error {}

export function createContext(options?: { device?: string; memoryLimitMb?: number } | string): VulkanContext;
export function getOrCreateContext(options?: { device?: string; memoryLimitMb?: number } | string | VulkanContext): VulkanContext;
export function isAvailable(): boolean;

export class SttExecutionPlan { static create(engine: any, ctx: VulkanContext): any; static attach(engine: any, ctx: VulkanContext): any; }
export class DiffusionExecutionPlan { static create(engine: any, ctx: VulkanContext): any; static attach(engine: any, ctx: VulkanContext): any; }
export class BitnetExecutionPlan { static create(engine: any, ctx: VulkanContext): any; static attach(engine: any, ctx: VulkanContext): any; }
export class LlamaCppExecutionPlan { static create(engine: any, ctx: VulkanContext): any; static attach(engine: any, ctx: VulkanContext): any; }
export class TtsExecutionPlan { static create(engine: any, ctx: VulkanContext): any; static attach(engine: any, ctx: VulkanContext): any; }
export class VisionExecutionPlan { static create(engine: any, ctx: VulkanContext): any; static attach(engine: any, ctx: VulkanContext): any; }

export const SttAdapter: typeof SttExecutionPlan;
export const DiffusionAdapter: typeof DiffusionExecutionPlan;
export const BitnetAdapter: typeof BitnetExecutionPlan;
export const LlamaCppAdapter: typeof LlamaCppExecutionPlan;
export const TtsAdapter: typeof TtsExecutionPlan;
export const VisionAdapter: typeof VisionExecutionPlan;

export interface SubprocessOptions {
  executable: string;
  args?: string[];
  env?: Record<string, string>;
  runtimeEnv?: {
    LD_LIBRARY_PATH?: string;
    VK_ICD_FILENAMES?: string;
  };
  allowedRuntimeRoots?: string[];
  cwd?: string;
  timeoutMs?: number;
  gracePeriodMs?: number;
  maxStdoutBytes?: number;
  maxStderrBytes?: number;
  maxTelemetryBytes?: number;
  maxTelemetryLineBytes?: number;
  maxTelemetryMessages?: number;
  signal?: AbortSignal;
  backendRequested?: string | null;
}

export interface SubprocessMetrics {
  processingDurationMs?: number;
  inputDurationMs?: number;
  tokensPerSecond?: number;
  rtf?: number;
}

export interface SubprocessTelemetryError {
  code: string;
  lineNumber: number;
  error?: string;
}

export interface SubprocessResult {
  spawned: boolean;
  pid: number | null;
  exitCode: number | null;
  signal: string | null;
  stdout: string;
  stderr: string;
  durationMs: number;
  terminationReason: "COMPLETED" | "EXIT_NONZERO" | "OUTPUT_LIMIT_EXCEEDED" | "ABORTED" | "TIMEOUT" | "SPAWN_ERROR";
  secondaryReasons: string[];
  timedOut: boolean;
  aborted: boolean;
  stdoutTruncated: boolean;
  stderrTruncated: boolean;
  spawnErrorCode: string | null;
  termSignalAttempted: boolean;
  termSignalDelivered: boolean;
  killSignalAttempted: boolean;
  killSignalDelivered: boolean;
  termSignalSent: boolean;
  killSignalSent: boolean;
  backendRequested: string | null;

  backendConfirmed: string | null;
  backendDevice: string | null;
  verificationSource: string;
  telemetryErrors: SubprocessTelemetryError[];
  metrics: SubprocessMetrics;
  backendHints: string[];
}

export function executeSubprocess(options: SubprocessOptions): Promise<SubprocessResult>;
export function validateExecutable(executable: string): Promise<string>;
export function toSubprocessOptions(plan: any, executable: string, extraArgs?: string[], options?: Partial<SubprocessOptions>): SubprocessOptions;
