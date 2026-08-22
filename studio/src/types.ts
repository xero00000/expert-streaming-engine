export type View = "models" | "chat" | "hub" | "apps" | "sweeper" | "settings";

export interface ChatStatus {
  active: boolean;
  modelId?: string;
}

export interface ModelProfile {
  id: string;
  name: string;
  path: string;
  sizeBytes?: number;
  available: boolean;
  architecture?: string;
  quantization?: string;
  context?: number;
  kvType?: string;
  batchSize?: number;
  ubatchSize?: number;
  source: string;
}

export interface AppProfile {
  id: string;
  name: string;
  command: string;
  args: string[];
  workingDirectory?: string;
  endpointAware: boolean;
}

export interface StudioSnapshot {
  platform: string;
  configPath: string;
  onboardingComplete: boolean;
  helpImproveEse: boolean;
  modelRoots: string[];
  models: ModelProfile[];
  apps: AppProfile[];
  eseBinary?: string;
}

export interface TerminalTab {
  id: string;
  sessionId?: string;
  name: string;
  command: string;
  status: "starting" | "running" | "exited" | "error";
  endpointChanged?: boolean;
}

export interface SweepPlan {
  modelPath: string;
  preset: string;
  objective: string;
  candidateContexts: number[];
  kvTypes: string[];
  batchSizes: number[];
  promotedContext: number;
  trialCount: number;
  requiresExclusiveGpu: boolean;
  safeMargin: number;
}

export interface SweepTrialResult {
  phase: string;
  context: number;
  kvType: string;
  batchSize: number;
  stable: boolean;
  tokensPerSecond?: number;
  elapsedSeconds: number;
  error?: string;
  logPath: string;
}

export interface SweepStatus {
  id: string;
  state: "preparing" | "running" | "cancelling" | "cancelled" | "complete" | "failed";
  phase: string;
  request: {
    modelPath: string;
    preset: string;
    objective: string;
    maxContext: number;
    safeMargin: number;
  };
  completedTrials: number;
  totalTrials: number;
  currentLabel: string;
  verifiedMaxContext?: number;
  promotedContext?: number;
  bestKvType?: string;
  bestBatchSize?: number;
  bestTokensPerSecond?: number;
  results: SweepTrialResult[];
  error?: string;
  checkpointPath: string;
}

export interface HubModel {
  id: string;
  downloads: number;
  likes: number;
  lastModified?: string;
  tags: string[];
}

export interface HardwareSummary {
  ramAvailableBytes: number;
  ramTotalBytes: number;
  vramFreeBytes: number;
  vramTotalBytes: number;
  gpuNames: string[];
  residentBudgetBytes: number;
  hybridBudgetBytes: number;
}

export interface QuantVariant {
  id: string;
  quantization: string;
  files: string[];
  fileSizes: Record<string, number>;
  totalSizeBytes: number;
  fit: "Resident" | "Hybrid/cache" | "Streamed";
  fitReason: string;
  recommended: boolean;
}

export interface HubModelDetails {
  model: HubModel;
  revision: string;
  variants: QuantVariant[];
  hardware: HardwareSummary;
}

export interface DownloadStatus {
  id: string;
  state: "downloading" | "complete" | "cancelled" | "failed";
  repoId: string;
  variantId: string;
  currentFile?: string;
  completedBytes: number;
  totalBytes: number;
  bytesPerSecond?: number;
  etaSeconds?: number;
  destination: string;
  error?: string;
}
