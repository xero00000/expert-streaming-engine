type Benchmark = {
  schemaVersion: number;
  eventId: string;
  installationId: string;
  recordedAtEpoch: number;
  eseVersion: string;
  platform: string;
  cpuModel?: string;
  logicalCpus: number;
  ramGib?: number;
  gpus: string[];
  modelSignature: string;
  modelSizeBytes?: number;
  architecture?: string;
  quantization?: string;
  preset: string;
  objective: string;
  safeMargin: number;
  verifiedMaxContext: number;
  promotedContext: number;
  bestKvType: string;
  bestBatchSize: number;
  bestTokensPerSecond: number;
  trials: BenchmarkTrial[];
};

type BenchmarkTrial = {
  phase: string;
  context: number;
  kvType: string;
  batchSize: number;
  stable: boolean;
  tokensPerSecond?: number;
  elapsedSeconds: number;
  errorClass?: string;
};

const json = (body: unknown, status = 200) => new Response(JSON.stringify(body), {
  status,
  headers: { "content-type": "application/json; charset=utf-8", "cache-control": "no-store" },
});

function validString(value: unknown, max = 160): value is string {
  return typeof value === "string" && value.length > 0 && value.length <= max;
}

function validOptionalString(value: unknown, max = 160): value is string | undefined {
  return value === undefined || validString(value, max);
}

function validTrial(value: unknown): value is BenchmarkTrial {
  if (!value || typeof value !== "object") return false;
  const trial = value as Partial<BenchmarkTrial>;
  return validString(trial.phase, 32)
    && Number.isInteger(trial.context) && trial.context! >= 512
    && validString(trial.kvType, 32)
    && Number.isInteger(trial.batchSize) && trial.batchSize! > 0
    && typeof trial.stable === "boolean"
    && (trial.tokensPerSecond === undefined || Number.isFinite(trial.tokensPerSecond))
    && Number.isFinite(trial.elapsedSeconds) && trial.elapsedSeconds! >= 0
    && validOptionalString(trial.errorClass, 32);
}

function isValid(value: unknown): value is Benchmark {
  if (!value || typeof value !== "object") return false;
  const item = value as Partial<Benchmark>;
  return item.schemaVersion === 1
    && validString(item.eventId, 64)
    && validString(item.installationId, 64)
    && validString(item.eseVersion, 32)
    && validString(item.platform, 32)
    && validString(item.modelSignature, 128)
    && validString(item.preset, 32)
    && validString(item.objective, 32)
    && validString(item.bestKvType, 32)
    && validOptionalString(item.cpuModel, 200)
    && validOptionalString(item.architecture, 80)
    && validOptionalString(item.quantization, 40)
    && Number.isFinite(item.recordedAtEpoch)
    && Number.isInteger(item.logicalCpus) && item.logicalCpus! > 0 && item.logicalCpus! <= 2048
    && Number.isFinite(item.safeMargin) && item.safeMargin! >= 0.5 && item.safeMargin! <= 1
    && Number.isInteger(item.verifiedMaxContext) && item.verifiedMaxContext! >= 512
    && Number.isInteger(item.promotedContext) && item.promotedContext! >= 512
    && Number.isInteger(item.bestBatchSize) && item.bestBatchSize! > 0
    && Number.isFinite(item.bestTokensPerSecond) && item.bestTokensPerSecond! > 0 && item.bestTokensPerSecond! < 1_000_000
    && Array.isArray(item.gpus) && item.gpus.length <= 16 && item.gpus.every((gpu) => validString(gpu, 200))
    && (item.ramGib === undefined || (Number.isInteger(item.ramGib) && item.ramGib! > 0))
    && (item.modelSizeBytes === undefined || (Number.isInteger(item.modelSizeBytes) && item.modelSizeBytes! > 0))
    && Array.isArray(item.trials) && item.trials.length <= 200 && item.trials.every(validTrial);
}

async function hashInstallation(id: string, salt: string) {
  const bytes = new TextEncoder().encode(`${salt}:${id}`);
  const digest = await crypto.subtle.digest("SHA-256", bytes);
  return [...new Uint8Array(digest)].map((byte) => byte.toString(16).padStart(2, "0")).join("");
}

async function secureEqual(left: string, right: string) {
  const encoder = new TextEncoder();
  const [leftHash, rightHash] = await Promise.all([
    crypto.subtle.digest("SHA-256", encoder.encode(left)),
    crypto.subtle.digest("SHA-256", encoder.encode(right)),
  ]);
  return crypto.subtle.timingSafeEqual(leftHash, rightHash);
}

async function ingest(request: Request, env: Env) {
  const length = Number(request.headers.get("content-length"));
  if (!Number.isInteger(length) || length <= 0) return json({ error: "content-length required" }, 411);
  if (length > 128 * 1024) return json({ error: "payload too large" }, 413);
  let payload: unknown;
  try { payload = await request.json(); } catch { return json({ error: "invalid JSON" }, 400); }
  if (!isValid(payload)) return json({ error: "invalid benchmark schema" }, 422);

  const installationHash = await hashInstallation(payload.installationId, env.INSTALLATION_HASH_SALT);
  const recent = await env.DB.prepare(
    "SELECT COUNT(*) AS count FROM benchmark_submissions WHERE installation_hash = ? AND received_at >= datetime('now', '-1 day')",
  ).bind(installationHash).first<{ count: number }>();
  if ((recent?.count ?? 0) >= 100) return json({ error: "daily submission limit reached" }, 429);

  const privatePayload = {
    schemaVersion: payload.schemaVersion,
    eventId: payload.eventId,
    recordedAtEpoch: payload.recordedAtEpoch,
    eseVersion: payload.eseVersion,
    platform: payload.platform,
    cpuModel: payload.cpuModel,
    logicalCpus: payload.logicalCpus,
    ramGib: payload.ramGib,
    gpus: payload.gpus,
    modelSignature: payload.modelSignature,
    modelSizeBytes: payload.modelSizeBytes,
    architecture: payload.architecture,
    quantization: payload.quantization,
    preset: payload.preset,
    objective: payload.objective,
    safeMargin: payload.safeMargin,
    verifiedMaxContext: payload.verifiedMaxContext,
    promotedContext: payload.promotedContext,
    bestKvType: payload.bestKvType,
    bestBatchSize: payload.bestBatchSize,
    bestTokensPerSecond: payload.bestTokensPerSecond,
    trials: payload.trials,
  };
  const result = await env.DB.prepare(`INSERT OR IGNORE INTO benchmark_submissions (
    event_id, installation_hash, recorded_at_epoch, ese_version, platform, cpu_model,
    logical_cpus, ram_gib, gpus_json, model_signature, model_size_bytes, architecture,
    quantization, preset, objective, safe_margin, verified_max_context, promoted_context,
    best_kv_type, best_batch_size, best_tokens_per_second, private_payload_json
  ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`)
    .bind(
      payload.eventId, installationHash, payload.recordedAtEpoch, payload.eseVersion, payload.platform,
      payload.cpuModel ?? null, payload.logicalCpus, payload.ramGib ?? null, JSON.stringify(payload.gpus),
      payload.modelSignature, payload.modelSizeBytes ?? null, payload.architecture ?? null,
      payload.quantization ?? null, payload.preset, payload.objective, payload.safeMargin,
      payload.verifiedMaxContext, payload.promotedContext, payload.bestKvType, payload.bestBatchSize,
      payload.bestTokensPerSecond, JSON.stringify(privatePayload),
    ).run();
  return result.meta.changes === 0 ? json({ accepted: true, duplicate: true }, 409) : json({ accepted: true }, 202);
}

async function exportAggregates(request: Request, env: Env) {
  const suppliedToken = request.headers.get("authorization")?.replace(/^Bearer\s+/i, "") ?? "";
  if (!await secureEqual(suppliedToken, env.EXPORT_TOKEN)) {
    return json({ error: "unauthorized" }, 401);
  }
  const result = await env.DB.prepare(`SELECT
      architecture, quantization, gpus_json AS gpus, model_size_bytes AS modelSizeBytes,
      promoted_context AS context, best_kv_type AS kvType, best_batch_size AS batchSize,
      COUNT(*) AS samples, ROUND(AVG(best_tokens_per_second), 2) AS averageTokensPerSecond,
      ROUND(MIN(best_tokens_per_second), 2) AS minimumTokensPerSecond,
      ROUND(MAX(best_tokens_per_second), 2) AS maximumTokensPerSecond
    FROM benchmark_submissions
    GROUP BY architecture, quantization, gpus_json, model_size_bytes, promoted_context,
      best_kv_type, best_batch_size
    HAVING COUNT(*) >= 3
    ORDER BY architecture, quantization, averageTokensPerSecond DESC`).all();
  return json({
    schemaVersion: 1,
    generatedAt: new Date().toISOString(),
    minimumGroupSize: 3,
    benchmarks: result.results.map((row) => ({ ...row, gpus: JSON.parse(String(row.gpus)) })),
  });
}

export default {
  async fetch(request: Request, env: Env): Promise<Response> {
    try {
      const url = new URL(request.url);
      if (request.method === "POST" && url.pathname === "/v1/benchmarks") return await ingest(request, env);
      if (request.method === "GET" && url.pathname === "/v1/export") return await exportAggregates(request, env);
      if (request.method === "GET" && url.pathname === "/health") return json({ status: "ok" });
      return json({ error: "not found" }, 404);
    } catch (error) {
      console.error(JSON.stringify({ message: "request failed", error: error instanceof Error ? error.message : "unknown" }));
      return json({ error: "internal error" }, 500);
    }
  },
} satisfies ExportedHandler<Env>;
