import { beforeEach, describe, expect, it } from "vitest";
import { env, exports } from "cloudflare:workers";

const benchmark = (eventId: string, overrides: Record<string, unknown> = {}) => ({
  schemaVersion: 1,
  eventId,
  installationId: "local-installation-id",
  recordedAtEpoch: 1_787_354_400,
  eseVersion: "0.1.0",
  platform: "linux",
  cpuModel: "Test CPU",
  logicalCpus: 16,
  ramGib: 64,
  gpus: ["Test GPU"],
  modelSignature: "test-model-signature",
  modelSizeBytes: 12_345_678,
  architecture: "test-moe",
  quantization: "Q4_K_M",
  preset: "standard",
  objective: "safe-context-speed",
  safeMargin: 0.9,
  verifiedMaxContext: 65_536,
  promotedContext: 58_368,
  bestKvType: "q8_0",
  bestBatchSize: 256,
  bestTokensPerSecond: 42.5,
  trials: [{
    phase: "speed",
    context: 58_368,
    kvType: "q8_0",
    batchSize: 256,
    stable: true,
    tokensPerSecond: 42.5,
    elapsedSeconds: 12.3,
  }],
  ...overrides,
});

const submit = (payload: unknown, headers: Record<string, string> = {}) => {
  const body = JSON.stringify(payload);
  return exports.default.fetch("https://collector.example/v1/benchmarks", {
    method: "POST",
    headers: {
      "content-type": "application/json",
      "content-length": String(new TextEncoder().encode(body).byteLength),
      ...headers,
    },
    body,
  });
};

beforeEach(async () => {
  await env.DB.exec("DELETE FROM benchmark_submissions");
});

describe("benchmark collector", () => {
  it("reports health and rejects unknown routes", async () => {
    const health = await exports.default.fetch("https://collector.example/health");
    expect(health.status).toBe(200);
    expect(await health.json()).toEqual({ status: "ok" });

    const missing = await exports.default.fetch("https://collector.example/missing");
    expect(missing.status).toBe(404);
  });

  it("enforces bounded JSON requests and the benchmark schema", async () => {
    const noLength = await exports.default.fetch("https://collector.example/v1/benchmarks", {
      method: "POST",
    });
    expect(noLength.status).toBe(411);

    const oversized = await submit(benchmark("oversized"), { "content-length": String(128 * 1024 + 1) });
    expect(oversized.status).toBe(413);

    const invalid = await submit(benchmark("invalid", { promotedContext: 128 }));
    expect(invalid.status).toBe(422);
  });

  it("hashes installation identity, stores sanitized data, and rejects duplicates", async () => {
    const first = await submit(benchmark("event-one"));
    expect(first.status).toBe(202);

    const duplicate = await submit(benchmark("event-one"));
    expect(duplicate.status).toBe(409);

    const row = await env.DB.prepare(
      "SELECT installation_hash, private_payload_json FROM benchmark_submissions WHERE event_id = ?",
    ).bind("event-one").first<{ installation_hash: string; private_payload_json: string }>();
    expect(row?.installation_hash).toMatch(/^[a-f0-9]{64}$/);
    expect(row?.installation_hash).not.toContain("local-installation-id");
    expect(row?.private_payload_json).not.toContain("installationId");
  });

  it("keeps exports private and publishes only groups with three samples", async () => {
    for (const id of ["group-one", "group-two", "group-three"]) {
      expect((await submit(benchmark(id))).status).toBe(202);
    }

    const unauthorized = await exports.default.fetch("https://collector.example/v1/export", {
      headers: { authorization: "Bearer wrong-token" },
    });
    expect(unauthorized.status).toBe(401);

    const exported = await exports.default.fetch("https://collector.example/v1/export", {
      headers: { authorization: "Bearer test-export-token" },
    });
    expect(exported.status).toBe(200);
    const payload = await exported.json<{
      minimumGroupSize: number;
      benchmarks: Array<Record<string, unknown>>;
    }>();
    expect(payload.minimumGroupSize).toBe(3);
    expect(payload.benchmarks).toHaveLength(1);
    expect(payload.benchmarks[0].samples).toBe(3);
    expect(JSON.stringify(payload)).not.toContain("installation");
    expect(JSON.stringify(payload)).not.toContain("eventId");
  });
});
