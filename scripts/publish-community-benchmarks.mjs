import { readFileSync, writeFileSync } from "node:fs";

const source = process.argv[2] ?? "benchmarks/community.json";
const target = process.argv[3] ?? "COMMUNITY_BENCHMARKS.md";
const data = JSON.parse(readFileSync(source, "utf8"));
const rows = data.benchmarks ?? [];
const lines = [
  "# ESE community benchmarks",
  "",
  "Organized, privacy-preserving results from users who enabled **Help improve ESE**. Raw submissions remain private. Groups with fewer than three verified sweeps are not published.",
  "",
  `Last updated: ${data.generatedAt ?? "unknown"}`,
  "",
  "| Architecture | Quant | GPU configuration | Model size | Context | KV | Batch | Average | Range | Samples |",
  "|---|---|---|---:|---:|---|---:|---:|---:|---:|",
];
for (const row of rows) {
  const size = row.modelSizeBytes ? `${(row.modelSizeBytes / 1024 ** 3).toFixed(1)} GiB` : "—";
  lines.push(`| ${row.architecture ?? "Unknown"} | ${row.quantization ?? "Unknown"} | ${(row.gpus ?? []).join(" + ") || "CPU"} | ${size} | ${Number(row.context).toLocaleString("en-US")} | ${row.kvType} | ${row.batchSize} | ${row.averageTokensPerSecond} tok/s | ${row.minimumTokensPerSecond}–${row.maximumTokensPerSecond} | ${row.samples} |`);
}
if (!rows.length) lines.push("| No groups have reached the three-sample privacy threshold yet. | | | | | | | | | |");
lines.push("", "The machine-readable aggregate is available in [`benchmarks/community.json`](benchmarks/community.json).", "");
writeFileSync(target, lines.join("\n"));
