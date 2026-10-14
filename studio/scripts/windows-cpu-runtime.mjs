import { cpSync, existsSync, mkdirSync, readFileSync, readdirSync, writeFileSync } from "node:fs";
import { basename, join, resolve } from "node:path";

export function stageWindowsCpuRuntime({ repoRoot, destination, buildDirectory }) {
  const roots = buildDirectory
    ? [resolve(buildDirectory)]
    : ["build-package-cpu", "build-release-runtime-cpu"].map((name) => join(repoRoot, name));
  const selected = roots.map((root) => ({
    root,
    server: [join(root, "bin", "llama-server.exe"), join(root, "bin", "Release", "llama-server.exe")].find(existsSync),
  })).find((candidate) => candidate.server);
  if (!selected) {
    if (buildDirectory) throw new Error("Requested Windows CPU runtime build is missing llama-server.exe.");
    return false;
  }
  const cache = join(selected.root, "CMakeCache.txt");
  if (!existsSync(cache) || !/^GGML_CUDA:BOOL=OFF\s*$/m.test(readFileSync(cache, "utf8"))) {
    throw new Error("Windows CPU fallback must be configured with GGML_CUDA=OFF.");
  }
  const target = join(destination, "build", "cpu", "bin");
  mkdirSync(target, { recursive: true });
  cpSync(selected.server, join(target, "llama-server.exe"));
  const pending = [selected.root];
  while (pending.length) {
    const directory = pending.pop();
    for (const entry of readdirSync(directory, { withFileTypes: true })) {
      const source = join(directory, entry.name);
      if (entry.isDirectory()) pending.push(source);
      else if (entry.name.toLowerCase().endsWith(".dll")) {
        if (/^(?:nvcuda|cudart|cublas|ggml-cuda)/i.test(entry.name)) {
          throw new Error(`CUDA DLL cannot be staged in the CPU-only runtime: ${entry.name}`);
        }
        cpSync(source, join(target, basename(source)));
      }
    }
  }
  writeFileSync(join(target, "ese-runtime.json"), `${JSON.stringify({ cuda: false }, null, 2)}\n`);
  return true;
}
