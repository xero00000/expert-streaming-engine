import assert from "node:assert/strict";
import { existsSync, mkdtempSync, mkdirSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import test from "node:test";
import { stageWindowsCpuRuntime } from "../scripts/windows-cpu-runtime.mjs";

function fixture(callback) {
  const root = mkdtempSync(join(tmpdir(), "ese-windows-cpu-runtime-"));
  try {
    const repoRoot = join(root, "repo");
    const destination = join(root, "runtime");
    const build = join(repoRoot, "build-package-cpu");
    const release = join(build, "bin", "Release");
    mkdirSync(release, { recursive: true });
    mkdirSync(join(destination, "build", "bin"), { recursive: true });
    writeFileSync(join(destination, "build", "bin", "ggml.dll"), "CUDA runtime preserved");
    writeFileSync(join(build, "CMakeCache.txt"), "GGML_CUDA:BOOL=OFF\n");
    writeFileSync(join(release, "llama-server.exe"), "CPU server");
    writeFileSync(join(release, "ggml.dll"), "CPU ggml");
    writeFileSync(join(release, "llama.dll"), "CPU llama");
    callback({ root, repoRoot, destination, build, release });
  } finally {
    rmSync(root, { recursive: true, force: true });
  }
}

test("Windows CPU runtime is isolated from same-named CUDA DLLs", () => {
  fixture(({ repoRoot, destination }) => {
    assert.equal(stageWindowsCpuRuntime({ repoRoot, destination }), true);
    const target = join(destination, "build", "cpu", "bin");
    assert.equal(readFileSync(join(target, "ggml.dll"), "utf8"), "CPU ggml");
    assert.equal(readFileSync(join(destination, "build", "bin", "ggml.dll"), "utf8"), "CUDA runtime preserved");
    assert.equal(JSON.parse(readFileSync(join(target, "ese-runtime.json"), "utf8")).cuda, false);
    assert.ok(existsSync(join(target, "llama-server.exe")));
  });
});

test("Windows CPU staging rejects a CUDA-configured build", () => {
  fixture(({ repoRoot, destination, build }) => {
    writeFileSync(join(build, "CMakeCache.txt"), "GGML_CUDA:BOOL=ON\n");
    assert.throws(() => stageWindowsCpuRuntime({ repoRoot, destination }), /GGML_CUDA=OFF/);
  });
});

test("Windows CPU staging rejects CUDA DLL payload contamination", () => {
  fixture(({ repoRoot, destination, release }) => {
    writeFileSync(join(release, "cudart64_12.dll"), "unexpected CUDA runtime");
    assert.throws(() => stageWindowsCpuRuntime({ repoRoot, destination }), /CUDA DLL cannot be staged/);
  });
});

test("Absent optional CPU bundle preserves existing staging; explicit missing build fails", () => {
  fixture(({ repoRoot, destination, build }) => {
    rmSync(build, { recursive: true });
    assert.equal(stageWindowsCpuRuntime({ repoRoot, destination }), false);
    assert.throws(() => stageWindowsCpuRuntime({ repoRoot, destination, buildDirectory: build }), /missing llama-server/);
  });
});
