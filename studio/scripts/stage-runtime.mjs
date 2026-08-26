import { chmodSync, cpSync, existsSync, mkdirSync, readFileSync, readdirSync, rmSync, writeFileSync } from "node:fs";
import { spawnSync } from "node:child_process";
import { basename, dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const studioRoot = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const repoRoot = resolve(studioRoot, "..");
const destination = join(studioRoot, "runtime", "ese");
const serverName = process.platform === "win32" ? "llama-server.exe" : "llama-server";
const buildRoots = [
  ...(process.env.ESE_RUNTIME_BUILD_DIR ? [resolve(process.env.ESE_RUNTIME_BUILD_DIR)] : []),
  ...["build-package", "build-release-runtime", "build-ci", "build"].map((name) => join(repoRoot, name)),
];

const serverCandidates = buildRoots.flatMap((root) => [
  join(root, "bin", serverName),
  join(root, "bin", "Release", serverName),
]);
const server = serverCandidates.find(existsSync);
if (!server) {
  throw new Error(
    `ESE runtime is not built. Run ${process.platform === "win32" ? "py -3" : "python3"} ../ese build --backend cpu --build-dir ../build-package first.`,
  );
}

rmSync(destination, { recursive: true, force: true });
mkdirSync(join(destination, "tools"), { recursive: true });
mkdirSync(join(destination, "build", "bin"), { recursive: true });
cpSync(join(repoRoot, "ese"), join(destination, "ese"));
cpSync(join(repoRoot, "ese.cmd"), join(destination, "ese.cmd"));
cpSync(join(repoRoot, "tools", "ese.py"), join(destination, "tools", "ese.py"));
cpSync(join(repoRoot, "tools", "hardware_profile.py"), join(destination, "tools", "hardware_profile.py"));
cpSync(join(repoRoot, "tools", "__init__.py"), join(destination, "tools", "__init__.py"));
if (process.platform === "win32") {
  const standaloneLauncher = join(repoRoot, "dist", "ese.exe");
  if (!existsSync(standaloneLauncher)) {
    throw new Error("Standalone ESE launcher is missing. Build dist/ese.exe with PyInstaller first.");
  }
  cpSync(standaloneLauncher, join(destination, "ese.exe"));
}
cpSync(server, join(destination, "build", "bin", serverName));
if (process.platform !== "win32") {
  chmodSync(join(destination, "ese"), 0o755);
  chmodSync(join(destination, "build", "bin", serverName), 0o755);
}

const runtimeExtensions = process.platform === "win32" ? [".dll"] : [".so"];
const serverBuildRoot = buildRoots.find((root) => server.startsWith(root));
if (serverBuildRoot) {
  const pending = [serverBuildRoot];
  while (pending.length) {
    const directory = pending.pop();
    for (const entry of readdirSync(directory, { withFileTypes: true })) {
      const source = join(directory, entry.name);
      if (entry.isDirectory()) {
        pending.push(source);
      } else if (runtimeExtensions.some((extension) => entry.name.includes(extension))) {
        cpSync(source, join(destination, "build", "bin", basename(source)));
      }
    }
  }

  const cache = join(serverBuildRoot, "CMakeCache.txt");
  const cuda = existsSync(cache) && /^GGML_CUDA:BOOL=ON\s*$/m.test(readFileSync(cache, "utf8"));
  writeFileSync(
    join(destination, "build", "bin", "ese-runtime.json"),
    `${JSON.stringify({ cuda }, null, 2)}\n`,
  );
}

if (process.platform === "win32" && process.env.CUDA_PATH) {
  const cudaBin = join(process.env.CUDA_PATH, "bin");
  if (existsSync(cudaBin)) {
    for (const entry of readdirSync(cudaBin)) {
      if (/^(?:cudart64_|cublas64_|cublasLt64_).*\.dll$/i.test(entry)) {
        cpSync(join(cudaBin, entry), join(destination, "build", "bin", entry));
      }
    }
  }
}

if (process.platform === "linux" && spawnSync("patchelf", ["--version"]).status === 0) {
  for (const entry of readdirSync(join(destination, "build", "bin"))) {
    if (entry === serverName || entry.includes(".so")) {
      const target = join(destination, "build", "bin", entry);
      const original = spawnSync("patchelf", ["--print-rpath", target], { encoding: "utf8" });
      const externalPaths = original.status === 0
        ? original.stdout.trim().split(":").filter((path) => path && !path.startsWith(repoRoot))
        : [];
      const runtimePath = ["$ORIGIN", ...externalPaths].join(":");
      const patched = spawnSync("patchelf", ["--set-rpath", runtimePath, target]);
      if (patched.status !== 0) throw new Error(`Could not make ${entry} relocatable.`);
    }
  }
}

console.log(`Staged ESE launcher and native runtime from ${server}`);
