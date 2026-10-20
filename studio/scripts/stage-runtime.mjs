import { chmodSync, cpSync, existsSync, mkdirSync, readFileSync, readdirSync, realpathSync, rmSync, writeFileSync } from "node:fs";
import { spawnSync } from "node:child_process";
import { basename, dirname, isAbsolute, join, relative, resolve, sep } from "node:path";
import { fileURLToPath } from "node:url";
import { stageWindowsCpuRuntime } from "./windows-cpu-runtime.mjs";

const studioRoot = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const repoRoot = resolve(studioRoot, "..");
const destination = join(studioRoot, "runtime", "ese");
const serverName = process.platform === "win32" ? "llama-server.exe" : "llama-server";
function within(path, root) {
  const value = relative(root, path);
  return value === "" || (value !== ".." && !value.startsWith(`..${sep}`) && !isAbsolute(value));
}
const buildRoots = [
  ...(process.env.ESE_RUNTIME_BUILD_DIR ? [resolve(process.env.ESE_RUNTIME_BUILD_DIR)]
    : ["build-package", "build-release-runtime", "build-ci", "build"].map((name) => join(repoRoot, name))),
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

const serverBuildRoot = buildRoots.find((root) => within(server, root));
const cache = join(serverBuildRoot, "CMakeCache.txt");
const configuredCuda = existsSync(cache) && readFileSync(cache, "utf8").match(/^GGML_CUDA:BOOL=(ON|OFF)\s*$/m);
if (!configuredCuda) throw new Error("Runtime staging requires an explicit GGML_CUDA build configuration.");
const runtimeCuda = configuredCuda[1] === "ON";
const packageVariant = runtimeCuda ? "cuda" : "cpu";
if (process.env.ESE_PACKAGE_VARIANT && process.env.ESE_PACKAGE_VARIANT !== packageVariant) {
  throw new Error(`Requested ${process.env.ESE_PACKAGE_VARIANT} package does not match the ${packageVariant} runtime build.`);
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

  writeFileSync(
    join(destination, "build", "bin", "ese-runtime.json"),
    `${JSON.stringify({ cuda: runtimeCuda, package_variant: packageVariant }, null, 2)}\n`,
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

if (process.platform === "win32" && runtimeCuda) {
  stageWindowsCpuRuntime({
    repoRoot,
    destination,
    buildDirectory: process.env.ESE_RUNTIME_CPU_BUILD_DIR,
  });
}

if (process.platform === "linux") {
  if (spawnSync("patchelf", ["--version"]).status !== 0) {
    throw new Error("Linux runtime staging requires patchelf.");
  }
  const runtimeBin = join(destination, "build", "bin");
  const projectElfs = readdirSync(runtimeBin).filter((entry) => entry === serverName || entry.includes(".so"));
  if (runtimeCuda) {
    const vendorLibraries = new Map();
    for (const entry of projectElfs) {
      const dependencies = spawnSync("ldd", [join(runtimeBin, entry)], { encoding: "utf8" });
      if (dependencies.status !== 0) throw new Error(`Could not inspect CUDA runtime dependencies for ${entry}.`);
      for (const line of dependencies.stdout.split("\n")) {
        const match = line.match(/^\s*(lib(?:cudart|cublas(?:Lt)?|nccl)\.so(?:\.\d+)*)\s+=>\s+(.+)$/);
        if (!match) continue;
        const source = match[2].replace(/\s+\(0x[0-9a-f]+\)\s*$/i, "").trim();
        if (!isAbsolute(source) || !existsSync(source)) {
          throw new Error(`Required CUDA redistributable ${match[1]} could not be resolved.`);
        }
        const actual = realpathSync(source);
        const previous = vendorLibraries.get(match[1]);
        if (previous && previous !== actual) throw new Error(`Conflicting runtime locations for ${match[1]}.`);
        vendorLibraries.set(match[1], actual);
      }
    }
    if (![...vendorLibraries.keys()].some((name) => name.startsWith("libcudart.so."))) {
      throw new Error("CUDA runtime staging found no CUDA runtime redistributable.");
    }
    for (const [soname, source] of vendorLibraries) {
      // Copy the resolved file under the loader's SONAME. Keep NVIDIA's
      // binaries byte-for-byte intact and leave libcuda to the host driver.
      cpSync(source, join(runtimeBin, soname));
    }
  }
  for (const entry of projectElfs) {
    const patched = spawnSync("patchelf", ["--set-rpath", "$ORIGIN", join(runtimeBin, entry)]);
    if (patched.status !== 0) throw new Error(`Could not make ${entry} relocatable.`);
  }
}

console.log(`Staged ESE launcher and native runtime from ${server}`);
