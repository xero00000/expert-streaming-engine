import { createHash } from "node:crypto";
import { spawnSync } from "node:child_process";
import { accessSync, chmodSync, constants, cpSync, existsSync, mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { homedir, tmpdir } from "node:os";
import { delimiter, dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const studioRoot = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const linuxdeployName = "linuxdeploy-x86_64.AppImage";
const linuxdeployUrl = `https://github.com/tauri-apps/binary-releases/releases/download/linuxdeploy/${linuxdeployName}`;
// Tauri 2.11.4's linuxdeploy 659c9db, with its AppImage identification bytes
// cleared in the same way as Tauri's prepare_tools().
const linuxdeploySha256 = "20eebde3c18ae2e44279bd624fc72482503aece216d5d77f10932235342f71c1";

export function linuxPackageConfiguration(buildRoot, requestedVariant) {
  const cache = readFileSync(join(buildRoot, "CMakeCache.txt"), "utf8");
  const configured = cache.match(/^GGML_CUDA:BOOL=(ON|OFF)\s*$/m);
  if (!configured) throw new Error("Linux packaging requires an explicit GGML_CUDA build configuration.");
  const variant = configured[1] === "ON" ? "cuda" : "cpu";
  if (requestedVariant && requestedVariant !== variant) {
    throw new Error(`Requested ${requestedVariant} package does not match the ${variant} runtime build.`);
  }
  return { variant, config: join(studioRoot, "src-tauri", `tauri.linux-${variant}.conf.json`) };
}

function executable(name) {
  for (const directory of (process.env.PATH || "").split(delimiter)) {
    const path = resolve(directory, name);
    try { accessSync(path, constants.X_OK); return path; } catch { /* try the next PATH entry */ }
  }
  throw new Error(`Linux packaging requires ${name}.`);
}

export function packagingEnvironment(cacheRoot, linuxdeploy, patchelf, base = process.env) {
  return {
    ...base,
    XDG_CACHE_HOME: cacheRoot,
    NO_STRIP: "1",
    APPIMAGE_EXTRACT_AND_RUN: "1",
    ESE_REAL_LINUXDEPLOY: linuxdeploy,
    ESE_LINUXDEPLOY_LAUNCHER: join(cacheRoot, "tauri", linuxdeployName),
    ESE_REAL_PATCHELF: patchelf,
    PATCHELF: join(studioRoot, "scripts", "patchelf-runtime.mjs"),
  };
}

export function gtkPluginWithRuntimeExclusions(source) {
  const callback = 'env LINUXDEPLOY_PLUGIN_MODE=1 "$LINUXDEPLOY"';
  if (!source.includes(callback)) throw new Error("The GTK packaging plugin has an unsupported linuxdeploy callback.");
  return source.replaceAll(callback, 'env LINUXDEPLOY_PLUGIN_MODE=1 "${ESE_LINUXDEPLOY_LAUNCHER:-$LINUXDEPLOY}"');
}

async function main() {
  if (process.platform !== "linux" || process.arch !== "x64") {
    throw new Error("Linux release packaging tools are pinned for x86_64.");
  }
  const buildRoot = process.env.ESE_RUNTIME_BUILD_DIR
    ? resolve(process.env.ESE_RUNTIME_BUILD_DIR)
    : ["build-package", "build-release-runtime", "build-ci", "build"]
      .map((name) => resolve(studioRoot, "..", name))
      .find((root) => existsSync(join(root, "bin", "llama-server")));
  if (!buildRoot) throw new Error("Build the Linux ESE runtime before packaging.");
  const packageConfig = linuxPackageConfiguration(buildRoot, process.env.ESE_PACKAGE_VARIANT);
  const realPatchelf = process.env.ESE_REAL_PATCHELF || process.env.PATCHELF || executable("patchelf");
  const parentCache = process.env.XDG_CACHE_HOME || join(homedir(), ".cache");
  const cacheRoot = mkdtempSync(join(tmpdir(), "ese-linux-package-"));
  const tools = join(cacheRoot, "tauri");
  mkdirSync(tools);
  try {
    const cached = join(parentCache, "tauri", linuxdeployName);
    const bytes = existsSync(cached) ? readFileSync(cached) : Buffer.from(await (await fetch(linuxdeployUrl)).arrayBuffer());
    if (bytes.subarray(0, 4).toString("hex") !== "7f454c46") throw new Error("Downloaded linuxdeploy is not an ELF executable.");
    bytes.fill(0, 8, 11);
    if (createHash("sha256").update(bytes).digest("hex") !== linuxdeploySha256) {
      throw new Error("linuxdeploy does not match the pinned Tauri packaging tool.");
    }
    const realLinuxdeploy = join(tools, "ese-linuxdeploy-original.AppImage");
    writeFileSync(realLinuxdeploy, bytes, { mode: 0o755 });
    for (const name of ["AppRun-x86_64", "linuxdeploy-plugin-gstreamer.sh", "linuxdeploy-plugin-appimage.AppImage"]) {
      const source = join(parentCache, "tauri", name);
      if (existsSync(source)) cpSync(source, join(tools, name));
    }
    const gtkCached = join(parentCache, "tauri", "linuxdeploy-plugin-gtk.sh");
    const gtkSource = existsSync(gtkCached) ? readFileSync(gtkCached, "utf8")
      : await (await fetch("https://raw.githubusercontent.com/tauri-apps/linuxdeploy-plugin-gtk/master/linuxdeploy-plugin-gtk.sh")).text();
    // The GTK plugin invokes linuxdeploy again to collect its dependencies.
    // Route that invocation through the same exclusions as the outer call.
    writeFileSync(join(tools, "linuxdeploy-plugin-gtk.sh"), gtkPluginWithRuntimeExclusions(gtkSource), { mode: 0o755 });
    const compiled = spawnSync(process.env.CC || executable("cc"), [
      "-O2", "-Wall", "-Wextra", "-Werror", join(studioRoot, "scripts", "linuxdeploy-runtime.c"),
      "-o", join(tools, linuxdeployName),
    ], { stdio: "inherit" });
    if (compiled.error) throw compiled.error;
    if (compiled.status !== 0) throw new Error("Could not build the linuxdeploy launcher.");
    chmodSync(join(studioRoot, "scripts", "patchelf-runtime.mjs"), 0o755);
    const result = spawnSync("pnpm", ["tauri", "build", "--config", packageConfig.config, ...process.argv.slice(2)], {
      cwd: studioRoot,
      stdio: "inherit",
      env: packagingEnvironment(cacheRoot, realLinuxdeploy, realPatchelf, {
        ...process.env, ESE_RUNTIME_BUILD_DIR: buildRoot, ESE_PACKAGE_VARIANT: packageConfig.variant,
      }),
    });
    if (result.error) throw result.error;
    process.exitCode = result.status ?? 1;
  } finally {
    rmSync(cacheRoot, { recursive: true, force: true });
  }
}

if (process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url)) await main();
