import assert from "node:assert/strict";
import { spawnSync } from "node:child_process";
import { chmodSync, existsSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { dirname, join, resolve } from "node:path";
import test from "node:test";
import { fileURLToPath } from "node:url";
import { gtkPluginWithRuntimeExclusions, packagingEnvironment, linuxPackageConfiguration } from "../scripts/package-linux.mjs";
import { protectsNvidiaLibrary } from "../scripts/patchelf-runtime.mjs";

const studioRoot = resolve(dirname(fileURLToPath(import.meta.url)), "..");

test("Linux package configuration preserves the installed runtime variant in its update feed", () => {
  const root = mkdtempSync(join(tmpdir(), "ese-package-variant-"));
  try {
    for (const [setting, variant] of [["OFF", "cpu"], ["ON", "cuda"]]) {
      writeFileSync(join(root, "CMakeCache.txt"), `GGML_CUDA:BOOL=${setting}\n`);
      const selected = linuxPackageConfiguration(root, variant);
      assert.equal(selected.variant, variant);
      const config = JSON.parse(readFileSync(selected.config, "utf8"));
      assert.deepEqual(config.plugins.updater.endpoints,
        [`https://github.com/xero00000/expert-streaming-engine/releases/latest/download/latest-${variant}.json`]);
      assert.throws(() => linuxPackageConfiguration(root, variant === "cpu" ? "cuda" : "cpu"), /does not match/);
    }
    writeFileSync(join(root, "CMakeCache.txt"), "# unknown configuration\n");
    assert.throws(() => linuxPackageConfiguration(root), /explicit GGML_CUDA/);
  } finally {
    rmSync(root, { recursive: true, force: true });
  }
});

test("Linux package environment preserves signing configuration and prevents stripping", () => {
  const env = packagingEnvironment("/fixture/cache", "/fixture/linuxdeploy", "/fixture/patchelf", {
    NO_STRIP: "0", TAURI_SIGNING_PRIVATE_KEY: "fixture-key", TAURI_SIGNING_PRIVATE_KEY_PASSWORD: "fixture-password",
  });
  assert.equal(env.NO_STRIP, "1");
  assert.equal(env.APPIMAGE_EXTRACT_AND_RUN, "1");
  assert.equal(env.XDG_CACHE_HOME, "/fixture/cache");
  assert.equal(env.ESE_REAL_LINUXDEPLOY, "/fixture/linuxdeploy");
  assert.equal(env.ESE_LINUXDEPLOY_LAUNCHER, "/fixture/cache/tauri/linuxdeploy-x86_64.AppImage");
  assert.equal(env.ESE_REAL_PATCHELF, "/fixture/patchelf");
  assert.equal(env.PATCHELF, join(studioRoot, "scripts", "patchelf-runtime.mjs"));
  assert.equal(env.TAURI_SIGNING_PRIVATE_KEY, "fixture-key");
  assert.equal(env.TAURI_SIGNING_PRIVATE_KEY_PASSWORD, "fixture-password");
});

test("GTK plugin callbacks retain the runtime exclusions", () => {
  const source = '#!/bin/bash\nenv LINUXDEPLOY_PLUGIN_MODE=1 "$LINUXDEPLOY" --appdir="$APPDIR" "${LIBRARIES[@]}"\n';
  assert.equal(gtkPluginWithRuntimeExclusions(source), '#!/bin/bash\nenv LINUXDEPLOY_PLUGIN_MODE=1 "${ESE_LINUXDEPLOY_LAUNCHER:-$LINUXDEPLOY}" --appdir="$APPDIR" "${LIBRARIES[@]}"\n');
  assert.throws(() => gtkPluginWithRuntimeExclusions("different plugin"), /unsupported linuxdeploy callback/);
});

test("NVIDIA protection matches SONAME and versioned runtime files", () => {
  for (const name of ["libcudart.so.12", "libcublas.so.12.8.3.14", "libcublasLt.so.12", "libnccl.so.2.25.1"]) {
    assert.equal(protectsNvidiaLibrary(`/fixture/${name}`), true);
  }
  for (const name of ["libggml.so", "libllama.so", "libmtmd.so", "libcublas-helper.so", "llama-server"]) {
    assert.equal(protectsNvidiaLibrary(`/fixture/${name}`), false);
  }
});

test("patchelf adapter preserves NVIDIA files, delegates reads and project writes, and rejects other vendor writes", () => {
  const root = mkdtempSync(join(tmpdir(), "ese-patchelf-adapter-"));
  try {
    const log = join(root, "calls.jsonl");
    const real = join(root, "real-patchelf.mjs");
    const vendor = join(root, "libcudart.so.12");
    const project = join(root, "libggml.so");
    writeFileSync(vendor, "original vendor bytes\n");
    writeFileSync(project, "project bytes\n");
    writeFileSync(real, `#!${process.execPath}\nimport fs from "node:fs";fs.appendFileSync(${JSON.stringify(log)},JSON.stringify(process.argv.slice(2))+"\\n");if(process.argv[2]==="--set-rpath")process.exit(7);\n`);
    chmodSync(real, 0o755);
    const invoke = (args) => spawnSync(process.execPath, [join(studioRoot, "scripts", "patchelf-runtime.mjs"), ...args], {
      env: { ...process.env, ESE_REAL_PATCHELF: real }, encoding: "utf8",
    });
    assert.equal(invoke(["--set-rpath", "$ORIGIN", vendor]).status, 0);
    assert.equal(existsSync(log), false);
    assert.equal(readFileSync(vendor, "utf8"), "original vendor bytes\n");
    assert.equal(invoke(["--print-rpath", vendor]).status, 0);
    assert.equal(invoke(["--set-rpath", "$ORIGIN", project]).status, 7);
    assert.notEqual(invoke(["--remove-rpath", vendor]).status, 0);
    assert.deepEqual(readFileSync(log, "utf8").trim().split("\n").map(JSON.parse), [
      ["--print-rpath", vendor], ["--set-rpath", "$ORIGIN", project],
    ]);
  } finally {
    rmSync(root, { recursive: true, force: true });
  }
});

test("linuxdeploy launcher adds literal exclusions and forwards the bundler arguments", { skip: process.platform !== "linux" }, () => {
  const root = mkdtempSync(join(tmpdir(), "ese-linuxdeploy-launcher-"));
  try {
    const wrapper = join(root, "linuxdeploy.AppImage");
    const compiled = spawnSync("cc", ["-Wall", "-Wextra", "-Werror", join(studioRoot, "scripts", "linuxdeploy-runtime.c"), "-o", wrapper], { encoding: "utf8" });
    assert.equal(compiled.status, 0, compiled.stderr);
    const bytes = readFileSync(wrapper);
    bytes.fill(0, 8, 11);
    writeFileSync(wrapper, bytes);
    chmodSync(wrapper, 0o755);
    const log = join(root, "args.json");
    const real = join(root, "real-linuxdeploy.mjs");
    writeFileSync(real, `#!${process.execPath}\nimport fs from "node:fs";fs.writeFileSync(${JSON.stringify(log)},JSON.stringify(process.argv.slice(2)));\n`);
    chmodSync(real, 0o755);
    const result = spawnSync(wrapper, ["--appimage-extract-and-run", "--appdir", "/fixture/space directory", "--plugin", "gtk", "--output", "appimage"], {
      env: { ...process.env, ESE_REAL_LINUXDEPLOY: real }, encoding: "utf8",
    });
    assert.equal(result.status, 0, result.stderr);
    const args = JSON.parse(readFileSync(log, "utf8"));
    assert.equal(args.filter((arg) => arg === "--appimage-extract-and-run").length, 1);
    for (const pattern of ["libcuda.so*", "libnvidia-*", "libcudart.so*", "libcublas.so*", "libcublasLt.so*", "libnccl.so*", "libggml.so*", "libllama.so*", "libmtmd.so*", "libgomp.so*"]) {
      assert.ok(args.some((arg, index) => arg === "--exclude-library" && args[index + 1] === pattern), pattern);
    }
    assert.deepEqual(args.slice(-6), ["--appdir", "/fixture/space directory", "--plugin", "gtk", "--output", "appimage"]);
    const callback = spawnSync(wrapper, ["--appdir=/fixture/plugin", "--library=/fixture/gtk.so"], {
      env: { ...process.env, ESE_REAL_LINUXDEPLOY: "/not-used-in-plugin-mode", LINUXDEPLOY_PLUGIN_MODE: "1", LINUXDEPLOY: real }, encoding: "utf8",
    });
    assert.equal(callback.status, 0, callback.stderr);
    const callbackArgs = JSON.parse(readFileSync(log, "utf8"));
    assert.ok(!callbackArgs.includes("--appimage-extract-and-run"));
    assert.deepEqual(callbackArgs.slice(-2), ["--appdir=/fixture/plugin", "--library=/fixture/gtk.so"]);
    for (const pattern of ["libcuda.so*", "libnvidia-*", "libcudart.so*", "libnccl.so*"]) {
      assert.ok(callbackArgs.some((arg, index) => arg === "--exclude-library" && callbackArgs[index + 1] === pattern), pattern);
    }
  } finally {
    rmSync(root, { recursive: true, force: true });
  }
});
