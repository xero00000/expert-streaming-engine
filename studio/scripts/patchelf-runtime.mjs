#!/usr/bin/env node
import { spawnSync } from "node:child_process";
import { existsSync } from "node:fs";
import { basename, resolve } from "node:path";
import { fileURLToPath } from "node:url";

export function protectsNvidiaLibrary(path) {
  return /^lib(?:cudart|cublas(?:Lt)?|nccl)\.so(?:\.\d+)*$/.test(basename(path));
}

if (process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  const args = process.argv.slice(2);
  const target = args.at(-1) || "";
  const real = process.env.ESE_REAL_PATCHELF;
  if (!real || resolve(real) === fileURLToPath(import.meta.url)) {
    throw new Error("ESE_REAL_PATCHELF must identify the system patchelf executable.");
  }
  if (protectsNvidiaLibrary(target)) {
    if (!existsSync(target)) throw new Error(`NVIDIA library ${target} is missing.`);
    if (args.length === 3 && args[0] === "--set-rpath") {
      console.error(`ESE: keeping NVIDIA library ${basename(target)} unmodified.`);
      process.exit(0);
    }
    if (!args[0]?.startsWith("--print-") && !["--version", "--help"].includes(args[0])) {
      throw new Error(`Refusing to modify NVIDIA library ${basename(target)}.`);
    }
  }
  const result = spawnSync(real, args, { stdio: "inherit" });
  if (result.error) throw result.error;
  process.exit(result.status ?? 1);
}
