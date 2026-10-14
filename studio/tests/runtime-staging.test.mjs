import assert from "node:assert/strict";
import { spawnSync } from "node:child_process";
import { chmodSync, cpSync, existsSync, mkdtempSync, mkdirSync, readFileSync, readdirSync, rmSync, symlinkSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { dirname, join, resolve } from "node:path";
import test from "node:test";
import { fileURLToPath } from "node:url";

const studioRoot = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const script = process.env.ESE_STAGE_TEST_SOURCE || join(studioRoot, "scripts", "stage-runtime.mjs");

function stageFixture(cuda, { missingVendor = false, missingPatcher = false } = {}) {
  const root = mkdtempSync(join(tmpdir(), "ese-runtime-stage-"));
  try {
    const repo = join(root, "repo");
    const scripts = join(repo, "studio", "scripts");
    const build = join(root, "external-build");
    const bin = join(root, "test-bin");
    const log = join(root, "patchelf.jsonl");
    for (const directory of [scripts, join(repo, "tools"), join(build, "bin"), bin]) mkdirSync(directory, { recursive: true });
    cpSync(script, join(scripts, "stage-runtime.mjs"));
    cpSync(join(dirname(script), "windows-cpu-runtime.mjs"), join(scripts, "windows-cpu-runtime.mjs"));
    for (const name of ["ese", "ese.cmd"]) writeFileSync(join(repo, name), "fixture launcher\n");
    for (const name of ["ese.py", "hardware_profile.py", "__init__.py"]) writeFileSync(join(repo, "tools", name), "# fixture\n");
    writeFileSync(join(build, "bin", "llama-server"), "fixture executable\n");
    writeFileSync(join(build, "libggml.so"), "fixture library\n");
    writeFileSync(join(build, "CMakeCache.txt"), `GGML_CUDA:BOOL=${cuda ? "ON" : "OFF"}\n`);
    const vendor = join(root, "vendor-runtime");
    mkdirSync(vendor);
    const vendors = ["libcudart.so.12", "libcublas.so.12", "libcublasLt.so.12", "libnccl.so.2"];
    for (const name of vendors) {
      writeFileSync(join(vendor, `${name}.fixture`), `unmodified vendor ${name}\n`);
      symlinkSync(`${name}.fixture`, join(vendor, name));
    }
    const originalRpath = [join(repo, "build", "src"), join(build, "ggml", "src"), vendor].join(":");
    const patcher = join(bin, "patchelf");
    writeFileSync(patcher, `#!${process.execPath}\nimport fs from "node:fs";\nconst args=process.argv.slice(2);\nif(args[0]==="--version") ${missingPatcher ? "process.exit(1)" : 'console.log("fixture patchelf")'};\nelse if(args[0]==="--print-rpath") console.log(${JSON.stringify(originalRpath)});\nelse if(args[0]==="--set-rpath") fs.appendFileSync(${JSON.stringify(log)},JSON.stringify(args)+"\\n");\nelse process.exit(1);\n`);
    chmodSync(patcher, 0o755);
    const dependencies = vendors.map((name) => `\t${name} => ${missingVendor && name === "libnccl.so.2" ? "not found" : `${join(vendor, name)} (0x1234)`}`).join("\n") + "\n\tlibcuda.so.1 => /host-driver/libcuda.so.1 (0x5678)\n";
    const ldd = join(bin, "ldd");
    writeFileSync(ldd, `#!${process.execPath}\nconsole.log(${JSON.stringify(dependencies)});\n`);
    chmodSync(ldd, 0o755);
    const result = spawnSync(process.execPath, [join(scripts, "stage-runtime.mjs")], {
      encoding: "utf8",
      env: { ...process.env, PATH: `${bin}:${process.env.PATH}`, ESE_RUNTIME_BUILD_DIR: build },
    });
    if (missingVendor || missingPatcher) {
      assert.notEqual(result.status, 0);
      assert.match(result.stderr, missingPatcher ? /requires patchelf/ : /Required CUDA redistributable libnccl\.so\.2 could not be resolved/);
      return;
    }
    assert.equal(result.status, 0, result.stderr || result.stdout);
    const metadata = JSON.parse(readFileSync(join(repo, "studio", "runtime", "ese", "build", "bin", "ese-runtime.json"), "utf8"));
    assert.equal(metadata.cuda, cuda);
    const calls = readFileSync(log, "utf8").trim().split("\n").map((line) => JSON.parse(line));
    assert.equal(calls.length, 2);
    assert.deepEqual(calls.map((args) => args[2].split("/").at(-1)).sort(), ["libggml.so", "llama-server"]);
    const runtimeBin = join(repo, "studio", "runtime", "ese", "build", "bin");
    for (const name of vendors) {
      assert.equal(existsSync(join(runtimeBin, name)), cuda);
      if (cuda) assert.equal(readFileSync(join(runtimeBin, name), "utf8"), `unmodified vendor ${name}\n`);
    }
    assert.ok(!readdirSync(runtimeBin).some((name) => /^libcuda\.so/.test(name)));
    return { paths: calls.map((args) => args[1]), expected: "$ORIGIN" };
  } finally {
    rmSync(root, { recursive: true, force: true });
  }
}

test("CPU runtime staging removes build and vendor search paths", { skip: process.platform !== "linux" }, () => {
  const fixture = stageFixture(false);
  for (const path of fixture.paths) assert.equal(path, fixture.expected);
});

test("CUDA runtime staging bundles resolved SONAME files without patching NVIDIA binaries", { skip: process.platform !== "linux" }, () => {
  const fixture = stageFixture(true);
  for (const path of fixture.paths) assert.equal(path, fixture.expected);
});

test("CUDA runtime staging rejects an unresolved redistributable", { skip: process.platform !== "linux" }, () => {
  stageFixture(true, { missingVendor: true });
});

test("Linux runtime staging requires patchelf", { skip: process.platform !== "linux" }, () => {
  stageFixture(false, { missingPatcher: true });
});
