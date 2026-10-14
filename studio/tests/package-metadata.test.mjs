import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { readFileSync, readdirSync } from "node:fs";
import test from "node:test";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const studioRoot = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const repoRoot = resolve(studioRoot, "..");
const tauriConfig = JSON.parse(readFileSync(join(studioRoot, "src-tauri", "tauri.conf.json"), "utf8"));
const studioWorkflow = readFileSync(join(repoRoot, ".github", "workflows", "studio-ci.yml"), "utf8");
const releaseWorkflow = readFileSync(join(repoRoot, ".github", "workflows", "release.yml"), "utf8");

test("package metadata bundles ESE and NVIDIA license notices", () => {
  assert.equal(tauriConfig.bundle.license, "MIT");
  assert.equal(tauriConfig.bundle.licenseFile, "../../THIRD_PARTY_NOTICES.txt");
  assert.equal(tauriConfig.bundle.resources["../../LICENSE"], "licenses/ESE-MIT-LICENSE.txt");
  assert.notEqual(tauriConfig.bundle.licenseFile, "../../LICENSE");
  assert.equal(
    tauriConfig.bundle.resources["../../THIRD_PARTY_NOTICES.txt"],
    "licenses/THIRD_PARTY_NOTICES.txt",
  );
  assert.equal(
    tauriConfig.bundle.resources["../licenses/NVIDIA-CUDA-12.4-LICENSE.txt"],
    "licenses/NVIDIA-CUDA-12.4-LICENSE.txt",
  );

  const nvidiaLicense = readFileSync(join(studioRoot, "licenses", "NVIDIA-CUDA-12.4-LICENSE.txt"));
  assert.equal(
    createHash("sha256").update(nvidiaLicense).digest("hex"),
    "e2c71babfd18a8e69542dd7e9ca018f9caa438094001a58e6bc4d8c999bf0d07",
  );

  const notices = readFileSync(join(repoRoot, "THIRD_PARTY_NOTICES.txt"), "utf8");
  const normalizedNotices = notices.replace(/\s+/g, " ");
  for (const required of [
    "cuda_cudart 12.4.127",
    "libcublas 12.4.5.8",
    "cuda_cudart/LICENSE.txt",
    "libcublas/LICENSE.txt",
    "not licensed under ESE's MIT License",
    "not legal advice or a statement of legal clearance",
  ]) {
    assert.match(normalizedNotices, new RegExp(required.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")));
  }
});

test("runtime staging includes the complete Python launcher package", () => {
  const stageRuntime = readFileSync(join(studioRoot, "scripts", "stage-runtime.mjs"), "utf8");
  for (const module of ["ese.py", "hardware_profile.py", "__init__.py"]) {
    assert.match(
      stageRuntime,
      new RegExp(`cpSync\\(join\\(repoRoot, "tools", "${module.replace(".", "\\.")}"\\)`),
    );
  }
  assert.match(studioWorkflow, /"tools\/hardware_profile\.py"/);
});

test("Linux package metadata declares launcher and OpenMP dependencies", () => {
  assert.ok(tauriConfig.bundle.linux.deb.depends.includes("python3 (>= 3.10)"));
  assert.ok(tauriConfig.bundle.linux.deb.depends.includes("libgomp1"));
  assert.ok(tauriConfig.bundle.linux.rpm.depends.includes("python3"));
  assert.ok(tauriConfig.bundle.linux.rpm.depends.includes("libgomp"));
  // Tauri CLI 2.11.4 uses rpm 0.16.0: compressed short writes corrupt
  // the raw payload digest and repeatedly hash the unconsumed input.
  assert.deepEqual(tauriConfig.bundle.linux.rpm.compression, { type: "none" });
});

test("Linux CUDA packages carry the pinned CUDA, NCCL, and NVTX licenses", () => {
  const expected = {
    "NVIDIA-CUDA-12.8-LICENSE.txt": "e2c71babfd18a8e69542dd7e9ca018f9caa438094001a58e6bc4d8c999bf0d07",
    "NVIDIA-NCCL-LICENSE.txt": "0f0174a6b4e0b33ac26375bf729533075b32f4c51a5b6802a3d742d7dcdc9a76",
    "NVIDIA-NVTX-LICENSE.txt": "abdfd8a6738b27cd1504e091f77b682dbae928339f040b8a6f59a548245991a0",
  };
  for (const [name, hash] of Object.entries(expected)) {
    assert.equal(tauriConfig.bundle.resources[`../licenses/${name}`], `licenses/${name}`);
    assert.equal(createHash("sha256").update(readFileSync(join(studioRoot, "licenses", name))).digest("hex"), hash);
  }
  const notices = readFileSync(join(repoRoot, "THIRD_PARTY_NOTICES.txt"), "utf8").replace(/\s+/g, " ");
  for (const text of ["cuda_cudart 12.8.57", "libcublas 12.8.3.14", "NVIDIA NCCL 2.25.1", "libcuda.so.1 is not bundled"]) {
    assert.ok(notices.includes(text), text);
  }
});

test("NSIS packages the large CUDA runtime without solid compression", () => {
  assert.equal(tauriConfig.bundle.windows.nsis.template, "installer.nsi");
  const template = readFileSync(join(studioRoot, "src-tauri", "installer.nsi"), "utf8");
  assert.match(template, /Vendored from tauri-apps\/tauri tauri-cli-v2\.11\.4/);
  assert.match(template, /SetCompressor "\{\{compression\}\}"/);
  assert.doesNotMatch(template, /SetCompressor\s+\/SOLID/);
});

test("WiX splits the CUDA runtime across bounded embedded cabinets", () => {
  assert.equal(tauriConfig.bundle.windows.wix.template, "wix/main.wxs");
  const template = readFileSync(join(studioRoot, "src-tauri", "wix", "main.wxs"), "utf8");
  assert.match(template, /Vendored from tauri-apps\/tauri tauri-v2\.11\.4/);
  assert.match(template, /<MediaTemplate/);
  assert.match(template, /CabinetTemplate="ese\{0\}\.cab"/);
  assert.match(template, /MaximumUncompressedMediaSize="768"/);
  assert.match(template, /MaximumCabinetSizeForLargeFileSplitting="768"/);
  assert.doesNotMatch(template, /<Media\s+Id="1"\s+Cabinet="app\.cab"/);
});

test("cached Windows builds still stage and enforce CUDA redistributables", () => {
  assert.match(
    studioWorkflow,
    /name: Studio frontend, Rust, and Windows installers[\s\S]*?timeout-minutes: 240/,
  );
  const cudaInstall = studioWorkflow.match(
    /- name: Install CUDA toolkit\n(?<body>(?: {8,}.*\n){1,5})/,
  );
  assert.ok(cudaInstall);
  assert.doesNotMatch(cudaInstall.groups.body, /cache-hit/);
  assert.match(
    studioWorkflow,
    /verify-windows-runtime\.py[^\n]*\n\s+if \(\$LASTEXITCODE -ne 0\) \{ throw "Staged Windows runtime verification failed" \}/,
  );
  assert.match(
    releaseWorkflow,
    /verify-windows-runtime\.py[^\n]*\n\s+if \(\$LASTEXITCODE -ne 0\) \{ throw "Staged Windows runtime verification failed" \}/,
  );
  assert.match(
    studioWorkflow,
    /- name: Verify staged Windows runtime[\s\S]*?- name: Save verified Windows CUDA runtime build\n\s+if: steps\.windows-cuda-cache\.outputs\.cache-hit != 'true'\n\s+uses: actions\/cache\/save@v4/,
  );
  for (const workflow of [studioWorkflow, releaseWorkflow]) {
    assert.match(workflow, /prepare-restored-build\.py --build-root \.\\build-package/);
    assert.match(workflow, /prepare-restored-build\.py --build-root build-package --record/);
    assert.match(workflow, /hashFiles\([^\n]*'include\/\*\*'[^\n]*'vendor\/\*\*'[^\n]*prepare-restored-build\.py/);
  }
  const cacheKey = /key: (ese-windows-cuda-v2-[^\n]+)/;
  assert.equal(studioWorkflow.match(cacheKey)?.[1], releaseWorkflow.match(cacheKey)?.[1]);
  for (const pattern of ["cudart64_*.dll", "cublas64_*.dll", "cublasLt64_*.dll"]) {
    assert.ok(studioWorkflow.split(pattern).length >= 3, `${pattern} is not gated in both installer jobs`);
  }
});

test("AppImage release packaging preserves modern ELF relocation sections", () => {
  assert.match(releaseWorkflow, /- name: Build packages[\s\S]*?NO_STRIP: "1"/);
});

test("release package jobs require the complete native regression suite", () => {
  assert.match(releaseWorkflow, /native-regressions:[\s\S]*?ctest --test-dir build-release-checks --output-on-failure/);
  assert.match(releaseWorkflow, /linux-studio:\n\s+name: Build Linux Studio packages\n\s+needs: native-regressions/);
  assert.match(releaseWorkflow, /windows-studio:\n\s+name: Build Windows Studio installers\n\s+needs: native-regressions/);
});

test("repository forms use ESE links, branding, and available labels", () => {
  const templateRoot = join(repoRoot, ".github", "ISSUE_TEMPLATE");
  const files = readdirSync(templateRoot)
    .filter((name) => name.endsWith(".yml"))
    .map((name) => readFileSync(join(templateRoot, name), "utf8"));
  files.push(readFileSync(join(repoRoot, ".github", "pull_request_template.md"), "utf8"));
  const templates = files.join("\n");

  assert.doesNotMatch(templates, /ikawrakow\/ik_llama|llama\.cpp|\/discussions|\/wiki\//i);
  const availableLabels = new Set(["bug", "documentation", "duplicate", "enhancement", "good first issue", "help wanted", "invalid", "question", "wontfix"]);
  for (const match of templates.matchAll(/^labels:\s*(\[[^\n]*\])/gm)) {
    for (const label of JSON.parse(match[1])) assert.ok(availableLabels.has(label), `Unavailable label: ${label}`);
  }
});

test("generated bug reports require review before submission", () => {
  const app = readFileSync(join(studioRoot, "src", "App.tsx"), "utf8");
  assert.match(app, /Review every field and remove any private or identifying information before submitting\./);
});

test("Studio presents reasoning and runtime actions from measured state", () => {
  const app = readFileSync(join(studioRoot, "src", "App.tsx"), "utf8");
  const types = readFileSync(join(studioRoot, "src", "types.ts"), "utf8");

  assert.match(app, /<details className="chat-reasoning">/);
  assert.match(app, /payload\.reasoning/);
  assert.match(app, /chatStatus\.ready \? "Model connected" : chatStatus\.active \? "Model starting"/);
  assert.match(app, /chatStatus\.active \? "Stop model & run" : "Run sweep"/);
  assert.match(app, /chatStatus\.modelPath === selected\.path \? "Restart model" : "Switch model"/);
  assert.match(app, /selectedLaunchable \? "Selected" : selected\.available \? "Validation gated" : "Missing"/);
  assert.match(app, />Included<\/span>/);
  assert.match(app, /Qwen 3\.8 · Flash Next/);
  assert.match(app, /support-badge">Experimental/);
  assert.match(app, /Text \$\{capabilityLabel\(selected\.runtimeSupport\.text\)\} · Vision/);
  assert.match(app, /!selectedLaunchable \? "Validation gated"/);
  assert.match(types, /ready: boolean;/);
  assert.match(types, /modelPath\?: string;/);
  assert.match(types, /runtimeSupport\?: RuntimeSupport;/);
  assert.doesNotMatch(app, />Scanning<\/span>/);
});
