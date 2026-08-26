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
    /name: Studio frontend, Rust, and Windows installers[\s\S]*?timeout-minutes: 180/,
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
  assert.match(app, /selected\.available \? "Selected" : "Missing"/);
  assert.match(app, />Included<\/span>/);
  assert.match(types, /ready: boolean;/);
  assert.match(types, /modelPath\?: string;/);
  assert.doesNotMatch(app, />Scanning<\/span>/);
});
