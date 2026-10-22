import assert from "node:assert/strict";
import test from "node:test";
import { checkPackageUpdate } from "../src/update-policy.ts";

test("DEB and RPM use the package manager without requesting an AppImage update", async () => {
  for (const format of ["deb", "rpm"]) {
    let requests = 0;
    const result = await checkPackageUpdate(async () => format, async () => { requests++; return {}; });
    assert.equal(requests, 0);
    assert.equal(result.kind, "manual");
    assert.match(result.instructions, /package manager/);
    assert.match(result.instructions, /matching CPU or CUDA/);
  }
});

test("AppImage and both Windows installers retain signed update checks", async () => {
  for (const format of ["appimage", "msi", "nsis"]) {
    let requests = 0;
    const update = { version: "0.2.2", downloadAndInstall() {} };
    const result = await checkPackageUpdate(async () => format, async () => { requests++; return update; });
    assert.equal(requests, 1);
    assert.equal(result.kind, "automatic");
    assert.equal(result.update, update);
  }
});

test("unidentified installations do not guess a download format", async () => {
  for (const format of [null, "unknown"]) {
    const result = await checkPackageUpdate(async () => format, async () => { throw new Error("must not contact feed"); });
    assert.equal(result.kind, "manual");
    assert.match(result.instructions, /could not be identified/);
  }
});

test("failed format detection does not contact the update feed", async () => {
  let requests = 0;
  await assert.rejects(checkPackageUpdate(async () => { throw new Error("IPC failed"); }, async () => { requests++; return null; }), /IPC failed/);
  assert.equal(requests, 0);
});

test("automatic checks preserve no-update and failure results", async () => {
  assert.deepEqual(await checkPackageUpdate(async () => "appimage", async () => null), { kind: "automatic", update: null });
  await assert.rejects(checkPackageUpdate(async () => "nsis", async () => { throw new Error("invalid signature"); }), /invalid signature/);
});
