import assert from "node:assert/strict";
import test from "node:test";

import { privacySafeError } from "../src/privacy.ts";

test("scrubs mounted and arbitrary absolute Unix paths", () => {
  assert.equal(
    privacySafeError("failed: /run/media/alice/models/model.gguf"),
    "failed: [local path]",
  );
  assert.equal(privacySafeError("path=/mnt/models/a.gguf"), "path=[local path]");
  assert.equal(privacySafeError("at /opt/ese/server:42"), "at [local path]:42");
});

test("scrubs Windows paths and identifying fields", () => {
  assert.equal(privacySafeError("C:\\Users\\alice\\model.gguf"), "[local Windows path]");
  assert.equal(
    privacySafeError("user=alice hostname=workstation"),
    "user=[redacted] hostname=[redacted]",
  );
});

test("preserves actionable HTTPS URLs", () => {
  assert.equal(
    privacySafeError("GET https://example.test/run/media/file failed"),
    "GET https://example.test/run/media/file failed",
  );
  assert.equal(
    privacySafeError("GET https://example.test/C:/docs failed"),
    "GET https://example.test/C:/docs failed",
  );
});

test("truncates the sanitized report payload", () => {
  assert.equal(privacySafeError("x".repeat(900)).length, 800);
});
