import assert from "node:assert/strict";
import test from "node:test";

import { privacySafeError } from "../src/privacy.ts";

test("scrubs mounted, quoted, and arbitrary absolute Unix paths", () => {
  assert.equal(
    privacySafeError("failed: /run/media/alice/models/model.gguf"),
    "failed: [local path]",
  );
  assert.equal(privacySafeError("path=/mnt/models/a.gguf"), "path=[local path]");
  assert.equal(privacySafeError("at /opt/ese/server:42"), "at [local path]");
  assert.equal(
    privacySafeError("path=/run/media/alice/My Models/private.gguf failed"),
    "path=[local path]",
  );
  assert.equal(
    privacySafeError('model="/run/media/alice/My Models/a.gguf"'),
    'model="[local path]"',
  );
  assert.equal(
    privacySafeError("at file:///run/media/alice/models/model.gguf"),
    "at [local file URL]",
  );
});

test("scrubs Windows drive and UNC paths", () => {
  assert.equal(privacySafeError("C:\\Users\\alice\\model.gguf"), "[local Windows path]");
  assert.equal(
    privacySafeError("load \\\\workstation\\models\\private.gguf failed"),
    "load [local Windows path]",
  );
  assert.equal(
    privacySafeError("cache=C:\\Users\\alice\\My Cache\\ese.bin failed"),
    "cache=[local Windows path]",
  );
  assert.equal(
    privacySafeError('model="C:\\My Models\\private.gguf"'),
    'model="[local Windows path]"',
  );
});

test("scrubs key-value and JSON identity fields", () => {
  assert.equal(
    privacySafeError("user=alice hostname=workstation"),
    "user=[redacted] hostname=[redacted]",
  );
  assert.equal(
    privacySafeError('{"username":"alice","hostname":"workstation","email":"alice@example.test"}'),
    '{"username":"[redacted]","hostname":"[redacted]","email":"[redacted]"}',
  );
  assert.equal(
    privacySafeError('{"username":"alice\\\"private","machine_name":"build\\\\host"}'),
    '{"username":"[redacted]","machine_name":"[redacted]"}',
  );
});

test("preserves URL origins and paths", () => {
  assert.equal(
    privacySafeError("GET https://example.test/run/media/file failed"),
    "GET https://example.test/run/media/file failed",
  );
  assert.equal(
    privacySafeError("GET https://example.test/C:/docs failed"),
    "GET https://example.test/C:/docs failed",
  );
});

test("scrubs URL userinfo, signed queries, and fragments", () => {
  assert.equal(
    privacySafeError("GET https://alice:secret@example.test/models/a.gguf?token=secret#session"),
    "GET https://example.test/models/a.gguf?[redacted]#[redacted]",
  );
  assert.equal(
    privacySafeError("https://cdn.example.test/model.gguf?X-Amz-Credential=private&X-Amz-Signature=secret"),
    "https://cdn.example.test/model.gguf?[redacted]",
  );
});

test("scrubs every line of a multiline error", () => {
  assert.equal(
    privacySafeError([
      "model: /run/media/alice/model.gguf",
      "cache: C:\\Users\\alice\\AppData\\Local\\ese",
      'identity: {"username":"alice"}',
      "download: https://alice@example.test/model.gguf?token=secret",
    ].join("\n")),
    [
      "model: [local path]",
      "cache: [local Windows path]",
      'identity: {"username":"[redacted]"}',
      "download: https://example.test/model.gguf?[redacted]",
    ].join("\n"),
  );
});

test("truncates the sanitized report payload", () => {
  assert.equal(privacySafeError("x".repeat(900)).length, 800);
});
