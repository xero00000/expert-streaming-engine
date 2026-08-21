#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
# Build and run the standalone Turbo4/Turbo8 CPU reference tests.

set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CXX_BIN="${CXX:-c++}"
BUILD_DIR="${ESE_TURBO_TEST_BUILD_DIR:-$(mktemp -d)}"
KEEP_BUILD="${ESE_TURBO_TEST_KEEP_BUILD:-0}"

cleanup() {
    if [[ "$KEEP_BUILD" != "1" ]]; then
        rm -rf "$BUILD_DIR"
    fi
}
trap cleanup EXIT

mkdir -p "$BUILD_DIR"
OUTPUT="$BUILD_DIR/test-turbo-kv"

"$CXX_BIN" \
    -std=c++17 \
    -O2 \
    -Wall \
    -Wextra \
    -Werror \
    -I"$ROOT/ggml/include" \
    "$ROOT/ggml/src/ggml-turbo-kv.cpp" \
    "$ROOT/tests/test-turbo-kv.cpp" \
    -o "$OUTPUT"

"$OUTPUT"
