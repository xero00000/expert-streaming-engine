#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
#
# Phase 1 Turbo KV pre-merge gate.
#
# This runs the Turbo-specific CPU reference, internal GGML type/dispatch, and
# visibility guards before delegating to the general ESE pre-merge harness.
# Pass every normal premerge-test.sh option through unchanged.

set -Eeuo pipefail
IFS=$'\n\t'

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

BUILD_JOBS="${ESE_TEST_JOBS:-$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 2)}"
if [[ "$BUILD_JOBS" -gt 16 ]]; then
    BUILD_JOBS=16
fi

TURBO_BUILD="$ROOT/.premerge-build/turbo-core"
TURBO_CUDA_BUILD="$ROOT/.premerge-build/turbo-cuda"
REQUIRE_TURBO_CUDA="${ESE_TURBO_REQUIRE_CUDA:-0}"

for argument in "$@"; do
    if [[ "$argument" == "--require-cuda" ]]; then
        REQUIRE_TURBO_CUDA=1
    fi
done

printf '%s\n' '== Turbo KV standalone reference =='
bash scripts/test-turbo-kv.sh

printf '\n%s\n' '== Turbo KV visibility guards =='
python3 -m unittest tests/test_turbo_kv_visibility.py -v

printf '\n%s\n' '== Turbo KV internal GGML integration =='
rm -rf "$TURBO_BUILD"
cmake -S . -B "$TURBO_BUILD" \
    -DCMAKE_BUILD_TYPE=Release \
    -DGGML_CUDA=OFF \
    -DLLAMA_BUILD_TESTS=ON \
    -DLLAMA_BUILD_SERVER=OFF
cmake --build "$TURBO_BUILD" --target test-turbo-kv-core -j "$BUILD_JOBS"
"$TURBO_BUILD/bin/test-turbo-kv-core"

if [[ "$REQUIRE_TURBO_CUDA" == "1" ]]; then
    printf '\n%s\n' '== Turbo KV native CUDA row codecs =='
    rm -rf "$TURBO_CUDA_BUILD"
    cmake -S . -B "$TURBO_CUDA_BUILD" \
        -DCMAKE_BUILD_TYPE=Release \
        -DGGML_CUDA=ON \
        -DLLAMA_BUILD_TESTS=ON \
        -DLLAMA_BUILD_SERVER=OFF
    cmake --build "$TURBO_CUDA_BUILD" --target test-turbo-kv-cuda -j "$BUILD_JOBS"
    "$TURBO_CUDA_BUILD/bin/test-turbo-kv-cuda"
fi

printf '\n%s\n' '== General ESE pre-merge gate =='
exec bash scripts/premerge-test.sh "$@"
