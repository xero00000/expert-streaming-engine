#!/usr/bin/env bash
# ESE pre-merge validation harness
#
# Safe default: source/static/unit/CPU-build checks only.
# Full hardware gate:
#   bash scripts/premerge-test.sh --full --model /path/model.gguf --require-cuda
#
# Recommended ESE workstation gate for a large MoE:
#   bash scripts/premerge-test.sh --full \
#     --model /models/model.gguf \
#     --require-cuda \
#     --policies auto,hybrid,stream \
#     --context 4096 \
#     --gpu-resident-moe 6
#
# Results are written under .premerge-results/<timestamp>/.

set -Eeuo pipefail
IFS=$'\n\t'

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

FULL=0
REQUIRE_CUDA=0
MODEL=""
POLICIES="auto"
CONTEXT=4096
GPU_RESIDENT_MOE=""
STARTUP_TIMEOUT=900
N_PREDICT=8
STRICT_PARITY=0
KEEP_SERVERS=0
BUILD_JOBS="${ESE_TEST_JOBS:-$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 2)}"
if [[ "$BUILD_JOBS" -gt 16 ]]; then BUILD_JOBS=16; fi

usage() {
  cat <<'EOF'
Usage: bash scripts/premerge-test.sh [options]

Core options:
  --full                    Run CUDA + real-model runtime gates.
  --model PATH              GGUF file (or first split shard) for runtime tests.
  --require-cuda            Fail if a working CUDA toolchain/GPU is unavailable.
  --policies LIST           Comma-separated: auto,resident,hybrid,stream.
                            Default: auto
  --context N               Runtime smoke context. Default: 4096
  --gpu-resident-moe N      Keep final N MoE blocks on GPU for hybrid/stream.
  --startup-timeout SEC     Per-policy server startup timeout. Default: 900
  --n-predict N             Deterministic smoke-generation length. Default: 8
  --strict-parity           Require generated text to match across tested policies.
  --jobs N                  Maximum build jobs. Default: min(CPUs,16)
  --keep-servers            Do not terminate test servers on script exit.
  -h, --help                Show this help.

Examples:
  # CI-equivalent + clean CPU build
  bash scripts/premerge-test.sh

  # Merge gate on a CUDA workstation
  bash scripts/premerge-test.sh --full --require-cuda --model /models/model.gguf

  # Exercise the ESE-specific MoE paths
  bash scripts/premerge-test.sh --full --require-cuda \
    --model /models/gpt-oss-120b-F16.gguf \
    --policies hybrid,stream --gpu-resident-moe 6 --context 4096

Exit status is non-zero when any required gate fails.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --full) FULL=1; shift ;;
    --require-cuda) REQUIRE_CUDA=1; shift ;;
    --model) MODEL="${2:?--model requires a path}"; shift 2 ;;
    --policies) POLICIES="${2:?--policies requires a list}"; shift 2 ;;
    --context) CONTEXT="${2:?--context requires a value}"; shift 2 ;;
    --gpu-resident-moe) GPU_RESIDENT_MOE="${2:?--gpu-resident-moe requires a value}"; shift 2 ;;
    --startup-timeout) STARTUP_TIMEOUT="${2:?--startup-timeout requires seconds}"; shift 2 ;;
    --n-predict) N_PREDICT="${2:?--n-predict requires a value}"; shift 2 ;;
    --strict-parity) STRICT_PARITY=1; shift ;;
    --jobs) BUILD_JOBS="${2:?--jobs requires a value}"; shift 2 ;;
    --keep-servers) KEEP_SERVERS=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if [[ "$FULL" -eq 1 && -z "$MODEL" ]]; then
  echo "ERROR: --full requires --model PATH" >&2
  exit 2
fi

for value in "$CONTEXT" "$STARTUP_TIMEOUT" "$N_PREDICT" "$BUILD_JOBS"; do
  [[ "$value" =~ ^[0-9]+$ ]] || { echo "ERROR: numeric option expected, got '$value'" >&2; exit 2; }
done

IFS=',' read -r -a POLICY_ARRAY <<< "$POLICIES"
for policy in "${POLICY_ARRAY[@]}"; do
  case "$policy" in auto|resident|hybrid|stream) ;; *) echo "ERROR: invalid policy '$policy'" >&2; exit 2 ;; esac
done

STAMP="$(date +%Y%m%d-%H%M%S)"
RESULT_DIR="$ROOT/.premerge-results/$STAMP"
CPU_BUILD="$ROOT/.premerge-build/cpu"
CUDA_BUILD="$ROOT/.premerge-build/cuda"
mkdir -p "$RESULT_DIR"
SUMMARY="$RESULT_DIR/SUMMARY.txt"
: > "$SUMMARY"

PASS=0
FAIL=0
WARN=0
CURRENT_SERVER_PID=""

color() { [[ -t 1 ]] && printf '\033[%sm' "$1" || true; }
reset_color() { [[ -t 1 ]] && printf '\033[0m' || true; }

record() { printf '%s\n' "$*" | tee -a "$SUMMARY"; }
pass() { PASS=$((PASS+1)); color '32;1'; printf 'PASS '; reset_color; record "$*"; }
warn() { WARN=$((WARN+1)); color '33;1'; printf 'WARN '; reset_color; record "$*"; }
fail() { FAIL=$((FAIL+1)); color '31;1'; printf 'FAIL '; reset_color; record "$*"; }

cleanup() {
  if [[ -n "$CURRENT_SERVER_PID" && "$KEEP_SERVERS" -eq 0 ]]; then
    kill "$CURRENT_SERVER_PID" 2>/dev/null || true
    wait "$CURRENT_SERVER_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

run_gate() {
  local name="$1"; shift
  local slug
  slug="$(printf '%s' "$name" | tr ' /:' '___' | tr -cd '[:alnum:]_.-')"
  local log="$RESULT_DIR/${slug}.log"
  record ""
  record "== $name =="
  record "+ $*"
  if "$@" > >(tee "$log") 2> >(tee -a "$log" >&2); then
    pass "$name"
    return 0
  fi
  fail "$name (see $log)"
  return 1
}

run_shell_gate() {
  local name="$1"; shift
  local cmd="$*"
  local slug
  slug="$(printf '%s' "$name" | tr ' /:' '___' | tr -cd '[:alnum:]_.-')"
  local log="$RESULT_DIR/${slug}.log"
  record ""
  record "== $name =="
  record "+ $cmd"
  if bash -o pipefail -c "$cmd" > >(tee "$log") 2> >(tee -a "$log" >&2); then
    pass "$name"
    return 0
  fi
  fail "$name (see $log)"
  return 1
}

require_cmd() {
  local cmd="$1"
  if command -v "$cmd" >/dev/null 2>&1; then
    pass "required command: $cmd ($(command -v "$cmd"))"
  else
    fail "required command missing: $cmd"
    return 1
  fi
}

record "Expert Streaming Engine pre-merge validation"
record "timestamp: $(date -Is)"
record "root: $ROOT"
record "results: $RESULT_DIR"
record "full hardware gate: $FULL"
record "requested policies: $POLICIES"
record "context: $CONTEXT"

# ---------------------------------------------------------------------------
# 1. Repository/source hygiene
# ---------------------------------------------------------------------------

require_cmd git || true
require_cmd python3 || true
require_cmd cmake || true
require_cmd c++ || true
require_cmd curl || true

if git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  record "commit: $(git rev-parse HEAD)"
  record "branch: $(git branch --show-current || true)"
  if [[ -n "$(git status --porcelain --untracked-files=no)" ]]; then
    warn "tracked working tree has local modifications; test results are not a pristine commit"
    git status --short | tee "$RESULT_DIR/git-status.txt" >/dev/null
  else
    pass "tracked working tree is clean"
  fi
  if git diff --check HEAD --; then pass "git diff --check"; else fail "git diff --check"; fi
  if git diff --name-only --diff-filter=U | grep -q .; then
    fail "unresolved merge conflicts exist"
  else
    pass "no unresolved merge conflicts"
  fi
fi

# `ese` is a Python entry point, not a shell script. Shell parsers only inspect
# shell files; the launcher is validated by the Python compile/help/doctor gates.
run_shell_gate "shell syntax" "bash -n scripts/*.sh" || true
if command -v shellcheck >/dev/null 2>&1; then
  run_shell_gate "shellcheck" "shellcheck scripts/*.sh" || true
else
  warn "shellcheck is not installed; shell syntax was still checked"
fi

run_gate "Python compile" python3 -m compileall -q ese tools tests || true
run_gate "Python unit tests" python3 -m unittest discover -s tests -p 'test_*.py' -v || true
run_gate "ESE help" ./ese --help || true
run_gate "ESE doctor" ./ese doctor --json || true

# ---------------------------------------------------------------------------
# 2. Native CPU build and parser-surface gate
# ---------------------------------------------------------------------------

rm -rf "$CPU_BUILD"
run_shell_gate "CPU configure" \
  "cmake -S . -B '$CPU_BUILD' -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_SERVER=ON" || true
run_shell_gate "CPU llama-server build" \
  "cmake --build '$CPU_BUILD' --target llama-server -j '$BUILD_JOBS'" || true

CPU_SERVER="$CPU_BUILD/bin/llama-server"
if [[ -x "$CPU_SERVER" ]]; then
  "$CPU_SERVER" --help > "$RESULT_DIR/cpu-native-help.txt" 2>&1 || true
  if [[ -s "$RESULT_DIR/cpu-native-help.txt" ]]; then pass "CPU binary emits help"; else fail "CPU binary produced no help output"; fi
  for flag in --fit --cpu-moe --n-cpu-moe --defer-experts; do
    if grep -Fq -- "$flag" "$RESULT_DIR/cpu-native-help.txt"; then
      pass "CPU native surface contains $flag"
    else
      fail "CPU native surface missing $flag"
    fi
  done
else
  fail "CPU llama-server binary was not produced"
fi

# ---------------------------------------------------------------------------
# 3. CUDA compile/runtime-loader gate
# ---------------------------------------------------------------------------

CUDA_AVAILABLE=0
if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L > "$RESULT_DIR/nvidia-smi-L.txt" 2>&1; then
  CUDA_AVAILABLE=1
  GPU_COUNT="$(grep -c '^GPU ' "$RESULT_DIR/nvidia-smi-L.txt" || true)"
  pass "NVIDIA driver sees $GPU_COUNT GPU(s)"
  nvidia-smi --query-gpu=index,name,driver_version,memory.total,memory.free,pci.bus_id \
    --format=csv,noheader > "$RESULT_DIR/gpus-before.csv" 2>&1 || true
else
  if [[ "$REQUIRE_CUDA" -eq 1 ]]; then fail "NVIDIA driver/GPU unavailable"; else warn "NVIDIA driver/GPU unavailable; CUDA runtime gate skipped"; fi
fi

if [[ "$FULL" -eq 1 || "$REQUIRE_CUDA" -eq 1 ]]; then
  if ! command -v nvcc >/dev/null 2>&1; then
    if [[ "$REQUIRE_CUDA" -eq 1 ]]; then fail "nvcc not found"; else warn "nvcc not found; attempting CMake CUDA detection anyway"; fi
  else
    nvcc --version > "$RESULT_DIR/nvcc-version.txt" 2>&1 || true
    pass "nvcc detected"
  fi

  rm -rf "$CUDA_BUILD"
  run_shell_gate "CUDA configure" \
    "cmake -S . -B '$CUDA_BUILD' -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_SERVER=ON" || true
  run_shell_gate "CUDA llama-server build" \
    "cmake --build '$CUDA_BUILD' --target llama-server -j '$BUILD_JOBS'" || true

  CUDA_SERVER="$CUDA_BUILD/bin/llama-server"
  if [[ -x "$CUDA_SERVER" ]]; then
    "$CUDA_SERVER" --help > "$RESULT_DIR/cuda-native-help.txt" 2>&1 || true
    if [[ -s "$RESULT_DIR/cuda-native-help.txt" ]]; then pass "CUDA binary emits help"; else fail "CUDA binary produced no help output"; fi
    for flag in --fit --cpu-moe --n-cpu-moe --defer-experts; do
      if grep -Fq -- "$flag" "$RESULT_DIR/cuda-native-help.txt"; then
        pass "CUDA native surface contains $flag"
      else
        fail "CUDA native surface missing $flag"
      fi
    done
    if command -v ldd >/dev/null 2>&1; then
      ldd "$CUDA_SERVER" > "$RESULT_DIR/cuda-ldd.txt" 2>&1 || true
      if grep -q 'not found' "$RESULT_DIR/cuda-ldd.txt"; then
        fail "CUDA binary has unresolved dynamic libraries"
      else
        pass "CUDA binary dynamic libraries resolve"
      fi
    fi
  else
    fail "CUDA llama-server binary was not produced"
  fi
fi

# ---------------------------------------------------------------------------
# 4. Real GGUF planning and actual inference
# ---------------------------------------------------------------------------

if [[ -n "$MODEL" ]]; then
  MODEL="$(realpath "$MODEL" 2>/dev/null || printf '%s' "$MODEL")"
  if [[ ! -f "$MODEL" ]]; then
    fail "model does not exist: $MODEL"
  else
    pass "model exists: $MODEL"
    run_gate "GGUF plan auto" ./ese plan "$MODEL" --policy auto -c "$CONTEXT" --json || true

    for policy in "${POLICY_ARRAY[@]}"; do
      PLAN_ARGS=(./ese plan "$MODEL" --policy "$policy" -c "$CONTEXT" --json)
      if [[ -n "$GPU_RESIDENT_MOE" && ( "$policy" == "hybrid" || "$policy" == "stream" ) ]]; then
        PLAN_ARGS+=(--gpu-resident-moe "$GPU_RESIDENT_MOE")
      fi
      run_gate "GGUF plan $policy" "${PLAN_ARGS[@]}" || true
    done
  fi
fi

server_ready() {
  local port="$1"
  local deadline=$((SECONDS + STARTUP_TIMEOUT))
  while (( SECONDS < deadline )); do
    if ! kill -0 "$CURRENT_SERVER_PID" 2>/dev/null; then
      return 2
    fi
    if curl --silent --show-error --max-time 2 "http://127.0.0.1:${port}/health" > /dev/null 2>&1; then
      return 0
    fi
    sleep 2
  done
  return 1
}

runtime_policy() {
  local policy="$1"
  local port="$2"
  local log="$RESULT_DIR/runtime-${policy}.log"
  local response="$RESULT_DIR/runtime-${policy}-response.json"
  local content="$RESULT_DIR/runtime-${policy}-content.txt"

  if [[ ! -x "${CUDA_SERVER:-}" ]]; then
    fail "runtime $policy: CUDA server is unavailable"
    return 1
  fi

  local args=(./ese serve "$MODEL" --policy "$policy" -c "$CONTEXT" --host 127.0.0.1 --port "$port")
  if [[ -n "$GPU_RESIDENT_MOE" && ( "$policy" == "hybrid" || "$policy" == "stream" ) ]]; then
    args+=(--gpu-resident-moe "$GPU_RESIDENT_MOE")
  fi

  record ""
  record "== runtime policy: $policy =="
  record "+ ESE_SERVER=$CUDA_SERVER ${args[*]}"
  ESE_SERVER="$CUDA_SERVER" "${args[@]}" > "$log" 2>&1 &
  CURRENT_SERVER_PID=$!

  local ready_rc=0
  server_ready "$port" || ready_rc=$?
  if [[ "$ready_rc" -ne 0 ]]; then
    if [[ "$ready_rc" -eq 2 ]]; then
      fail "runtime $policy: server exited before becoming healthy (see $log)"
    else
      fail "runtime $policy: startup exceeded ${STARTUP_TIMEOUT}s (see $log)"
    fi
    tail -n 120 "$log" > "$RESULT_DIR/runtime-${policy}-tail.txt" 2>/dev/null || true
    cleanup
    CURRENT_SERVER_PID=""
    return 1
  fi
  pass "runtime $policy: /health became ready"

  local payload
  payload="$(python3 - "$N_PREDICT" <<'PY'
import json, sys
print(json.dumps({
    "prompt": "Return exactly one short sentence describing what a CPU does.",
    "n_predict": int(sys.argv[1]),
    "temperature": 0,
    "seed": 1,
    "cache_prompt": False,
}))
PY
)"

  local http_code
  http_code="$(curl --silent --show-error --max-time "$STARTUP_TIMEOUT" \
    -o "$response" -w '%{http_code}' \
    -H 'Content-Type: application/json' \
    -d "$payload" \
    "http://127.0.0.1:${port}/completion" || true)"

  if [[ "$http_code" != "200" ]]; then
    fail "runtime $policy: /completion returned HTTP ${http_code:-curl-error}"
  elif python3 - "$response" "$content" <<'PY'
import json, pathlib, sys
p = pathlib.Path(sys.argv[1])
out = pathlib.Path(sys.argv[2])
data = json.loads(p.read_text())
content = data.get("content")
if not isinstance(content, str) or not content.strip():
    raise SystemExit("missing/empty completion content")
out.write_text(content)
print("content:", repr(content))
for key in ("timings", "tokens_predicted", "tokens_evaluated"):
    if key in data:
        print(key + ":", data[key])
PY
  then
    pass "runtime $policy: deterministic completion returned valid content"
  else
    fail "runtime $policy: completion JSON/content validation failed"
  fi

  if grep -Eqi 'CUDA error|illegal memory access|GGML_ASSERT|segmentation fault|core dumped|nan[^a-z]' "$log"; then
    fail "runtime $policy: fatal/assert/NaN signature found in server log"
  else
    pass "runtime $policy: no fatal CUDA/assert/NaN signature in server log"
  fi

  kill "$CURRENT_SERVER_PID" 2>/dev/null || true
  wait "$CURRENT_SERVER_PID" 2>/dev/null || true
  CURRENT_SERVER_PID=""
  sleep 2
  return 0
}

if [[ "$FULL" -eq 1 && -f "$MODEL" ]]; then
  if [[ "$CUDA_AVAILABLE" -ne 1 ]]; then
    fail "full runtime gate requires a visible NVIDIA GPU"
  else
    BASE_PORT=18880
    idx=0
    for policy in "${POLICY_ARRAY[@]}"; do
      runtime_policy "$policy" $((BASE_PORT + idx)) || true
      idx=$((idx + 1))
    done

    # Optional cross-policy greedy parity. This is useful for exact-regression
    # testing but is not universally guaranteed across different backend math,
    # so it is a warning unless --strict-parity was requested.
    mapfile -t CONTENT_FILES < <(find "$RESULT_DIR" -maxdepth 1 -name 'runtime-*-content.txt' -type f | sort)
    if (( ${#CONTENT_FILES[@]} >= 2 )); then
      reference="${CONTENT_FILES[0]}"
      parity_ok=1
      for candidate in "${CONTENT_FILES[@]:1}"; do
        if ! cmp -s "$reference" "$candidate"; then parity_ok=0; fi
      done
      if [[ "$parity_ok" -eq 1 ]]; then
        pass "cross-policy greedy text parity"
      elif [[ "$STRICT_PARITY" -eq 1 ]]; then
        fail "cross-policy greedy text differs (--strict-parity)"
      else
        warn "cross-policy greedy text differs; inspect runtime responses (backend numeric divergence may be legitimate)"
      fi
    fi

    nvidia-smi --query-gpu=index,name,memory.total,memory.free,temperature.gpu,power.draw \
      --format=csv,noheader > "$RESULT_DIR/gpus-after.csv" 2>&1 || true
  fi
fi

# ---------------------------------------------------------------------------
# 5. Result bundle + final gate
# ---------------------------------------------------------------------------

{
  echo
  echo "System snapshot"
  echo "---------------"
  uname -a || true
  python3 --version || true
  cmake --version | head -1 || true
  c++ --version | head -1 || true
  command -v nvcc >/dev/null 2>&1 && nvcc --version | tail -4 || true
  command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L || true
  echo
  echo "git HEAD: $(git rev-parse HEAD 2>/dev/null || echo unknown)"
} > "$RESULT_DIR/system.txt" 2>&1

record ""
record "============================================================"
record "PRE-MERGE RESULT"
record "PASS=$PASS  WARN=$WARN  FAIL=$FAIL"
record "Evidence bundle: $RESULT_DIR"
record "============================================================"

if [[ "$FAIL" -ne 0 ]]; then
  record "NOT MERGE-READY: $FAIL required gate(s) failed."
  exit 1
fi

if [[ "$FULL" -eq 1 ]]; then
  record "MERGE GATE PASSED: source, CPU, CUDA, GGUF planning, and requested runtime policies passed."
else
  record "QUICK GATE PASSED. Run with --full --require-cuda --model PATH before merging CUDA/MoE changes."
fi
exit 0