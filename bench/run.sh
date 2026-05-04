#!/usr/bin/env bash
# Run the official k6 harness against a local API at http://localhost:9999.
# Usage:
#   bench/run.sh [--profile=smoke|short|medium|full] [--label=<name>]
# Profiles control the load shape — only 'full' matches the upstream test (120s ramp to 900 RPS).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PROFILE="short"
LABEL=""
for arg in "$@"; do
  case "$arg" in
    --profile=*) PROFILE="${arg#*=}" ;;
    --label=*)   LABEL="${arg#*=}" ;;
    -h|--help)
      sed -n '2,7p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
      exit 0
      ;;
    *) echo "Unknown arg: $arg" >&2; exit 2 ;;
  esac
done

case "$PROFILE" in
  smoke)  DURATION="10s";  TARGET=300  ;;  # warmup-style, low rate
  short)  DURATION="30s";  TARGET=900  ;;  # iteration loop
  medium) DURATION="60s";  TARGET=900  ;;
  full)   DURATION="120s"; TARGET=900  ;;  # matches official test.js
  *) echo "Unknown profile: $PROFILE" >&2; exit 2 ;;
esac

K6="${K6:-$HOME/.local/bin/k6}"
[[ -x "$K6" ]] || K6="$(command -v k6 || true)"
[[ -n "$K6" ]] || { echo "k6 not found (export K6=/path/to/k6)" >&2; exit 1; }

# Ensure API is reachable.
if ! curl -fsS http://localhost:9999/ready -o /dev/null; then
  echo "API not reachable at http://localhost:9999/ready" >&2
  exit 1
fi

mkdir -p bench/results
SHA="$(git rev-parse --short HEAD 2>/dev/null || echo no-git)"
TS="$(date -u +%Y%m%dT%H%M%SZ)"
SUFFIX="${LABEL:+-$LABEL}"
NAME="${TS}-${SHA}-${PROFILE}${SUFFIX}"
OUT_JSON="bench/results/${NAME}.json"

echo ">> profile=$PROFILE duration=$DURATION target=$TARGET → $OUT_JSON"

K6_DURATION="$DURATION" K6_TARGET="$TARGET" K6_OUT_FILE="$OUT_JSON" \
  "$K6" run --quiet \
    -e RUN_DURATION="$DURATION" \
    -e RUN_TARGET="$TARGET" \
    -e RESULT_FILE="$OUT_JSON" \
    bench/k6/test.runner.js

# Print top-level summary
if command -v jq >/dev/null 2>&1; then
  jq '{p99, scoring}' "$OUT_JSON"
else
  cat "$OUT_JSON"
fi
