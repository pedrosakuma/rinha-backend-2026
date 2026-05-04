#!/usr/bin/env bash
# Attach dotnet-trace to the running API process and collect a CPU sample,
# then convert to speedscope.json (drag-drop into https://speedscope.app).
# Usage: bench/profile.sh [seconds]
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
SECONDS_ARG="${1:-15}"

command -v dotnet-trace >/dev/null 2>&1 || dotnet tool install -g dotnet-trace

PID="$(pgrep -f 'Rinha.Api' | head -n1 || true)"
[[ -n "$PID" ]] || { echo "no Rinha.Api process found"; exit 1; }

TS="$(date -u +%Y%m%dT%H%M%SZ)"
OUT="bench/results/trace-${TS}.nettrace"
mkdir -p bench/results
echo ">> tracing PID=$PID for ${SECONDS_ARG}s → $OUT"
dotnet-trace collect --process-id "$PID" --duration "00:00:${SECONDS_ARG}" \
  --providers Microsoft-DotNETCore-SampleProfiler \
  -o "$OUT"

echo ">> converting to speedscope"
dotnet-trace convert --format speedscope -o "${OUT%.nettrace}.speedscope.json" "$OUT"
echo "open: ${OUT%.nettrace}.speedscope.json"
