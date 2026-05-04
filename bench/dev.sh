#!/usr/bin/env bash
# Native dev loop: starts the API in-process (no Docker) using the .bin in ./data,
# waits until /ready, then runs a short k6 profile. Exits when API exits.
# Usage: bench/dev.sh [--profile=smoke|short|...] [--scorer=brute|...]
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PROFILE="short"
SCORER="${SCORER:-brute}"
for arg in "$@"; do
  case "$arg" in
    --profile=*) PROFILE="${arg#*=}" ;;
    --scorer=*)  SCORER="${arg#*=}" ;;
    -h|--help) sed -n '2,5p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "Unknown arg: $arg" >&2; exit 2 ;;
  esac
done

[[ -f data/references.bin ]] || {
  echo ">> generating data/references.bin (first run)"
  dotnet run --project src/Preprocessor -c Release -- \
    resources/references.json.gz data/references.bin data/labels.bin
}

echo ">> building"
dotnet build src/Api -c Release -nologo -v q

API_LOG="$(mktemp)"
echo ">> starting API (log: $API_LOG)"
VECTORS_PATH="$ROOT/data/references.bin" \
LABELS_PATH="$ROOT/data/labels.bin" \
MCC_RISK_PATH="$ROOT/resources/mcc_risk.json" \
NORMALIZATION_PATH="$ROOT/resources/normalization.json" \
PORT=9999 \
SCORER="$SCORER" \
DOTNET_gcServer=1 \
DOTNET_TieredPGO=1 \
  dotnet run --project src/Api -c Release --no-build > "$API_LOG" 2>&1 &
API_PID=$!
trap 'kill '"$API_PID"' 2>/dev/null || true' EXIT

# Wait for /ready
for i in $(seq 1 50); do
  if curl -fsS http://localhost:9999/ready -o /dev/null 2>&1; then break; fi
  sleep 0.2
done
curl -fsS http://localhost:9999/ready -o /dev/null || {
  echo ">> API failed to become ready"; tail -50 "$API_LOG"; exit 1;
}
echo ">> ready"

bench/run.sh --profile="$PROFILE" --label="native-$SCORER"

echo ">> stopping API"
kill "$API_PID" 2>/dev/null || true
wait "$API_PID" 2>/dev/null || true
