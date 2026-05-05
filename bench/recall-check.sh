#!/usr/bin/env bash
# Compares the IVF scorer against brute-force on N synthetic queries.
# Fails (exit 1) if disagreement rate exceeds tolerance — meant to be run
# locally before promoting algorithmic changes (or in CI).
#
# Usage: bench/recall-check.sh [-- <args passed to Bench --recall>]
# Examples:
#   bench/recall-check.sh
#   bench/recall-check.sh -- --n=5000 --tol=0.005
#   bench/recall-check.sh -- --early-stop-pct=60   # check a candidate change

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PASSTHRU=()
if [ "${1-}" = "--" ]; then shift; PASSTHRU=("$@"); fi

dotnet run -c Release --project src/Bench/Bench.csproj -- --recall "${PASSTHRU[@]}"
