#!/usr/bin/env bash
# Attach dotnet-counters to the running API and stream key counters.
# Usage: bench/counters.sh
set -euo pipefail
command -v dotnet-counters >/dev/null 2>&1 || dotnet tool install -g dotnet-counters
PID="$(pgrep -f 'Rinha.Api' | head -n1 || true)"
[[ -n "$PID" ]] || { echo "no Rinha.Api process found"; exit 1; }
exec dotnet-counters monitor --process-id "$PID" \
    System.Runtime \
    Microsoft.AspNetCore.Hosting \
    Microsoft.AspNetCore.Server.Kestrel
