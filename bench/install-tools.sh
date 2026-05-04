#!/usr/bin/env bash
# One-shot setup for benchmark tooling (k6, dotnet-trace, dotnet-counters).
set -euo pipefail
mkdir -p "$HOME/.local/bin"

if ! command -v k6 >/dev/null 2>&1 && [[ ! -x "$HOME/.local/bin/k6" ]]; then
  echo ">> installing k6 v0.55.0 → ~/.local/bin/k6"
  curl -sL https://github.com/grafana/k6/releases/download/v0.55.0/k6-v0.55.0-linux-amd64.tar.gz \
    | tar -xz -C /tmp
  mv /tmp/k6-v0.55.0-linux-amd64/k6 "$HOME/.local/bin/k6"
fi

# Make sure ~/.dotnet/tools is on PATH for the tools below.
export PATH="$PATH:$HOME/.dotnet/tools"

for tool in dotnet-trace dotnet-counters; do
  if ! command -v "$tool" >/dev/null 2>&1; then
    echo ">> installing $tool"
    dotnet tool install -g "$tool"
  fi
done

echo "done. Add to your shell rc if not already:"
echo '  export PATH="$PATH:$HOME/.local/bin:$HOME/.dotnet/tools"'
