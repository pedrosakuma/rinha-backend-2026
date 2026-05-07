#!/usr/bin/env bash
# Compare TCP-loopback vs UDS transport between nginx LB and Kestrel API replicas.
# Runs N alternated cycles (TCP, UDS, TCP, UDS, ...), 3 k6 runs per cycle, then
# aggregates across all runs (and separately for "warm-only", dropping the first
# k6 run of each cycle).
#
# Why alternated (not block A then block B):
#   thermal/turbo decay and host noise drift over time → alternation cancels it.
#
# Usage:
#   bench/uds-vs-tcp.sh [--cycles=2] [--profile=short]
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

CYCLES=2
PROFILE="short"
for arg in "$@"; do
  case "$arg" in
    --cycles=*)  CYCLES="${arg#*=}" ;;
    --profile=*) PROFILE="${arg#*=}" ;;
    -h|--help)
      sed -n '2,12p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "Unknown arg: $arg" >&2; exit 2 ;;
  esac
done

mkdir -p bench/results
TS="$(date -u +%Y%m%dT%H%M%SZ)"
SHA="$(git rev-parse --short HEAD 2>/dev/null || echo no-git)"
RUN_ID="${TS}-${SHA}"

UDS_COMPOSE=(-f docker-compose.yml -f docker-compose.uds.yml)
TCP_COMPOSE=(-f docker-compose.yml)

bring_down() {
  docker compose down -v --remove-orphans >/dev/null 2>&1 || true
}

bring_up() {
  local mode="$1"  # tcp | uds
  local compose_args
  if [[ "$mode" == "uds" ]]; then
    compose_args=("${UDS_COMPOSE[@]}")
  else
    compose_args=("${TCP_COMPOSE[@]}")
  fi
  echo ">> bringing up $mode stack"
  docker compose "${compose_args[@]}" up -d >/dev/null
  for i in $(seq 1 60); do
    if curl -fsS http://localhost:9999/ready -o /dev/null 2>&1; then
      echo "   ready after ${i}s"
      return 0
    fi
    sleep 1
  done
  echo "API never became ready ($mode)" >&2
  docker compose "${compose_args[@]}" logs --tail=50 >&2 || true
  return 1
}

declare -a TCP_RUNS=()
declare -a UDS_RUNS=()

for c in $(seq 1 "$CYCLES"); do
  for mode in tcp uds; do
    bring_down
    bring_up "$mode"
    LABEL="${mode}-c${c}"
    echo ">> running k6 x3 ($LABEL)"
    bash bench/run-x3.sh --profile="$PROFILE" --label="$LABEL" >/dev/null
    # The aggregate file produced by run-x3.sh:
    AGG="$(ls -t bench/results/*-"${PROFILE}-${LABEL}-x3".json | head -1)"
    if [[ "$mode" == "tcp" ]]; then
      TCP_RUNS+=("$AGG")
    else
      UDS_RUNS+=("$AGG")
    fi
    echo "   → $AGG"
  done
done

bring_down

OUT="bench/results/${RUN_ID}-uds-vs-tcp.json"
python3 - "$OUT" "${TCP_RUNS[@]}" "__SEP__" "${UDS_RUNS[@]}" <<'PY'
import json, sys, statistics, os

out_path = sys.argv[1]
rest = sys.argv[2:]
sep = rest.index("__SEP__")
tcp_files = rest[:sep]
uds_files = rest[sep+1:]

def collect(files):
    all_runs = []     # every individual k6 run final score
    warm_runs = []    # all but first run of each cycle (per-cycle warmup drop)
    p50s, p90s, p99s, maxs = [], [], [], []
    for f in files:
        d = json.load(open(f))
        per = d["scoring"]["per_run_final"]
        all_runs.extend(per)
        warm_runs.extend(per[1:])
        p50s.append(d["latency"]["p50"])
        p90s.append(d["latency"]["p90"])
        p99s.append(d["latency"]["p99"])
        maxs.append(d["latency"]["max"])
    return all_runs, warm_runs, p50s, p90s, p99s, maxs

def stats(label, vals):
    if not vals:
        return f"  {label}: (empty)"
    if len(vals) >= 2:
        return (f"  {label}: median={statistics.median(vals):.1f} "
                f"mean={statistics.mean(vals):.1f} "
                f"σ={statistics.stdev(vals):.1f} n={len(vals)}")
    return f"  {label}: value={vals[0]:.1f} n=1"

tcp_all, tcp_warm, t50, t90, t99, tmx = collect(tcp_files)
uds_all, uds_warm, u50, u90, u99, umx = collect(uds_files)

def safe_med(xs): return statistics.median(xs) if xs else float('nan')
def safe_mean(xs): return statistics.mean(xs) if xs else float('nan')

summary = {
    "tcp_files": [os.path.basename(f) for f in tcp_files],
    "uds_files": [os.path.basename(f) for f in uds_files],
    "tcp": {
        "final_all":  {"median": safe_med(tcp_all),  "mean": safe_mean(tcp_all),  "n": len(tcp_all)},
        "final_warm": {"median": safe_med(tcp_warm), "mean": safe_mean(tcp_warm), "n": len(tcp_warm)},
        "latency_per_cycle": {
            "p50_median": safe_med(t50), "p90_median": safe_med(t90),
            "p99_median": safe_med(t99), "max_median": safe_med(tmx),
        },
        "per_run_final": tcp_all,
    },
    "uds": {
        "final_all":  {"median": safe_med(uds_all),  "mean": safe_mean(uds_all),  "n": len(uds_all)},
        "final_warm": {"median": safe_med(uds_warm), "mean": safe_mean(uds_warm), "n": len(uds_warm)},
        "latency_per_cycle": {
            "p50_median": safe_med(u50), "p90_median": safe_med(u90),
            "p99_median": safe_med(u99), "max_median": safe_med(umx),
        },
        "per_run_final": uds_all,
    },
    "delta_uds_minus_tcp": {
        "final_median_all":  safe_med(uds_all)  - safe_med(tcp_all),
        "final_median_warm": safe_med(uds_warm) - safe_med(tcp_warm),
        "p99_median":        safe_med(u99) - safe_med(t99),
    },
}
json.dump(summary, open(out_path, "w"), indent=2)

print()
print(f"=== UDS vs TCP comparison ({len(tcp_files)} cycles each) ===")
print(f"TCP latency (median across cycles): p50={safe_med(t50):.2f} p90={safe_med(t90):.2f} p99={safe_med(t99):.2f} max={safe_med(tmx):.2f}")
print(f"UDS latency (median across cycles): p50={safe_med(u50):.2f} p90={safe_med(u90):.2f} p99={safe_med(u99):.2f} max={safe_med(umx):.2f}")
print()
print("Final score, all runs:")
print(stats("TCP", tcp_all))
print(stats("UDS", uds_all))
print(f"  Δ (UDS-TCP) median: {safe_med(uds_all)-safe_med(tcp_all):+.1f}")
print()
print("Final score, warm-only (first run of each cycle dropped):")
print(stats("TCP", tcp_warm))
print(stats("UDS", uds_warm))
print(f"  Δ (UDS-TCP) median: {safe_med(uds_warm)-safe_med(tcp_warm):+.1f}")
print()
print(f"wrote {out_path}")
PY
