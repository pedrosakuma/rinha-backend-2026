#!/usr/bin/env bash
# Run bench/run.sh N times and aggregate the median into a single consolidated JSON.
# Usage: bench/run-x3.sh [--profile=short] [--label=name] [--n=3]
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PROFILE="short"
LABEL=""
N=3
for arg in "$@"; do
  case "$arg" in
    --profile=*) PROFILE="${arg#*=}" ;;
    --label=*)   LABEL="${arg#*=}" ;;
    --n=*)       N="${arg#*=}" ;;
    -h|--help)
      sed -n '2,4p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "Unknown arg: $arg" >&2; exit 2 ;;
  esac
done

mkdir -p bench/results
SHA="$(git rev-parse --short HEAD 2>/dev/null || echo no-git)"
TS="$(date -u +%Y%m%dT%H%M%SZ)"
SUFFIX="${LABEL:+-$LABEL}"
RUN_FILES=()

for i in $(seq 1 "$N"); do
  echo ">> run $i/$N"
  ITER_LABEL="${LABEL:+$LABEL-}r${i}"
  bash bench/run.sh --profile="$PROFILE" --label="$ITER_LABEL" >/dev/null
  LAST="$(ls -t bench/results/*-"${PROFILE}-${ITER_LABEL}".json | head -1)"
  RUN_FILES+=("$LAST")
  echo "   → $LAST"
done

OUT="bench/results/${TS}-${SHA}-${PROFILE}${SUFFIX}-x${N}.json"
python3 - "$OUT" "${RUN_FILES[@]}" <<'PY'
import json, sys, statistics
out_path, *paths = sys.argv[1:]
runs = [json.load(open(p)) for p in paths]

def med(vals): return round(statistics.median(vals), 2)
def mean(vals): return round(statistics.mean(vals), 2)
def sdev(vals): return round(statistics.stdev(vals), 2) if len(vals) >= 2 else 0.0

p50 = med([r["latency"]["p50"] for r in runs])
p90 = med([r["latency"]["p90"] for r in runs])
p99 = med([r["latency"]["p99"] for r in runs])
mx  = med([r["latency"]["max"] for r in runs])
final_scores = [r["scoring"]["final_score"] for r in runs]
p99_scores   = [r["scoring"]["p99_score"]["value"]      for r in runs]
det_scores   = [r["scoring"]["detection_score"]["value"] for r in runs]
fps   = sum(r["scoring"]["breakdown"]["fp"] for r in runs)
fns   = sum(r["scoring"]["breakdown"]["fn"] for r in runs)
errs  = sum(r["scoring"]["breakdown"]["http_errors"] for r in runs)
reqs  = sum(r["run"]["requests_total"] for r in runs)

agg = {
    "kind": "aggregated",
    "n": len(runs),
    "sources": [p.split("/")[-1] for p in paths],
    "run":     {"profile": runs[0]["run"], "requests_total": reqs},
    "latency": {"p50": p50, "p90": p90, "p99": p99, "max": mx},
    "scoring": {
        "p99_score":       {"value": med(p99_scores)},
        "detection_score": {"value": med(det_scores)},
        "final_score":     med(final_scores),
        "final_mean":      mean(final_scores),
        "final_stddev":    sdev(final_scores),
        "breakdown":       {"fp": fps, "fn": fns, "http_errors": errs},
        "per_run_final":   final_scores,
    },
}
json.dump(agg, open(out_path, "w"), indent=2)
print(f"wrote {out_path}")
print(f"median p50={p50}ms p90={p90}ms p99={p99}ms final={agg['scoring']['final_score']} "
      f"mean={agg['scoring']['final_mean']} σ={agg['scoring']['final_stddev']}")
PY
