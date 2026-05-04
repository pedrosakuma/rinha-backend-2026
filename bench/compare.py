#!/usr/bin/env python3
"""Compare two bench/results/*.json files side by side. Supports aggregated (kind=aggregated) and single-run shapes."""
import json, sys
from pathlib import Path

def load(p):
    return json.loads(Path(p).read_text())

def fmt_pct(a, b):
    if a == 0: return "  -"
    delta = (b - a) / a * 100
    sign = "+" if delta >= 0 else ""
    return f"{sign}{delta:.1f}%"

def reqs_of(j):
    rt = j.get("run", {}).get("requests_total")
    return rt if rt is not None else 0

def main(a_path, b_path):
    a, b = load(a_path), load(b_path)
    label_a = "A" + (" (med)" if a.get("kind") == "aggregated" else "")
    label_b = "B" + (" (med)" if b.get("kind") == "aggregated" else "")
    rows = [
        ("p50 (ms)",    a["latency"]["p50"],  b["latency"]["p50"]),
        ("p90 (ms)",    a["latency"]["p90"],  b["latency"]["p90"]),
        ("p99 (ms)",    a["latency"]["p99"],  b["latency"]["p99"]),
        ("max (ms)",    a["latency"]["max"],  b["latency"]["max"]),
        ("requests",    reqs_of(a), reqs_of(b)),
        ("p99_score",   a["scoring"]["p99_score"]["value"],       b["scoring"]["p99_score"]["value"]),
        ("det_score",   a["scoring"]["detection_score"]["value"], b["scoring"]["detection_score"]["value"]),
        ("final_score", a["scoring"]["final_score"],              b["scoring"]["final_score"]),
        ("FP",          a["scoring"]["breakdown"]["fp"], b["scoring"]["breakdown"]["fp"]),
        ("FN",          a["scoring"]["breakdown"]["fn"], b["scoring"]["breakdown"]["fn"]),
        ("HTTP errors", a["scoring"]["breakdown"]["http_errors"], b["scoring"]["breakdown"]["http_errors"]),
    ]
    print(f"{'metric':14}  {label_a:>12}  {label_b:>12}  {'Δ':>10}")
    print("-" * 56)
    for name, av, bv in rows:
        print(f"{name:14}  {av:>12}  {bv:>12}  {fmt_pct(av, bv):>10}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("usage: bench/compare.py <a.json> <b.json>", file=sys.stderr)
        sys.exit(2)
    main(sys.argv[1], sys.argv[2])
