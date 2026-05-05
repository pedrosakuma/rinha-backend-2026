#!/usr/bin/env python3
"""Compare two bench/results/*.json files side by side. Supports aggregated (kind=aggregated) and single-run shapes."""
import json, math, sys, statistics
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

def per_run_finals(j):
    """Return list of per-run final_scores when available."""
    s = j.get("scoring", {})
    pr = s.get("per_run_final")
    if pr:
        return list(pr)
    fs = s.get("final_score")
    return [fs] if fs is not None else []

def welch_pvalue(t, df):
    """Two-sided p-value via the regularised incomplete beta. Uses math.erf as a fallback
    only when df is large; for small df we use Student-t survival via series."""
    # Cumulative t-distribution survival approx using regularized incomplete beta:
    # p_two_sided = I_{df/(df+t^2)}(df/2, 1/2)
    x = df / (df + t * t)
    a, b = df / 2.0, 0.5
    # Use math.lgamma based incomplete beta via continued fraction (Lentz).
    def betacf(a, b, x, max_iter=200, eps=1e-12):
        qab, qap, qam = a + b, a + 1.0, a - 1.0
        c, d = 1.0, 1.0 - qab * x / qap
        if abs(d) < 1e-30: d = 1e-30
        d = 1.0 / d
        h = d
        for m in range(1, max_iter + 1):
            m2 = 2 * m
            aa = m * (b - m) * x / ((qam + m2) * (a + m2))
            d = 1.0 + aa * d
            if abs(d) < 1e-30: d = 1e-30
            c = 1.0 + aa / c
            if abs(c) < 1e-30: c = 1e-30
            d = 1.0 / d
            h *= d * c
            aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
            d = 1.0 + aa * d
            if abs(d) < 1e-30: d = 1e-30
            c = 1.0 + aa / c
            if abs(c) < 1e-30: c = 1e-30
            d = 1.0 / d
            delta = d * c
            h *= delta
            if abs(delta - 1.0) < eps: break
        return h
    log_bt = (math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
              + a * math.log(x) + b * math.log(1.0 - x))
    bt = math.exp(log_bt)
    if x < (a + 1.0) / (a + b + 2.0):
        ireg = bt * betacf(a, b, x) / a
    else:
        ireg = 1.0 - bt * betacf(b, a, 1.0 - x) / b
    return ireg  # already two-sided for symmetric t

def welch(a_vals, b_vals):
    if len(a_vals) < 2 or len(b_vals) < 2:
        return None
    mA, mB = statistics.mean(a_vals), statistics.mean(b_vals)
    sA, sB = statistics.stdev(a_vals), statistics.stdev(b_vals)
    nA, nB = len(a_vals), len(b_vals)
    se2 = (sA * sA / nA) + (sB * sB / nB)
    if se2 <= 0:
        return None
    se = math.sqrt(se2)
    delta = mB - mA
    t = delta / se
    # Welch–Satterthwaite df.
    num = se2 * se2
    den = ((sA * sA / nA) ** 2) / (nA - 1) + ((sB * sB / nB) ** 2) / (nB - 1)
    df = num / den
    p = welch_pvalue(t, df)
    return {"meanA": mA, "meanB": mB, "sA": sA, "sB": sB,
            "nA": nA, "nB": nB, "delta": delta, "se": se,
            "t": t, "df": df, "p": p, "sigma": delta / se}

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

    # Welch t-test on final_score across runs (requires aggregated with per_run_final).
    a_runs = per_run_finals(a)
    b_runs = per_run_finals(b)
    w = welch(a_runs, b_runs)
    print()
    if w is None:
        print(f"welch t-test: SKIPPED (need ≥2 runs each; got nA={len(a_runs)} nB={len(b_runs)})")
    else:
        sig = "***" if w["p"] < 0.001 else ("**" if w["p"] < 0.01 else ("*" if w["p"] < 0.05 else "ns"))
        print(f"welch t-test on final_score (per-run):")
        print(f"  A: n={w['nA']:>2}  mean={w['meanA']:>8.2f}  σ={w['sA']:>6.2f}  runs={a_runs}")
        print(f"  B: n={w['nB']:>2}  mean={w['meanB']:>8.2f}  σ={w['sB']:>6.2f}  runs={b_runs}")
        print(f"  Δ (B−A)={w['delta']:+.2f}  SE={w['se']:.2f}  t={w['t']:+.2f}  "
              f"df={w['df']:.1f}  p={w['p']:.4f}  σ-conf={w['sigma']:+.2f}σ  [{sig}]")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("usage: bench/compare.py <a.json> <b.json>", file=sys.stderr)
        sys.exit(2)
    main(sys.argv[1], sys.argv[2])
