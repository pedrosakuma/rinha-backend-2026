#!/usr/bin/env python3
"""Extract per-query features and labels from bench/k6/test-data.json.

Mirrors src/Api/Vectorizer.cs (the single source of truth for the production
14-d embedding) and computes top-3 IVF cell distances using
data/ivf_centroids.bin. Output is a single .npz consumed by cascade_train.py.

Features produced (per entry):
  v[0..13]       float32  — same dims emitted by Vectorizer
  c0, c1, c2     int32    — top-3 nearest centroid IDs (by squared L2)
  d0, d1, d2     float32  — corresponding squared L2 distances
  d_gap          float32  — d1 - d0 (centroid margin)
  d_ratio        float32  — d0 / d1 (centroid ratio, 0..1)
  expected       float32  — ground truth fraud score (canonical)
  label          int8     — 1 if expected >= 0.6 else 0
"""
import argparse
import datetime as dt
import json
import os
import struct
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent.parent
TEST_DATA = ROOT / "bench" / "k6" / "test-data.json"
NORM_PATH = ROOT / "resources" / "normalization.json"
MCC_PATH  = ROOT / "resources" / "mcc_risk.json"
CENTS_PATH = ROOT / "data" / "ivf_centroids.bin"

NLIST = 256
DIM = 14
PADDED_DIM = 16
MCC_DEFAULT = 0.5


def load_centroids():
    raw = CENTS_PATH.read_bytes()
    expected = NLIST * PADDED_DIM * 4
    if len(raw) != expected:
        sys.exit(f"centroids file size {len(raw)} != expected {expected}")
    arr = np.frombuffer(raw, dtype=np.float32).reshape(NLIST, PADDED_DIM)
    return arr[:, :DIM].copy()  # discard padding


def load_norm():
    return json.loads(NORM_PATH.read_text())


def load_mcc():
    return json.loads(MCC_PATH.read_text())


def parse_iso8601(s):
    # All timestamps in test-data are ISO8601 with Z suffix; treat as UTC.
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    return dt.datetime.fromisoformat(s)


def monday_zero(weekday):
    # Python: Monday=0..Sunday=6 already matches MondayZero in Vectorizer.
    return weekday


def clamp01(x):
    if x < 0.0: return 0.0
    if x > 1.0: return 1.0
    return x


def vectorize(req, norm, mcc, out):
    """Mirror Vectorizer.cs Vectorize(req, dst)."""
    tx = req["transaction"]
    customer = req["customer"]
    merchant = req["merchant"]
    terminal = req["terminal"]

    amount = float(tx["amount"])
    out[0] = clamp01(amount / norm["max_amount"])

    installments = float(tx["installments"])
    out[1] = clamp01(installments / norm["max_installments"])

    avg = float(customer["avg_amount"])
    if avg > 0.0:
        ratio = (amount / avg) / norm["amount_vs_avg_ratio"]
    else:
        ratio = 1.0
    out[2] = clamp01(ratio)

    utc = parse_iso8601(tx["requested_at"])
    out[3] = utc.hour / 23.0
    out[4] = monday_zero(utc.weekday()) / 6.0

    last = req.get("last_transaction")
    if last is not None:
        last_ts = parse_iso8601(last["timestamp"])
        minutes = (utc - last_ts).total_seconds() / 60.0
        if minutes < 0.0:
            minutes = 0.0
        out[5] = clamp01(minutes / norm["max_minutes"])
        out[6] = clamp01(float(last["km_from_current"]) / norm["max_km"])
    else:
        out[5] = -1.0
        out[6] = -1.0

    out[7] = clamp01(float(terminal["km_from_home"]) / norm["max_km"])
    out[8] = clamp01(float(customer["tx_count_24h"]) / norm["max_tx_count_24h"])
    out[9]  = 1.0 if terminal["is_online"] else 0.0
    out[10] = 1.0 if terminal["card_present"] else 0.0

    known = customer.get("known_merchants") or []
    out[11] = 0.0 if merchant["id"] in known else 1.0
    out[12] = mcc.get(merchant["mcc"], MCC_DEFAULT)
    out[13] = clamp01(float(merchant["avg_amount"]) / norm["max_merchant_avg_amount"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "cascade" / "dataset.npz"))
    ap.add_argument("--limit", type=int, default=0, help="debug: only N entries")
    args = ap.parse_args()

    print(f"loading {TEST_DATA}…")
    data = json.loads(TEST_DATA.read_text())
    entries = data["entries"]
    if args.limit:
        entries = entries[:args.limit]
    n = len(entries)
    print(f"  {n} entries")

    norm = load_norm()
    mcc = load_mcc()
    cents = load_centroids()  # (256, 14)
    print(f"  centroids shape={cents.shape}")

    feats = np.zeros((n, DIM), dtype=np.float32)
    expected = np.zeros(n, dtype=np.float32)
    for i, e in enumerate(entries):
        vectorize(e["request"], norm, mcc, feats[i])
        expected[i] = float(e["expected_fraud_score"])
        if (i + 1) % 10000 == 0:
            print(f"  vectorized {i+1}/{n}")

    # Distances to all centroids: (n, 256) squared L2.
    # ||q-c||^2 = ||q||^2 + ||c||^2 - 2 q·c
    print("computing centroid distances…")
    q2 = (feats * feats).sum(axis=1, keepdims=True)              # (n,1)
    c2 = (cents * cents).sum(axis=1, keepdims=True).T            # (1,256)
    dot = feats @ cents.T                                        # (n,256)
    d2 = q2 + c2 - 2.0 * dot                                     # (n,256)
    np.maximum(d2, 0.0, out=d2)

    # Top-3 (smallest) per row.
    idx_part = np.argpartition(d2, 3, axis=1)[:, :3]             # (n,3) unsorted
    rows = np.arange(n)[:, None]
    d_part = d2[rows, idx_part]                                  # (n,3)
    order = np.argsort(d_part, axis=1)                           # sort within top-3
    top3_ids = idx_part[rows, order].astype(np.int32)            # (n,3)
    top3_d   = d_part[rows, order].astype(np.float32)            # (n,3)

    d_gap   = (top3_d[:, 1] - top3_d[:, 0]).astype(np.float32)
    d_ratio = np.where(top3_d[:, 1] > 0,
                       top3_d[:, 0] / top3_d[:, 1],
                       np.float32(1.0)).astype(np.float32)

    label = (expected >= 0.6).astype(np.int8)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path,
                        feats=feats,
                        top3_ids=top3_ids,
                        top3_d=top3_d,
                        d_gap=d_gap,
                        d_ratio=d_ratio,
                        expected=expected,
                        label=label)
    print(f"wrote {out_path}  ({out_path.stat().st_size/1024:.0f} KB)")
    print(f"label balance: fraud={int(label.sum())} / legit={int((1-label).sum())}  "
          f"(fraud_rate={label.mean():.3f})")
    print(f"expected histogram: "
          f"min={expected.min():.3f} max={expected.max():.3f} "
          f"unique={len(np.unique(expected))} values")


if __name__ == "__main__":
    main()
