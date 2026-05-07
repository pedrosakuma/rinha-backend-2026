#!/usr/bin/env python3
"""Probe the running API on every entry of test-data.json and identify borderline queries.

A borderline query is one whose fraud_score is exactly at or near the 0.6 decision
boundary, and/or whose API decision disagrees with expected_approved.

Outputs a CSV-style report with the most relevant rows.
"""
import json, sys, os, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from http.client import HTTPConnection

URL_HOST = "localhost"
URL_PORT = 9999
PATH = "/fraud-score"
WORKERS = 32

def load_test_data(path):
    with open(path) as f:
        d = json.load(f)
    return d["entries"]

def score_one(entry):
    body = json.dumps(entry["request"]).encode()
    conn = HTTPConnection(URL_HOST, URL_PORT, timeout=5)
    try:
        conn.request("POST", PATH, body=body, headers={"Content-Type": "application/json"})
        resp = conn.getresponse()
        data = resp.read()
        if resp.status != 200:
            return (entry["request"]["id"], None, None, entry.get("expected_approved"), f"http {resp.status}")
        j = json.loads(data)
        return (entry["request"]["id"], j["approved"], j["fraud_score"], entry.get("expected_approved"), None)
    finally:
        conn.close()

def main(test_data_path, n_repeat=1):
    entries = load_test_data(test_data_path)
    print(f"Loaded {len(entries)} entries; sending {len(entries)*n_repeat} requests with {WORKERS} workers", file=sys.stderr)

    # Repeat each entry n_repeat times to detect non-determinism
    work = []
    for rep in range(n_repeat):
        for e in entries:
            work.append(e)

    # Aggregate per-id: list of (approved, score)
    per_id = {}  # id -> {"expected": bool, "responses": [(approved, score, err)]}
    t0 = time.time()
    done = 0
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futs = [ex.submit(score_one, e) for e in work]
        for fut in as_completed(futs):
            tx_id, approved, score, expected, err = fut.result()
            d = per_id.setdefault(tx_id, {"expected": expected, "responses": []})
            d["responses"].append((approved, score, err))
            done += 1
            if done % 10000 == 0:
                elapsed = time.time() - t0
                rate = done / elapsed
                print(f"  {done}/{len(work)} ({rate:.0f} req/s)", file=sys.stderr)
    elapsed = time.time() - t0
    print(f"Done in {elapsed:.1f}s ({len(work)/elapsed:.0f} req/s)", file=sys.stderr)

    # Analysis
    score_buckets = {}
    fns = []      # expected fraud (expected=False), got approved=True
    fps = []      # expected legit (expected=True), got approved=False
    flippy = []   # same id, different responses across repeats
    errors = []
    for tx_id, d in per_id.items():
        exp = d["expected"]
        responses = d["responses"]
        # Detect flippiness
        approveds = set(r[0] for r in responses if r[2] is None)
        scores = set(r[1] for r in responses if r[2] is None)
        for r in responses:
            if r[2] is not None:
                errors.append((tx_id, r[2]))
                continue
            approved, score = r[0], r[1]
            score_buckets[score] = score_buckets.get(score, 0) + 1
            if exp is False and approved is True:
                fns.append((tx_id, score, exp, approved))
            elif exp is True and approved is False:
                fps.append((tx_id, score, exp, approved))
        if len(approveds) > 1 or len(scores) > 1:
            flippy.append((tx_id, exp, responses))

    print()
    print("=== Score distribution ===")
    for s in sorted(score_buckets):
        print(f"  score={s:.4f}  count={score_buckets[s]}")
    print()
    print(f"=== False negatives (expected fraud, got approved) — total {len(fns)} ===")
    for tx_id, score, exp, approved in fns[:30]:
        print(f"  id={tx_id} score={score} expected={exp} approved={approved}")
    print()
    print(f"=== False positives (expected legit, got rejected) — total {len(fps)} ===")
    for tx_id, score, exp, approved in fps[:30]:
        print(f"  id={tx_id} score={score} expected={exp} approved={approved}")
    print()
    print(f"=== Flippy across repeats (n_repeat={n_repeat}) — total {len(flippy)} ===")
    for tx_id, exp, responses in flippy[:30]:
        print(f"  id={tx_id} expected={exp}: {responses}")
    print()
    print(f"=== HTTP errors — total {len(errors)} ===")
    for tx_id, err in errors[:10]:
        print(f"  id={tx_id} {err}")

if __name__ == "__main__":
    n_repeat = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    main(sys.argv[1], n_repeat)
