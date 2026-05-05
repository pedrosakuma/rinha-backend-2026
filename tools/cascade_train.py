#!/usr/bin/env python3
"""Train a DecisionTreeClassifier as a pre-classifier for the IVF scorer.

The cascade philosophy:
  - Tree decides on "easy" queries (pure leaves with high coverage).
  - "Difficult" queries (impure leaves) fall through to IvfScorer.

Output:
  cascade/tree.json   — serialized tree + leaf metadata for C# emission.

Reports per depth: coverage (% queries the tree decides), accuracy on those,
and predicted score gain (cascade time saved × queries decided).
"""
import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import KFold


ROOT = Path(__file__).resolve().parent.parent
DATASET = ROOT / "cascade" / "dataset.npz"
OUT_TREE = ROOT / "cascade" / "tree.json"

FEATURE_NAMES = [f"v{i}" for i in range(14)] + [
    "c0_id", "c1_id", "c2_id",
    "d0", "d1", "d2",
    "d_gap", "d_ratio",
]


def build_X(npz):
    feats = npz["feats"]                     # (n,14)
    top3_ids = npz["top3_ids"].astype(np.float32)  # (n,3)
    top3_d = npz["top3_d"]                   # (n,3)
    d_gap = npz["d_gap"][:, None]
    d_ratio = npz["d_ratio"][:, None]
    return np.hstack([feats, top3_ids, top3_d, d_gap, d_ratio]).astype(np.float32)


def evaluate_tree(clf, X, y, purity_threshold):
    """For each leaf in the trained tree, decide if it's pure enough to commit.
    Pure leaves contribute to coverage. Returns (coverage, errors_in_decided).
    """
    # leaf id for each sample.
    leaves = clf.apply(X)
    leaf_stats = {}
    for leaf_id in np.unique(leaves):
        mask = leaves == leaf_id
        n = int(mask.sum())
        frauds = int(y[mask].sum())
        # majority class label & purity.
        if frauds * 2 >= n:
            pred, count = 1, frauds
        else:
            pred, count = 0, n - frauds
        purity = count / n if n > 0 else 0.0
        decided = purity >= purity_threshold
        leaf_stats[int(leaf_id)] = {
            "n": n, "frauds": frauds, "pred": pred, "purity": purity,
            "decided": decided,
        }
    # coverage & errors.
    decided_mask = np.array([leaf_stats[int(l)]["decided"] for l in leaves])
    coverage = decided_mask.mean()
    if decided_mask.any():
        preds = np.array([leaf_stats[int(l)]["pred"] for l in leaves])
        errors = int(((preds != y) & decided_mask).sum())
    else:
        errors = 0
    return coverage, errors, leaf_stats


def cv_evaluate(X, y, depth, min_samples_leaf, purity, n_splits=5, seed=42):
    """K-fold to confirm coverage and decided-error generalize."""
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    covs, errs_decided = [], []
    for train_idx, test_idx in kf.split(X):
        clf = DecisionTreeClassifier(max_depth=depth,
                                     min_samples_leaf=min_samples_leaf,
                                     random_state=seed)
        clf.fit(X[train_idx], y[train_idx])
        # Decide which leaves to commit using TRAIN purity, then measure on TEST.
        train_leaves = clf.apply(X[train_idx])
        leaf_decided = {}
        leaf_pred = {}
        for leaf_id in np.unique(train_leaves):
            mask = train_leaves == leaf_id
            n = int(mask.sum())
            frauds = int(y[train_idx][mask].sum())
            if frauds * 2 >= n:
                pred, count = 1, frauds
            else:
                pred, count = 0, n - frauds
            p = count / n if n > 0 else 0.0
            leaf_decided[int(leaf_id)] = p >= purity
            leaf_pred[int(leaf_id)] = pred
        test_leaves = clf.apply(X[test_idx])
        decided_mask = np.array([leaf_decided.get(int(l), False) for l in test_leaves])
        preds = np.array([leaf_pred.get(int(l), 0) for l in test_leaves])
        if decided_mask.any():
            err = int(((preds != y[test_idx]) & decided_mask).sum())
        else:
            err = 0
        covs.append(decided_mask.mean())
        errs_decided.append(err)
    return float(np.mean(covs)), int(sum(errs_decided))


def export_tree(clf, leaf_stats, purity_threshold, out_path):
    """Serialize tree (preorder) for C# emission."""
    t = clf.tree_
    nodes = []
    for i in range(t.node_count):
        if t.children_left[i] == t.children_right[i]:
            # Leaf.
            stats = leaf_stats.get(i, {})
            nodes.append({
                "node_id": i,
                "is_leaf": True,
                "decided": bool(stats.get("decided", False)),
                "pred":    int(stats.get("pred", 0)),
                "n":       int(stats.get("n", 0)),
                "purity":  float(stats.get("purity", 0.0)),
            })
        else:
            nodes.append({
                "node_id": i,
                "is_leaf": False,
                "feature_idx": int(t.feature[i]),
                "feature_name": FEATURE_NAMES[int(t.feature[i])],
                "threshold": float(t.threshold[i]),
                "left":  int(t.children_left[i]),
                "right": int(t.children_right[i]),
            })
    payload = {
        "feature_names": FEATURE_NAMES,
        "purity_threshold": purity_threshold,
        "n_features": int(clf.n_features_in_),
        "max_depth": int(clf.get_depth()),
        "n_leaves": int(clf.get_n_leaves()),
        "nodes": nodes,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"wrote {out_path}  ({out_path.stat().st_size/1024:.0f} KB)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--purity", type=float, default=1.0,
                    help="leaf purity threshold to commit a decision (1.0=must be pure)")
    ap.add_argument("--min-leaf", type=int, default=20)
    ap.add_argument("--depths", default="4,5,6,7,8,10,12")
    ap.add_argument("--final-depth", type=int, default=0,
                    help="depth to actually export (0=picks best by CV coverage with 0 errors)")
    args = ap.parse_args()

    npz = np.load(DATASET)
    X = build_X(npz)
    y = npz["label"].astype(np.int8)
    n = len(y)
    print(f"dataset: n={n}  features={X.shape[1]}  fraud_rate={y.mean():.3f}")

    depths = [int(d) for d in args.depths.split(",")]
    print(f"\nsweeping depths {depths} (purity≥{args.purity}, min_leaf={args.min_leaf})")
    print(f"{'depth':>5} {'leaves':>7} {'cov_train':>10} {'err_train':>10} "
          f"{'cov_cv':>8} {'err_cv':>8}")
    print("-" * 60)
    rows = []
    for d in depths:
        clf = DecisionTreeClassifier(max_depth=d,
                                     min_samples_leaf=args.min_leaf,
                                     random_state=42)
        clf.fit(X, y)
        cov_train, err_train, _ = evaluate_tree(clf, X, y, args.purity)
        cov_cv, err_cv = cv_evaluate(X, y, d, args.min_leaf, args.purity)
        print(f"{d:>5} {clf.get_n_leaves():>7} {cov_train:>10.3%} {err_train:>10} "
              f"{cov_cv:>8.3%} {err_cv:>8}")
        rows.append((d, cov_cv, err_cv))

    if args.final_depth:
        chosen = args.final_depth
    else:
        # Pick depth with max CV coverage subject to err_cv == 0.
        zero_err = [r for r in rows if r[2] == 0]
        chosen = max(zero_err, key=lambda r: r[1])[0] if zero_err else rows[-1][0]

    print(f"\nchosen depth = {chosen}")
    clf = DecisionTreeClassifier(max_depth=chosen,
                                 min_samples_leaf=args.min_leaf,
                                 random_state=42)
    clf.fit(X, y)
    cov, err, leaf_stats = evaluate_tree(clf, X, y, args.purity)
    print(f"final tree: leaves={clf.get_n_leaves()} "
          f"coverage={cov:.3%}  errors_decided={err}")
    decided_leaves = sum(1 for s in leaf_stats.values() if s["decided"])
    reject_leaves = len(leaf_stats) - decided_leaves
    print(f"leaves decided={decided_leaves}  rejected={reject_leaves}")

    export_tree(clf, leaf_stats, args.purity, OUT_TREE)


if __name__ == "__main__":
    main()
