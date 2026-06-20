"""
Multi-label fault detector (v2).

One independent binary classifier PER fault (battery, gps, compass, motor),
wrapped in MultiOutputClassifier. A window can be positive for several faults
at once, which is what makes this multi-LABEL instead of single-class.

Trained on noisy multi-fault flights so it learns to tolerate noise. Evaluated
per fault on held-out FLIGHTS (no leakage). Also reports the effect of temporal
smoothing: requiring a fault to persist K consecutive windows before alarming,
which is the main noise / false-positive killer.

Run:
    python ml/train_multi.py
"""

import glob
import json
import os
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import precision_score, recall_score, f1_score

sys.path.insert(0, os.path.dirname(__file__))
from features import _window_features, WINDOW, STRIDE  # noqa: E402

DATA = "data/multi"
FAULTS = ["battery", "gps", "compass", "motor"]
LBL = [f"lbl_{f}" for f in FAULTS]
RATE_HZ = 10
SMOOTH_K = 3          # consecutive positive windows required to confirm
THRESH = 0.5


def windows_for(df):
    X, Y = [], []
    for s in range(0, len(df) - WINDOW + 1, STRIDE):
        w = df.iloc[s:s + WINDOW]
        X.append(_window_features(w))
        Y.append([int(w[c].iloc[-1]) for c in LBL])
    return pd.DataFrame(X), np.array(Y)


def smooth(binary_seq, k):
    """Confirm a positive only after k consecutive raw positives."""
    out = np.zeros_like(binary_seq)
    run = 0
    for i, v in enumerate(binary_seq):
        run = run + 1 if v else 0
        out[i] = 1 if run >= k else 0
    return out


def main():
    os.makedirs("models", exist_ok=True)
    os.makedirs("reports", exist_ok=True)

    paths = sorted(p for p in glob.glob(os.path.join(DATA, "*.csv"))
                   if "_manifest" not in p)
    frames = [pd.read_csv(p) for p in paths]

    rng = np.random.default_rng(0)
    idx = rng.permutation(len(frames))
    cut = int(0.30 * len(frames))
    test_f = [frames[i] for i in idx[:cut]]
    train_f = [frames[i] for i in idx[cut:]]

    # build train/test window matrices
    Xtr_list, Ytr_list = [], []
    for f in train_f:
        X, Y = windows_for(f); Xtr_list.append(X); Ytr_list.append(Y)
    Xtr = pd.concat(Xtr_list, ignore_index=True); Ytr = np.vstack(Ytr_list)
    Xte_list, Yte_list, te_meta = [], [], []
    for f in test_f:
        X, Y = windows_for(f); Xte_list.append(X); Yte_list.append(Y)
    Xte = pd.concat(Xte_list, ignore_index=True); Yte = np.vstack(Yte_list)

    print(f"Train flights {len(train_f)}  Test flights {len(test_f)}")
    print(f"Train windows {len(Xtr)}  Test windows {len(Xte)}")

    base = GradientBoostingClassifier(random_state=0)
    clf = MultiOutputClassifier(base)
    clf.fit(Xtr, Ytr)

    # raw per-fault metrics on held-out windows
    proba = np.stack([p[:, 1] for p in clf.predict_proba(Xte)], axis=1)
    pred_raw = (proba >= THRESH).astype(int)

    print("\nPer-fault (raw window-level):")
    metrics = {}
    for j, f in enumerate(FAULTS):
        p = precision_score(Yte[:, j], pred_raw[:, j], zero_division=0)
        r = recall_score(Yte[:, j], pred_raw[:, j], zero_division=0)
        f1 = f1_score(Yte[:, j], pred_raw[:, j], zero_division=0)
        metrics[f] = {"precision": round(p, 3), "recall": round(r, 3), "f1": round(f1, 3)}
        print(f"  {f:8s} P={p:.2f} R={r:.2f} F1={f1:.2f}")

    # effect of temporal smoothing, measured per test flight then pooled
    fp_raw = fp_smooth = pos_windows = 0
    for f in test_f:
        X, Y = windows_for(f)
        pr = np.stack([pp[:, 1] for pp in clf.predict_proba(X)], axis=1) >= THRESH
        for j in range(len(FAULTS)):
            raw = pr[:, j].astype(int)
            sm = smooth(raw, SMOOTH_K)
            truth = Y[:, j]
            # false positives = predicted 1 where truth 0
            fp_raw += int(((raw == 1) & (truth == 0)).sum())
            fp_smooth += int(((sm == 1) & (truth == 0)).sum())
            pos_windows += int((truth == 1).sum())

    print(f"\nFalse-positive windows  raw: {fp_raw}   "
          f"after smoothing(K={SMOOTH_K}): {fp_smooth}   "
          f"({100*(1-fp_smooth/max(fp_raw,1)):.0f}% reduction)")

    joblib.dump({"model": clf, "features": list(Xtr.columns),
                 "faults": FAULTS, "thresh": THRESH, "smooth_k": SMOOTH_K},
                "models/fault_multi.joblib")
    json.dump({"per_fault": metrics, "fp_raw": fp_raw, "fp_smooth": fp_smooth,
               "smooth_k": SMOOTH_K, "thresh": THRESH},
              open("models/metrics_multi.json", "w"), indent=2)
    print("\nSaved models/fault_multi.joblib")


if __name__ == "__main__":
    main()
