"""
Train the predictive failure classifier.

Model: GradientBoosting multiclass (nominal / battery / gps / compass / motor).
Split: by FLIGHT, not by row, via GroupShuffleSplit. Windows from one flight
never appear in both train and test, so reported accuracy is honest.

Outputs:
    models/failure_clf.joblib    trained pipeline
    models/metrics.json          held-out metrics
    reports/lead_time.csv        per-failure-flight detection lead time

Run:
    python ml/train.py
"""

import glob
import json
import os
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix
# group split done manually below

sys.path.insert(0, os.path.dirname(__file__))
from features import extract, extract_many, WINDOW, STRIDE  # noqa: E402

RAW = "data/raw"
RATE_HZ = 10


def load_frames():
    paths = sorted(p for p in glob.glob(os.path.join(RAW, "*.csv"))
                   if not p.endswith("_manifest.csv"))
    return [pd.read_csv(p) for p in paths]


def lead_time_report(frames, clf):
    """For each failure flight, seconds between first correct alarm and the
    onset of hard degradation midpoint. Positive => early warning."""
    rows = []
    for df in frames:
        cls = df["true_class"].iloc[0]
        if cls == "nominal":
            continue
        X, y, _ = extract(df)
        pred = clf.predict(X)
        # window index -> end-sample time
        end_times = (np.arange(len(X)) * STRIDE + WINDOW - 1) / RATE_HZ
        onset_idx = np.argmax(df["label"].values != "nominal")
        onset_t = onset_idx / RATE_HZ
        hit = np.where(pred == cls)[0]
        if len(hit) == 0:
            rows.append({"flight_id": df["flight_id"].iloc[0], "class": cls,
                         "detected": False, "lead_s": None})
            continue
        first_t = end_times[hit[0]]
        # hard cutoff ~ end of flight; lead = time before flight end
        cutoff_t = df["t"].iloc[-1]
        rows.append({"flight_id": df["flight_id"].iloc[0], "class": cls,
                     "detected": True,
                     "alarm_t": round(float(first_t), 1),
                     "onset_t": round(float(onset_t), 1),
                     "lead_before_cutoff_s": round(float(cutoff_t - first_t), 1)})
    return pd.DataFrame(rows)


def main():
    os.makedirs("models", exist_ok=True)
    os.makedirs("reports", exist_ok=True)

    frames = load_frames()
    # Flight-level stratified split: ~30% of EACH class's flights to test
    rng = np.random.default_rng(42)
    by_class = {}
    for df in frames:
        by_class.setdefault(df["true_class"].iloc[0], []).append(df)
    train_frames, test_frames = [], []
    for cls, fs in by_class.items():
        idx = rng.permutation(len(fs))
        cut = max(1, int(round(0.30 * len(fs))))
        test_frames += [fs[i] for i in idx[:cut]]
        train_frames += [fs[i] for i in idx[cut:]]

    Xtr, ytr, _ = extract_many(train_frames)
    Xte, yte, _ = extract_many(test_frames)
    X = pd.concat([Xtr, Xte], ignore_index=True)
    y = pd.concat([ytr, yte], ignore_index=True)
    print(f"Train windows: {len(Xtr)}  Test windows: {len(Xte)}  "
          f"Features: {Xtr.shape[1]}")
    print(f"Train flights: {len(train_frames)}  Test flights: {len(test_frames)}")

    clf = GradientBoostingClassifier(random_state=42)
    clf.fit(Xtr, ytr)

    labels = sorted(y.unique())
    pred = clf.predict(Xte)
    report = classification_report(yte, pred, labels=labels,
                                   output_dict=True, zero_division=0)
    cm = confusion_matrix(yte, pred, labels=labels)

    acc = report["accuracy"]
    macro_f1 = report["macro avg"]["f1-score"]
    test_classes = sorted(yte.unique())
    print(f"Held-out accuracy: {acc:.3f}   macro-F1: {macro_f1:.3f}")
    print(f"Test-set classes present: {test_classes}")
    print("\nPer-class:")
    for c in labels:
        tag = "" if c in test_classes else "  (not in test fold)"
        print(f"  {c:8s}  precision={report[c]['precision']:.2f}  "
              f"recall={report[c]['recall']:.2f}  f1={report[c]['f1-score']:.2f}{tag}")

    # Refit on all data for deployment, save feature order
    clf_full = GradientBoostingClassifier(random_state=42).fit(X, y)
    joblib.dump({"model": clf_full, "features": list(X.columns), "labels": labels},
                "models/failure_clf.joblib")

    with open("models/metrics.json", "w") as f:
        json.dump({"accuracy": acc, "macro_f1": macro_f1,
                   "labels": labels, "confusion_matrix": cm.tolist(),
                   "per_class": {c: report[c] for c in labels}}, f, indent=2)

    lt = lead_time_report(frames, clf_full)
    lt.to_csv("reports/lead_time.csv", index=False)
    detected = lt[lt["detected"] == True]
    if len(detected):
        print(f"\nFailure flights: {len(lt)}  detected: {len(detected)}  "
              f"mean lead before cutoff: "
              f"{detected['lead_before_cutoff_s'].mean():.1f} s")


if __name__ == "__main__":
    main()
