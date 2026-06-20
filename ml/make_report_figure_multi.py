"""
Results figure for the MULTI-LABEL model.

Three panels suited to independent per-fault detectors (unlike the single-class
figure):
  (A) per-fault precision / recall / F1 (the real headline numbers)
  (B) top features aggregated across the four detectors
  (C) multi-fault timeline on the stress flight: several faults rising in sequence

Run after train_multi.py:
    python ml/make_report_figure_multi.py
Writes: reports/results_overview_multi.png
"""

import json
import os
import sys

import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
from features import _window_features, WINDOW, STRIDE  # noqa: E402

RATE_HZ = 10
plt.rcParams.update({"font.size": 9, "figure.dpi": 130})
COLORS = {"battery": "#e67e22", "gps": "#3498db",
          "compass": "#9b59b6", "motor": "#e74c3c"}


def main():
    bundle = joblib.load("models/fault_multi.joblib")
    clf, FEATS, FAULTS = bundle["model"], bundle["features"], bundle["faults"]
    metrics = json.load(open("models/metrics_multi.json"))
    per = metrics["per_fault"]

    fig = plt.figure(figsize=(13.5, 4.3), constrained_layout=True)
    gs = fig.add_gridspec(1, 3, width_ratios=[1.1, 1.1, 1.4])

    # (A) per-fault P/R/F1 grouped bars
    ax = fig.add_subplot(gs[0])
    x = np.arange(len(FAULTS))
    w = 0.26
    P = [per[f]["precision"] for f in FAULTS]
    R = [per[f]["recall"] for f in FAULTS]
    F = [per[f]["f1"] for f in FAULTS]
    ax.bar(x - w, P, w, label="Precision", color="#2b6cb0")
    ax.bar(x, R, w, label="Recall", color="#48bb78")
    ax.bar(x + w, F, w, label="F1", color="#ed8936")
    ax.set_xticks(x); ax.set_xticklabels(FAULTS, rotation=20)
    ax.set_ylim(0.8, 1.01); ax.set_ylabel("score")
    fpr = 100 * (1 - metrics["fp_smooth"] / max(metrics["fp_raw"], 1))
    ax.set_title(f"(A) Per-fault performance\nFP windows {metrics['fp_raw']}→"
                 f"{metrics['fp_smooth']} ({fpr:.0f}% less) via smoothing")
    ax.legend(fontsize=7, loc="lower right")

    # (B) feature importance averaged across the four detectors
    ax = fig.add_subplot(gs[1])
    imp = np.zeros(len(FEATS))
    for est in clf.estimators_:
        imp += est.feature_importances_
    imp /= len(clf.estimators_)
    s = pd.Series(imp, index=FEATS).sort_values()[-12:]
    ax.barh(range(len(s)), s.values, color="#2b6cb0")
    ax.set_yticks(range(len(s))); ax.set_yticklabels(s.index, fontsize=7)
    ax.set_xlabel("mean importance across detectors")
    ax.set_title("(B) Top features (all faults)")

    # (C) multi-fault timeline on the stress flight
    ax = fig.add_subplot(gs[2])
    df = pd.read_csv("data/raw/STRESS_multi.csv")
    times, probs = [], {f: [] for f in FAULTS}
    for st in range(0, len(df) - WINDOW + 1, STRIDE):
        wdf = df.iloc[st:st + WINDOW]
        xrow = pd.DataFrame([_window_features(wdf)])[FEATS]
        pr = np.stack([p[:, 1] for p in clf.predict_proba(xrow)], axis=1)[0]
        times.append(wdf["t"].iloc[-1])
        for j, f in enumerate(FAULTS):
            probs[f].append(pr[j])
    for f in FAULTS:
        ax.plot(times, probs[f], label=f, lw=1.7, color=COLORS[f])
    for onset, lab in [(90, "battery"), (150, "compass"), (210, "motor")]:
        ax.axvline(onset, ls="--", color="grey", lw=0.8)
    ax.set_xlabel("Flight time (s)"); ax.set_ylabel("P(fault)")
    ax.set_ylim(0, 1.05)
    ax.set_title("(C) Multi-fault timeline — stress flight\n(battery→compass→motor onsets)")
    ax.legend(fontsize=7, loc="center left")

    os.makedirs("reports", exist_ok=True)
    out = "reports/results_overview_multi.png"
    fig.savefig(out, bbox_inches="tight")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
