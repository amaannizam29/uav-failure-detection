"""
Render a 3-panel results figure for the slide deck and README:
  (A) confusion matrix on held-out flights
  (B) top feature importances
  (C) live failure-probability timeline for one example failure flight

Run after train.py:
    python ml/make_report_figure.py
"""

import glob
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
from features import extract, WINDOW, STRIDE  # noqa: E402

RATE_HZ = 10
plt.rcParams.update({"font.size": 9, "figure.dpi": 130})


def main():
    bundle = joblib.load("models/failure_clf.joblib")
    clf, feats, labels = bundle["model"], bundle["features"], bundle["labels"]
    metrics = json.load(open("models/metrics.json"))
    cm = np.array(metrics["confusion_matrix"])

    fig = plt.figure(figsize=(13, 4.2), constrained_layout=True)
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1.1, 1.3])

    # (A) Confusion matrix (row-normalised)
    ax = fig.add_subplot(gs[0])
    cmn = cm / cm.sum(axis=1, keepdims=True).clip(min=1)
    im = ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(labels))); ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right"); ax.set_yticklabels(labels)
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, f"{cmn[i, j]:.2f}", ha="center", va="center",
                    color="white" if cmn[i, j] > 0.5 else "black", fontsize=8)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title(f"(A) Confusion matrix\nacc={metrics['accuracy']:.3f}  "
                 f"macroF1={metrics['macro_f1']:.3f}")

    # (B) Feature importance
    ax = fig.add_subplot(gs[1])
    imp = pd.Series(clf.feature_importances_, index=feats).sort_values()[-12:]
    ax.barh(range(len(imp)), imp.values, color="#2b6cb0")
    ax.set_yticks(range(len(imp))); ax.set_yticklabels(imp.index, fontsize=7)
    ax.set_xlabel("Gini importance"); ax.set_title("(B) Top features")

    # (C) Live probability timeline on one motor-failure flight
    ax = fig.add_subplot(gs[2])
    mf = sorted(glob.glob("data/raw/*_motor.csv"))[0]
    df = pd.read_csv(mf)
    X, y, _ = extract(df)
    proba = clf.predict_proba(X[feats])
    end_t = (np.arange(len(X)) * STRIDE + WINDOW - 1) / RATE_HZ
    for k, lab in enumerate(labels):
        ax.plot(end_t, proba[:, k], label=lab, lw=1.6)
    onset_idx = np.argmax(df["label"].values != "nominal")
    ax.axvline(onset_idx / RATE_HZ, ls="--", color="grey", lw=1)
    ax.text(onset_idx / RATE_HZ, 1.02, "degradation onset", fontsize=7,
            ha="center", color="grey")
    ax.set_xlabel("Flight time (s)"); ax.set_ylabel("P(class)")
    ax.set_ylim(0, 1.08); ax.set_title("(C) Live failure probability — motor flight")
    ax.legend(fontsize=7, loc="center left")

    os.makedirs("reports", exist_ok=True)
    out = "reports/results_overview.png"
    fig.savefig(out, bbox_inches="tight")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
