"""
Export a decision stream for the operator hub.

Runs the multi-label detector over a flight, applies temporal smoothing to
confirm faults, derives battery-critical state, asks the policy for the
recommended action, and writes a compact JSON timeline the HTML hub replays.

Run:
    python ml/export_stream.py --flight data/raw/STRESS_multi.csv --out dashboard/stream.json
"""

import argparse
import json
import os
import sys

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from features import _window_features, WINDOW, STRIDE  # noqa: E402
from decision_policy import decide  # noqa: E402

RATE_HZ = 10
BATT_CRITICAL_V = 21.6


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--flight", default="data/raw/STRESS_multi.csv")
    ap.add_argument("--out", default="dashboard/stream.json")
    args = ap.parse_args()

    b = joblib.load("models/fault_multi.joblib")
    clf, FEATS, FAULTS = b["model"], b["features"], b["faults"]
    K, THRESH = b["smooth_k"], b["thresh"]

    df = pd.read_csv(args.flight)
    run = {f: 0 for f in FAULTS}     # consecutive-positive counters
    frames = []

    for s in range(0, len(df) - WINDOW + 1, STRIDE):
        w = df.iloc[s:s + WINDOW]
        x = pd.DataFrame([_window_features(w)])[FEATS]
        probs = np.stack([p[:, 1] for p in clf.predict_proba(x)], axis=1)[0]
        t = float(w["t"].iloc[-1])
        volt = float(w["bat_volt"].iloc[-1])

        active = []
        for j, f in enumerate(FAULTS):
            if probs[j] >= THRESH:
                run[f] += 1
            else:
                run[f] = 0
            if run[f] >= K:          # confirmed only after K in a row
                active.append(f)

        crit = ("battery" in active) and (volt <= BATT_CRITICAL_V)
        d = decide(active, battery_critical=crit)

        frames.append({
            "t": round(t, 1),
            "volt": round(volt, 1),
            "probs": {f: round(float(probs[j]), 2) for j, f in enumerate(FAULTS)},
            "active": active,
            "critical": crit,
            "decision": {
                "default": d["default"],
                "countdown": d["countdown_s"],
                "severity": d["severity"],
                "options": d["options"],
                "reason": d["reason"],
            },
        })

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump({"flight": os.path.basename(args.flight),
               "rate_hz": RATE_HZ, "faults": FAULTS, "frames": frames},
              open(args.out, "w"))
    n_alarm = sum(1 for f in frames if f["active"])
    first = next((f["t"] for f in frames if f["active"]), None)
    print(f"Wrote {args.out}  frames={len(frames)}  "
          f"first confirmed fault at t={first}s  alarm frames={n_alarm}")


if __name__ == "__main__":
    main()
