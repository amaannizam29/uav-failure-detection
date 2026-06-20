"""
Shared stream core.

One place that turns a window of telemetry into a decision frame, used by both
the offline exporter and the live server, so CSV replay and live MAVLink produce
identical frame shapes and identical smoothing / policy behaviour.
"""

import os
import sys

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from features import _window_features, WINDOW, STRIDE  # noqa: E402
from decision_policy import decide  # noqa: E402

BATT_CRITICAL_V = 21.6
_MODEL = None


def load_model(path="models/fault_multi.joblib"):
    global _MODEL
    if _MODEL is None:
        _MODEL = joblib.load(path)
    return _MODEL


class StreamState:
    """Holds the consecutive-positive counters so smoothing works across calls
    (needed for live, where frames arrive one at a time)."""
    def __init__(self, faults):
        self.run = {f: 0 for f in faults}


def frame_from_window(window_df, state, bundle):
    """window_df: DataFrame with WINDOW raw samples (same columns as the logger).
    Returns one decision frame dict."""
    clf, FEATS = bundle["model"], bundle["features"]
    FAULTS, K, THRESH = bundle["faults"], bundle["smooth_k"], bundle["thresh"]

    x = pd.DataFrame([_window_features(window_df)])[FEATS]
    probs = np.stack([p[:, 1] for p in clf.predict_proba(x)], axis=1)[0]
    t = float(window_df["t"].iloc[-1])
    volt = float(window_df["bat_volt"].iloc[-1])

    active = []
    for j, f in enumerate(FAULTS):
        state.run[f] = state.run[f] + 1 if probs[j] >= THRESH else 0
        if state.run[f] >= K:
            active.append(f)

    crit = ("battery" in active) and (volt <= BATT_CRITICAL_V)
    d = decide(active, battery_critical=crit)
    return {
        "t": round(t, 1), "volt": round(volt, 1),
        "probs": {f: round(float(probs[j]), 2) for j, f in enumerate(FAULTS)},
        "active": active, "critical": crit,
        "decision": {"default": d["default"], "countdown": d["countdown_s"],
                     "severity": d["severity"], "options": d["options"],
                     "reason": d["reason"]},
    }


def compute_stream(df, bundle=None):
    """Offline: full flight DataFrame -> list of frames."""
    bundle = bundle or load_model()
    state = StreamState(bundle["faults"])
    frames = []
    for s in range(0, len(df) - WINDOW + 1, STRIDE):
        frames.append(frame_from_window(df.iloc[s:s + WINDOW], state, bundle))
    return frames
