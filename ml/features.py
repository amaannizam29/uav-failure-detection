"""
Feature extraction.

Converts raw per-sample telemetry into fixed-length sliding windows, then
computes physically meaningful features per window. Windowing is what makes the
model "predictive": each window sees a short slice of recent behaviour and must
call the failure class before hard cutoff.

A window is labelled by its LAST sample's label, so a window that overlaps the
degradation onset is already labelled with the failure class.
"""

import numpy as np
import pandas as pd

WINDOW = 30          # samples per window (3 s at 10 Hz)
STRIDE = 10          # hop (1 s)

SIGNALS = ["bat_volt", "bat_curr", "bat_soc", "gps_nsats", "gps_hdop",
           "gps_innov", "mag_field", "mag_innov", "att_err", "vibe"]
MOTORS = ["mot1", "mot2", "mot3", "mot4"]


def _slope(x):
    n = len(x)
    if n < 2:
        return 0.0
    t = np.arange(n)
    return np.polyfit(t, x, 1)[0]


def _window_features(w):
    f = {}
    for s in SIGNALS:
        v = w[s].values
        f[f"{s}_mean"] = v.mean()
        f[f"{s}_std"] = v.std()
        f[f"{s}_slope"] = _slope(v)
        f[f"{s}_min"] = v.min()
        f[f"{s}_max"] = v.max()
    # Motor-spread features: a single failing motor breaks symmetry
    mot = w[MOTORS].values
    f["mot_spread_mean"] = (mot.max(axis=1) - mot.min(axis=1)).mean()
    f["mot_spread_max"] = (mot.max(axis=1) - mot.min(axis=1)).max()
    f["mot_max_slope"] = max(_slope(w[m].values) for m in MOTORS)
    return f


def extract(df):
    """Return (X dataframe, y series, group series) for one flight."""
    rows, labels, groups = [], [], []
    n = len(df)
    for start in range(0, n - WINDOW + 1, STRIDE):
        w = df.iloc[start:start + WINDOW]
        rows.append(_window_features(w))
        labels.append(w["label"].iloc[-1])
        groups.append(df["flight_id"].iloc[0])
    X = pd.DataFrame(rows)
    return X, pd.Series(labels, name="label"), pd.Series(groups, name="flight_id")


def extract_many(frames):
    Xs, ys, gs = [], [], []
    for df in frames:
        X, y, g = extract(df)
        Xs.append(X); ys.append(y); gs.append(g)
    return (pd.concat(Xs, ignore_index=True),
            pd.concat(ys, ignore_index=True),
            pd.concat(gs, ignore_index=True))
