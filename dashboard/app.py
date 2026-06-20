"""
Live failure-detection dashboard.

Replays a flight CSV sample-by-sample (or loads a real logged flight) and shows:
  - current telemetry gauges
  - rolling failure-probability for each class
  - a banner alarm when any failure class crosses threshold

Run:
    streamlit run dashboard/app.py

Pick any file from data/raw in the sidebar. The model in models/ must exist
(run ml/train.py first).
"""

import glob
import os
import sys

import joblib
import numpy as np
import pandas as pd
import streamlit as st

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "ml"))
from features import _window_features, WINDOW, STRIDE  # noqa: E402

RATE_HZ = 10
ALARM_THRESH = 0.6

st.set_page_config(page_title="UAV Failure Detection", layout="wide")
st.title("UAV Predictive Failure Detection")


@st.cache_resource
def load_model():
    return joblib.load("models/failure_clf.joblib")


bundle = load_model()
clf, FEATS, LABELS = bundle["model"], bundle["features"], bundle["labels"]

files = sorted(p for p in glob.glob("data/raw/*.csv") if "_manifest" not in p)
sel = st.sidebar.selectbox("Flight log", files,
                           format_func=lambda p: os.path.basename(p))
speed = st.sidebar.slider("Replay speed (x realtime)", 1, 50, 20)
run = st.sidebar.button("Run replay")

df = pd.read_csv(sel)
st.sidebar.write(f"Rows: {len(df)}   Duration: {df['t'].iloc[-1]:.0f} s")

col1, col2, col3, col4 = st.columns(4)
g_volt = col1.empty(); g_curr = col2.empty()
g_sats = col3.empty(); g_vibe = col4.empty()
banner = st.empty()
chart = st.empty()

if run:
    import time
    prob_hist = {lab: [] for lab in LABELS}
    times = []
    triggered = False
    for start in range(0, len(df) - WINDOW + 1, STRIDE):
        w = df.iloc[start:start + WINDOW]
        feats = pd.DataFrame([_window_features(w)])[FEATS]
        proba = clf.predict_proba(feats)[0]
        t_now = w["t"].iloc[-1]
        times.append(t_now)
        for k, lab in enumerate(LABELS):
            prob_hist[lab].append(proba[k])

        last = w.iloc[-1]
        g_volt.metric("Battery (V)", f"{last['bat_volt']:.1f}")
        g_curr.metric("Current (A)", f"{last['bat_curr']:.1f}")
        g_sats.metric("GPS sats", int(last["gps_nsats"]))
        g_vibe.metric("Vibration", f"{last['vibe']:.1f}")

        top = LABELS[int(np.argmax(proba))]
        top_p = float(np.max(proba))
        if top != "nominal" and top_p >= ALARM_THRESH:
            banner.error(f"ALARM: {top.upper()} failure predicted  "
                         f"(p={top_p:.2f})  at t={t_now:.0f}s")
            triggered = True
        elif not triggered:
            banner.success(f"Nominal  (p={top_p:.2f})  t={t_now:.0f}s")

        chart.line_chart(pd.DataFrame(prob_hist, index=times))
        time.sleep(max(0.0, (STRIDE / RATE_HZ) / speed))

    st.write("Replay complete.")
else:
    st.info("Pick a flight and press Run replay. "
            "Failure flights show the probability flipping to the failure "
            "class before the flight ends.")
