"""
Stress-test flight: heavy sensor noise + MULTIPLE overlapping failures.

This deliberately violates the model's single-fault training assumption, so you
can watch it behave under conditions it was never trained for. Expect:
  - the prediction to flicker between classes instead of a clean step
  - lower confidence (probabilities not pinned at 1.0)
  - it may latch onto whichever fault has the strongest signal at a given moment

Run:
    python sim/make_stress_flight.py
Writes: data/raw/STRESS_multi.csv   (drop it in the dashboard dropdown)
"""

import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
import generate_flight_data as g  # noqa: E402

RATE_HZ = g.RATE_HZ


def main():
    rng = np.random.default_rng(123)
    dur_s = 300
    n = int(dur_s * RATE_HZ)
    phase = g._phase_profile(n)
    df = g._base_flight(n, phase, rng)

    # 1) Crank baseline sensor noise well above normal flights
    df["bat_volt"] += rng.normal(0, 0.25, n)        # ~6x voltage noise
    df["bat_curr"] += rng.normal(0, 4.0, n)         # ~3x current noise
    df["gps_hdop"] += np.abs(rng.normal(0, 0.18, n))
    df["gps_innov"] += np.abs(rng.normal(0, 0.6, n))
    df["mag_field"] += rng.normal(0, 18, n)         # ~3x mag noise
    df["mag_innov"] += np.abs(rng.normal(0, 0.10, n))
    df["att_err"] += np.abs(rng.normal(0, 2.5, n))
    df["vibe"] += np.abs(rng.normal(0, 1.5, n))
    for m in ["mot1", "mot2", "mot3", "mot4"]:
        df[m] += rng.normal(0, 25, n)               # ~3x motor noise

    label = np.array(["nominal"] * n, dtype=object)

    # 2) Overlapping failures at different onsets
    # --- Battery sag from t=90s, accelerating
    b0 = 90 * RATE_HZ
    kb = np.arange(n - b0)
    df.loc[b0:, "bat_volt"] = df.loc[b0:, "bat_volt"].values - 0.0011 * kb ** 1.3
    df.loc[b0:, "bat_curr"] = df.loc[b0:, "bat_curr"].values + 0.04 * kb
    label[b0:] = "battery"

    # --- Compass drift from t=150s (now overlapping battery)
    c0 = 150 * RATE_HZ
    kc = np.arange(n - c0)
    df.loc[c0:, "mag_field"] = df.loc[c0:, "mag_field"].values + 1.6 * kc
    df.loc[c0:, "mag_innov"] = df.loc[c0:, "mag_innov"].values + 0.011 * kc
    # keep battery label where battery is also active; mark overlap separately
    label[c0:] = "battery+compass"

    # --- Single motor saturating from t=210s (triple overlap)
    m0 = 210 * RATE_HZ
    km = np.arange(n - m0)
    df.loc[m0:, "mot2"] = np.clip(df.loc[m0:, "mot2"].values + 3.2 * km, 1000, 2000)
    df.loc[m0:, "att_err"] = df.loc[m0:, "att_err"].values + 0.09 * km
    df.loc[m0:, "vibe"] = df.loc[m0:, "vibe"].values + 0.06 * km
    label[m0:] = "battery+compass+motor"

    df["label"] = label
    df["flight_id"] = "STRESS"
    df["true_class"] = "multi"

    out = "data/raw/STRESS_multi.csv"
    df.to_csv(out, index=False)
    print(f"Wrote {out}  ({n} rows, {dur_s}s)")
    print("Onsets: battery@90s, compass@150s, motor@210s (overlapping)")


if __name__ == "__main__":
    main()
