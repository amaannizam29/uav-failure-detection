"""
Multi-fault flight generator (v2).

Differences from v1:
  - a flight can contain 0..3 simultaneous faults, each with its own onset
  - labels are PER-FAULT binary columns (lbl_battery, lbl_gps, lbl_compass,
    lbl_motor), not a single class. This is what enables multi-LABEL detection.
    - each flight gets a random noise scale, so the model trains across clean AND
    noisy flights and learns to ignore noise instead of over-reacting to it.

Run:
    python sim/generate_multifault.py --flights 120 --out data/multi
"""

import argparse
import os
import numpy as np
import pandas as pd

import sys
sys.path.insert(0, os.path.dirname(__file__))
import generate_flight_data as g  # reuse base flight + phase  # noqa: E402

RATE_HZ = g.RATE_HZ
FAULTS = ["battery", "gps", "compass", "motor"]


def add_noise(df, rng, scale):
    n = len(df)
    df["bat_volt"] += rng.normal(0, 0.04 * scale, n)
    df["bat_curr"] += rng.normal(0, 1.2 * scale, n)
    df["gps_hdop"] += np.abs(rng.normal(0, 0.06 * scale, n))
    df["gps_innov"] += np.abs(rng.normal(0, 0.25 * scale, n))
    df["mag_field"] += rng.normal(0, 6 * scale, n)
    df["mag_innov"] += np.abs(rng.normal(0, 0.04 * scale, n))
    df["att_err"] += np.abs(rng.normal(0, 1.0 * scale, n))
    df["vibe"] += np.abs(rng.normal(0, 0.6 * scale, n))
    for m in ["mot1", "mot2", "mot3", "mot4"]:
        df[m] += rng.normal(0, 8 * scale, n)
    return df


def inj_battery(df, onset):
    k = np.arange(len(df) - onset)
    df.loc[onset:, "bat_volt"] = df.loc[onset:, "bat_volt"].values - 0.0009 * k ** 1.35
    df.loc[onset:, "bat_curr"] = df.loc[onset:, "bat_curr"].values + 0.05 * k
    return df


def inj_gps(df, onset):
    k = np.arange(len(df) - onset)
    df.loc[onset:, "gps_nsats"] = np.clip(g.NOMINAL_NSATS - (k * 0.06).astype(int), 0, g.NOMINAL_NSATS)
    df.loc[onset:, "gps_hdop"] = df.loc[onset:, "gps_hdop"].values + 0.012 * k
    df.loc[onset:, "gps_innov"] = df.loc[onset:, "gps_innov"].values + 0.04 * k
    return df


def inj_compass(df, onset):
    k = np.arange(len(df) - onset)
    df.loc[onset:, "mag_field"] = df.loc[onset:, "mag_field"].values + 1.8 * k
    df.loc[onset:, "mag_innov"] = df.loc[onset:, "mag_innov"].values + 0.012 * k
    return df


def inj_motor(df, onset, rng):
    k = np.arange(len(df) - onset)
    col = f"mot{rng.integers(1,5)}"
    df.loc[onset:, col] = np.clip(df.loc[onset:, col].values + 3.0 * k, 1000, 2000)
    df.loc[onset:, "att_err"] = df.loc[onset:, "att_err"].values + 0.08 * k
    df.loc[onset:, "vibe"] = df.loc[onset:, "vibe"].values + 0.05 * k
    return df


INJECT = {"battery": inj_battery, "gps": inj_gps, "compass": inj_compass}


def make_flight(fid, seed):
    rng = np.random.default_rng(seed)
    dur = rng.uniform(200, 320)
    n = int(dur * RATE_HZ)
    phase = g._phase_profile(n)
    df = g._base_flight(n, phase, rng)

    # random noise level: 0.5 (clean) .. 3.0 (very noisy)
    scale = rng.uniform(0.5, 3.0)
    df = add_noise(df, rng, scale)

    # choose 0..3 faults
    n_faults = rng.choice([0, 1, 1, 2, 2, 3])
    chosen = list(rng.choice(FAULTS, size=n_faults, replace=False)) if n_faults else []

    labels = {f"lbl_{f}": np.zeros(n, dtype=int) for f in FAULTS}
    for f in chosen:
        onset = int(rng.uniform(0.4, 0.7) * n)
        if f == "motor":
            df = inj_motor(df, onset, rng)
        else:
            df = INJECT[f](df, onset)
        labels[f"lbl_{f}"][onset:] = 1

    for c, v in labels.items():
        df[c] = v
    df["flight_id"] = fid
    df["noise_scale"] = round(scale, 2)
    df["fault_set"] = "+".join(chosen) if chosen else "nominal"
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--flights", type=int, default=120)
    ap.add_argument("--out", default="data/multi")
    ap.add_argument("--seed", type=int, default=11)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    manifest = []
    for i in range(args.flights):
        fid = f"M{i:03d}"
        df = make_flight(fid, int(rng.integers(0, 1_000_000)))
        path = os.path.join(args.out, f"{fid}.csv")
        df.to_csv(path, index=False)
        manifest.append({"flight_id": fid, "fault_set": df["fault_set"].iloc[0],
                         "noise": df["noise_scale"].iloc[0], "rows": len(df)})
    mf = pd.DataFrame(manifest)
    mf.to_csv(os.path.join(args.out, "_manifest.csv"), index=False)
    print(f"Wrote {len(mf)} flights to {args.out}")
    print("Fault-set distribution:")
    print(mf["fault_set"].value_counts().head(12).to_string())


if __name__ == "__main__":
    main()
