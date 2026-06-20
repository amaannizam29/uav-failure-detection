"""
Flight telemetry generator.

Produces per-flight CSV logs whose columns mirror ArduPilot DataFlash / MAVLink
telemetry, so the same downstream pipeline works on real SITL logs without code
changes (see logger/mavlink_logger.py for the real-data path).

Failure modes injected: nominal, battery, gps, compass, motor.
Each non-nominal flight degrades progressively, so a model can learn to flag
the failure BEFORE hard cutoff (predictive maintenance, not post-mortem).

Run:
    python sim/generate_flight_data.py --flights 48 --out data/raw
"""

import argparse
import os
import numpy as np
import pandas as pd

RATE_HZ = 10                      # telemetry log rate
DT = 1.0 / RATE_HZ
FAILURE_CLASSES = ["nominal", "battery", "gps", "compass", "motor"]

# Nominal sensor baselines (loosely Talon-1400-class fixed-wing + lift quad)
BATT_FULL_V = 25.2                # 6S pack
BATT_EMPTY_V = 21.0
HOVER_CURRENT_A = 42.0            # peak lift draw
CRUISE_CURRENT_A = 9.0
PACK_CAPACITY_AH = 8.0
NOMINAL_NSATS = 16
NOMINAL_HDOP = 0.8
MAG_FIELD_MGAUSS = 500.0         # total field magnitude
MOTOR_PWM_HOVER = 1550           # us, lift motors near hover
MOTOR_PWM_CRUISE = 1200


def _rng(seed):
    return np.random.default_rng(seed)


def _phase_profile(n):
    """Return per-sample flight phase: 0 takeoff/hover, 1 cruise, 2 hover/land."""
    phase = np.ones(n, dtype=int)
    t0 = int(0.12 * n)
    t1 = int(0.85 * n)
    phase[:t0] = 0
    phase[t1:] = 2
    return phase


def _base_flight(n, phase, rng):
    """Build a clean nominal flight, all failure injectors mutate this in place."""
    t = np.arange(n) * DT

    # Power: current depends on phase, voltage sags with energy used + load
    current = np.where(phase == 1, CRUISE_CURRENT_A, HOVER_CURRENT_A)
    current = current + rng.normal(0, 1.2, n)
    used_ah = np.cumsum(current) * DT / 3600.0
    soc = np.clip(1.0 - used_ah / PACK_CAPACITY_AH, 0.0, 1.0)
    sag = 0.018 * current                       # ohmic sag under load
    volt = BATT_EMPTY_V + (BATT_FULL_V - BATT_EMPTY_V) * soc - sag
    volt += rng.normal(0, 0.04, n)

    # GPS
    nsats = np.full(n, NOMINAL_NSATS) + rng.integers(-1, 2, n)
    hdop = NOMINAL_HDOP + np.abs(rng.normal(0, 0.06, n))
    gps_innov = np.abs(rng.normal(0, 0.25, n))  # EKF position innovation (m)

    # Compass / magnetometer
    mag_field = MAG_FIELD_MGAUSS + rng.normal(0, 6, n)
    mag_innov = np.abs(rng.normal(0, 0.04, n))  # EKF mag innovation
    yaw = (np.cumsum(rng.normal(0, 0.3, n)) % 360)

    # Motors (4 lift). Cruise => low, hover => high. Symmetric outputs.
    base_pwm = np.where(phase == 1, MOTOR_PWM_CRUISE, MOTOR_PWM_HOVER).astype(float)
    motors = np.stack([base_pwm + rng.normal(0, 8, n) for _ in range(4)], axis=1)

    # Attitude error (deg) and vibration: small when healthy
    att_err = np.abs(rng.normal(0, 1.0, n))
    vibe = np.abs(rng.normal(2.0, 0.6, n))      # m/s/s

    df = pd.DataFrame({
        "t": t,
        "phase": phase,
        "bat_volt": volt,
        "bat_curr": current,
        "bat_soc": soc,
        "gps_nsats": nsats,
        "gps_hdop": hdop,
        "gps_innov": gps_innov,
        "mag_field": mag_field,
        "mag_innov": mag_innov,
        "yaw": yaw,
        "mot1": motors[:, 0],
        "mot2": motors[:, 1],
        "mot3": motors[:, 2],
        "mot4": motors[:, 3],
        "att_err": att_err,
        "vibe": vibe,
    })
    return df


def _label_window(n, onset, label):
    """Label samples from onset onward as the failure class (predictive horizon
    starts at degradation onset, not at hard cutoff)."""
    y = np.array(["nominal"] * n, dtype=object)
    y[onset:] = label
    return y


def inject_battery(df, rng):
    n = len(df)
    onset = int(rng.uniform(0.55, 0.75) * n)
    k = np.arange(n - onset)
    # Accelerating voltage collapse + rising internal resistance
    droop = 0.0009 * k ** 1.35
    df.loc[onset:, "bat_volt"] = df.loc[onset:, "bat_volt"].values - droop
    df.loc[onset:, "bat_curr"] = df.loc[onset:, "bat_curr"].values + 0.05 * k
    return df, _label_window(n, onset, "battery")


def inject_gps(df, rng):
    n = len(df)
    onset = int(rng.uniform(0.5, 0.7) * n)
    k = np.arange(n - onset)
    df.loc[onset:, "gps_nsats"] = np.clip(NOMINAL_NSATS - (k * 0.06).astype(int), 0, NOMINAL_NSATS)
    df.loc[onset:, "gps_hdop"] = df.loc[onset:, "gps_hdop"].values + 0.012 * k
    df.loc[onset:, "gps_innov"] = df.loc[onset:, "gps_innov"].values + 0.04 * k
    return df, _label_window(n, onset, "gps")


def inject_compass(df, rng):
    n = len(df)
    onset = int(rng.uniform(0.5, 0.72) * n)
    k = np.arange(n - onset)
    # Field magnitude drifts off baseline (interference / failing sensor)
    df.loc[onset:, "mag_field"] = df.loc[onset:, "mag_field"].values + 1.8 * k
    df.loc[onset:, "mag_innov"] = df.loc[onset:, "mag_innov"].values + 0.012 * k
    df.loc[onset:, "yaw"] = (df.loc[onset:, "yaw"].values + 0.25 * k) % 360
    return df, _label_window(n, onset, "compass")


def inject_motor(df, rng):
    n = len(df)
    onset = int(rng.uniform(0.55, 0.75) * n)
    k = np.arange(n - onset)
    bad = rng.integers(1, 5)                     # which motor degrades
    col = f"mot{bad}"
    # Failing motor saturates toward max as FC commands more to hold attitude
    df.loc[onset:, col] = np.clip(df.loc[onset:, col].values + 3.0 * k, 1000, 2000)
    df.loc[onset:, "att_err"] = df.loc[onset:, "att_err"].values + 0.08 * k
    df.loc[onset:, "vibe"] = df.loc[onset:, "vibe"].values + 0.05 * k
    return df, _label_window(n, onset, "motor")


INJECTORS = {
    "battery": inject_battery,
    "gps": inject_gps,
    "compass": inject_compass,
    "motor": inject_motor,
}


def make_flight(flight_id, fclass, seed):
    rng = _rng(seed)
    dur_s = rng.uniform(180, 300)                # 3 to 5 min
    n = int(dur_s * RATE_HZ)
    phase = _phase_profile(n)
    df = _base_flight(n, phase, rng)
    if fclass == "nominal":
        y = np.array(["nominal"] * n, dtype=object)
    else:
        df, y = INJECTORS[fclass](df, rng)
    df["label"] = y
    df["flight_id"] = flight_id
    df["true_class"] = fclass
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--flights", type=int, default=48)
    ap.add_argument("--out", default="data/raw")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    rng = _rng(args.seed)
    # Balanced-ish: ~40% nominal, rest split across 4 failures
    plan = []
    for i in range(args.flights):
        if i % 5 == 0 or i % 5 == 1:
            plan.append("nominal")
        else:
            plan.append(FAILURE_CLASSES[1 + (i % 4)])

    manifest = []
    for i, fclass in enumerate(plan):
        fid = f"F{i:03d}"
        df = make_flight(fid, fclass, seed=int(rng.integers(0, 1_000_000)))
        path = os.path.join(args.out, f"{fid}_{fclass}.csv")
        df.to_csv(path, index=False)
        manifest.append({"flight_id": fid, "class": fclass, "rows": len(df), "path": path})

    pd.DataFrame(manifest).to_csv(os.path.join(args.out, "_manifest.csv"), index=False)
    counts = pd.Series([m["class"] for m in manifest]).value_counts().to_dict()
    print(f"Wrote {len(manifest)} flights to {args.out}")
    print("Class counts:", counts)


if __name__ == "__main__":
    main()
