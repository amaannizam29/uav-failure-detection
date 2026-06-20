# UAV Predictive Failure Detection

A predictive maintenance and failure-detection pipeline for fixed-wing / VTOL
UAVs. It ingests MAVLink flight telemetry, extracts windowed features, and a
classifier flags four degradation modes (battery, GPS, compass, motor) **before
hard cutoff**, giving an early-warning margin rather than a post-crash log.

The pipeline is schema-compatible with ArduPilot DataFlash / MAVLink, so it runs
on real ArduPilot SITL logs and on a synthetic generator with identical columns.

## Result (held-out flights, flight-level split)

| Metric | Value |
|---|---|
| Accuracy | 0.99 |
| Macro-F1 | 0.99 |
| Failure flights detected | 36 / 36 |
| Mean early warning before cutoff | ~94 s |

Top predictive features are physically interpretable: GPS HDOP slope, battery
voltage slope, motor-output spread, magnetometer innovation slope.

![results](reports/results_overview.png)

> Note on the numbers: synthetic flights are cleaner and more separable than
> real flight logs. The same pipeline on real SITL/DataFlash data is expected to
> score lower. The value here is the architecture and the early-warning margin,
> not the headline accuracy.

## Architecture

```
SITL / real FC ──MAVLink──> logger/mavlink_logger.py ──CSV──┐
                                                            ├─> ml/features.py ─> ml/train.py ─> models/
sim/generate_flight_data.py ──CSV (same schema)─────────────┘                                    │
                                                                                                 v
                                                              dashboard/app.py  (live probabilities + alarm)
```

## Quickstart

```bash
pip install -r requirements.txt

# 1. generate a synthetic dataset (or skip and use real SITL logs)
python sim/generate_flight_data.py --flights 60 --out data/raw

# 2. train + evaluate (writes models/ and reports/)
python ml/train.py

# 3. render the results figure
python ml/make_report_figure.py

# 4. live dashboard
streamlit run dashboard/app.py
```

## Real-data path (ArduPilot SITL)

```bash
# terminal 1: launch QuadPlane SITL
sim_vehicle.py -v ArduPlane -f quadplane --console --map

# terminal 2: log live telemetry to the same CSV schema
python logger/mavlink_logger.py --conn udp:127.0.0.1:14550 --out data/raw/live.csv
```

Then re-run `ml/train.py` (or just inference in the dashboard) on the real log.

## Failure modes modelled

| Mode | Signature the model keys on |
|---|---|
| Battery | accelerating voltage collapse, rising internal resistance under load |
| GPS | satellite dropout, HDOP climb, EKF position innovation rise |
| Compass | magnetometer field-magnitude drift, EKF mag innovation, yaw drift |
| Motor | one motor PWM saturating toward max, attitude-error and vibration rise |

## Repo layout

```
sim/        synthetic flight generator (ArduPilot-schema CSV)
logger/     pymavlink logger for live SITL / real FC
ml/         feature extraction, training, reporting
dashboard/  Streamlit live failure dashboard
data/       generated / logged flight CSVs
models/     trained model + metrics
reports/    figures + lead-time table
docs/       architecture, requirements, risk assessment
```

## Limitations

- Synthetic data is a stand-in for real flight logs; validate on SITL/real logs.
- Failure injectors are deterministic profiles, not high-fidelity fault physics.
- Single-fault assumption; concurrent faults are out of scope for v1.
