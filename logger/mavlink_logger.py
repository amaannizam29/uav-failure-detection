"""
Real telemetry logger for ArduPilot SITL (or a real flight controller).

Connects over MAVLink, samples the same fields the synthetic generator uses,
and writes a CSV with an identical schema, so ml/ and dashboard/ run unchanged
on real flight data.

Typical SITL launch (separate terminal):
    sim_vehicle.py -v ArduPlane -f quadplane --console --map

Then:
    python logger/mavlink_logger.py --conn udp:127.0.0.1:14550 --out data/raw/live.csv

Requires: pymavlink  (pip install pymavlink)
"""

import argparse
import csv
import math
import time

try:
    from pymavlink import mavutil
except ImportError:
    mavutil = None

FIELDS = ["t", "phase", "bat_volt", "bat_curr", "bat_soc",
          "gps_nsats", "gps_hdop", "gps_innov",
          "mag_field", "mag_innov", "yaw",
          "mot1", "mot2", "mot3", "mot4", "att_err", "vibe"]


def request_streams(m, rate_hz):
    m.mav.request_data_stream_send(
        m.target_system, m.target_component,
        mavutil.mavlink.MAV_DATA_STREAM_ALL, rate_hz, 1)


def collect(conn, out, rate_hz, duration):
    if mavutil is None:
        raise SystemExit("pymavlink not installed. Run: pip install pymavlink")

    m = mavutil.mavlink_connection(conn)
    print("Waiting for heartbeat...")
    m.wait_heartbeat()
    print(f"Heartbeat from sys {m.target_system}. Logging to {out}")
    request_streams(m, rate_hz)

    state = {k: 0.0 for k in FIELDS}
    motors = [0, 0, 0, 0]
    t0 = time.time()

    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        last_write = 0.0
        while time.time() - t0 < duration:
            msg = m.recv_match(blocking=True, timeout=2)
            if msg is None:
                continue
            typ = msg.get_type()

            if typ == "SYS_STATUS":
                state["bat_volt"] = msg.voltage_battery / 1000.0
                state["bat_curr"] = msg.current_battery / 100.0
                state["bat_soc"] = max(msg.battery_remaining, 0) / 100.0
            elif typ == "GPS_RAW_INT":
                state["gps_nsats"] = msg.satellites_visible
                state["gps_hdop"] = msg.eph / 100.0
            elif typ == "ATTITUDE":
                state["yaw"] = math.degrees(msg.yaw) % 360
            elif typ == "SCALED_IMU2" or typ == "RAW_IMU":
                mx, my, mz = msg.xmag, msg.ymag, msg.zmag
                state["mag_field"] = math.sqrt(mx * mx + my * my + mz * mz)
            elif typ == "VIBRATION":
                state["vibe"] = (msg.vibration_x + msg.vibration_y +
                                 msg.vibration_z) / 3.0
            elif typ == "SERVO_OUTPUT_RAW":
                motors = [msg.servo1_raw, msg.servo2_raw,
                          msg.servo3_raw, msg.servo4_raw]
            elif typ == "EKF_STATUS_REPORT":
                state["gps_innov"] = getattr(msg, "pos_horiz_variance", 0.0)
                state["mag_innov"] = getattr(msg, "compass_variance", 0.0)

            now = time.time() - t0
            if now - last_write >= 1.0 / rate_hz:
                state["t"] = round(now, 2)
                state["mot1"], state["mot2"] = motors[0], motors[1]
                state["mot3"], state["mot4"] = motors[2], motors[3]
                # phase heuristic: motors high => hover, else cruise
                hi = max(motors) if max(motors) else 0
                state["phase"] = 0 if hi > 1450 else 1
                w.writerow({k: state.get(k, 0) for k in FIELDS})
                last_write = now

    print("Done.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--conn", default="udp:127.0.0.1:14550")
    ap.add_argument("--out", default="data/raw/live.csv")
    ap.add_argument("--rate", type=int, default=10)
    ap.add_argument("--duration", type=float, default=600)
    args = ap.parse_args()
    collect(args.conn, args.out, args.rate, args.duration)


if __name__ == "__main__":
    main()
