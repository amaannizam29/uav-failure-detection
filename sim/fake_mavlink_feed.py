"""
Fake MAVLink feed.

Replays a flight CSV as live MAVLink telemetry on udp:127.0.0.1:14550, so the
hub's LIVE mode can connect and detect faults in real time WITHOUT ArduPilot.

LIMITATION: this is telemetry only. It does NOT run a flight stack, so it cannot
respond to LAND/RTL commands. For a closed loop (commands actually act on the
vehicle) you need real SITL. This is for demoing live DETECTION, not control.

Setup:
    pip install pymavlink

Run (terminal 1):
    python sim/fake_mavlink_feed.py --flight data/raw/STRESS_multi.csv

Then start the server (terminal 2):
    python dashboard/server.py
Open http://localhost:8000, pick MAVLink [LIVE], Load.
"""

import argparse
import time

import pandas as pd
from pymavlink import mavutil


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--flight", default="data/raw/STRESS_multi.csv")
    ap.add_argument("--conn", default="udpout:127.0.0.1:14550")
    ap.add_argument("--rate", type=float, default=10.0)
    ap.add_argument("--speed", type=float, default=1.0,
                    help="playback speed multiplier (1.0 = realtime)")
    ap.add_argument("--loop", action="store_true", help="repeat when finished")
    args = ap.parse_args()

    df = pd.read_csv(args.flight)
    m = mavutil.mavlink_connection(args.conn, source_system=1)
    print(f"Streaming {args.flight} -> {args.conn}  "
          f"({len(df)} rows @ {args.rate}Hz x{args.speed})")
    dt = 1.0 / args.rate / args.speed

    def send_once():
        for _, r in df.iterrows():
            now_ms = int(time.time() * 1000) & 0xFFFFFFFF

            # heartbeat: pretend to be an ArduPilot quadplane, armed
            m.mav.heartbeat_send(
                mavutil.mavlink.MAV_TYPE_FIXED_WING,
                mavutil.mavlink.MAV_AUTOPILOT_ARDUPILOTMEGA,
                mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED, 0,
                mavutil.mavlink.MAV_STATE_ACTIVE)

            # battery: SYS_STATUS (voltage mV, current cA, remaining %)
            m.mav.sys_status_send(
                0, 0, 0, 0,
                int(r["bat_volt"] * 1000), int(r["bat_curr"] * 100),
                int(r["bat_soc"] * 100), 0, 0, 0, 0, 0, 0)

            # GPS: sats + HDOP (eph in cm)
            m.mav.gps_raw_int_send(
                now_ms * 1000, 3, 0, 0, 0,
                int(r["gps_hdop"] * 100), 65535, 0, 0,
                int(r["gps_nsats"]))

            # attitude (yaw in rad)
            yaw_rad = float(r["yaw"]) * 3.14159 / 180.0
            m.mav.attitude_send(now_ms, 0, 0, yaw_rad, 0, 0, 0)

            # magnetometer via SCALED_IMU2 (split field magnitude across axes)
            mag = float(r["mag_field"]) / (3 ** 0.5)
            m.mav.scaled_imu2_send(now_ms, 0, 0, 0, 0, 0, 0,
                                   int(mag), int(mag), int(mag))

            # motor outputs via SERVO_OUTPUT_RAW
            m.mav.servo_output_raw_send(
                (now_ms * 1000) & 0xFFFFFFFF, 0,
                int(r["mot1"]), int(r["mot2"]), int(r["mot3"]), int(r["mot4"]),
                0, 0, 0, 0)

            # vibration
            v = float(r["vibe"])
            m.mav.vibration_send(now_ms * 1000, v, v, v, 0, 0, 0)

            # EKF innovations carry gps_innov / mag_innov (the model uses these)
            m.mav.ekf_status_report_send(
                1, 0.0,
                float(r.get("gps_innov", 0.0)), 0.0,
                float(r.get("mag_innov", 0.0)), 0.0)

            time.sleep(dt)

    try:
        while True:
            send_once()
            if not args.loop:
                break
            print("loop restart")
    except KeyboardInterrupt:
        print("\nstopped")


if __name__ == "__main__":
    main()
