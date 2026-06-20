"""
Operator hub backend (v2: adds command-out path).

Serves the hub UI and APIs:
  GET /                       the hub client page
  GET /api/flights            list flight CSVs (+ LIVE entry)
  GET /api/stream?flight=..   compute decision stream for one CSV (replay)
  GET /api/live               SSE: live decisions from MAVLink (telemetry IN)
  GET /api/command?cmd=LAND   send a real command to the vehicle (command OUT)

CSV replay = detection only; commands there are logged client-side, flight keeps
playing (no vehicle to act on). LIVE = closed loop: a command actually changes
the connected vehicle (SITL or real), and you see the mode change come back in
telemetry.

Run:
    python dashboard/server.py
Then open http://localhost:8000
Live mode needs ArduPilot SITL (or a real link) on udp:127.0.0.1:14550.
"""

import glob
import json
import math
import os
import queue
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
sys.path.insert(0, os.path.join(ROOT, "ml"))

import pandas as pd  # noqa: E402
from stream_core import (load_model, compute_stream, frame_from_window,  # noqa: E402
                         StreamState, WINDOW)

RATE_HZ = 10
PORT = 8000
DATA_DIRS = ["data/raw", "data/multi"]
LIVE_CONN = os.environ.get("MAVLINK_CONN", "udp:127.0.0.1:14550")
CLIENT_HTML = os.path.join("dashboard", "operator_hub_client.html")

# action code -> candidate ArduPilot mode names (first one the vehicle has wins).
# Covers Plane / QuadPlane / Copter naming so it adapts to the connected vehicle.
MODE_CANDIDATES = {
    "LAND":    ["QLAND", "LAND"],
    "RTL":     ["QRTL", "RTL"],
    "POSHOLD": ["QLOITER", "LOITER", "POSHOLD"],
    "ALTHOLD": ["QHOVER", "ALT_HOLD", "FBWA"],
    "CONTINUE":["AUTO", "QLOITER", "LOITER"],
}


def list_flights():
    out = []
    for d in DATA_DIRS:
        for p in sorted(glob.glob(os.path.join(d, "*.csv"))):
            if "_manifest" not in p:
                out.append(p.replace("\\", "/"))
    return out


def safe_flight_path(flight):
    flight = flight.replace("\\", "/")
    if ".." in flight or not any(flight.startswith(d) for d in DATA_DIRS):
        return None
    return flight if os.path.exists(flight) else None


# --------------------------------------------------------------------------
# Persistent MAVLink link: one connection shared by the SSE reader and the
# command sender, so a command goes out on the same link telemetry comes in.
# --------------------------------------------------------------------------
class LiveLink:
    def __init__(self):
        self.master = None
        self.running = False
        self.error = None
        self.mode_name = "?"
        self.rev_modes = {}        # id -> name
        self.fwd_modes = {}        # name -> id
        self.q = queue.Queue(maxsize=200)
        self._lock = threading.Lock()

    def start(self, conn):
        with self._lock:
            if self.running:
                return
            self.error = None
            try:
                from pymavlink import mavutil
            except ImportError:
                self.error = "pymavlink not installed (pip install pymavlink)"
                return
            self.mavutil = mavutil
            self.master = mavutil.mavlink_connection(conn)
            hb = self.master.wait_heartbeat(timeout=8)
            if hb is None:
                self.error = f"no MAVLink heartbeat on {conn} (is SITL running?)"
                self.master = None
                return
            mm = self.master.mode_mapping() or {}
            self.fwd_modes = {k.upper(): v for k, v in mm.items()}
            self.rev_modes = {v: k for k, v in mm.items()}
            self.master.mav.request_data_stream_send(
                self.master.target_system, self.master.target_component,
                mavutil.mavlink.MAV_DATA_STREAM_ALL, RATE_HZ, 1)
            self.running = True
            threading.Thread(target=self._reader, daemon=True).start()

    def _reader(self):
        bundle = load_model()
        state = StreamState(bundle["faults"])
        st = {"bat_volt": 25.0, "bat_curr": 0.0, "bat_soc": 1.0, "gps_nsats": 0,
              "gps_hdop": 1.0, "gps_innov": 0.0, "mag_field": 500.0,
              "mag_innov": 0.0, "yaw": 0.0, "mot1": 1000, "mot2": 1000,
              "mot3": 1000, "mot4": 1000, "att_err": 0.0, "vibe": 0.0}
        buf = []
        t0 = time.time()
        last = 0.0
        while self.running:
            msg = self.master.recv_match(blocking=True, timeout=2)
            if msg is None:
                continue
            typ = msg.get_type()
            if typ == "HEARTBEAT":
                self.mode_name = self.rev_modes.get(msg.custom_mode, self.mode_name)
            elif typ == "SYS_STATUS":
                st["bat_volt"] = msg.voltage_battery / 1000.0
                st["bat_curr"] = msg.current_battery / 100.0
                st["bat_soc"] = max(msg.battery_remaining, 0) / 100.0
            elif typ == "GPS_RAW_INT":
                st["gps_nsats"] = msg.satellites_visible
                st["gps_hdop"] = msg.eph / 100.0
            elif typ == "ATTITUDE":
                st["yaw"] = math.degrees(msg.yaw) % 360
            elif typ in ("RAW_IMU", "SCALED_IMU2"):
                st["mag_field"] = math.sqrt(msg.xmag**2 + msg.ymag**2 + msg.zmag**2)
            elif typ == "VIBRATION":
                st["vibe"] = (msg.vibration_x + msg.vibration_y + msg.vibration_z) / 3.0
            elif typ == "SERVO_OUTPUT_RAW":
                st["mot1"], st["mot2"] = msg.servo1_raw, msg.servo2_raw
                st["mot3"], st["mot4"] = msg.servo3_raw, msg.servo4_raw
            elif typ == "EKF_STATUS_REPORT":
                st["gps_innov"] = getattr(msg, "pos_horiz_variance", 0.0)
                st["mag_innov"] = getattr(msg, "compass_variance", 0.0)

            now = time.time() - t0
            if now - last >= 1.0 / RATE_HZ:
                last = now
                hi = max(st["mot1"], st["mot2"], st["mot3"], st["mot4"])
                sample = dict(st)
                sample["t"] = round(now, 2)
                sample["phase"] = 0 if hi > 1450 else 1
                buf.append(sample)
                if len(buf) > WINDOW:
                    buf.pop(0)
                if len(buf) == WINDOW:
                    frame = frame_from_window(pd.DataFrame(buf), state, bundle)
                    frame["uav_mode"] = self.mode_name
                    try:
                        self.q.put_nowait(frame)
                    except queue.Full:
                        try:
                            self.q.get_nowait()
                        except queue.Empty:
                            pass
                        self.q.put_nowait(frame)

    def send_action(self, code):
        """Set the vehicle mode that realises the operator's action."""
        if not self.running or self.master is None:
            return False, "no live link"
        names = MODE_CANDIDATES.get(code, [])
        chosen = next((n for n in names if n in self.fwd_modes), None)
        if chosen is None:
            avail = ", ".join(sorted(self.fwd_modes)[:8])
            return False, f"{code}: no matching mode on vehicle (have: {avail})"
        mode_id = self.fwd_modes[chosen]
        self.master.mav.set_mode_send(
            self.master.target_system,
            self.mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED, mode_id)
        return True, f"mode set to {chosen}"


LINK = LiveLink()


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _json(self, obj, code=200):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        u = urlparse(self.path)
        if u.path in ("/", "/index.html"):
            return self._file(CLIENT_HTML, "text/html")
        if u.path == "/api/flights":
            return self._json({"flights": list_flights(), "live": "LIVE"})
        if u.path == "/api/stream":
            flight = (parse_qs(u.query).get("flight") or [""])[0]
            path = safe_flight_path(flight)
            if not path:
                return self._json({"error": "flight not found"}, 404)
            frames = compute_stream(pd.read_csv(path))
            return self._json({"flight": os.path.basename(path), "rate_hz": RATE_HZ,
                               "faults": load_model()["faults"], "frames": frames})
        if u.path == "/api/command":
            code = (parse_qs(u.query).get("cmd") or [""])[0].upper()
            ok, detail = LINK.send_action(code)
            return self._json({"sent": ok, "code": code, "detail": detail})
        if u.path == "/api/live":
            return self._live()
        self.send_response(404); self.end_headers()

    def _file(self, path, ctype):
        if not os.path.exists(path):
            self.send_response(404); self.end_headers()
            self.wfile.write(b"client page missing"); return
        data = open(path, "rb").read()
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _live(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        LINK.start(LIVE_CONN)
        if LINK.error:
            self._sse({"error": LINK.error}); return
        self._sse({"info": f"live MAVLink connected on {LIVE_CONN}"})
        try:
            while True:
                try:
                    frame = LINK.q.get(timeout=1)
                    self._sse({"frame": frame})
                except queue.Empty:
                    self.wfile.write(b": keepalive\n\n"); self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            return

    def _sse(self, obj):
        self.wfile.write(f"data: {json.dumps(obj)}\n\n".encode())
        self.wfile.flush()


def main():
    load_model()
    print(f"Operator hub on http://localhost:{PORT}")
    print(f"Flights: {len(list_flights())} CSVs found")
    print(f"Live MAVLink: {LIVE_CONN} (telemetry in + command out)")
    ThreadingHTTPServer(("127.0.0.1", PORT), Handler).serve_forever()


if __name__ == "__main__":
    main()
