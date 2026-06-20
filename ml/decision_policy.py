"""
Decision policy (deterministic, not ML).

Maps the set of currently-active faults to a ranked list of operator actions,
a default action, a countdown, and a severity. This is the safety layer that
sits on top of the detector.

Action vocabulary (and the real ArduPilot command each maps to):
  RTL       Return To Launch        MAV_CMD_NAV_RETURN_TO_LAUNCH  (needs GPS+compass)
  LAND      Land at current pos     MAV_CMD_NAV_LAND
  POSHOLD   Hold position           needs GPS
  ALTHOLD   Hold altitude, manual   no GPS needed
  CONTINUE  Keep mission, monitor   no command

Safety rules baked in:
  - GPS fault     -> RTL and POSHOLD are UNSAFE (both need GPS). Prefer ALTHOLD/LAND.
  - Compass fault -> RTL is UNSAFE (heading unreliable). Prefer LAND.
  - Battery fault -> energy budget shrinking. Critical -> LAND now (RTL may not make it).
  - Motor fault   -> reduced control authority -> LAND immediately.
  - Multiple      -> take the highest-severity action across the active set.
"""

SEVERITY = {"battery": 3, "motor": 3, "gps": 2, "compass": 2, None: 0}

# Each action carries a risk weight; lower is safer for a given situation.
ALL_ACTIONS = {
    "RTL":      "Return to launch",
    "LAND":     "Land at current position",
    "POSHOLD":  "Hold position",
    "ALTHOLD":  "Hold altitude (manual horizontal)",
    "CONTINUE": "Continue mission, monitor",
}


def decide(active, battery_critical=False):
    """
    active: set/list of active fault names, e.g. {"gps", "battery"}
    battery_critical: True if battery has degraded past a hard threshold
    returns dict: severity, default, countdown_s, options (ranked list of (code,label))
    """
    active = set(active)

    # Start with everything, then remove options the faults make unsafe.
    allowed = set(ALL_ACTIONS)

    if "gps" in active:
        allowed.discard("RTL")        # cannot navigate home without GPS
        allowed.discard("POSHOLD")    # position hold needs GPS
    if "compass" in active:
        allowed.discard("RTL")        # heading unreliable -> RTL can fly off course
    if "motor" in active:
        allowed.discard("CONTINUE")   # never keep flying a mission on a bad motor
    if battery_critical:
        allowed.discard("CONTINUE")
        allowed.discard("RTL")        # may not have energy to reach home

    # No faults -> nominal
    if not active:
        return {"severity": 0, "default": "CONTINUE", "countdown_s": 0,
                "options": [("CONTINUE", ALL_ACTIONS["CONTINUE"])],
                "reason": "All systems nominal."}

    sev = max(SEVERITY[f] for f in active)

    # Pick default: safest decisive action available, severity-driven.
    if sev >= 3 or battery_critical:
        allowed.discard("CONTINUE")   # too severe to keep flying the mission
        default = "LAND" if "LAND" in allowed else "ALTHOLD"
        countdown = 5
    else:
        # moderate: prefer LAND if RTL unsafe, else RTL
        if "RTL" in allowed:
            default = "RTL"
        elif "LAND" in allowed:
            default = "LAND"
        else:
            default = "ALTHOLD"
        countdown = 8

    # Rank options: default first, then remaining allowed by a sensible order
    order = ["LAND", "RTL", "ALTHOLD", "POSHOLD", "CONTINUE"]
    ranked = [default] + [a for a in order if a in allowed and a != default]
    options = [(a, ALL_ACTIONS[a]) for a in ranked]

    reason = _reason(active, battery_critical, default)
    return {"severity": sev, "default": default, "countdown_s": countdown,
            "options": options, "reason": reason}


def _reason(active, crit, default):
    bits = []
    if "battery" in active:
        bits.append("battery degrading" + (" (CRITICAL)" if crit else ""))
    if "motor" in active:
        bits.append("motor authority reduced")
    if "gps" in active:
        bits.append("GPS unreliable, RTL/PosHold disabled")
    if "compass" in active:
        bits.append("heading unreliable, RTL disabled")
    why = "; ".join(bits)
    return f"{why}. Recommended default: {ALL_ACTIONS[default]}."


if __name__ == "__main__":
    for case in [set(), {"battery"}, {"gps"}, {"compass"},
                 {"gps", "compass"}, {"battery", "motor"},
                 {"battery", "gps", "compass"}]:
        print(case if case else "{nominal}")
        d = decide(case, battery_critical=("battery" in case and "motor" in case))
        print(f"   default={d['default']}  countdown={d['countdown_s']}s  "
              f"sev={d['severity']}")
        print(f"   options={[o[0] for o in d['options']]}")
        print(f"   {d['reason']}\n")
