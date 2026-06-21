# System Architecture — UAV Predictive Failure Detection

**Repository:** uav-failure-detection
**Builds on:** [TriFlight Avionics Research](https://github.com/amaannizam29/triflight-avionics-research) — the pre-experimental avionics architecture, power/signal mapping, and MAVLink telemetry logger this project extends.
**Status:** Working failure-detection software + integration CAD. Airframe is existing; this is systems/integration work, not airframe design.
**Note:** Values marked *(target)* are design intents carried from the TriFlight research stage, not flight-validated. No hardware flight results are claimed.

---

## 1. Background and relationship to TriFlight

This project is the failure-detection extension of the **TriFlight** research
initiative. TriFlight is a pre-experimental UAV avionics and systems-integration
study for a fixed-wing platform; it produced the avionics architecture, the
power and signal mapping, a firmware comparison (ArduPilot / INAV / PX4), a
read-only MAVLink telemetry logger, and a bench/flight-test validation plan. It
remained at the research and software stage due to lack of hardware funding, and
claims no flight results.

This repository takes that foundation and adds what TriFlight did not include: a
machine-learning failure-detection subsystem, a safety-aware decision policy, an
operator response console with live MAVLink command-out, and integration CAD
components. In short:

- **From TriFlight:** avionics architecture, power/signal mapping, MAVLink
  telemetry logging, validation methodology, system requirements.
- **Added here:** multi-fault ML detection, decision policy, operator hub,
  command-out, and integration hardware (battery-retention and lift-motor mounts).

---

## 2. System summary

The system reads UAV telemetry (replayed, simulated, or live over MAVLink),
detects component faults during their early onset, and presents a ground operator
with safety-aware recovery actions and a 5-second auto-default. It is designed
around the TriFlight avionics architecture and is schema-compatible with
ArduPilot DataFlash / MAVLink.

Design philosophy: integrate proven components (ArduPilot, MAVLink) rather than
reinvent them; keep safety logic deterministic and auditable; keep detection
data-driven; ensure every fault has a defined detection path and response.

---

## 3. System requirements

These targets are carried from the TriFlight research stage. They are design
intents for evaluating the architecture, not flight-validated results.

### 3.1 Performance (TriFlight-derived targets)

| ID | Requirement | Target | Source / rationale |
|---|---|---|---|
| P-1 | Endurance | > 60 min *(target)* | TriFlight mission target |
| P-2 | Range | > 20 km *(target)* | TriFlight BVLOS mission target |
| P-3 | Payload | > 500 g *(target)* | TriFlight sensor-payload target |
| P-4 | Cruise speed | 15–25 m/s *(target)* | efficient fixed-wing transit |

### 3.2 Functional requirements

| ID | Requirement |
|---|---|
| F-1 | Autonomous waypoint navigation (autopilot) |
| F-2 | Stream telemetry to a ground station over MAVLink |
| F-3 | Detect battery, GPS, compass, and motor faults during early onset |
| F-4 | Present ranked recovery actions with a 5 s auto-default |
| F-5 | Execute operator/auto commands over the live link (with hardening, see 6.4) |

### 3.3 Safety requirements

| ID | Requirement |
|---|---|
| S-1 | RTL must be disabled when GPS or compass is unreliable (cannot navigate home) |
| S-2 | Battery-critical state must force an immediate land |
| S-3 | A confirmed fault must never depend on a single noisy reading (smoothing) |
| S-4 | The safety pilot's RC override always takes precedence over any GCS command |

---

## 4. Avionics context (from TriFlight)

The avionics architecture and power/signal mapping originate in TriFlight. See
`AVIONICS_ARCHITECTURE.md` in this repo for the block diagram and the power,
sensor, and communication tables as applied here. Summary:

```
Battery → Power module → Flight controller → ESC(s) → Motor(s)
                              ↑
        GPS · Compass · IMU · Barometer · Airspeed · Telemetry · RC receiver
```

The four detected fault modes map onto these subsystems: power module → battery,
GPS → GPS, compass → compass, ESC/IMU signatures → motor. The telemetry link is
bidirectional MAVLink: telemetry down, commands up.

---

## 5. Autopilot architecture

The autopilot runs ArduPilot. Three layers, slowest to fastest (full flowcharts
in the control-architecture diagrams):

- **State estimation:** GPS, IMU, barometer, compass fused by an EKF into a state
  estimate. The EKF innovation (predicted vs measured) is the signal the detector
  uses for GPS and compass faults.
- **Navigation:** mission → waypoint manager → guidance → target. RTL depends on
  this layer computing a path home, which needs GPS and compass (hence S-1).
- **Control loop:** target vs measured → PID → servo/motor commands → feedback.

---

## 6. Predictive failure-detection subsystem

Full detail in `SYSTEM_DOCUMENTATION.md`.

### 6.1 Function
Reads telemetry, detects single and simultaneous faults during early onset, and
drives the operator console.

### 6.2 Detection
Telemetry → 3 s windows → 53 features (means, slopes, spreads) → four independent
classifiers (battery, GPS, compass, motor) → per-fault probabilities. A fault
confirms only after persisting 3 consecutive windows (smoothing), which cut
false-positive windows ~91%.

### 6.3 Decision and response
A deterministic policy maps confirmed faults to ranked actions, removing unsafe
options (RTL on GPS/compass loss). Operator chooses, or the safe default
auto-fires after 5 s.

### 6.4 Command-out (status)
Sends a real MAVLink mode command, but is not yet hardened: no COMMAND_ACK
confirmation, no pre-arm/failsafe/RC-override checks. Required before any flight
test (see `SYSTEM_DOCUMENTATION.md` section 8). The TriFlight logger this builds
on is read-only by design; command-out is new here and therefore the least
validated part.

### 6.5 Validation status
Detection is trained and tested on synthetic data (per-fault F1 0.96–0.99,
flight-level split). Synthetic data is more separable than real logs, so these
numbers will drop on real data. Validation on real or higher-fidelity SITL data
is the open step, consistent with TriFlight's stated validation plan.

---

## 7. Risk assessment

Each risk maps to how it is detected and the planned response.

| Risk | Cause | Effect if unhandled | Detection | Response |
|---|---|---|---|---|
| GPS loss | Multipath, jamming, receiver fault | Cannot navigate, RTL unsafe | Sat dropout, HDOP climb, EKF position innovation rise | Disable RTL; hold altitude or land; alert operator |
| Battery failure | Cell degradation, over-draw | Power loss mid-flight | Voltage sag, rising draw under load | If critical, force immediate land; alert operator |
| Motor failure | ESC fault, mechanical, prop damage | Loss of control authority | One motor saturating, vibration + attitude error rise | Immediate land; alert operator |
| Telemetry loss | Range, interference, radio fault | Operator blind, no command path | Link timeout / heartbeat loss | Autopilot failsafe (RTL/land per config) |
| Compass failure | Magnetic interference, sensor drift | Wrong heading, RTL flies off course | Field-magnitude drift, EKF mag innovation | Disable RTL; land; alert operator |

### 7.1 Priorities
- Highest severity: battery and motor (direct loss of flight) → both force land.
- Highest subtlety: GPS and compass (gradual) → the detector's early-onset focus
  targets exactly these.
- Telemetry loss is handled by the autopilot's own failsafe, independent of the
  detector, and is a reason to consider onboard deployment.

---

## 8. Limitations and honest scope

- This repo extends TriFlight; it does not claim TriFlight's hardware was built.
  TriFlight remained pre-experimental (no flight results), and neither does this.
- The airframe is existing; this is integration and systems work.
- Performance requirements (section 3) are TriFlight-derived targets, not measured.
- Failure-detection accuracy is from synthetic data; real-data validation is open.
- Command-out is protocol-correct but not safety-hardened; not for hardware yet.
- The CAD parts are integration components to demonstrate capability, not a full
  aircraft.

---

## 9. Future development

1. Validate failure detection on real or high-fidelity SITL fault-injection data.
2. Harden command-out (ACK confirmation, pre-arm/failsafe/RC-override checks).
3. Complete the avionics integration CAD set.
4. Carry the TriFlight bench/flight-test validation plan into hardware when funded.
5. Consider onboard deployment of the detector on a companion computer.

---

## Appendix: related documents and repos

- [TriFlight Avionics Research](https://github.com/amaannizam29/triflight-avionics-research) — the foundation this builds on
- `README.md` — this project's overview
- `SYSTEM_DOCUMENTATION.md` — full failure-detection reference + real-drone roadmap
- `AVIONICS_ARCHITECTURE.md` — block diagram + power/sensor/comms tables
- Control-architecture diagrams — state estimation, navigation, control loop
