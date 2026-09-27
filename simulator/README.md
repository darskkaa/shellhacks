# MechDog bridge simulation

See [compatibility tests and detailed evidence](COMPATIBILITY.md) for the full verification scope.
See [rendered-camera YOLO experiment](VISION.md) for person/car assets and vision control.
Start here: [traffic simulation run instructions](RUN_TRAFFIC.md), including NixOS setup.
See [guide-dog crossing gated by the blind-escort YOLO model](ESCORT.md) for the WALK-signal escort demo.

## Real Miami data demos

Built from one real Edgewater, Miami block: OpenStreetMap geometry plus SafeRoute Miami's hazard layers
(Miami-Dade 311 flooding, FDOT work zones and crashes), cached in `simulator/data/` by `real_block.py`.
Only the robot hardware is simulated (PyBullet); the car's final pull-in runs in Waymo's Waymax.
Install with `uv pip install --python .venv/bin/python -r simulator/requirements-real.txt`.

| Demo | Run (`.venv/bin/python -m ...`) | Doc | Output (`simulator/artifacts/`) |
|---|---|---|---|
| Pickup curb scoring | `simulator.pickup_choice` | this section | `data/pickup.json` |
| Walk to the Waymo | `simulator.gemini_waymo_sim --offline` | module docstring | `waymo_escort/` |
| Obstacle course video | `simulator.escort_obstacles_video --offline` | [ESCORT_OBSTACLES.md](ESCORT_OBSTACLES.md) | `escort_obstacles/` |
| Flooded sidewalk video | `simulator.escort_flood_video --offline` | [ESCORT_FLOOD.md](ESCORT_FLOOD.md) | `escort_flood/` |
| Signalized crossing video | `simulator.crossing_video --assets <dir> < /dev/null` | [REAL_CROSSWALK.md](REAL_CROSSWALK.md) | `real_crosswalk/` |
| Waymo pickup + Waymax | `simulator.waymo_pickup_sim` | [PICKUP.md](PICKUP.md) | `waymo_pickup/` |

`simulator/video.py` writes WebM and MP4 with the ffmpeg bundled in `imageio-ffmpeg`. The rendered videos
are published on the static page in `showcase/`.

PyBullet's bundled eight-motor Minitaur is an **uncalibrated MechDog surrogate**.
The scene exercises motors, contact and rendering. It does not reproduce MechDog
walking/action groups, the frozen Hiwonder library, or the ESP32 firmware image.

## Install and run

From the repository root; verified on Linux x86-64, Python 3.11.14:

```bash
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python --only-binary=:all: -r simulator/requirements.txt
.venv/bin/python simulator/run.py
.venv/bin/python simulator/run.py --gui --seconds 30
```

The wheel includes models. No ROS/GPU/Gym required for headless mode.
`run.py` accepts `--seconds` (finite 4–300; default 10), `--gui`, and `--output`
(default `simulator/artifacts/`). It writes `minitaur.png` and `report.json`.
Require a fresh report with `passed: true`: native GUI failures can exit 0.
Checks cover eight motors, four leg constraints, contacts, motor motion,
body clearance, upright final pose and rendered robot pixels; NaNs abort.
Reports contain model/version/connection, steps/time, contacts, motor excursion
(radians), final position (metres), roll/pitch (radians), pixels, checks and passed.
Recorded headless and GUI results: `verification.json`, `verification-gui.json`.

## Existing control panel

```bash
.venv/bin/python simulator/bridge.py --port 5005
# Second terminal:
.venv/bin/python tools/dog_panel.py --wifi 127.0.0.1 --wifi-only
```

`bridge.py` loads unchanged `robot_dump/main.py` Bridge and DogHAL code, with
simulated SDK devices and CPython clock/poll adapters. It reads only
`config.example.py`, binds localhost, and never discovers/connects to hardware.
`--port` accepts 0–65535 (default 5005); 0 chooses a free port. `--gui` is optional.
Ctrl-C stops it. Panel requires port 5005; close other panels using HTTP port 8000.
`caps.simulation` is true. Move/stop/heartbeat/telemetry/safety logic runs; motors
perform small excursions rather than calibrated travel. Sonar is a ray cast;
IMU angles come from physics, with DogHAL's synthetic raw-data fallback.
Battery is fixed at 7.4 V. Height/gait/posture/action return unsuccessful ACKs;
RGB/buzzer do nothing despite native successful ACKs. Reset only stops motion.
The panel ignores action ACK failures. WiFi, MicroPython timing/wraparound and
real hardware calibration remain unverified. Host poll adapter requires Linux.

## NixOS GUI

Desktop display access and dynamically discoverable libGL/libGLX are required.
On the verified machine, prepend the installed libglvnd directory to
`LD_LIBRARY_PATH` for the command:
`/nix/store/kfy4qljyaacifhdfrr0m9wcbig4jjihq-libglvnd-1.7.0/lib`.
That path is generation-specific; headless mode needs no OpenGL configuration.
