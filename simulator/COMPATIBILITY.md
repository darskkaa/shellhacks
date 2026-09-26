# Robot-dog simulator

[PyBullet 3.2.7](https://pypi.org/project/pybullet/) with its bundled Minitaur
model provides a working local quadruped physics sandbox. Minitaur has eight
actuated motors and four linked legs. The demo loads its URDF and meshes,
closes the leg constraints, settles on a floor, exercises every motor, then
returns to its standing target.

**This is an uncalibrated surrogate for our Hiwonder MechDog.** The official
[MechDog repository](https://github.com/Hiwonder/MechDog) and
[documentation](https://docs.hiwonder.com/projects/MechDog/en/latest/), plus
targeted searches on 2026-09-26, did not reveal a ready-made MechDog physics
model. The vendored SDK also contains no URDF/MJCF/SDF model. Matching motor
count alone does not establish matching mechanics.

Use this setup to explore rigid-body physics, joint control, contacts and
rendering. It does not emulate ESP32/MicroPython or the frozen Hiwonder SDK.
The standalone motion demo is a gentle leg-extension exercise, not a verified walking controller. Masses,
dimensions, torque limits and sensors have not been calibrated to MechDog;
do not transfer these motor targets to hardware.

## Current custom firmware compatibility

`bridge.py` loads this checkout's **unchanged `robot_dump/main.py`** and runs its
actual `Bridge` command handlers, sensor filter, safety checks, and `DogHAL`.
PyBullet devices supply the SDK interfaces used by `DogHAL`: movement, sonar,
IMU angles, and battery millivolts. The original capability probing, sensor
conversions and IMU fallback execute unchanged. Small host adapters provide
MicroPython's clock and socket-poll conventions. It loads `config.example.py`, changes only
the listening port, and restricts binding to localhost. It never reads the real
`config.py`, connects to robot WiFi, or accesses USB.

The smoke test passed **29 checks**, including real TCP and the existing
`tools/dog_panel.py` `WifiDog` client: connection, telemetry, move/stop, clamping,
invalid JSON, heartbeat, watchdog, link loss, low battery, fall/recovery, and
obstacle stopping using a real PyBullet ray cast and collision object. Additional
checks cover the original `DogHAL` class, millivolt conversion, unavailable
battery readings, invalid sonar readings, derived IMU data, capability flags,
fragmented/batched TCP commands, reset, and rejection of unsupported commands.

That verifies the tested bridge behavior against simulated hardware. It does
not execute the ESP32 firmware image, WiFi stack, or Hiwonder's
frozen motion library. It also does not verify the settings currently flashed
on the robot, MicroPython clock wraparound, or real hardware timing.

| Capability | Simulator behavior |
| --- | --- |
| `move`, `stop`, `hb`, `ping`, `sub` | Actual bridge logic; movement maps to small motor excursions, not calibrated travel/steering |
| Sonar, IMU | Physics ray cast and body orientation through original `DogHAL`; its fallback derives angular rates and supplies synthetic acceleration (0, 0, 1), with raw-IMU capability disabled |
| Battery | Explicit 7.4 V constant; test can inject low voltage |
| `reset` | Stops motion; requested height/gait restoration unsupported |
| `action`, `height`, `gait`, `posture` | Unsupported; capability flags disabled, handlers return unsuccessful ACKs |
| `rgb`, `buzzer` | No effect; the unchanged bridge sends successful ACKs even with false capability flags |

The existing panel ignores action ACK failures, so its action buttons can appear
accepted despite no simulated effect. Use it for the tested movement/telemetry
path, not action-group validation. The default scene is an empty floor; sonar
reports unknown (`null`) when its ray hits nothing.

After installation, from the checkout root:

```bash
.venv/bin/python simulator/smoke_bridge.py
.venv/bin/python simulator/bridge.py --port 5005
# In a second terminal:
.venv/bin/python tools/dog_panel.py --wifi 127.0.0.1 --wifi-only
```

`bridge.py` supports `--gui` and `--port` (0 chooses a free port; default 5005).
It serves only `127.0.0.1`; Ctrl-C stops it. The panel expects port 5005 and uses
HTTP port 8000, so close any other local panel before starting this one. The
test uses an ephemeral port and imports the panel client without starting its
web server or hardware discovery. Linux is required for the host poll adapter.

`artifacts/bridge-report.json` contains `scope`, `firmware_sha256`, named boolean
`checks`, and `passed`. [`verification-bridge.json`](verification-bridge.json)
records the verified run. Its firmware SHA-256 matched both this worktree and
the original checkout on 2026-09-26. Rerun the test after firmware changes;
this result applies only to the recorded source and template configuration.

The hello message adds `caps.simulation: true`; clients can distinguish this
adapter from hardware. Other flags come from the original `DogHAL` probes.
The battery capability names the simulated `Battery_power` SDK function; the
normal 7.4 V reading is a constant, not a battery discharge model.

For simulator code, `create_runtime()` exposes `runtime.hal` as the original
firmware `DogHAL`, and `runtime.hardware` as the PyBullet devices. Scene IDs,
motor state, `distance_cm_override` and `battery_v_override` live on
`runtime.hardware`. Set the distance override to `None` for ray-cast readings;
set the battery override to `None` to simulate an unavailable battery reading.

## Install

Run from this checkout's root. Use a separate Python 3.11 environment: the
published Linux x86-64 PyBullet wheel supports 3.11, while this repo's hardware
notes describe a newer Python environment. The wheel includes robot assets,
so no second model download, ROS, GPU or Gym installation is required.

```bash
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python --only-binary=:all: -r simulator/requirements.txt
```

Alternatively, with Python 3.11 already installed:

```bash
python3.11 -m venv .venv
.venv/bin/python -m pip install --only-binary=:all: -r simulator/requirements.txt
```

Linux x86-64 is the verified platform. `--only-binary` fails promptly on platforms
without a matching wheel instead of silently starting a source build. Dependencies
are pinned; upstream models remain in the installed package with upstream licensing.

## Run and verify

```bash
.venv/bin/python simulator/run.py
.venv/bin/python simulator/run.py --gui --seconds 30 --output simulator/artifacts/gui
```

Default run: ten simulated seconds, headless, fixed 1/240-second steps. GUI mode
opens a 3D window and paces the same simulation near real time. `--seconds`
accepts finite values from 4 through 300. `--output` selects the artifact directory;
the default is `simulator/artifacts/`, regardless of the current directory.

Each completed run writes `minitaur.png` (640×480 CPU rendering) and `report.json`.
Require a **fresh report with `passed: true`**, in addition to exit status 0:
PyBullet's native GUI initialization can exit with code 0 on a graphics failure.
Use a distinct output directory when checking a new graphics configuration.

The smoke test checks eight motors, four leg constraints, ground contact during
more than half the run, motion greater than 0.05 radians in every motor, body
height above 0.08 m throughout, final roll/pitch below 0.3 radians, and more than
100 rendered robot pixels. Non-finite body/joint states abort immediately.
Failed checks produce exit status 1; invalid arguments produce exit status 2.

Report fields: `model`, `pybullet_version`, `connection` (`DIRECT` or `GUI`),
`steps`, `simulated_seconds`, `contact_steps`, `motor_excursion_rad` (eight
values, front-left/back-left/front-right/back-right, L then R),
`final_position_m` (XYZ), `final_roll_pitch_rad`, `robot_pixels`, `checks`
(named booleans), and `passed` (all checks passed).

Verified on 2026-09-26, NixOS x86-64, Python 3.11.14:

- Headless: 2,400 steps; contact in 2,366 steps; all seven checks passed.
- Every motor traversed approximately 0.24 radians; robot ended upright.
- PNG rendered and visually inspected; 17,305 robot pixels.
- GUI: 960 steps; all seven checks passed using the NixOS loader path below.
- Ruff and mypy passed for all three simulator Python files.

The checked-in [`verification.json`](verification.json) records that headless run.
[`verification-gui.json`](verification-gui.json) records the graphical run.
New runs write separate reports under the ignored artifact directory.

These are execution checks, not locomotion-quality or sim-to-real benchmarks.

## NixOS graphical viewer

The Python wheel dynamically loads OpenGL. On this machine its default library
path omitted `libGL.so` and `libGLX.so.0`, producing `Error in gladLoadGLX`.
This process-local setting enabled the Intel Mesa viewer:

```bash
LD_LIBRARY_PATH=/nix/store/kfy4qljyaacifhdfrr0m9wcbig4jjihq-libglvnd-1.7.0/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH} \
  .venv/bin/python simulator/run.py --gui --seconds 30
```

That Nix store path is specific to this machine/generation. Other NixOS installs
need their installed `libglvnd` library directory. A sandbox also needs access to
the desktop display. Headless mode uses the CPU renderer and needs neither setting.

## Model provenance

Joint names, initial pose, motor directions, knee pivots and nominal motor force
follow the [upstream Minitaur implementation](https://github.com/bulletphysics/bullet3/blob/master/examples/pybullet/gym/pybullet_envs/bullet/minitaur.py).
The loader uses the public PyBullet API directly, avoiding the bundled legacy Gym
registration layer. A faithful MechDog simulation still needs measured geometry,
mass/inertia, servo calibration, linkage constraints and a validated controller.
