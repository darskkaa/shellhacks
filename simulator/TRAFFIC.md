# Traffic-light scene

Run from the vision worktree, with the environment and downloaded assets from
[VISION.md](VISION.md) and [ASSETS.md](ASSETS.md):

```sh
.venv/bin/python simulator/traffic_demo.py --port 0
# NixOS desktop (path specific to this installed generation):
env LD_LIBRARY_PATH=/nix/store/kfy4qljyaacifhdfrr0m9wcbig4jjihq-libglvnd-1.7.0/lib \
  .venv/bin/python simulator/traffic_demo.py --gui --port 5005 --seconds 60
```

The scene includes two textured cars, a crosswalk, a three-color vehicle signal,
a pedestrian indicator, a person, and the existing robot/firmware bridge.
Zebra stripes run parallel to car travel, spaced across the pedestrian route.
Cars travel in one direction. The cycle is 6 seconds red, 6 green, 2 amber.
New cars stop before the crossing on red or amber. Cars already past the stop
line clear the crossing. Cars maintain 4.5 m center spacing; after leaving the
scene they reappear upstream. Pedestrian green requires red vehicle lights AND
an empty crossing. The person remains stationary.

Cars use scripted position updates at 1.8 m/s, with instantaneous stopping;
this is not vehicle dynamics or a model of a particular real intersection.
Signal colors and the image overlay are simulator state, not YOLO predictions.
The robot stays on the sidewalk unless controlled through the existing panel;
this demo does not autonomously cross, perceive traffic signals, or avoid cars.
Car collisions are static mesh approximations, not validated impact physics.

CLI: `--gui` opens the viewer and paces physics steps to wall time (rendering may
slow playback); default headless runs as fast as possible. `--seconds` accepts
finite 1–60, default 24. `--port` accepts 0–65535, default 5005; 0 selects a free
localhost port. `--assets` defaults to `simulator/artifacts/assets`; `--output`
defaults to `simulator/artifacts/traffic`. Do not run a second bridge on the same
port. The viewer closes after the requested simulated duration; Ctrl-C stops early.

Headless output: `first.png`, `last.png`, `traffic.gif` (10 frames per simulated
second), and `report.json`. GUI mode writes only the report, avoiding image
readback that stalled this desktop viewer. `recorded_images` distinguishes the
two modes; existing images in a reused directory may belong to an earlier run. The report records completion, checks, firmware hash, elapsed
simulation duration, stop-line crossings/violations, minimum center spacing,
and a trace of timestamps, signal phases, pedestrian state, and car positions.
A successful full-cycle run requires all three phases, cars stopped on red,
cars entering on green, no red/amber entries, and maintained spacing. Short runs
may exit 1 because they cannot satisfy all checks. A fresh `completed: true` and
`passed: true` report is required: native GUI failures can return process exit 0.

```sh
.venv/bin/python -m unittest simulator.test_traffic
```

Tests use actual PyBullet bodies and local meshes to check queue stopping,
green release, crossing clearance during amber/red, and upstream recycling.

Verified on this NixOS desktop: headless and GUI 14-second full-cycle runs both
passed all checks, with zero prohibited entries. See `verification-traffic.json`.
Crosswalk orientation was inspected in a corrected render. GUI lamp materials
update only on signal changes; per-physics-step updates stalled the viewer.
