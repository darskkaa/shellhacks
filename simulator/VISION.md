# Rendered-camera YOLO experiment

This experiment sends real YOLO predictions from PyBullet RGB frames through the
unchanged firmware's localhost TCP bridge. A detected target controls a planar
motion proxy with animated joints. This does **not** validate MechDog walking.
Depth comes from the simulator, not a monocular real-world range estimator.
There is no route planner or car/obstacle avoidance.

## Install

Use a separate worktree and Python 3.11 environment. From its repository root:

```sh
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python --only-binary=:all: \
  --index-url https://download.pytorch.org/whl/cpu \
  torch==2.14.0+cpu torchvision==0.29.0+cpu
uv pip install --python .venv/bin/python --only-binary=:all: \
  -r simulator/requirements-vision.txt
```

Fetch/convert meshes using [ASSETS.md](ASSETS.md). Models must already exist
locally; automatic weight downloads and dependency installation are disabled.
The default is the repository's `models/yolo26n.pt`, whose labels include person
and car. The separate blind-escort custom ONNX model has 20 classes but no
individual-person class; use it with `--detect-only`.

On NixOS, OpenCV requires libGL and GLib even for headless inference. Supply
installed library directories through `LD_LIBRARY_PATH`; on this machine:

```sh
export LD_LIBRARY_PATH=/nix/store/kfy4qljyaacifhdfrr0m9wcbig4jjihq-libglvnd-1.7.0/lib:/nix/store/7mf69bavdjazjvbhflj0s40d2a6mk5wb-glib-2.88.3/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
```

These paths change with the NixOS generation.

## Run and inspect

```sh
.venv/bin/python simulator/vision_demo.py --detect-only
.venv/bin/python simulator/vision_demo.py --frames 80
.venv/bin/python simulator/vision_demo.py --hide-person --frames 3 \
  --output simulator/artifacts/vision-no-person
.venv/bin/python simulator/vision_demo.py --model /path/to/custom.onnx --detect-only
.venv/bin/python -m unittest simulator.test_vision_controller
```

CLI: `--model` local weights, `--assets` converted asset directory, `--output`
artifact directory, `--frames` 1–300 (default 80), `--target` label (default
`person`), `--conf` finite (0,1] (default 0.35), `--detect-only` one frame without
movement, `--hide-person` remove the person for a negative check.

The controller follows the highest-confidence target. It turns toward the box
center, advances only when roughly centered, and stops at depth ≤1.5 m. Missing
target or invalid depth means zero stride and turn. It does not track individual
identity. Each inference is followed by 0.2 s simulated motion; wall-clock timing
and real-time control latency are not validated. Firmware safety gates still
apply to commands. The camera height is fixed at 0.8 m, an experimental viewpoint.

Outputs are ignored by Git: `camera.png`, annotated `first.png`/`last.png`,
`approach.gif`, and `report.json`. The report includes completion/reached flags,
model and firmware hashes, model labels, target/confidence/detect-only settings,
scene object names, final pose, and per-frame predictions/commands/poses.
`completed` means the run finished, not that detection was correct. Approach mode
exits 1 unless it reaches a detected target; absent-target checks therefore expect
exit 1 and zero movement. Detect-only exits 0 even with empty/wrong predictions.
Never infer successful person recognition merely from process exit status.

## Verified result

Recorded in `verification-vision.json`: generic YOLO26n detected the person in
all 31 frames of one fixed scene. The planar proxy moved 2.16 m, reducing sampled
camera depth from 3.656 m to 1.496 m, then stopped. A three-frame person-absent run
stayed at the origin. These are functional checks, not accuracy estimates.

The car was correctly detected in one approach frame and mislabeled as airplane
in others. The custom blind-escort ONNX runs, but its staircase predictions on
this scene are false positives; it cannot supply an individual-person target.

Earlier scenes failed: the stylized CesiumMan was not recognized; Soldier facing
backward lost detection after 7 cm. Facing the person toward the camera fixed
that view. A 1.2 m stop trial lost recognition when the person became cropped;
1.5 m stand-off keeps the figure visible. Other viewpoints need further testing.
