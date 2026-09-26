# Run the traffic-light simulation

## On Sean's machine: already installed

Use the **vision worktree**, not `/tmp/shellhack-mechdog-simulator` (that older
worktree does not contain the traffic scene). Paste these commands into Fish or Bash:

```sh
cd /tmp/shellhack-sim-vision
env LD_LIBRARY_PATH=/nix/store/kfy4qljyaacifhdfrr0m9wcbig4jjihq-libglvnd-1.7.0/lib \
  .venv/bin/python simulator/traffic_demo.py --gui --port 0 --seconds 60
```

A window opens with two moving cars, a crosswalk, a person, the robot, and traffic
lights. Cars stop on red/amber and move on green. The pedestrian indicator turns
green only while vehicles have red and the crossing is clear. The robot and
person do not cross automatically. The lights repeat every 14 simulated seconds.
The window closes after 60 simulated seconds; Ctrl-C stops it earlier.
`--port 0` avoids conflicts with any existing robot-control server.

## Fresh checkout: one-time setup

Requires `git`, `curl`, `uv`, and access to this private GitHub repository.
Clone into a new directory so other agents' worktrees remain untouched:

```sh
git clone --branch feat/mechdog-sim-traffic --single-branch \
  https://github.com/darskkaa/shellhacks.git shellhacks-traffic
cd shellhacks-traffic
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python --only-binary=:all: \
  -r simulator/requirements.txt trimesh==5.1.0
mkdir -p simulator/artifacts/assets
curl --fail --location --connect-timeout 10 --max-time 60 --max-filesize 12000000 \
  -o simulator/artifacts/assets/Soldier.glb \
  https://raw.githubusercontent.com/mrdoob/three.js/dev/examples/models/gltf/Soldier.glb
curl --fail --location --connect-timeout 10 --max-time 60 --max-filesize 12000000 \
  -o simulator/artifacts/assets/ToyCar.glb \
  https://raw.githubusercontent.com/KhronosGroup/glTF-Sample-Assets/main/Models/ToyCar/glTF-Binary/ToyCar.glb
.venv/bin/python simulator/scene_assets.py
```

Preparation verifies download hashes and converts the meshes locally. Asset
attributions and terms are in [ASSETS.md](ASSETS.md). Stop if either download or
preparation fails. Traffic alone needs **no Torch, YOLO weights, or GPU inference**.

On a Linux desktop with OpenGL libraries available:

```sh
.venv/bin/python simulator/traffic_demo.py --gui --port 0 --seconds 60
```

On this NixOS machine, use the `env LD_LIBRARY_PATH=...` command above, from your
new checkout directory. That `/nix/store` path is specific to this installed
system generation; other NixOS systems need their own installed `libglvnd` path.

## Record a GIF without opening a window

```sh
.venv/bin/python simulator/traffic_demo.py --port 0 --seconds 24
.venv/bin/python -c 'import json; r=json.load(open("simulator/artifacts/traffic/report.json")); assert r["completed"] and r["passed"]; print("Traffic checks passed")'
```

Open `simulator/artifacts/traffic/traffic.gif` to watch the animation.
`first.png` and `last.png` are still frames. A full-cycle run checks red-light
stopping, green-light entry, spacing, and all three signal phases.
GUI mode writes `report.json` only; use headless mode to record images.

## Optional: connect the existing control panel

Start the simulation with `--port 5005` instead of `--port 0`. Stop any older
bridge on that port first. In another terminal, enter the same checkout directory:

```sh
.venv/bin/python tools/dog_panel.py --wifi 127.0.0.1 --wifi-only
```

Open the local address printed by the panel. Robot commands exercise the
unchanged firmware bridge and simulated joints; this is not calibrated MechDog
walking or autonomous street crossing.

## If something fails

- `Error in gladLoadGLX`: use the NixOS `env LD_LIBRARY_PATH=...` command.
- `can't open file ...traffic_demo.py`: wrong checkout; use the vision worktree
  above or clone `feat/mechdog-sim-traffic`.
- `Missing ...Soldier.glb` / `ToyCar.glb`: run both downloads and preparation.
- `Address already in use`: use `--port 0`, or stop the older bridge for panel use.
- No desktop/display: run without `--gui` and inspect the GIF.
- Exit code 1 after a short run: use at least 14 seconds to cover all phases.
- A native GUI error can exit 0; trust a **fresh** report with `completed: true`
  and `passed: true`, not process exit alone.

The traffic motion and light states are scripted. This does not verify YOLO
recognition of lights or cars. See [TRAFFIC.md](TRAFFIC.md) for behavior and CLI
details, or [VISION.md](VISION.md) for the separate YOLO person-approach experiment.
