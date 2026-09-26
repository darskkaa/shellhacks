"""Scripted traffic signals and queued cars beside the firmware-driven robot."""

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import pybullet as p
from PIL import Image, ImageDraw

if __package__:
    from .bridge import create_runtime
    from .run import TIME_STEP
    from .scene_assets import load_scene
else:
    from bridge import create_runtime
    from run import TIME_STEP
    from scene_assets import load_scene

ROOT = Path(__file__).resolve().parents[1]
STOP_Y = -3.0  # 1.5 m car half-length leaves space before the crossing.
SPEED = 1.8
GAP = 4.5
COLORS = {"red": [1, 0.03, 0.02, 1], "amber": [1, 0.6, 0, 1], "green": [0, 1, 0.1, 1]}


CYCLE = (6.0, 6.0, 2.0)  # red, green, amber seconds


def phase_at(seconds: float, cycle: tuple[float, float, float] = CYCLE) -> str:
    red, green, amber = cycle
    phase = seconds % (red + green + amber)
    return "red" if phase < red else "green" if phase < red + green else "amber"


def box(position, half_extents, color) -> int:
    shape = p.createVisualShape(p.GEOM_BOX, halfExtents=half_extents, rgbaColor=color)
    return p.createMultiBody(
        baseMass=0, baseVisualShapeIndex=shape, basePosition=position
    )


class TrafficScene:
    """Own rendered state and deterministic traffic updates; never infer detections."""

    def __init__(self, assets: Path, cycle: tuple[float, float, float] = CYCLE):
        self.cycle = cycle
        objects = load_scene(assets)
        self.person = objects["person"]
        p.resetBasePositionAndOrientation(
            objects["person"], [8, 0, 0], p.getQuaternionFromEuler([0, 0, math.pi])
        )
        box([5, 0, 0.005], [2, 16, 0.005], [0.15, 0.17, 0.2, 1])
        for x in np.arange(3.2, 7.0, 0.45):
            box([float(x), 0, 0.02], [0.12, 0.9, 0.01], [0.95, 0.95, 0.9, 1])
        box([5, -1.4, 0.02], [1.9, 0.07, 0.01], [1, 1, 1, 1])
        box([7.4, -2, 1.3], [0.06, 0.06, 1.3], [0.3, 0.3, 0.3, 1])
        box([7.4, -2, 2.5], [0.22, 0.18, 0.63], [0.04, 0.04, 0.04, 1])
        self.lamps = {}
        for index, name in enumerate(COLORS):
            shape = p.createVisualShape(
                p.GEOM_SPHERE, radius=0.16, rgbaColor=COLORS[name]
            )
            self.lamps[name] = p.createMultiBody(
                baseMass=0,
                baseVisualShapeIndex=shape,
                basePosition=[7.15, -2.15, 2.9 - index * 0.4],
            )
        self.pedestrian_pole = box([2.7, 1.5, 0.9], [0.06, 0.06, 0.9], [0.3, 0.3, 0.3, 1])
        self.pedestrian = box([2.7, 1.5, 1.8], [0.18, 0.12, 0.2], COLORS["red"])
        # Reuse the verified converted car mesh, including its diffuse materials.
        shape = p.createVisualShape(
            p.GEOM_MESH,
            fileName=str(assets.resolve() / "car/model.obj"),
            rgbaColor=[1, 1, 1, 1],
        )
        collision = p.createCollisionShape(
            p.GEOM_MESH,
            fileName=str(assets.resolve() / "car/model.obj"),
            flags=p.GEOM_FORCE_CONCAVE_TRIMESH,
        )
        second = p.createMultiBody(
            baseMass=0, baseVisualShapeIndex=shape, baseCollisionShapeIndex=collision
        )
        self.cars = [objects["car"], second]
        self.positions = [-4.8, -9.3]
        self.elapsed = 0.0
        self.phase = "red"
        self.walk = False
        self.violations = 0
        self.crossings = 0
        self.stopped_red_steps = 0
        self.minimum_gap = GAP
        self._displayed_state: tuple[str, bool] | None = None
        self.update(0)

    def update(self, dt: float) -> None:
        if not math.isfinite(dt) or not 0 <= dt <= 0.1:
            raise ValueError(
                "Traffic step must be finite and between 0 and 0.1 seconds"
            )
        self.phase = phase_at(self.elapsed, self.cycle)
        leader = math.inf
        for index in sorted(
            range(len(self.cars)), key=lambda i: self.positions[i], reverse=True
        ):
            before = self.positions[index]
            after = min(before + SPEED * dt, leader - GAP)
            if self.phase != "green" and before <= STOP_Y:
                after = min(after, STOP_Y)
            if before <= STOP_Y < after:
                self.crossings += 1
                self.violations += self.phase != "green"
            if self.phase == "red" and abs(after - STOP_Y) < 1e-8:
                self.stopped_red_steps += 1
            self.positions[index] = after
            leader = after
        self.minimum_gap = min(
            self.minimum_gap, abs(self.positions[0] - self.positions[1])
        )
        for index, y in enumerate(self.positions):
            if y > 12:
                self.positions[index] = min(-9.3, min(self.positions) - GAP)
            p.resetBasePositionAndOrientation(
                self.cars[index],
                [5, self.positions[index], 0.025],
                p.getQuaternionFromEuler([0, 0, math.pi]),
            )
        clear = all(abs(y) > 2.5 for y in self.positions)
        self.walk = self.phase == "red" and clear
        if (self.phase, self.walk) != self._displayed_state:
            for name, body in self.lamps.items():
                p.changeVisualShape(
                    body,
                    -1,
                    rgbaColor=COLORS[name]
                    if name == self.phase
                    else [0.09, 0.09, 0.09, 1],
                )
            p.changeVisualShape(
                self.pedestrian, -1, rgbaColor=COLORS["green" if self.walk else "red"]
            )
            self._displayed_state = (self.phase, self.walk)
        self.elapsed += dt

    def frame(self) -> Image.Image:
        view = p.computeViewMatrix([-4, -10, 8], [5, 0, 0.6], [0, 0, 1])
        projection = p.computeProjectionMatrixFOV(58, 640 / 480, 0.1, 60)
        _, _, rgba, _, _ = p.getCameraImage(
            640, 480, view, projection, renderer=p.ER_TINY_RENDERER
        )
        frame = Image.fromarray(np.asarray(rgba, dtype=np.uint8)).convert("RGB")
        draw = ImageDraw.Draw(frame)
        draw.rectangle([0, 0, 640, 36], fill="black")
        draw.text(
            (10, 5),
            f"SIM STATE | Cars: {self.phase.upper()} | Pedestrians: {'WALK' if self.walk else 'WAIT'} | {self.elapsed:.1f}s",
            fill="white",
        )
        draw.text(
            (10, 20),
            "Scripted traffic, not YOLO predictions. Robot controlled separately.",
            fill="white",
        )
        return frame


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gui", action="store_true")
    parser.add_argument("--port", type=int, default=5005)
    parser.add_argument("--seconds", type=float, default=24)
    parser.add_argument(
        "--assets", type=Path, default=ROOT / "simulator/artifacts/assets"
    )
    parser.add_argument(
        "--output", type=Path, default=ROOT / "simulator/artifacts/traffic"
    )
    args = parser.parse_args()
    if not math.isfinite(args.seconds) or not 1 <= args.seconds <= 60:
        parser.error("--seconds must be finite in [1, 60]")
    if not 0 <= args.port <= 65535:
        parser.error("--port must be in [0, 65535]")
    args.output.mkdir(parents=True, exist_ok=True)
    report_path = args.output / "report.json"
    report_path.write_text(json.dumps({"completed": False, "passed": False}) + "\n")
    runtime = create_runtime(args.port, args.gui)
    frames = []
    trace = []
    phases = set()
    try:
        scene = TrafficScene(args.assets)
        p.resetBasePositionAndOrientation(
            runtime.hardware.robot, [1.5, 0, 0.2], [0, 0, 0, 1]
        )
        p.resetDebugVisualizerCamera(12, 40, -40, [5, 0, 0])
        print(f"Bridge listening on 127.0.0.1:{runtime.bridge.port}", flush=True)
        for step in range(round(args.seconds / TIME_STEP)):
            start = time.monotonic()
            scene.update(TIME_STEP)
            runtime.step()
            phases.add(scene.phase)
            if step % 24 == 0:
                # Avoid synchronous image readback stalling the desktop viewer.
                if not args.gui:
                    frames.append(scene.frame())
                trace.append(
                    {
                        "seconds": round(scene.elapsed, 3),
                        "phase": scene.phase,
                        "walk": scene.walk,
                        "car_y": list(scene.positions),
                    }
                )
                if args.gui:
                    p.addUserDebugText(
                        f"Cars {scene.phase.upper()} / Pedestrians {'WALK' if scene.walk else 'WAIT'}",
                        [4, 0, 3.6],
                        textSize=1.5,
                        lifeTime=0.15,
                    )
            if args.gui:
                time.sleep(max(0, TIME_STEP - (time.monotonic() - start)))
        checks = {
            "no_red_or_amber_entries": scene.violations == 0,
            "cars_keep_gap": scene.minimum_gap >= GAP - 1e-8,
            "all_phases_observed": phases == set(COLORS),
            "cars_stopped_on_red": scene.stopped_red_steps > 0,
            "cars_entered_on_green": scene.crossings > 0,
        }
        if frames:
            frames[0].save(args.output / "first.png")
            frames[-1].save(args.output / "last.png")
            frames[0].save(
                args.output / "traffic.gif",
                save_all=True,
                append_images=frames[1:],
                duration=100,
                loop=0,
            )
        report = {
            "completed": True,
            "passed": all(checks.values()),
            "checks": checks,
            "seconds": args.seconds,
            "recorded_images": bool(frames),
            "firmware_sha256": runtime.firmware_sha256,
            "crossings": scene.crossings,
            "violations": scene.violations,
            "minimum_center_gap_m": scene.minimum_gap,
            "trace": trace,
            "limitations": "Kinematic cars; signal state is scripted, not perceived. No autonomous robot crossing or collision avoidance.",
        }
        report_path.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps({"passed": report["passed"], "checks": checks}), flush=True)
    finally:
        runtime.close()
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
