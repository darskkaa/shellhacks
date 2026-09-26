"""Guide-dog street crossing gated by the blind-escort YOLO model's pedestrian-signal detections.

The dog waits at the curb with its handler on its right. It crosses only after
the custom ONNX model reports ped_signal_walk on several consecutive camera
frames, then leads the handler to the far curb. The simulator's own signal state
is used only to grade the model and the crossing, never to drive the dog.
"""

import argparse
import hashlib
import json
import math
import socket
import sys
import time
from pathlib import Path

import numpy as np
import pybullet as p
from PIL import Image, ImageDraw

if __package__:
    from .bridge import create_runtime
    from .run import TIME_STEP
    from .traffic_demo import TrafficScene, box
else:
    from bridge import create_runtime
    from run import TIME_STEP
    from traffic_demo import TrafficScene, box

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from blind_escort_yolo.inference import BlindEscortDetector  # noqa: E402

CYCLE = (22.0, 8.0, 2.0)  # red, green, amber: long red so a slow escort can finish
WALK_SECONDS = 10.0  # WALK shown at the start of red; the rest is DON'T WALK clearance
START_OFFSET = 22.0  # begin at the start of green so the dog first sees DON'T WALK
ROAD_X = (3.0, 7.0)
START_X, GOAL_X = 2.4, 8.2
HANDLER_OFFSET = (-0.35, -0.6)  # behind and to the dog's right (guide dog on handler's left)
SIGNAL_POS = (7.6, -1.3, 2.4)
SIGNAL_HALF = 0.4  # 0.8 m face: larger than a real ~0.45 m head, see ESCORT.md
DECISION_STEPS = 48  # 0.2 s of simulated time per camera frame
WIDTH, HEIGHT = 640, 480

WALK_CONF = 0.4  # this synthetic WALK face peaks near 0.47; DON'T WALK frames score 0.0 walk
MARGIN = 0.1
CONFIRM_FRAMES = 3
CONFLICT_CONF = 0.5
CROSS_STRIDE = 100

WHITE, ORANGE, BLACK = (245, 248, 255), (255, 118, 16), (6, 6, 6)


def signal_face(state: str, size: int = 256) -> Image.Image:
    """Lunar-white walking person or orange raised hand on a black lens."""
    im = Image.new("RGB", (size, size), BLACK)
    d = ImageDraw.Draw(im)
    s = size / 256

    def seg(a, b, w):
        d.line([a[0] * s, a[1] * s, b[0] * s, b[1] * s], fill=WHITE, width=int(w * s))
        for q in (a, b):
            r = w * s / 2
            d.ellipse([q[0] * s - r, q[1] * s - r, q[0] * s + r, q[1] * s + r], fill=WHITE)

    if state == "walk":
        d.ellipse([112 * s, 22 * s, 148 * s, 58 * s], fill=WHITE)
        seg((128, 72), (120, 142), 34)
        seg((126, 80), (96, 116), 20)
        seg((96, 116), (86, 150), 18)
        seg((128, 80), (158, 108), 20)
        seg((158, 108), (180, 124), 18)
        seg((120, 142), (100, 186), 24)
        seg((100, 186), (80, 228), 22)
        seg((122, 142), (156, 182), 24)
        seg((156, 182), (164, 230), 22)
    else:
        d.rounded_rectangle([84 * s, 112 * s, 178 * s, 222 * s], radius=int(24 * s), fill=ORANGE)
        for x, top in zip([88, 111, 134, 157], [62, 42, 46, 64]):
            d.rounded_rectangle([x * s, top * s, (x + 20) * s, 132 * s], radius=int(10 * s), fill=ORANGE)
        d.polygon([(88 * s, 152 * s), (52 * s, 110 * s), (38 * s, 122 * s), (82 * s, 194 * s)], fill=ORANGE)
    return im


def vote(detections: list[dict], walk_conf: float = WALK_CONF) -> dict:
    """Classify one frame as walk / dont_walk / unknown from model detections alone."""
    best = lambda name: max((d["confidence"] for d in detections if d["class"] == name), default=0.0)
    walk, stop, conflict = best("ped_signal_walk"), best("ped_signal_stop"), best("conflict_vehicle_cyclist")
    if walk >= walk_conf and walk >= stop + MARGIN and conflict < CONFLICT_CONF:
        label = "walk"
    elif stop > 0 or conflict >= CONFLICT_CONF:
        label = "dont_walk"
    else:
        label = "unknown"
    return {"vote": label, "walk_conf": round(walk, 3), "stop_conf": round(stop, 3), "conflict_conf": round(conflict, 3)}


class EscortController:
    """Wait at the curb until WALK is confirmed, then commit to crossing to the far curb."""

    def __init__(self) -> None:
        self.state = "waiting"
        self.streak = 0

    def update(self, frame_vote: dict, x: float) -> dict:
        if self.state == "waiting":
            self.streak = self.streak + 1 if frame_vote["vote"] == "walk" else 0
            if self.streak >= CONFIRM_FRAMES:
                self.state = "crossing"
        if self.state == "crossing" and x >= GOAL_X:
            self.state = "arrived"
        stride = CROSS_STRIDE if self.state == "crossing" else 0
        return {"state": self.state, "streak": self.streak, "stride": stride, "angle": 0}


class SignalHead:
    def __init__(self, textures_dir: Path):
        x, y, z = SIGNAL_POS
        h = SIGNAL_HALF
        box([x + 0.1, y, z / 2], [0.05, 0.05, z / 2], [0.3, 0.3, 0.3, 1])
        box([x + 0.06, y, z], [0.05, h + 0.03, h + 0.03], [0.05, 0.05, 0.05, 1])
        shape = p.createVisualShape(
            p.GEOM_MESH,
            vertices=[[0, h, -h], [0, -h, -h], [0, -h, h], [0, h, h]],
            indices=[0, 1, 2, 0, 2, 3],
            uvs=[[0, 0], [1, 0], [1, 1], [0, 1]],
            normals=[[-1, 0, 0]] * 4,
            rgbaColor=[1, 1, 1, 1],
        )
        self.body = p.createMultiBody(baseMass=0, baseVisualShapeIndex=shape, basePosition=[x, y, z])
        textures_dir.mkdir(parents=True, exist_ok=True)
        self.textures = {}
        for state in ("walk", "stop"):
            path = textures_dir / f"ped_signal_{state}.png"  # distinct names: PyBullet caches by path
            signal_face(state).save(path)
            self.textures[state] = p.loadTexture(str(path))
        self.shown = None

    def show(self, walk: bool) -> None:
        state = "walk" if walk else "stop"
        if state != self.shown:
            p.changeVisualShape(self.body, -1, textureUniqueId=self.textures[state])
            self.shown = state


def dog_camera(x: float, y: float, yaw: float) -> Image.Image:
    eye = [x + 0.2 * math.cos(yaw), y + 0.2 * math.sin(yaw), 0.8]
    look = [eye[0] + math.cos(yaw) * math.cos(0.28), eye[1] + math.sin(yaw) * math.cos(0.28), 0.8 + math.sin(0.28)]
    view = p.computeViewMatrix(eye, look, [0, 0, 1])
    proj = p.computeProjectionMatrixFOV(65, WIDTH / HEIGHT, 0.05, 20)
    _, _, rgba, _, _ = p.getCameraImage(WIDTH, HEIGHT, view, proj, renderer=p.ER_TINY_RENDERER)
    return Image.fromarray(np.asarray(rgba, dtype=np.uint8)).convert("RGB")


def overhead() -> Image.Image:
    view = p.computeViewMatrix([-1.5, -9, 7], [5, 0, 0.6], [0, 0, 1])
    proj = p.computeProjectionMatrixFOV(55, WIDTH / HEIGHT, 0.1, 60)
    _, _, rgba, _, _ = p.getCameraImage(WIDTH, HEIGHT, view, proj, renderer=p.ER_TINY_RENDERER)
    return Image.fromarray(np.asarray(rgba, dtype=np.uint8)).convert("RGB")


def compose(top: Image.Image, cam: Image.Image, detections, fv, ctl, gt_walk, phase, t) -> Image.Image:
    cam = cam.copy()
    d = ImageDraw.Draw(cam)
    for det in detections:
        color = {"ped_signal_walk": (40, 220, 90), "ped_signal_stop": (255, 90, 40)}.get(det["class"], (150, 150, 150))
        d.rectangle(det["box"], outline=color, width=3 if color != (150, 150, 150) else 1)
        d.text((det["box"][0], max(0, det["box"][1] - 12)), f"{det['class']} {det['confidence']:.2f}", fill=color)
    decision = {"waiting": f"WAIT ({ctl['streak']}/{CONFIRM_FRAMES} walk frames)", "crossing": "CROSSING", "arrived": "ARRIVED"}[ctl["state"]]
    d.rectangle([0, 0, WIDTH, 40], fill="black")
    d.text((8, 4), f"DOG CAMERA + YOLO | walk {fv['walk_conf']:.2f}  dont_walk {fv['stop_conf']:.2f}  -> {fv['vote'].upper()}", fill="white")
    d.text((8, 22), f"Dog decision: {decision}", fill=(80, 255, 120) if ctl["state"] != "waiting" else (255, 200, 80))
    top = top.copy()
    d = ImageDraw.Draw(top)
    d.rectangle([0, 0, WIDTH, 40], fill="black")
    d.text((8, 4), f"SIM GROUND TRUTH | t={t:5.1f}s  cars: {phase.upper()}  ped signal: {'WALK' if gt_walk else 'DONT WALK'}", fill="white")
    d.text((8, 22), "Handler on the dog's right. Dog is driven only by YOLO, not by this state.", fill=(200, 200, 200))
    out = Image.new("RGB", (WIDTH * 2, HEIGHT))
    out.paste(top, (0, 0))
    out.paste(cam, (WIDTH, 0))
    return out.resize((WIDTH * 3 // 2, HEIGHT * 3 // 4))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gui", action="store_true")
    parser.add_argument("--port", type=int, default=0)
    parser.add_argument("--seconds", type=float, default=30)
    parser.add_argument("--model", type=Path, default=ROOT / "blind_escort_yolo/weights/yolo11n_blind_escort.onnx")
    parser.add_argument("--assets", type=Path, default=ROOT / "simulator/artifacts/assets")
    parser.add_argument("--output", type=Path, default=ROOT / "simulator/artifacts/escort")
    parser.add_argument("--walk-conf", type=float, default=WALK_CONF,
                        help="minimum ped_signal_walk confidence per frame (recalibrate on real footage)")
    parser.add_argument("--signal", choices=["cycle", "stop"], default="cycle",
                        help="'stop' holds DON'T WALK forever: the dog must never cross")
    args = parser.parse_args()
    if not math.isfinite(args.seconds) or not 5 <= args.seconds <= 90:
        parser.error("--seconds must be finite in [5, 90]")
    if not 0 < args.walk_conf <= 1:
        parser.error("--walk-conf must be in (0, 1]")
    if not args.model.is_file():
        parser.error("Model must be an existing local file")
    args.output.mkdir(parents=True, exist_ok=True)
    report_path = args.output / "report.json"
    report_path.write_text(json.dumps({"completed": False, "passed": False}) + "\n")

    detector = BlindEscortDetector(str(args.model))
    runtime = create_runtime(args.port, args.gui)
    client = None
    frames, trace = [], []
    try:
        scene = TrafficScene(args.assets, cycle=CYCLE)
        scene.elapsed = START_OFFSET
        scene.update(0)
        for body in (scene.pedestrian, scene.pedestrian_pole):  # replaced by the far-side signal head
            p.resetBasePositionAndOrientation(body, [0, 30, -10], [0, 0, 0, 1])
        # Vehicle lamps face oncoming cars (-y), as on a real street, not the crosswalk.
        for index, body in enumerate(scene.lamps.values()):
            p.resetBasePositionAndOrientation(body, [7.4, -2.2, 2.9 - index * 0.4], [0, 0, 0, 1])
        signal = SignalHead(args.output / "textures")
        total = sum(CYCLE)
        x, y, yaw = START_X, 0.0, 0.0

        def place() -> None:
            p.resetBasePositionAndOrientation(runtime.hardware.robot, [x, y, 0.2], p.getQuaternionFromEuler([0, 0, yaw]))
            p.resetBaseVelocity(runtime.hardware.robot, [0, 0, 0], [0, 0, 0])
            hx = x + HANDLER_OFFSET[0] * math.cos(yaw) - HANDLER_OFFSET[1] * math.sin(yaw)
            hy = y + HANDLER_OFFSET[0] * math.sin(yaw) + HANDLER_OFFSET[1] * math.cos(yaw)
            p.resetBasePositionAndOrientation(scene.person, [hx, hy, 0], p.getQuaternionFromEuler([0, 0, yaw]))

        place()
        if args.gui:
            p.resetDebugVisualizerCamera(9, 20, -35, [5, 0, 0])
        client = socket.create_connection(("127.0.0.1", runtime.bridge.port), timeout=2)
        runtime.step()
        client.recv(8192)
        client.setblocking(False)

        controller = EscortController()
        gt_walk = False
        stats = {"frames": 0, "walk_votes_on_walk": 0, "walk_votes_on_dont_walk": 0,
                 "dont_walk_votes_on_dont_walk": 0, "non_walk_votes_on_walk": 0}
        road_violations, min_car_gap, max_handler_gap = 0, math.inf, 0.0
        cross_start = None
        first_gt_walk = None
        step = 0
        while step * TIME_STEP < args.seconds:
            cam = dog_camera(x, y, yaw)
            result = detector.predict(cam, conf_thresh=0.25)
            detections = result["detections"]
            fv = vote(detections, args.walk_conf)
            was_waiting = controller.state == "waiting"
            ctl = controller.update(fv, x)
            if was_waiting:  # grade the model only while it is the thing deciding
                stats["frames"] += 1
                if gt_walk:
                    stats["walk_votes_on_walk" if fv["vote"] == "walk" else "non_walk_votes_on_walk"] += 1
                else:
                    stats["walk_votes_on_dont_walk" if fv["vote"] == "walk" else "dont_walk_votes_on_dont_walk"] += 1
            if was_waiting and ctl["state"] == "crossing":
                cross_start = {"seconds": round(step * TIME_STEP, 2), "ground_truth_walk": gt_walk, "phase": scene.phase}
            client.sendall((json.dumps({"t": "move", "stride": ctl["stride"], "angle": ctl["angle"]}) + "\n").encode())
            t = step * TIME_STEP
            trace.append({"seconds": round(t, 2), "ground_truth_walk": gt_walk, "phase": scene.phase, **fv, **ctl,
                          "dog_x": round(x, 3), "detections": detections})
            if not args.gui:
                frames.append(compose(overhead(), cam, detections, fv, ctl, gt_walk, scene.phase, t))
            for _ in range(DECISION_STEPS):
                start = time.monotonic()
                red_time = scene.elapsed % total
                scene.update(TIME_STEP)
                gt_walk = args.signal == "cycle" and scene.walk and red_time < WALK_SECONDS
                if first_gt_walk is None and gt_walk:
                    first_gt_walk = round(step * TIME_STEP, 2)
                signal.show(gt_walk)
                runtime.step()
                speed = runtime.hardware.stride / 100 * 0.6  # same planar proxy as vision_demo.py
                yaw += math.radians(runtime.hardware.angle) * TIME_STEP
                x += speed * math.cos(yaw) * TIME_STEP
                y += speed * math.sin(yaw) * TIME_STEP
                place()
                hx, hy, _ = p.getBasePositionAndOrientation(scene.person)[0]
                max_handler_gap = max(max_handler_gap, math.hypot(hx - x, hy - y))
                for px, py in ((x, y), (hx, hy)):
                    if ROAD_X[0] < px < ROAD_X[1]:
                        road_violations += scene.phase != "red"
                        for car_y in scene.positions:
                            min_car_gap = min(min_car_gap, math.hypot(5 - px, car_y - py))
                step += 1
                if args.gui:
                    time.sleep(max(0, TIME_STEP - (time.monotonic() - start)))
            try:
                client.recv(65536)
            except BlockingIOError:
                pass
        runtime.hal.stop()
        hx = p.getBasePositionAndOrientation(scene.person)[0][0]
        crossed = cross_start is not None
        if args.signal == "stop":
            checks = {
                "no_walk_votes_on_dont_walk": stats["walk_votes_on_dont_walk"] == 0,
                "stayed_on_curb": not crossed and x < ROAD_X[0],
            }
        else:
            checks = {
                "no_walk_votes_on_dont_walk": stats["walk_votes_on_dont_walk"] == 0,
                "crossing_started_on_true_walk": crossed and cross_start["ground_truth_walk"],
                "in_road_only_while_cars_held_on_red": crossed and road_violations == 0,
                "dog_reached_far_curb": x >= GOAL_X,
                "handler_reached_far_curb": hx >= ROAD_X[1],
                "handler_stayed_beside_dog": max_handler_gap <= 0.8,
            }
        if frames:
            frames[0].save(args.output / "first.png")
            frames[-1].save(args.output / "last.png")
            frames[0].save(args.output / "escort.gif", save_all=True, append_images=frames[1:], duration=200, loop=0)
        report = {
            "completed": True,
            "passed": all(checks.values()),
            "checks": checks,
            "signal_mode": args.signal,
            "seconds": args.seconds,
            "model": str(args.model),
            "model_sha256": hashlib.sha256(args.model.read_bytes()).hexdigest(),
            "firmware_sha256": runtime.firmware_sha256,
            "decision_rule": {"walk_conf": args.walk_conf, "margin_over_dont_walk": MARGIN,
                              "confirm_frames": CONFIRM_FRAMES, "conflict_veto": CONFLICT_CONF},
            "model_grading_while_waiting": stats,
            "first_ground_truth_walk_s": first_gt_walk,
            "crossing_start": cross_start,
            "detection_latency_s": round(cross_start["seconds"] - first_gt_walk, 2) if crossed and first_gt_walk is not None else None,
            "road_violations_steps": road_violations,
            "min_car_center_distance_in_road_m": None if math.isinf(min_car_gap) else round(min_car_gap, 2),
            "max_handler_gap_m": round(max_handler_gap, 3),
            "final_dog_x": round(x, 3),
            "final_handler_x": round(hx, 3),
            "trace": trace,
            "limitations": "Synthetic signal face (0.8 m, oversized), rendered TinyRenderer frames, planar motion proxy, "
                           "static handler mesh. Grades the model on this scene only; not evidence of real-street accuracy.",
        }
        report_path.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps({k: report[k] for k in ("passed", "checks", "model_grading_while_waiting",
                                                 "crossing_start", "detection_latency_s", "max_handler_gap_m")}), flush=True)
    finally:
        if client is not None:
            client.close()
        runtime.close()
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
