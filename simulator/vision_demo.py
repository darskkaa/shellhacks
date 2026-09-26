"""Render a person/car scene and approach a YOLO target using simulated planar motion."""

import argparse
import hashlib
import json
import math
import os
import socket
from pathlib import Path

import numpy as np
import pybullet as p
from PIL import Image

if __package__:
    from .bridge import create_runtime
    from .run import TIME_STEP
    from .scene_assets import load_scene
else:
    from bridge import create_runtime
    from run import TIME_STEP
    from scene_assets import load_scene

ROOT = Path(__file__).resolve().parents[1]
NEAR = 0.05
FAR = 20.0
WIDTH, HEIGHT = 640, 480


def capture(x: float, y: float, yaw: float) -> tuple[Image.Image, np.ndarray]:
    eye = [x + 0.2 * math.cos(yaw), y + 0.2 * math.sin(yaw), 0.8]
    look = [eye[0] + math.cos(yaw), eye[1] + math.sin(yaw), eye[2]]
    view = p.computeViewMatrix(eye, look, [0, 0, 1])
    projection = p.computeProjectionMatrixFOV(65, WIDTH / HEIGHT, NEAR, FAR)
    _, _, rgba, depth, _ = p.getCameraImage(
        WIDTH,
        HEIGHT,
        viewMatrix=view,
        projectionMatrix=projection,
        renderer=p.ER_TINY_RENDERER,
    )
    rgb = Image.fromarray(np.asarray(rgba, dtype=np.uint8)).convert("RGB")
    buffer = np.asarray(depth)
    distance = FAR * NEAR / (FAR - (FAR - NEAR) * buffer)
    return rgb, distance


def decide(detections: list[dict], depth: np.ndarray, target: str) -> dict:
    candidates = [item for item in detections if item["class"] == target]
    if not candidates:
        return {"stride": 0, "angle": 0, "reason": "target_lost", "distance_m": None}
    best = max(candidates, key=lambda item: item["confidence"])
    x1, y1, x2, y2 = best["box"]
    cx, cy = (x1 + x2) / 2, y1 + (y2 - y1) * 0.4
    ix, iy = int(np.clip(cx, 2, WIDTH - 3)), int(np.clip(cy, 2, HEIGHT - 3))
    distance = float(np.median(depth[iy - 2 : iy + 3, ix - 2 : ix + 3]))
    if not math.isfinite(distance) or not NEAR < distance < FAR * 0.95:
        return {"stride": 0, "angle": 0, "reason": "invalid_depth", "distance_m": None}
    error = (cx / WIDTH - 0.5) * 2
    angle = int(np.clip(-error * 30, -30, 30))
    # Keep a stand-off that leaves the person visible in this fixed-height camera.
    reached = distance <= 1.5
    return {
        "stride": 0 if reached or abs(error) > 0.45 else 60,
        "angle": 0 if reached else angle,
        "reason": "reached" if reached else "approach",
        "distance_m": round(distance, 3),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=ROOT / "models/yolo26n.pt")
    parser.add_argument(
        "--assets", type=Path, default=ROOT / "simulator/artifacts/assets"
    )
    parser.add_argument(
        "--output", type=Path, default=ROOT / "simulator/artifacts/vision"
    )
    parser.add_argument("--frames", type=int, default=80)
    parser.add_argument("--target", default="person")
    parser.add_argument("--conf", type=float, default=0.35)
    parser.add_argument("--detect-only", action="store_true")
    parser.add_argument(
        "--hide-person",
        action="store_true",
        help="Remove person to verify lost-target stopping",
    )
    args = parser.parse_args()
    if not args.model.is_file():
        parser.error(
            "Model must be an existing local file; automatic download disabled"
        )
    if (
        not 1 <= args.frames <= 300
        or not math.isfinite(args.conf)
        or not 0 < args.conf <= 1
    ):
        parser.error("frames must be 1..300 and conf must be finite in (0, 1]")
    config_dir = ROOT / "simulator/artifacts/yolo-config"
    (config_dir / "Ultralytics").mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("YOLO_CONFIG_DIR", str(config_dir))
    os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "simulator/artifacts/matplotlib"))
    os.environ.setdefault("YOLO_AUTOINSTALL", "false")
    from ultralytics import YOLO

    model = YOLO(str(args.model), task="detect")
    if not args.detect_only and args.target not in model.names.values():
        parser.error(
            f"Target {args.target!r} absent from this model's labels; use --detect-only to inspect predictions"
        )
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "report.json").write_text(
        json.dumps({"completed": False, "reached": False}) + "\n"
    )
    runtime = create_runtime()
    client = None
    frames = []
    trace: list[dict] = []
    x = y = yaw = 0.0
    try:
        objects = load_scene(args.assets)
        if args.hide_person:
            p.removeBody(objects.pop("person"))
        client = socket.create_connection(("127.0.0.1", runtime.bridge.port), timeout=2)
        runtime.step()
        client.recv(8192)
        client.setblocking(False)
        for index in range(1 if args.detect_only else args.frames):
            rgb, depth = capture(x, y, yaw)
            result = model.predict(
                rgb, device="cpu", imgsz=640, conf=args.conf, verbose=False
            )[0]
            detections = [
                {
                    "class": result.names[int(box.cls.item())],
                    "confidence": float(box.conf.item()),
                    "box": box.xyxy[0].tolist(),
                }
                for box in result.boxes
            ]
            command = decide(detections, depth, args.target)
            if args.detect_only:
                command.update(stride=0, angle=0, reason="detect_only")
            client.sendall(
                (
                    json.dumps(
                        {
                            "t": "move",
                            "stride": command["stride"],
                            "angle": command["angle"],
                        }
                    )
                    + "\n"
                ).encode()
            )
            annotated = Image.fromarray(result.plot()[..., ::-1])
            frames.append(annotated)
            if index == 0:
                rgb.save(args.output / "camera.png")
                annotated.save(args.output / "first.png")
            trace.append(
                {
                    "frame": index,
                    "detections": detections,
                    "command": command,
                    "pose_xy_yaw": [x, y, yaw],
                }
            )
            print(
                json.dumps(
                    {
                        "frame": index,
                        "labels": [d["class"] for d in detections],
                        **command,
                    }
                ),
                flush=True,
            )
            for _ in range(48):
                runtime.step()
                # Explicit planar proxy; locomotion physics is not calibrated for MechDog.
                yaw += math.radians(runtime.hardware.angle) * TIME_STEP
                speed = runtime.hardware.stride / 100 * 0.6
                x += speed * math.cos(yaw) * TIME_STEP
                y += speed * math.sin(yaw) * TIME_STEP
                p.resetBasePositionAndOrientation(
                    runtime.hardware.robot,
                    [x, y, 0.2],
                    p.getQuaternionFromEuler([0, 0, yaw]),
                )
                p.resetBaseVelocity(runtime.hardware.robot, [0, 0, 0], [0, 0, 0])
            try:
                client.recv(65536)
            except BlockingIOError:
                pass
            if command["reason"] == "reached":
                break
        runtime.hal.stop()
        report = {
            "completed": True,
            "motion": "planar kinematic proxy with animated joints, not validated walking",
            "model": str(args.model),
            "model_sha256": hashlib.sha256(args.model.read_bytes()).hexdigest(),
            "firmware_sha256": runtime.firmware_sha256,
            "labels": model.names,
            "target": args.target,
            "confidence_threshold": args.conf,
            "detect_only": args.detect_only,
            "reached": any(step["command"]["reason"] == "reached" for step in trace),
            "final_pose_xy_yaw": [x, y, yaw],
            "scene_objects": list(objects),
            "trace": trace,
        }
        frames[-1].save(args.output / "last.png")
        frames[0].save(
            args.output / "approach.gif",
            save_all=True,
            append_images=frames[1:],
            duration=200,
            loop=0,
        )
        (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    finally:
        if client is not None:
            client.close()
        runtime.close()
    if not args.detect_only and not report["reached"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
