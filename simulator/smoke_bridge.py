"""Exercise the repo's actual Bridge over loopback TCP with a PyBullet adapter."""

import importlib.util
import json
import socket
import sys
import threading
import time
from pathlib import Path

import pybullet as p

from bridge import create_runtime
from run import TIME_STEP


def main() -> None:
    runtime = create_runtime(port=0)
    client = socket.create_connection(("127.0.0.1", runtime.bridge.port), timeout=2)
    client.setblocking(False)
    buffer = b""
    messages: list[dict] = []
    checks: dict[str, bool] = {}
    panel_dogs: list = []

    def pump(seconds: float) -> None:
        nonlocal buffer
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            runtime.step()
            if client.fileno() >= 0:
                try:
                    chunk = client.recv(65536)
                except BlockingIOError:
                    chunk = b""
                buffer += chunk
                while b"\n" in buffer:
                    line, buffer = buffer.split(b"\n", 1)
                    messages.append(json.loads(line))
            time.sleep(TIME_STEP)

    def send(message: dict) -> None:
        client.sendall((json.dumps(message) + "\n").encode())

    def expect(name: str, condition: bool) -> None:
        checks[name] = bool(condition)
        if not condition:
            raise AssertionError(
                f"{name}: mode={runtime.bridge.mode}, recent={messages[-5:]}"
            )

    def received(kind: str, **fields: object) -> bool:
        return any(
            message.get("t") == kind
            and all(message.get(key) == value for key, value in fields.items())
            for message in messages
        )

    try:
        pump(0.15)
        expect("hello", received("hello", proto=1, name="MechDog"))
        send({"t": "sub", "hz": 20})
        pump(0.2)
        expect("telemetry", received("tel") and received("ack", hz=20, ok=True))

        client.sendall(b"{bad json}\n")
        pump(0.05)
        expect("bad_json_rejected", received("err", msg="bad json"))
        send({"t": "move", "stride": 1000, "angle": -200})
        pump(0.05)
        expect(
            "command_clamping",
            received("ack", clamped=True, ok=True)
            and (runtime.hal.stride, runtime.hal.angle) == (100, -30),
        )

        before = [s[0] for s in p.getJointStates(runtime.hal.robot, runtime.hal.motors)]
        send({"t": "move", "stride": 60, "angle": 0})
        pump(0.25)
        after = [s[0] for s in p.getJointStates(runtime.hal.robot, runtime.hal.motors)]
        expect(
            "move_reaches_physics",
            max(abs(a - b) for a, b in zip(after, before)) > 0.01,
        )
        for _ in range(6):
            send({"t": "hb"})
            pump(0.3)
        expect("heartbeat_keeps_command", runtime.bridge.mode == "walk")
        send({"t": "stop"})
        pump(0.05)
        expect("stop", runtime.hal.stride == runtime.hal.angle == 0)

        messages.clear()
        send({"t": "move", "stride": 60, "angle": 0})
        pump(1.8)
        expect(
            "watchdog_stop",
            received("event", name="watchdog_stop", reason="watchdog")
            and runtime.hal.stride == runtime.hal.angle == 0,
        )

        messages.clear()
        position, orientation = p.getBasePositionAndOrientation(runtime.hal.robot)
        obstacle_position, _ = p.multiplyTransforms(
            position, orientation, [0.4, 0, 0.03], [0, 0, 0, 1]
        )
        shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.05, 0.2, 0.2])
        obstacle = p.createMultiBody(
            baseMass=0, baseCollisionShapeIndex=shape, basePosition=obstacle_position
        )
        distance = runtime.hal.dist_cm()
        expect("physics_sonar", distance is not None and 0 < distance < 20)
        send({"t": "move", "stride": 60, "angle": 0})
        pump(0.5)
        expect(
            "obstacle_stop",
            received("event", name="obstacle_stop") and runtime.hal.stride == 0,
        )
        p.removeBody(obstacle)
        runtime.hal.distance_cm_override = 200.0
        runtime.hal.battery_v_override = 6.0
        pump(0.4)
        send({"t": "move", "stride": 60, "angle": 0})
        pump(0.1)
        expect("low_battery_rejects_move", received("ack", ok=False, msg="low_battery"))

        messages.clear()
        runtime.hal.battery_v_override = 7.4
        position, _ = p.getBasePositionAndOrientation(runtime.hal.robot)
        p.resetBasePositionAndOrientation(
            runtime.hal.robot, position, p.getQuaternionFromEuler([1.2, 0, 0])
        )
        runtime.bridge.poll_sensors()
        pump(0.02)
        send({"t": "move", "stride": 60, "angle": 0})
        pump(0.02)
        expect(
            "fall_rejects_move",
            received("event", name="fall") and received("ack", ok=False, msg="fallen"),
        )

        p.resetBasePositionAndOrientation(runtime.hal.robot, [0, 0, 0.2], [0, 0, 0, 1])
        p.resetBaseVelocity(runtime.hal.robot, [0, 0, 0], [0, 0, 0])
        runtime.bridge.poll_sensors()
        pump(0.15)
        expect("upright_event", received("event", name="upright"))
        send({"t": "move", "stride": 60, "angle": 0})
        pump(0.02)
        expect("walking_before_disconnect", runtime.bridge.mode == "walk")
        client.close()
        pump(0.1)
        expect(
            "disconnect_stops",
            runtime.bridge.mode == "safe" and runtime.hal.stride == 0,
        )

        panel_path = Path(__file__).resolve().parents[1] / "tools" / "dog_panel.py"
        spec = importlib.util.spec_from_file_location("tested_dog_panel", panel_path)
        if spec is None or spec.loader is None:
            raise RuntimeError("Cannot import existing control panel")
        panel = importlib.util.module_from_spec(spec)
        original_args = sys.argv
        try:
            # Import classes without running main(), scanning WiFi or opening USB.
            sys.argv = [str(panel_path), "--wifi", "127.0.0.1", "--wifi-only"]
            spec.loader.exec_module(panel)
        finally:
            sys.argv = original_args
        vars(panel)["BRIDGE_PORT"] = runtime.bridge.port
        errors: list[Exception] = []

        def connect_panel() -> None:
            try:
                panel_dogs.append(panel.WifiDog("127.0.0.1"))
            except (OSError, ValueError) as error:
                errors.append(error)

        connector = threading.Thread(target=connect_panel, daemon=True)
        connector.start()
        pump(0.5)
        connector.join(timeout=0.1)
        expect("panel_connects", len(panel_dogs) == 1 and not errors)
        dog = panel_dogs[0]
        expect("panel_reads_telemetry", dog.sensors() == (200.0, 7.4))
        dog.move(40, 10)
        pump(0.1)
        expect("panel_move", (runtime.hal.stride, runtime.hal.angle) == (40, 10))
        dog.stop()
        pump(0.1)
        expect("panel_stop", runtime.hal.stride == runtime.hal.angle == 0)
    finally:
        client.close()
        for dog in panel_dogs:
            dog.close()
        runtime.close()
        report = {
            "scope": "Unmodified Bridge logic on CPython with surrogate hardware; not ESP32 firmware emulation",
            "firmware_sha256": runtime.firmware_sha256,
            "checks": checks,
            "passed": len(checks) == 19 and all(checks.values()),
        }
        output = Path(__file__).parent / "artifacts" / "bridge-report.json"
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
