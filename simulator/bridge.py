"""Run the unchanged ESP32 Bridge with a local, uncalibrated PyBullet HAL."""

import argparse
import hashlib
import importlib.util
import math
import select
import socket
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pybullet as p
import pybullet_data

if __package__:
    from .run import TIME_STEP, load_robot
else:
    from run import TIME_STEP, load_robot


class SocketPoll:
    """MicroPython poll returns objects; CPython returns file descriptors."""

    def __init__(self) -> None:
        self.poller = select.poll()
        self.sockets: dict[int, socket.socket] = {}

    def register(self, sock: socket.socket, events: int) -> None:
        self.poller.register(sock, events)
        self.sockets[sock.fileno()] = sock

    def unregister(self, sock: socket.socket) -> None:
        self.poller.unregister(sock)
        self.sockets.pop(sock.fileno(), None)

    def poll(self, timeout: int = 0) -> list[tuple[socket.socket, int]]:
        return [
            (self.sockets[fd], events)
            for fd, events in self.poller.poll(timeout)
            if fd in self.sockets
        ]


class LoopbackSocket(socket.socket):
    def bind(self, address: Any) -> None:
        super().bind(("127.0.0.1", address[1]))


def load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_firmware(port: int) -> tuple[Any, str]:
    folder = Path(__file__).resolve().parents[1] / "robot_dump"
    config = load_module("simulator_config_example", folder / "config.example.py")
    config.PORT = port
    previous = sys.modules.get("config")
    sys.modules["config"] = config
    try:
        firmware = load_module("simulator_firmware", folder / "main.py")
    finally:
        if previous is None:
            sys.modules.pop("config", None)
        else:
            sys.modules["config"] = previous
    firmware.time = SimpleNamespace(
        ticks_ms=lambda: time.monotonic_ns() // 1_000_000,
        ticks_diff=lambda now, before: now - before,
        ticks_add=lambda now, delta: now + delta,
        sleep_ms=lambda milliseconds: time.sleep(milliseconds / 1000),
    )
    firmware.select = SimpleNamespace(poll=SocketPoll, POLLIN=select.POLLIN)
    firmware.socket = SimpleNamespace(
        socket=LoopbackSocket,
        AF_INET=socket.AF_INET,
        SOCK_STREAM=socket.SOCK_STREAM,
        SOL_SOCKET=socket.SOL_SOCKET,
        SO_REUSEADDR=socket.SO_REUSEADDR,
    )
    return firmware, hashlib.sha256((folder / "main.py").read_bytes()).hexdigest()


class BulletHAL:
    def __init__(self) -> None:
        self.robot, self.motors, self.directions = load_robot()
        self.stride = 0
        self.angle = 0
        self.elapsed = 0.0
        self.distance_cm_override: float | None = None
        self.battery_v_override: float | None = 7.4
        self.caps = {
            "move": True,
            "action": None,
            "gait": None,
            "height": None,
            "posture": None,
            "imu_raw": None,
            "imu_angle": True,
            "sonar": True,
            "rgb": False,
            "buzzer": False,
            "battery": "surrogate_constant_7.4V",
            "sonar_unit": "cm",
        }

    def move(self, stride: int, angle: int) -> None:
        self.stride = max(-100, min(100, int(stride)))
        self.angle = max(-30, min(30, int(angle)))

    def stop(self) -> None:
        self.move(0, 0)

    @staticmethod
    def unsupported(*args: Any) -> bool:
        return False

    action = unsupported
    gait = unsupported
    height = unsupported
    posture = unsupported
    # The unchanged firmware ACKs these commands unconditionally despite false caps.
    rgb = unsupported
    beep = unsupported

    def dist_cm(self) -> float | None:
        if self.distance_cm_override is not None:
            return self.distance_cm_override
        position, orientation = p.getBasePositionAndOrientation(self.robot)
        start, _ = p.multiplyTransforms(
            position, orientation, [0.25, 0, 0.03], [0, 0, 0, 1]
        )
        end, _ = p.multiplyTransforms(
            position, orientation, [3.25, 0, 0.03], [0, 0, 0, 1]
        )
        hit = p.rayTest(start, end)[0]
        return round(hit[2] * 300, 1) if hit[0] >= 0 else None

    def imu_read(self) -> tuple[None, list[float]]:
        _, orientation = p.getBasePositionAndOrientation(self.robot)
        return None, [
            math.degrees(value) for value in p.getEulerFromQuaternion(orientation)
        ]

    def battery_v(self) -> float | None:
        return self.battery_v_override

    def apply_motors(self) -> None:
        # These gentle joint excursions exercise commands, not calibrated walking.
        wave = math.sin(2 * math.pi * self.elapsed)
        targets = [
            sign
            * (
                math.pi / 2
                + 0.12
                * wave
                * max(-1, min(1, self.stride / 100 + sign * self.angle / 30))
            )
            for sign in self.directions
        ]
        p.setJointMotorControlArray(
            self.robot,
            self.motors,
            p.POSITION_CONTROL,
            targetPositions=targets,
            forces=[3.5] * 8,
            positionGains=[1.0] * 8,
            velocityGains=[1.0] * 8,
        )
        self.elapsed += TIME_STEP


class Runtime:
    def __init__(self, port: int, gui: bool) -> None:
        if p.isConnected():
            raise RuntimeError("Run one simulator runtime per process")
        self.firmware, self.firmware_sha256 = load_firmware(port)
        self.client = p.connect(p.GUI if gui else p.DIRECT)
        if self.client < 0:
            raise RuntimeError("Could not connect to PyBullet")
        try:
            p.setAdditionalSearchPath(pybullet_data.getDataPath())
            p.setGravity(0, 0, -9.81)
            p.setTimeStep(TIME_STEP)
            p.setPhysicsEngineParameter(numSolverIterations=100)
            p.loadURDF("plane.urdf")
            self.hal = BulletHAL()
            self.bridge = self.firmware.Bridge(self.hal)
            p.resetDebugVisualizerCamera(1.1, 45, -25, [0, 0, 0.15])
        except BaseException:
            p.disconnect(self.client)
            raise
        self.closed = False

    def step(self) -> None:
        if self.closed:
            raise RuntimeError("Simulator runtime is closed")
        for sock, event in self.bridge.poller.poll(0):
            if sock is self.bridge.srv:
                self.bridge.accept()
            elif event & select.POLLIN:
                self.bridge.read(sock)
            else:
                self.bridge.drop(sock)
        self.bridge.tick()
        self.hal.apply_motors()
        p.stepSimulation()

    def close(self) -> None:
        if self.closed:
            return
        self.bridge.running = False
        self.hal.stop()
        for client in list(self.bridge.clients):
            self.bridge.drop(client)
        self.bridge.poller.unregister(self.bridge.srv)
        self.bridge.srv.close()
        p.disconnect(self.client)
        self.closed = True


def create_runtime(port: int = 0, gui: bool = False) -> Runtime:
    if not 0 <= port <= 65535:
        raise ValueError("port must be between 0 and 65535")
    return Runtime(port, gui)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=5005)
    parser.add_argument("--gui", action="store_true")
    args = parser.parse_args()
    if not 0 <= args.port <= 65535:
        parser.error("--port must be between 0 and 65535")
    runtime = create_runtime(args.port, args.gui)
    print(f"Bridge listening on 127.0.0.1:{runtime.bridge.port}", flush=True)
    print(
        "Uncalibrated Minitaur surrogate: motor exercise only; battery fixed at 7.4 V."
    )
    print(
        "Height/gait/action/posture unsupported. RGB/buzzer no-op; firmware still ACKs them."
    )
    print(f"Original firmware SHA256: {runtime.firmware_sha256}", flush=True)
    try:
        while runtime.bridge.running:
            start = time.monotonic()
            runtime.step()
            time.sleep(max(0, TIME_STEP - (time.monotonic() - start)))
    except KeyboardInterrupt:
        pass
    finally:
        runtime.close()


if __name__ == "__main__":
    main()
