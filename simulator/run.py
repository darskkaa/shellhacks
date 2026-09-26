"""Exercise PyBullet's eight-motor Minitaur surrogate; no hardware connection."""

import argparse
import json
import math
import time
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pybullet as p
import pybullet_data
from PIL import Image

TIME_STEP = 1 / 240
LEGS = ("front_left", "back_left", "front_right", "back_right")


def load_robot() -> tuple[int, list[int], list[int]]:
    robot = p.loadURDF("quadruped/minitaur.urdf", [0, 0, 0.2])
    joints = {
        p.getJointInfo(robot, index)[1].decode(): index
        for index in range(p.getNumJoints(robot))
    }
    motors, directions = [], []
    for leg_index, leg in enumerate(LEGS):
        direction = -1 if leg_index < 2 else 1
        for side in ("L", "R"):
            motor = joints[f"motor_{leg}{side}_joint"]
            knee = joints[f"knee_{leg}{side}_link"]
            motors.append(motor)
            directions.append(direction)
            p.resetJointState(robot, motor, direction * math.pi / 2)
            p.resetJointState(robot, knee, direction * -2.1834)
            p.setJointMotorControl2(robot, knee, p.VELOCITY_CONTROL, force=0)
        # The URDF is a tree; these four constraints close its physical leg loops.
        p.createConstraint(
            robot,
            joints[f"knee_{leg}R_link"],
            robot,
            joints[f"knee_{leg}L_link"],
            p.JOINT_POINT2POINT,
            [0, 0, 0],
            [0, 0.005, 0.2],
            [0, 0.01, 0.2],
        )
    return robot, motors, directions


def run(gui: bool, seconds: float, output: Path) -> dict:
    client = p.connect(p.GUI if gui else p.DIRECT)
    if client < 0:
        raise RuntimeError("Could not connect to PyBullet")
    try:
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        p.setTimeStep(TIME_STEP)
        p.setPhysicsEngineParameter(numSolverIterations=100)
        ground = p.loadURDF("plane.urdf")
        robot, motors, directions = load_robot()
        p.resetDebugVisualizerCamera(1.1, 45, -25, [0, 0, 0.15])
        positions, angles = [], []
        contact_steps = 0
        steps = round(seconds / TIME_STEP)
        for step in range(steps):
            elapsed = step * TIME_STEP
            # Gentle symmetric extension verifies actuation without claiming a gait.
            offset = (
                0.12 * math.sin(2 * math.pi * (elapsed - 1))
                if 1 <= elapsed < seconds - 1
                else 0.0
            )
            p.setJointMotorControlArray(
                robot,
                motors,
                p.POSITION_CONTROL,
                targetPositions=[sign * (math.pi / 2 + offset) for sign in directions],
                forces=[3.5] * 8,
                positionGains=[1.0] * 8,
                velocityGains=[1.0] * 8,
            )
            p.stepSimulation()
            position, orientation = p.getBasePositionAndOrientation(robot)
            joint_angles = [state[0] for state in p.getJointStates(robot, motors)]
            if not all(
                math.isfinite(value)
                for value in (*position, *orientation, *joint_angles)
            ):
                raise RuntimeError(f"Non-finite physics state at step {step}")
            positions.append(position)
            angles.append(joint_angles)
            contact_steps += bool(p.getContactPoints(robot, ground))
            if gui:
                time.sleep(TIME_STEP)

        position_array = np.asarray(positions)
        # Exclude the initial settling interval from the motor-motion measurement.
        excursion = np.ptp(np.asarray(angles)[240:-240], axis=0)
        roll, pitch, _ = p.getEulerFromQuaternion(orientation)
        view = p.computeViewMatrixFromYawPitchRoll(position, 1.1, 45, -25, 0, 2)
        projection = p.computeProjectionMatrixFOV(55, 640 / 480, 0.01, 10)
        _, _, rgba, _, segmentation = p.getCameraImage(
            640,
            480,
            viewMatrix=view,
            projectionMatrix=projection,
            renderer=p.ER_TINY_RENDERER,
        )
        robot_pixels = int(
            np.count_nonzero((np.asarray(segmentation) & 0xFFFFFF) == robot)
        )
        checks = {
            "eight_motors": len(motors) == 8,
            "four_leg_constraints": p.getNumConstraints() == 4,
            "ground_contact": contact_steps > steps // 2,
            "all_motors_moved": bool(np.all(excursion > 0.05)),
            "body_above_ground": bool(np.all(position_array[:, 2] > 0.08)),
            "upright_at_end": abs(roll) < 0.3 and abs(pitch) < 0.3,
            "robot_rendered": robot_pixels > 100,
        }
        report = {
            "model": "PyBullet Minitaur (MechDog surrogate, uncalibrated)",
            "pybullet_version": version("pybullet"),
            "connection": "GUI" if gui else "DIRECT",
            "steps": steps,
            "simulated_seconds": steps * TIME_STEP,
            "contact_steps": contact_steps,
            "motor_excursion_rad": excursion.tolist(),
            "final_position_m": list(position),
            "final_roll_pitch_rad": [roll, pitch],
            "robot_pixels": robot_pixels,
            "checks": checks,
            "passed": all(checks.values()),
        }
        output.mkdir(parents=True, exist_ok=True)
        Image.fromarray(np.asarray(rgba, dtype=np.uint8)).save(output / "minitaur.png")
        (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        return report
    finally:
        p.disconnect(client)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--gui", action="store_true", help="Open a 3D window; requires a display"
    )
    parser.add_argument(
        "--seconds", type=float, default=10, help="Simulation duration, 4–300 s"
    )
    parser.add_argument(
        "--output", type=Path, default=Path(__file__).parent / "artifacts"
    )
    args = parser.parse_args()
    if not math.isfinite(args.seconds) or not 4 <= args.seconds <= 300:
        parser.error("--seconds must be finite and between 4 and 300")
    report = run(args.gui, args.seconds, args.output)
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
