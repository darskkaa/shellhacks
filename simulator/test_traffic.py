"""Exercise actual PyBullet traffic bodies and signal transitions."""

import unittest

import pybullet as p

from simulator.traffic_demo import ROOT, TrafficScene, phase_at


class TrafficTests(unittest.TestCase):
    def test_phases(self):
        for seconds, expected in [
            (0, "red"),
            (5.9, "red"),
            (6, "green"),
            (12, "amber"),
            (14, "red"),
        ]:
            self.assertEqual(phase_at(seconds), expected)

    def test_queue_clearance_and_recycling(self):
        client = p.connect(p.DIRECT)
        try:
            scene = TrafficScene(ROOT / "simulator/artifacts/assets")
            scene.positions = [-3, -7.5]
            scene.update(0.1)
            self.assertEqual(scene.positions, [-3, -7.5])
            self.assertTrue(scene.walk)
            scene.elapsed = 6
            scene.update(0.1)
            self.assertGreater(scene.positions[0], -3)
            self.assertFalse(scene.walk)
            scene.elapsed = 12
            scene.positions = [-1, -7.5]
            scene.update(0.1)
            self.assertGreater(scene.positions[0], -1)
            scene.elapsed = 14
            scene.update(0.1)
            self.assertFalse(scene.walk)  # Red alone is insufficient: car in crossing.
            scene.positions = [12.1, 7.6]
            scene.update(0)
            self.assertLess(scene.positions[0], -3)
            self.assertEqual(
                p.getBasePositionAndOrientation(scene.cars[0])[0][1], scene.positions[0]
            )
            with self.assertRaises(ValueError):
                scene.update(float("nan"))
        finally:
            p.disconnect(client)
