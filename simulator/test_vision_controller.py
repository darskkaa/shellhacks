"""Guard controller stopping and steering independently of model detections."""

import unittest

import numpy as np

from simulator.vision_demo import HEIGHT, WIDTH, decide


class VisionControllerTests(unittest.TestCase):
    def test_depth_and_loss(self):
        detections = [
            {"class": "person", "confidence": 0.9, "box": [270, 100, 370, 400]}
        ]
        for distance, reason, stride in [
            (3.0, "approach", 60),
            (1.0, "reached", 0),
            (float("nan"), "invalid_depth", 0),
            (-1.0, "invalid_depth", 0),
            (20.0, "invalid_depth", 0),
        ]:
            with self.subTest(distance=distance):
                result = decide(
                    detections, np.full((HEIGHT, WIDTH), distance), "person"
                )
                self.assertEqual(
                    (result["reason"], result["stride"], result["angle"]),
                    (reason, stride, 0),
                )
        result = decide([], np.ones((HEIGHT, WIDTH)), "person")
        self.assertEqual(
            (result["reason"], result["stride"], result["angle"]), ("target_lost", 0, 0)
        )

    def test_turn_before_advancing(self):
        for box, sign in [([0, 100, 100, 400], 1), ([540, 100, 640, 400], -1)]:
            result = decide(
                [{"class": "person", "confidence": 0.9, "box": box}],
                np.full((HEIGHT, WIDTH), 3.0),
                "person",
            )
            self.assertEqual(result["stride"], 0)
            self.assertGreater(result["angle"] * sign, 0)
