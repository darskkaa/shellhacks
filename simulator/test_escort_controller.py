"""Guard the crossing decision rule independently of the model and renderer."""

import unittest

from simulator.escort_demo import CONFIRM_FRAMES, GOAL_X, START_X, EscortController, vote


def det(name, conf):
    return {"class": name, "confidence": conf, "box": [0, 0, 1, 1]}


class EscortControllerTests(unittest.TestCase):
    def test_vote(self):
        cases = [
            ([det("ped_signal_walk", 0.47)], "walk"),
            ([det("ped_signal_walk", 0.35)], "unknown"),  # below threshold
            ([det("ped_signal_walk", 0.6), det("ped_signal_stop", 0.55)], "dont_walk"),  # ambiguous
            ([det("ped_signal_walk", 0.9), det("conflict_vehicle_cyclist", 0.6)], "dont_walk"),  # vehicle veto
            ([det("ped_signal_stop", 0.4)], "dont_walk"),
            ([det("staircase_steps", 0.9)], "unknown"),
            ([], "unknown"),
        ]
        for detections, expected in cases:
            with self.subTest(detections=detections):
                self.assertEqual(vote(detections)["vote"], expected)

    def test_requires_consecutive_walk_frames_then_commits(self):
        walk, stop = vote([det("ped_signal_walk", 0.5)]), vote([det("ped_signal_stop", 0.8)])
        ctl = EscortController()
        for _ in range(CONFIRM_FRAMES - 1):
            self.assertEqual(ctl.update(walk, START_X)["stride"], 0)
        self.assertEqual(ctl.update(stop, START_X)["stride"], 0)  # streak resets
        for _ in range(CONFIRM_FRAMES - 1):
            self.assertEqual(ctl.update(walk, START_X)["stride"], 0)
        self.assertGreater(ctl.update(walk, START_X)["stride"], 0)
        # Committed: a DON'T WALK mid-crossing (clearance interval) does not strand the pair in the road.
        self.assertGreater(ctl.update(stop, 5.0)["stride"], 0)
        result = ctl.update(stop, GOAL_X)
        self.assertEqual((result["state"], result["stride"]), ("arrived", 0))

    def test_never_moves_without_walk(self):
        ctl = EscortController()
        for detections in ([], [det("ped_signal_stop", 0.9)], [det("ped_signal_walk", 0.2)]):
            for _ in range(10):
                self.assertEqual(ctl.update(vote(detections), START_X)["stride"], 0)
