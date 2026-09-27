"""Measure the dog's real walking speed with its own sonar: face a wall 1-3 m away, then for each stride walk
backward for a fixed time and then forward again (so it ends where it started and the forward leg begins
farther from the wall), comparing the median sonar distance before and after each leg.

  python tools/measure_speed.py --serial COM4 [--strides 80 50] [--seconds 1.5]

Uses the gait signal_approach.py drives with (Hiwonder's default 150/200/25). Aborts before a forward leg if the
sonar has no echo (nothing within ~4 m) or the wall is closer than MIN_START_CM.
"""
import argparse
import json
import statistics
import threading
import time

import serial

GAIT = (150, 200, 25)
MIN_START_CM = 45
NO_ECHO_CM = 400


class Dog:
    def __init__(self, port):
        self.s = serial.Serial()
        self.s.port, self.s.baudrate, self.s.timeout = port, 115200, 0.2
        self.s.dtr = self.s.rts = False
        self.s.open()
        self.dist = []  # (t, cm)
        self.alive = True
        threading.Thread(target=self._read, daemon=True).start()

    def _read(self):
        buf = b""
        while self.alive:
            buf += self.s.read(self.s.in_waiting or 1)
            while b"\n" in buf:
                line, buf = buf.split(b"\n", 1)
                if line.startswith(b"{"):
                    try:
                        m = json.loads(line)
                    except ValueError:
                        continue
                    if m.get("t") == "tel" and isinstance(m.get("dist_cm"), (int, float)):
                        self.dist.append((time.monotonic(), float(m["dist_cm"])))

    def send(self, msg):
        self.s.write((json.dumps(msg) + "\n").encode())

    def distance(self, settle_s=1.0, window_s=0.8):
        """Median of valid readings over window_s, after settling."""
        time.sleep(settle_s)
        t0 = time.monotonic()
        time.sleep(window_s)
        vals = [d for t, d in self.dist if t >= t0 and d < NO_ECHO_CM]
        return statistics.median(vals) if len(vals) >= 3 else None

    def walk(self, stride, seconds):
        t_end = time.monotonic() + seconds
        while time.monotonic() < t_end:
            self.send({"t": "move", "stride": stride, "angle": 0})
            time.sleep(0.2)  # well inside the bridge's 1.5 s watchdog
        self.send({"t": "stop"})

    def close(self):
        self.send({"t": "stop"})
        self.send({"t": "sub", "hz": 0})
        time.sleep(0.2)
        self.alive = False
        self.s.close()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--serial", default="COM4")
    ap.add_argument("--strides", type=int, nargs="+", default=[80, 50])
    ap.add_argument("--seconds", type=float, default=1.5)
    args = ap.parse_args()

    dog = Dog(args.serial)
    try:
        dog.send({"t": "hello"})
        dog.send({"t": "stop"})
        dog.send({"t": "gait", "lift_ms": GAIT[0], "contact_ms": GAIT[1], "lift_mm": GAIT[2]})
        dog.send({"t": "sub", "hz": 10})
        results = []
        for stride in args.strides:
            start = dog.distance(settle_s=1.5)
            if start is None:
                print("abort: sonar has no echo; face the dog at a wall within ~3 m")
                break
            if start < MIN_START_CM:
                print(f"abort: wall at {start:.0f} cm is closer than {MIN_START_CM} cm")
                break
            dog.walk(-stride, args.seconds)
            mid = dog.distance()
            if mid is None:
                print(f"stride {stride}: lost the sonar echo after backing up")
                break
            dog.walk(stride, args.seconds)
            end = dog.distance()
            if end is None:
                print(f"stride {stride}: lost the sonar echo mid-test")
                continue
            back = (mid - start) / 100 / args.seconds
            fwd = (mid - end) / 100 / args.seconds
            theory = stride / GAIT[1]
            results.append((stride, fwd, back, theory))
            print(f"stride {stride:>3} mm: {start:.0f} -> {mid:.0f} -> {end:.0f} cm | forward {fwd:.2f} m/s, "
                  f"back {back:.2f} m/s | theory {theory:.2f} m/s -> {fwd / theory:.0%} of theory")
        if results:
            ratio = statistics.mean(r[1] / r[3] for r in results)
            print(f"measured forward speed averages {ratio:.0%} of stride/ground-time theory")
    finally:
        dog.close()


if __name__ == "__main__":
    main()
