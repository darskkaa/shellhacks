"""Minimal physical walk check for the MechDog bridge: reset stance, walk forward, log telemetry.

Run: nix-shell -p python3Packages.pyserial --run 'python3 tools/test_walk_forward.py --serial /dev/ttyUSB0'
Sonar distance falling while walking (facing a wall) and gyro activity are the movement evidence.
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from real_world_escort import BridgeLink  # noqa: E402

tel = []
acks = []


def on_msg(m):
    (tel if m.get("t") == "tel" else acks).append(m)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--serial", default="/dev/ttyUSB0")
    ap.add_argument("--stride", type=int, default=50)
    ap.add_argument("--gait", default="reset", help="'reset' (bridge default), 'none' (leave as is) or 'lift,contact,mm'")
    ap.add_argument("--action", type=int, help="action index to run before walking (15 = normal_attitude)")
    ap.add_argument("--seconds", type=float, default=4.0)
    ap.add_argument("--hb", type=float, default=0.5, help="heartbeat period while walking")
    a = ap.parse_args()

    link = BridgeLink(serial_port=a.serial, on_msg=on_msg, tag="Test")
    link.start()
    t0 = time.time()
    while link.hellos == 0 and time.time() - t0 < 5:
        time.sleep(0.1)
    if link.hellos == 0:
        print("no hello from bridge")
        return 1
    link.send({"t": "sub", "hz": 10})
    if a.gait == "reset":
        link.send({"t": "reset"})
    elif a.gait != "none":
        l, c, h = (int(x) for x in a.gait.split(","))
        link.send({"t": "gait", "lift_ms": l, "contact_ms": c, "lift_mm": h})
    if a.action is not None:
        link.send({"t": "action", "id": a.action})
        time.sleep(3.0)
    time.sleep(1.0)
    before = [m for m in tel if m.get("dist_cm") is not None]
    d0 = before[-1]["dist_cm"] if before else None
    print(f"baseline dist_cm={d0} batt={tel[-1].get('batt_v') if tel else None}")

    n0 = len(tel)
    link.send({"t": "move", "stride": a.stride, "angle": 0})
    t0 = time.time()
    while time.time() - t0 < a.seconds:
        time.sleep(a.hb)
        link.send({"t": "hb"})
    link.send({"t": "stop"})
    time.sleep(0.5)

    walk = tel[n0:]
    for m in walk:
        g = m.get("imu") or {}
        print(f"t={m.get('ts')} mode={m.get('mode')} stride={m.get('stride')} dist={m.get('dist_cm')} "
              f"ang={m.get('ang')} g=({g.get('gx')},{g.get('gy')},{g.get('gz')})")
    d1 = next((m["dist_cm"] for m in reversed(walk) if m.get("dist_cm") is not None), None)
    print(f"acks: {[ (x.get('for'), x.get('ok'), x.get('msg'), x.get('name'), x.get('reason')) for x in acks]}")
    print(f"dist_cm before={d0} after={d1} delta={None if None in (d0, d1) else round(d0 - d1, 1)}")
    modes = {m.get("mode") for m in walk}
    print(f"modes seen while walking: {modes}")
    link.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
