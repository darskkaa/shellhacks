"""Walk the dog right up to the nearest detected person.

The camera (tools/vision_stream.py, http://localhost:8001/detections) is used
for steering: it keeps the person centered in frame. The dog's own onboard
sonar (read back from tools/dog_panel.py's /status as dist_cm) decides when
it has arrived, since it measures true distance instead of frame size. Both
of those servers must already be running.

Run:  python tools/follow_person.py
"""
import argparse
import json
import time
import urllib.request

ap = argparse.ArgumentParser()
# 127.0.0.1, not localhost: on Windows "localhost" tries IPv6 first and adds ~200ms to every request.
ap.add_argument("--vision", default="http://127.0.0.1:8001")
ap.add_argument("--panel", default="http://127.0.0.1:8000")
ap.add_argument("--stride", type=int, default=90, help="forward speed while approaching")
ap.add_argument("--turn-gain", type=float, default=90.0)
ap.add_argument("--stop-cm", type=float, default=22.0, help="sonar distance to stop at (right up to them)")
ap.add_argument("--slow-cm", type=float, default=35.0, help="start easing off stride below this sonar distance")
ap.add_argument("--min-stride", type=int, default=45, help="slowest forward stride while easing in")
ap.add_argument("--close-area", type=float, default=0.92, help="fallback stop if the person fills this much of "
                 "the frame (wide-angle lens fills most of the frame from over half a meter out, so this is a "
                 "last-resort safety, not the real stop signal - the sonar decides)")
ap.add_argument("--min-conf", type=float, default=0.4)
ap.add_argument("--lost-after", type=float, default=1.0, help="seconds without a person before stopping")
ap.add_argument("--smoothing", type=float, default=0.5, help="EMA weight on each new cx reading (1 = no smoothing)")
ap.add_argument("--hz", type=float, default=10.0)
args = ap.parse_args()

# The panel stops the walk if it goes 0.7s without a move command (tools/dog_panel.py MOVE_TIMEOUT_S).
# Our loop must keep resending well inside that even on frames where YOLO briefly misses the person.


def get_json(url):
    with urllib.request.urlopen(url, timeout=1.5) as r:
        return json.loads(r.read())


def post_json(url, obj):
    data = json.dumps(obj).encode()
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=1.5) as r:
        return json.loads(r.read())


def best_person(dets):
    people = [d for d in dets if d["label"] == "person" and d["conf"] >= args.min_conf]
    return max(people, key=lambda d: d["area"]) if people else None


def main():
    period = 1.0 / args.hz
    last_seen = 0.0
    held_cx = None       # EMA-smoothed heading, carried across frames where the detector briefly misses
    held_area = None
    stopped = True
    print("Following: watching %s/detections, driving %s (Ctrl-C to stop)" % (args.vision, args.panel), flush=True)
    while True:
        t0 = time.time()
        try:
            dets = get_json(args.vision + "/detections")["dets"]
        except Exception:
            dets = []
        person = best_person(dets)
        try:
            dist_cm = get_json(args.panel + "/status").get("dist_cm")
        except Exception:
            dist_cm = None

        now = time.time()
        if person is not None:
            last_seen = now
            held_area = person["area"]
            held_cx = person["cx"] if held_cx is None else (
                args.smoothing * person["cx"] + (1 - args.smoothing) * held_cx)
        have_target = held_cx is not None and (now - last_seen) <= args.lost_after

        try:
            if not have_target:
                if not stopped:
                    post_json(args.panel, {"t": "stop"})
                    stopped = True
                    print("no person in view - stopped", flush=True)
                held_cx = held_area = None
            else:
                arrived = (dist_cm is not None and dist_cm <= args.stop_cm) or (
                    person is not None and held_area >= args.close_area)
                if arrived:
                    if not stopped:
                        post_json(args.panel, {"t": "stop"})
                        stopped = True
                    print("reached person (dist=%s cm, area=%.2f)" % (dist_cm, held_area), flush=True)
                else:
                    angle = max(-30, min(30, round((0.5 - held_cx) * args.turn_gain)))
                    stride = args.stride
                    if abs(held_cx - 0.5) >= 0.35:
                        stride = max(args.min_stride, stride // 2)
                    if dist_cm is not None and dist_cm < args.slow_cm:
                        frac = max(0.0, (dist_cm - args.stop_cm) / (args.slow_cm - args.stop_cm))
                        stride = max(args.min_stride, round(stride * frac))
                    # Coarse steps: over USB every changed value is a fresh d.move() REPL call that can hitch the gait.
                    angle = 5 * round(angle / 5)
                    stride = 10 * round(stride / 10)
                    post_json(args.panel, {"t": "move", "stride": stride, "angle": angle})
                    stopped = False
                    tag = "seen" if person is not None else "held"
                    print("%s cx=%.2f area=%.3f dist=%s -> stride=%d angle=%d" %
                          (tag, held_cx, held_area or 0, dist_cm, stride, angle), flush=True)
        except Exception as e:
            print("command failed: %s" % e, flush=True)
        dt = time.time() - t0
        if dt < period:
            time.sleep(period - dt)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        try:
            post_json(args.panel, {"t": "stop"})
        except Exception:
            pass
