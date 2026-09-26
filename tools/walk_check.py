"""Point the webcam at a pedestrian signal and report whether it is safe to walk.

Uses the blind-escort ONNX model and the same decision rule as
simulator/escort_demo.py: a frame votes WALK when ped_signal_walk >= --walk-conf,
beats ped_signal_stop by 0.1, and no conflict_vehicle_cyclist >= 0.5. Three
WALK votes -> READY TO WALK.

Accuracy: frames are letterboxed (not squashed) before the model, and a second
zoomed pass runs on a crop of the full-res frame: the centre until a signal is
found, then a box that follows the signal.
Resilience: up to --grace frames with no signal don't reset the WALK streak, and
once WALK is confirmed it is held for --latch-seconds even if the signal leaves
view (assume the light stays). A real DON'T WALK still cancels it before commit.

Without --drive this never moves the dog. With --drive it sends the crossing
through tools/dog_panel.py (must already be running): wait at the curb, then on
READY TO WALK walk straight at full stride until the first end condition:
  1. END clicked on the live view page (POST /end)
  2. sonar reads <= --stop-cm (something right in front at the far side)
  3. the model sees the far curb (curb ramp / step up / drop-off) filling the
     bottom of the frame for --curb-frames frames (camera on the dog, facing forward)
  4. --cross-seconds safety cap
Once in the road it does not stop for the signal changing (stopping mid-street
is the unsafe choice, see simulator/ESCORT.md); the panel's own sonar obstacle
stop still applies.

Run:  python tools/walk_check.py --cam 1            # live, Ctrl-C to quit
      python tools/walk_check.py --cam 1 --drive    # live + drive the dog across
      python tools/walk_check.py --image frame.jpg  # one still image
Live view (camera + boxes + decision): http://127.0.0.1:8002
"""
import argparse
import io
import json
import sys
import threading
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import cv2
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from blind_escort_yolo.inference import BlindEscortDetector  # noqa: E402

MARGIN = 0.1
CONFLICT_CONF = 0.5
CONFIRM_FRAMES = 3
TARGET_TTL = 2.0  # seconds the zoom keeps following a signal after it was last seen
# Only these classes feed vote(); everything else is drawn only with --show-all.
DECISION_CLASSES = {"ped_signal_walk": "lime", "ped_signal_stop": "orange", "conflict_vehicle_cyclist": "red"}
# Seeing one of these filling the bottom of the frame while crossing means the far curb is right in front.
CURB_CLASSES = {"curb_ramp_ada": "yellow", "curb_step_up": "yellow", "curb_drop_off_hazard": "yellow"}
END = threading.Event()  # set by the END button on the live view


def vote(detections, walk_conf):
    best = lambda name: max((d["confidence"] for d in detections if d["class"] == name), default=0.0)
    walk, stop, conflict = best("ped_signal_walk"), best("ped_signal_stop"), best("conflict_vehicle_cyclist")
    if walk >= walk_conf and walk >= stop + MARGIN and conflict < CONFLICT_CONF:
        label = "walk"
    elif stop > 0 or conflict >= CONFLICT_CONF:
        label = "dont_walk"
    else:
        label = "unknown"
    return label, walk, stop, conflict


def post(panel, msg):
    req = urllib.request.Request(panel + "/cmd", data=json.dumps(msg).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=1.5) as r:
        return json.loads(r.read())


PAGE = """<!doctype html><html><head><meta charset="utf-8"><title>Walk Check</title>
<style>body{margin:0;background:#111;color:#eee;font:16px system-ui;text-align:center}
#s{font:700 48px system-ui;padding:14px}#h{font:600 18px system-ui;color:#ffd166;min-height:24px}
#d{font:14px monospace;color:#aaa;padding:6px}img{max-width:100%;border:4px solid #333}</style></head><body>
<div id="s">...</div><div id="h"></div>
<button id="e" onclick="fetch('/end',{method:'POST'})" style="font:700 28px system-ui;padding:12px 40px;margin:8px;background:#c0392b;color:#fff;border:0;border-radius:10px;cursor:pointer">END</button><br><img id="v" src="/stream"><div id="d"></div>
<script>const C={"WAIT":"#e67e22","READY TO WALK":"#2ecc71","CROSSING":"#3498db","ARRIVED":"#9b59b6"};
// An MJPEG <img> never reconnects on its own after the server restarts, so reopen it when the server comes back.
const v=document.getElementById("v");let down=false;const reopen=()=>{v.src="/stream?"+Date.now()};
v.onerror=()=>{down=true;setTimeout(reopen,1000)};
setInterval(async()=>{const s=document.getElementById("s");try{const j=await (await fetch("/state")).json();
if(down){down=false;reopen()}
s.textContent=j.state||"...";s.style.background=C[j.state]||"#333";
document.getElementById("h").textContent=j.held?"HELD: signal out of view, assuming the light stays":"";
document.getElementById("d").textContent=`vote=${j.vote} walk=${j.walk} stop=${j.stop} conflict=${j.conflict} streak=${j.streak} miss=${j.miss} ${j.ms}ms drive=${j.drive}${j.why?' | '+j.why:''}`}
catch(e){down=true;s.textContent="OFFLINE";s.style.background="#333"}},200)</script>
</body></html>"""
LIVE = {"jpeg": b"", "state": {}}
LIVE_LOCK = threading.Condition()


class Web(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/state":
            data = json.dumps(LIVE["state"]).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")  # the dog panel page on :8000 polls this
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(data)
        elif self.path.startswith("/stream"):
            self.send_response(200)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=f")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            try:
                while True:
                    with LIVE_LOCK:
                        LIVE_LOCK.wait(2)
                        jpg = LIVE["jpeg"]
                    if jpg:
                        self.wfile.write(b"--f\r\nContent-Type: image/jpeg\r\nContent-Length: %d\r\n\r\n" % len(jpg))
                        self.wfile.write(jpg + b"\r\n")
                        self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                pass
        else:
            data = PAGE.encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(data)

    def do_POST(self):
        if self.path == "/end":
            END.set()
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Content-Length", "0")
        self.end_headers()

    def log_message(self, *a):
        pass


def publish(img, state):
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=75)
    with LIVE_LOCK:
        LIVE["jpeg"], LIVE["state"] = buf.getvalue(), state
        LIVE_LOCK.notify_all()


def letterbox(img):
    """Pad to a square (bottom/right) so the model's square resize doesn't squash the frame.

    inference.py stretches whatever it gets to imgsz x imgsz; a 16:9 frame stretched to 1:1 distorts the signal
    face. Padding only bottom/right keeps box coordinates identical to the unpadded image.
    """
    side = max(img.size)
    if img.size == (side, side):
        return img
    sq = Image.new("RGB", (side, side), (114, 114, 114))
    sq.paste(img, (0, 0))
    return sq


def iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def zoom_region(size, target, zoom):
    """Square crop: around the last seen signal if we have one, else the centre 1/zoom of the frame."""
    W, H = size
    if target is not None:
        x1, y1, x2, y2 = target
        side = max(x2 - x1, y2 - y1) * 4  # keep context around the head so the model sees a signal, not a blob
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    else:
        side = min(W, H) / zoom
        cx, cy = W / 2, H / 2
    side = int(max(256, min(side, min(W, H))))
    x0 = int(min(max(cx - side / 2, 0), W - side))
    y0 = int(min(max(cy - side / 2, 0), H - side))
    return x0, y0, x0 + side, y0 + side


def detect(det, img, args, target):
    """Full-frame pass plus a zoomed pass; the crop gives a small, distant signal several times more pixels."""
    t0 = time.time()
    dets = det.predict(letterbox(img), conf_thresh=args.det_conf, imgsz=args.imgsz)["detections"]
    region = None
    if args.zoom > 1:
        region = zoom_region(img.size, target, args.zoom)
        x0, y0 = region[0], region[1]
        for d in det.predict(img.crop(region), conf_thresh=args.det_conf, imgsz=args.imgsz)["detections"]:
            b = d["box"]
            # An object cut by the crop edge gives a clipped box glued to that edge; the full-frame pass covers it.
            W, H, side, m = img.width, img.height, region[2] - region[0], 4
            if ((b[0] < m and x0 > 0) or (b[1] < m and y0 > 0) or
                    (b[2] > side - m and region[2] < W) or (b[3] > side - m and region[3] < H)):
                continue
            d = dict(d, box=[b[0] + x0, b[1] + y0, b[2] + x0, b[3] + y0], zoom=True)
            same = [i for i, e in enumerate(dets) if e["class"] == d["class"] and iou(e["box"], d["box"]) > 0.4]
            if not same:
                dets.append(d)
            elif d["confidence"] > dets[same[0]]["confidence"]:
                dets[same[0]] = d
    return dets, region, (time.time() - t0) * 1000


def signal_box(dets):
    sig = [d for d in dets if d["class"] in ("ped_signal_walk", "ped_signal_stop")]
    return max(sig, key=lambda d: d["confidence"])["box"] if sig else None


def _font(size):
    for name in ("arialbd.ttf", "DejaVuSans-Bold.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    return ImageFont.load_default()


FONT, FONT_BIG = _font(15), _font(17)


def annotate(img, detections, text, path, show_all=False, region=None, walk_conf=0.4, curb_conf=0.4):
    # Shrink first, then draw at display resolution so labels stay readable (drawing on 1080p and shrinking
    # afterwards turned them into ~5 px smudges).
    k = 1.0
    if img.width > 960:  # keep the web stream light; detection already ran on the full-res frame
        k = 960 / img.width
        img = img.resize((960, round(img.height * k)))
    d = ImageDraw.Draw(img)
    sc = lambda b: [b[0] * k, b[1] * k, b[2] * k, b[3] * k]
    if region is not None:
        r = sc(region)
        d.rectangle(r, outline="cyan", width=2)
        d.text((r[0] + 4, r[3] - 20), "zoom", fill="cyan", font=FONT)
    for det in detections:
        cls, conf = det["class"], det["confidence"]
        if cls in DECISION_CLASSES:
            color = DECISION_CLASSES[cls]
            width = 4 if conf >= walk_conf else 1  # thin = seen but too weak to count
        elif cls in CURB_CLASSES and conf >= curb_conf:
            color, width = CURB_CLASSES[cls], 3
        elif show_all:
            color, width = "gray", 1
        else:
            continue
        b = sc(det["box"])
        d.rectangle(b, outline=color, width=width)
        label = f"{cls} {conf:.2f}"
        tw = d.textlength(label, font=FONT)
        ty = b[1] - 19 if b[1] >= 19 else b[1]
        d.rectangle([b[0], ty, b[0] + tw + 6, ty + 18], fill=color)
        d.text((b[0] + 3, ty + 1), label, fill="black", font=FONT)
    d.rectangle([0, 0, img.width, 24], fill="black")
    d.text((6, 3), text, fill="white", font=FONT_BIG)
    img.save(path)
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cam", type=int, default=1, help="camera index (1 = Brio 105, 0 = laptop webcam)")
    ap.add_argument("--image", help="classify one image instead of the camera")
    ap.add_argument("--walk-conf", type=float, default=0.4, help="tuned on the sim face; recalibrate on real signals")
    ap.add_argument("--det-conf", type=float, default=0.1, help="model threshold, low so weak WALK scores are visible")
    ap.add_argument("--show-all", action="store_true", help="draw all 20 classes, not just WALK/DON'T WALK/conflict")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--width", type=int, default=1920, help="capture width; more pixels make the zoom pass real zoom")
    ap.add_argument("--height", type=int, default=1080)
    ap.add_argument("--zoom", type=float, default=2.5, help="centre zoom for the second pass until a signal is found "
                    "(then it follows the signal); 1 = off, which roughly doubles the frame rate")
    ap.add_argument("--grace", type=int, default=3, help="frames with no signal allowed before the WALK streak resets")
    ap.add_argument("--latch-seconds", type=float, default=10.0, help="once WALK is confirmed, hold READY TO WALK this "
                    "long even if the signal leaves view (assume the light stays); DON'T WALK still cancels it")
    ap.add_argument("--drive", action="store_true", help="actually walk the dog across on READY TO WALK")
    # 127.0.0.1, not localhost: on Windows "localhost" tries IPv6 first and adds ~200ms per request.
    ap.add_argument("--panel", default="http://127.0.0.1:8000")
    ap.add_argument("--stride", type=int, default=100)
    ap.add_argument("--cross-seconds", type=float, default=15.0, help="safety cap: stop after this long no matter what")
    ap.add_argument("--stop-cm", type=float, default=25.0, help="sonar distance that counts as arrived")
    ap.add_argument("--curb-conf", type=float, default=0.4, help="min confidence for a far-curb detection")
    ap.add_argument("--curb-ignore", type=float, default=4.0, help="ignore curbs this long after starting, so the "
                    "near curb the dog starts from doesn't count as arrived")
    ap.add_argument("--curb-frames", type=int, default=2, help="consecutive near-curb frames that count as arrived")
    ap.add_argument("--web-port", type=int, default=8002, help="live view port (0 = off)")
    ap.add_argument("--snapshot", default="walk_check.jpg", help="annotated latest frame is written here")
    args = ap.parse_args()

    det = BlindEscortDetector(use_cpu=True)

    if args.image:
        img = Image.open(args.image).convert("RGB")
        dets, region, _ = detect(det, img, args, None)
        label, w, s, c = vote(dets, args.walk_conf)
        line = f"{label.upper()}  walk={w:.2f} stop={s:.2f} conflict={c:.2f}"
        print(line)
        for d in dets:
            print(f"  {d['class']:<26} {d['confidence']:.2f}{'  (zoom)' if d.get('zoom') else ''}")
        annotate(img, dets, line, args.snapshot, args.show_all, region, args.walk_conf, args.curb_conf)
        return

    cap = cv2.VideoCapture(args.cam, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))  # raw 1080p over USB drops to a few fps
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    if not cap.isOpened():
        raise SystemExit(f"camera {args.cam} did not open (try --cam 0)")
    print(f"camera {args.cam}: {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}",
          flush=True)
    if args.web_port:
        srv = ThreadingHTTPServer(("127.0.0.1", args.web_port), Web)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        print(f"live view: http://127.0.0.1:{args.web_port}", flush=True)
    if args.drive:
        status = json.loads(urllib.request.urlopen(args.panel + "/status", timeout=2).read())
        if not status.get("connected"):
            raise SystemExit("panel says dog is not connected: %s" % status.get("conn"))
        post(args.panel, {"t": "stop"})
        print("driving via %s (%s)" % (args.panel, status.get("conn")), flush=True)
    streak, misses, state, cross_start, latch_until = 0, 0, None, None, 0.0
    target, target_seen, curb_hits = None, 0.0, 0
    try:
        while True:
            if cross_start is not None:
                # Straight at full stride until the first end condition. The panel stops the gait after 0.7 s
                # without a move, so this resends every frame.
                try:
                    dist = json.loads(urllib.request.urlopen(args.panel + "/status", timeout=1).read()).get("dist_cm")
                except Exception:
                    dist = None
                why = None
                if END.is_set():
                    why = "END clicked"
                elif dist is not None and dist <= args.stop_cm:
                    why = "sonar %.0f cm" % dist
                elif curb_hits >= args.curb_frames and time.time() - cross_start >= args.curb_ignore:
                    why = "far curb in view"
                elif time.time() - cross_start >= args.cross_seconds:
                    why = "time cap %.0fs" % args.cross_seconds
                else:
                    r = post(args.panel, {"t": "move", "stride": args.stride, "angle": 0})
                    print(f"    drive stride={args.stride} sonar={dist} curb={curb_hits} "
                          f"t={time.time() - cross_start:.1f}s", flush=True)
                    if r.get("ok") is False:
                        why = "panel obstacle stop"
                if why is not None:
                    post(args.panel, {"t": "stop"})
                    print("--- state -> ARRIVED: %s, dog stopped" % why, flush=True)
                    LIVE["state"] = dict(LIVE["state"], state="ARRIVED", held=False, why=why)
                    time.sleep(3)  # leave the ARRIVED banner up briefly for the live view
                    return
            elif END.is_set():
                END.clear()  # END before crossing just cancels a held WALK
                latch_until, streak = 0.0, 0
            ok, frame = cap.read()
            if not ok:
                print("frame grab failed")
                time.sleep(0.2)
                continue
            img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            now = time.time()
            dets, region, ms = detect(det, img, args, target if now - target_seen < TARGET_TTL else None)
            box = signal_box(dets)
            if box is not None:
                target, target_seen = box, now
            near_curb = any(d["class"] in CURB_CLASSES and d["confidence"] >= args.curb_conf
                            and d["box"][3] >= 0.8 * img.height and d["box"][2] - d["box"][0] >= 0.25 * img.width
                            for d in dets)
            curb_hits = curb_hits + 1 if near_curb else 0
            label, w, s, c = vote(dets, args.walk_conf)

            # One missed frame (blur, glare, the score dipping under threshold) shouldn't throw away the WALK
            # streak; only an explicit DON'T WALK / conflict, or more than --grace misses in a row, resets it.
            if label == "walk":
                streak, misses = streak + 1, 0
            elif label == "dont_walk":
                streak, misses = 0, 0
                if cross_start is None:
                    latch_until = 0.0  # a real DON'T WALK cancels a held WALK before we commit
            else:
                misses += 1
                if misses > args.grace:
                    streak = 0
            if streak >= CONFIRM_FRAMES and label == "walk":
                latch_until = now + args.latch_seconds

            if cross_start is not None:
                new_state = "CROSSING"
            elif now < latch_until:
                new_state = "READY TO WALK"
            else:
                new_state = "WAIT"
            held = new_state == "READY TO WALK" and label != "walk"
            line = (f"{new_state}{' (held %.0fs)' % (latch_until - now) if held else ''}  vote={label} "
                    f"walk={w:.2f} stop={s:.2f} conflict={c:.2f} streak={streak} miss={misses} {ms:.0f}ms")
            print(line, flush=True)
            if new_state != state:
                print(f"--- state -> {new_state}", flush=True)
                state = new_state
                if new_state == "READY TO WALK" and args.drive:
                    cross_start = time.time()
                    post(args.panel, {"t": "move", "stride": args.stride, "angle": 0})
                    print("--- state -> CROSSING for %.0fs" % args.cross_seconds, flush=True)
            publish(annotate(img, dets, line, args.snapshot, args.show_all, region, args.walk_conf, args.curb_conf),
                    {"state": new_state, "held": held, "vote": label, "walk": round(w, 2), "stop": round(s, 2),
                     "conflict": round(c, 2), "streak": streak, "miss": misses, "drive": args.drive,
                     "ms": round(ms)})
    except KeyboardInterrupt:
        pass
    finally:
        if args.drive:
            try:
                post(args.panel, {"t": "stop"})
            except Exception as e:
                print("STOP FAILED, stop the dog by hand: %s" % e, flush=True)
        cap.release()


if __name__ == "__main__":
    main()
