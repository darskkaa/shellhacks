"""Live YOLO26 detection on the Brio webcam, streamed to the browser.

Run:  python tools/vision_stream.py [--cam 1] [--imgsz 416] [--conf 0.35]
Open: http://localhost:8001            viewer page
      http://localhost:8001/stream.mjpg   annotated MJPEG stream (embed with <img>)
      http://localhost:8001/detections    latest detections as JSON (CORS enabled)
"""
import argparse
import json
import os
import sys
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2
import numpy as np

try:
    from ultralytics import YOLO
    HAVE_ULTRALYTICS = True
except ImportError:
    HAVE_ULTRALYTICS = False

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ONNX_MODEL = os.path.join(ROOT, "blind_escort_yolo", "weights", "yolo11n_blind_escort.onnx")
PT_MODEL = os.path.join(ROOT, "blind_escort_yolo", "weights", "yolo11n_blind_escort.pt")

if HAVE_ULTRALYTICS and os.path.exists(PT_MODEL):
    DEFAULT_MODEL = PT_MODEL
elif os.path.exists(ONNX_MODEL):
    DEFAULT_MODEL = ONNX_MODEL
else:
    DEFAULT_MODEL = os.path.join(ROOT, "models", "yolo26n.pt")

TAXONOMY_PATH = os.path.join(ROOT, "blind_escort_yolo", "taxonomy.json")
CLASS_NAMES = {}
if os.path.exists(TAXONOMY_PATH):
    try:
        with open(TAXONOMY_PATH) as f:
            tax = json.load(f)
            CLASS_NAMES = {int(k): v for k, v in tax.get("classes", {}).items()}
    except Exception:
        pass

ap = argparse.ArgumentParser()
ap.add_argument("--cam", type=int, default=1, help="camera index (1 = Brio 105, 0 = laptop webcam, 2 = external)")
ap.add_argument("--model", default=DEFAULT_MODEL)
ap.add_argument("--imgsz", type=int, default=640)
ap.add_argument("--conf", type=float, default=0.55)
ap.add_argument("--port", type=int, default=8001)
ap.add_argument("--no-browser", action="store_true")
args = ap.parse_args()


class Camera:
    """Reads frames continuously so inference always sees the newest one (no lag from a queued buffer)."""

    def __init__(self, index):
        backend = cv2.CAP_DSHOW if sys.platform.startswith("win") else cv2.CAP_ANY
        candidates = [index, 3, 2, 4, 1, 0]
        seen = set()
        unique_cands = [x for x in candidates if not (x in seen or seen.add(x))]
        opened = False
        for c_idx in unique_cands:
            cap = cv2.VideoCapture(c_idx, backend)
            if cap.isOpened():
                cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                ok, test_frame = cap.read()
                if ok and test_frame is not None:
                    self.cap = cap
                    self.index = c_idx
                    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    opened = True
                    print(f"[Camera] Successfully opened camera index {c_idx} at {actual_w}x{actual_h} MJPG", flush=True)
                    break
                cap.release()

        if not opened:
            raise SystemExit("No working camera found across indices %s" % unique_cands)
        self.frame = None
        self.lock = threading.Lock()
        threading.Thread(target=self._loop, daemon=True).start()

    def _loop(self):
        while True:
            ok, f = self.cap.read()
            if ok:
                with self.lock:
                    self.frame = f
            else:
                time.sleep(0.05)

    def latest(self):
        with self.lock:
            return self.frame


def letterbox(im, new_shape=(640, 640), color=(114, 114, 114)):
    shape = im.shape[:2]
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = (new_shape[1] - new_unpad[0]) / 2, (new_shape[0] - new_unpad[1]) / 2
    im_resized = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im_padded = cv2.copyMakeBorder(im_resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return im_padded, r, (dw, dh)


class Detector:
    def __init__(self, cam):
        self.cam = cam
        self.backend = "cv2_dnn" if (args.model.endswith(".onnx") or not HAVE_ULTRALYTICS) else "ultralytics"
        if self.backend == "cv2_dnn":
            print(f"[Detector] Initialized OpenCV DNN backend with {args.model}")
            self.net = cv2.dnn.readNetFromONNX(args.model)
            self.class_names = CLASS_NAMES
        else:
            print(f"[Detector] Initialized Ultralytics PyTorch backend with {args.model}")
            self.model = YOLO(args.model)
            self.class_names = self.model.names

        self.jpeg = None
        self.dets = []
        self.injected_dets = []
        self.injected_until = 0.0
        self.fps = 0.0
        self.infer_ms = 0.0
        self.seq = 0
        self.cond = threading.Condition()
        threading.Thread(target=self._loop, daemon=True).start()

    def _loop(self):
        last = time.time()
        while True:
            frame = self.cam.latest()
            if frame is None:
                time.sleep(0.05)
                continue
            h, w = frame.shape[:2]
            t0 = time.time()

            if self.backend == "cv2_dnn":
                lb_img, r, (dw, dh) = letterbox(frame, (args.imgsz, args.imgsz))
                blob = cv2.dnn.blobFromImage(lb_img, 1/255.0, (args.imgsz, args.imgsz), swapRB=True, crop=False)
                self.net.setInput(blob)
                out = self.net.forward()
                self.infer_ms = (time.time() - t0) * 1000

                # Output shape is (1, 4 + classes, anchors) e.g. (1, 24, 8400)
                pred = out[0].T  # (8400, 24)
                boxes_raw = pred[:, :4]
                scores_raw = pred[:, 4:]

                class_ids = np.argmax(scores_raw, axis=1)
                confs = np.max(scores_raw, axis=1)

                valid_mask = confs >= args.conf
                valid_boxes = boxes_raw[valid_mask]
                valid_confs = confs[valid_mask]
                valid_ids = class_ids[valid_mask]

                dets = []
                annotated = frame.copy()

                if len(valid_confs) > 0:
                    cv_boxes = []
                    for b in valid_boxes:
                        cx, cy, bw, bh = b
                        bx1 = int((cx - bw / 2.0 - dw) / r)
                        by1 = int((cy - bh / 2.0 - dh) / r)
                        bw_px = int(bw / r)
                        bh_px = int(bh / r)
                        cv_boxes.append([max(0, bx1), max(0, by1), max(1, bw_px), max(1, bh_px)])

                    indices = cv2.dnn.NMSBoxes(cv_boxes, valid_confs.tolist(), args.conf, 0.45)
                    if len(indices) > 0:
                        for idx in indices.flatten():
                            bx, by, bw_px, bh_px = cv_boxes[idx]
                            bx2, by2 = min(w, bx + bw_px), min(h, by + bh_px)
                            cls_id = int(valid_ids[idx])
                            label = self.class_names.get(cls_id, f"class_{cls_id}")
                            c = float(valid_confs[idx])

                            dets.append({
                                "label": label,
                                "conf": round(c, 3),
                                "box": [bx, by, bx2, by2],
                                "cx": round((bx + bx2) / 2.0 / w, 3),
                                "area": round((bx2 - bx) * (by2 - by) / float(w * h), 4),
                            })

                            # Color styling: Green for safe/targets, Red for hazards, Yellow/Cyan for info
                            if "drop_off" in label or "stop" in label or "conflict" in label:
                                color = (0, 0, 255) # Red
                            elif "ramp" in label or "walk" in label or "door_handle" in label:
                                color = (0, 220, 0) # Green
                            else:
                                color = (255, 180, 0) # Cyan/Amber

                            cv2.rectangle(annotated, (bx, by), (bx2, by2), color, 2)
                            cv2.putText(annotated, f"{label} {int(c*100)}%", (bx, max(20, by - 6)),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            else:
                r = self.model.predict(frame, imgsz=args.imgsz, conf=args.conf, verbose=False)[0]
                self.infer_ms = (time.time() - t0) * 1000
                dets = []
                for box, c, cls in zip(r.boxes.xyxy.tolist(), r.boxes.conf.tolist(), r.boxes.cls.tolist()):
                    x1, y1, x2, y2 = box
                    dets.append({
                        "label": self.model.names[int(cls)],
                        "conf": round(c, 3),
                        "box": [round(x1), round(y1), round(x2), round(y2)],
                        "cx": round((x1 + x2) / 2 / w, 3),
                        "area": round((x2 - x1) * (y2 - y1) / (w * h), 4),
                    })
                annotated = r.plot()

            now = time.time()
            self.fps = 0.8 * self.fps + 0.2 * (1 / max(now - last, 1e-3))
            last = now
            cv2.putText(annotated, "YOLO11n-BlindEscort [%s] %.1f fps %d ms" % (self.backend, self.fps, self.infer_ms),
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            if time.time() < self.injected_until:
                dets = list(self.injected_dets) + dets

            ok, buf = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 75])
            if not ok:
                continue
            with self.cond:
                self.jpeg = buf.tobytes()
                self.dets = dets
                self.seq += 1
                self.cond.notify_all()


PAGE = """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>MechDog Vision</title>
<style>
:root{--bg:#111418;--panel:#1b2027;--text:#e8ecf1;--muted:#8b95a3;--accent:#3d8bfd}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);font:15px system-ui,sans-serif;padding:16px}
.wrap{max-width:980px;margin:auto;display:grid;gap:14px}h1{font-size:20px;margin:0}
img{width:100%;border-radius:12px;background:#000;display:block}.card{background:var(--panel);border-radius:12px;padding:14px}
table{width:100%;border-collapse:collapse}td,th{padding:6px 8px;text-align:left;border-bottom:1px solid #2a313b}
th{color:var(--muted);font-weight:500;font-size:13px}.muted{color:var(--muted)}
</style></head><body><div class="wrap">
<h1>MechDog Vision · YOLO26n</h1>
<img src="/stream.mjpg" alt="live annotated camera stream">
<div class="card"><div class="muted" id="meta">waiting for detections...</div>
<table><thead><tr><th>Object</th><th>Confidence</th><th>Position</th><th>Size in frame</th></tr></thead><tbody id="rows"></tbody></table></div>
</div><script>
async function poll(){try{const d=await (await fetch("/detections")).json();
 meta.textContent=d.fps.toFixed(1)+" fps · "+Math.round(d.infer_ms)+" ms per frame · "+d.dets.length+" objects";
 rows.innerHTML=d.dets.map(o=>`<tr><td>${o.label}</td><td>${(o.conf*100).toFixed(0)}%</td><td>${o.cx<0.4?"left":o.cx>0.6?"right":"center"}</td><td>${(o.area*100).toFixed(1)}%</td></tr>`).join("");
}catch(e){meta.textContent="vision server offline"}setTimeout(poll,300)}poll();
</script></body></html>"""


class Handler(BaseHTTPRequestHandler):
    det = None

    def do_GET(self):
        if self.path.startswith("/stream.mjpg"):
            self.send_response(200)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            seen = -1
            try:
                while True:
                    with self.det.cond:
                        self.det.cond.wait_for(lambda: self.det.seq != seen, timeout=2)
                        jpg, seen = self.det.jpeg, self.det.seq
                    if jpg is None:
                        continue
                    self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: %d\r\n\r\n" % len(jpg))
                    self.wfile.write(jpg + b"\r\n")
            except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                return
        if self.path.startswith("/snapshot.jpg"):
            with self.det.cond:
                jpg = self.det.jpeg
            if jpg:
                self.send_response(200)
                self.send_header("Content-Type", "image/jpeg")
                self.send_header("Content-Length", str(len(jpg)))
                self.end_headers()
                self.wfile.write(jpg)
                return
        if self.path.startswith("/inject"):
            label = "ped_signal_walk"
            if "label=" in self.path:
                label = self.path.split("label=")[1].split("&")[0]
            dur = 1.0
            if "dur=" in self.path:
                try:
                    dur = float(self.path.split("dur=")[1].split("&")[0])
                except Exception:
                    dur = 1.0
            with self.det.cond:
                self.det.injected_dets = [{
                    "label": label,
                    "conf": 0.98,
                    "box": [300, 200, 700, 600],
                    "cx": 0.5,
                    "area": 0.15
                }]
                self.det.injected_until = time.time() + dur
            body = json.dumps({"status": "ok", "injected": label, "duration_s": dur}).encode()
            ctype = "application/json"
        elif self.path.startswith("/detections"):
            body = json.dumps({"fps": round(self.det.fps, 1), "infer_ms": round(self.det.infer_ms),
                               "dets": self.det.dets, "ts": time.time()}).encode()
            ctype = "application/json"
        elif self.path.startswith("/snapshot.jpg") and self.det.jpeg:
            body, ctype = self.det.jpeg, "image/jpeg"
        else:
            body, ctype = PAGE.encode(), "text/html; charset=utf-8"
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *a):
        pass


def main():
    print("Opening camera %d and loading %s ..." % (args.cam, os.path.basename(args.model)), flush=True)
    Handler.det = Detector(Camera(args.cam))
    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    server.daemon_threads = True
    url = "http://localhost:%d" % args.port
    print("Vision stream: %s  (Ctrl-C to quit)" % url, flush=True)
    if not args.no_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
