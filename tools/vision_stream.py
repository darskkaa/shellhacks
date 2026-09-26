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
from ultralytics import YOLO

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_MODEL = os.path.join(ROOT, "blind_escort_yolo", "weights", "yolo11n_blind_escort.pt")
if not os.path.exists(DEFAULT_MODEL):
    DEFAULT_MODEL = os.path.join(ROOT, "models", "yolo26n.pt")

ap = argparse.ArgumentParser()
ap.add_argument("--cam", type=int, default=1, help="camera index (1 = Brio 105, 0 = laptop webcam)")
ap.add_argument("--model", default=DEFAULT_MODEL)
ap.add_argument("--imgsz", type=int, default=416)
ap.add_argument("--conf", type=float, default=0.35)
ap.add_argument("--port", type=int, default=8001)
ap.add_argument("--no-browser", action="store_true")
args = ap.parse_args()


class Camera:
    """Reads frames continuously so inference always sees the newest one (no lag from a queued buffer)."""

    def __init__(self, index):
        backend = cv2.CAP_DSHOW if sys.platform.startswith("win") else cv2.CAP_ANY
        self.cap = cv2.VideoCapture(index, backend)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if not self.cap.isOpened():
            raise SystemExit("camera %d did not open" % index)
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


class Detector:
    def __init__(self, cam):
        self.cam = cam
        self.model = YOLO(args.model)
        self.jpeg = None
        self.dets = []
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
            t0 = time.time()
            r = self.model.predict(frame, imgsz=args.imgsz, conf=args.conf, verbose=False)[0]
            self.infer_ms = (time.time() - t0) * 1000
            h, w = frame.shape[:2]
            dets = []
            for box, c, cls in zip(r.boxes.xyxy.tolist(), r.boxes.conf.tolist(), r.boxes.cls.tolist()):
                x1, y1, x2, y2 = box
                dets.append({
                    "label": self.model.names[int(cls)],
                    "conf": round(c, 3),
                    "box": [round(x1), round(y1), round(x2), round(y2)],
                    "cx": round((x1 + x2) / 2 / w, 3),        # 0 = left edge, 1 = right edge
                    "area": round((x2 - x1) * (y2 - y1) / (w * h), 4),  # fraction of the frame (proxy for closeness)
                })
            annotated = r.plot()
            now = time.time()
            self.fps = 0.8 * self.fps + 0.2 * (1 / max(now - last, 1e-3))
            last = now
            cv2.putText(annotated, "YOLO26n  %.1f fps  %d ms" % (self.fps, self.infer_ms), (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
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
        if self.path.startswith("/detections"):
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
