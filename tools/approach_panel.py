"""Browser control page for --go-signal mode, served at http://127.0.0.1:8002.

- Live status: state, sonar, latest Gemini verdict, latest YOLO26 detections.
- Target: "walk_signal" (Gemini reads the pedestrian signal) or any YOLO26 (COCO) class, e.g. "cell phone" (YOLO26
  detects it; Gemini stays idle). Plus YOLO26's minimum confidence. The target picks the detector, and
  fills in while Gemini is unavailable or may trigger on its own.
- Pause / resume, and a switch for stopping on the escort model's hazard classes.
- Voice: speaks guidance with ElevenLabs through Puter.js (free, no API key; the listener signs in to Puter, which
  covers the cost). Puter.js only runs in a browser, hence a page. Falls back to the browser's built-in voice.
"""
import json
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

PANEL_PORT = 8002
CROSSING_UI = "http://127.0.0.1:3000/crossing.html"  # the operator interface (saferoute/, npm start)
REPEAT_S = 4.0  # the same line again within this window is dropped, so a 4 Hz loop cannot spam the speaker

DEFAULT_SETTINGS = {
    "target": "walk_signal",          # "walk_signal" -> Gemini; a YOLO26 class name (e.g. "cell phone") -> YOLO26
    "min_conf": 0.5,                  # YOLO26 confidence needed for its target to count
    # Off by default: the custom escort model reports "hazards" in empty rooms, which would halt a crossing at random.
    "hazard_stop": False,             # stop on the escort model's hazard classes (curb drop-off, vehicle/cyclist)
    "paused": True,                   # start disarmed: no Gemini requests (free quota is 500/day) until Arm is pressed
    # Walk-signal demo timer: this many seconds after Arm, start crossing even if Gemini hasn't said GO yet (press the
    # crosswalk's push button and Arm together; the WALK light comes on a fixed time later). 0 = off.
    "go_timer_s": 0.0,                # off: Gemini starts it (6.72 matched the real crosswalk's push button, 2026-09-27)
    # Walking tuning from the operator page: body pitch / height (the bridge's posture and height commands) and a
    # reverse switch (every walk, crossing or W, runs backwards; the dog walked better backwards on smooth concrete).
    "pitch": 0,                       # degrees, -15..15
    "height": 80,                     # body height mm, 50..120 (80 = stock)
    "shift": 0,                       # body shift along the walking axis, mm, -25..25 (weight onto front / back feet)
    # Steering trim while it walks "straight", degrees, + = left, - = right. The real dog drifted left (2026-09-27).
    "trim": -3,
    "reverse": False,
    # Where a walk-signal crossing ends. Street: 0 = only on Stop, 180 s cap. Table demo: stop this many cm before the
    # "far curb" object at the end of the table (sonar), with a short cap so it can never walk off the edge.
    # The main demo (2026-09-27): the walk sign on a device at the end of a table or floor run, so the device demo is
    # the default; the page's "Street" button switches to finish_cm 0 / cross_max_s 180.
    "finish_cm": 25,
    "cross_max_s": 60,                # a backstop only: it walks until the sonar says it's there (8 s cut it off halfway)
    "celebration": "paw",             # "paw": jingle + raised paw; "full": beeps, bow, hind-leg stand; "bow"; "handshake"               # 2026-09-27 demo: the crosswalk shows WALK ~6.72 s after its button is pressed
}

PAGE = """<!doctype html>
<html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>MechDog Crossing</title>
<script src="https://js.puter.com/v2/"></script>
<style>
:root{--bg:#f6f6f4;--fg:#1b1b1b;--muted:#6b6b6b;--card:#fff;--line:#e2e2de;--go:#1a7f37;--stop:#c62828;--warn:#b26a00}
@media (prefers-color-scheme:dark){:root{--bg:#141414;--fg:#ececec;--muted:#9a9a9a;--card:#1e1e1e;--line:#2e2e2e;
--go:#3fb950;--stop:#ff6b6b;--warn:#e3a008}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.5 system-ui,sans-serif}
main{max-width:720px;margin:0 auto;padding:20px 16px}
h1{font-size:20px;margin:0 0 12px}h2{font-size:14px;text-transform:uppercase;letter-spacing:.04em;color:var(--muted);
margin:0 0 8px}
section{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:14px;margin:0 0 12px}
#state{font-size:28px;font-weight:700}#state.walk{color:var(--go)}#state.wait,#state.arrived{color:var(--muted)}
#state.hazard,#state.paused{color:var(--stop)}
.row{display:flex;gap:12px;flex-wrap:wrap;align-items:center;margin:6px 0}
.kv{color:var(--muted)}.kv b{color:var(--fg);font-weight:600}
label{display:flex;gap:6px;align-items:center}select,input{font:inherit;padding:6px 8px;border-radius:6px;
border:1px solid var(--line);background:var(--bg);color:var(--fg);max-width:100%}
button{font:inherit;padding:9px 16px;border-radius:8px;border:0;cursor:pointer;color:#fff;background:var(--go)}
button.stop{background:var(--stop)}button:disabled{opacity:.5}
ol{list-style:none;padding:0;margin:0}li{border-top:1px solid var(--line);padding:6px 0}li small{color:var(--muted);
margin-left:8px}
</style></head><body><main>
<h1>MechDog Crossing</h1>
<section>
  <div id="state">...</div>
  <div class="row kv"><span>sonar <b id="sonar">-</b></span><span>stride <b id="stride">-</b></span>
    <span>walking for <b id="elapsed">-</b></span></div>
  <div class="kv">Gemini: <b id="gemini">-</b></div>
  <div class="kv">YOLO26 sees: <b id="yolo">-</b></div>
  <div class="kv">steer <b id="steer">-</b> · est. distance <b id="dist">-</b></div>
  <div class="row"><button id="go">Force go (W)</button><button id="end" class="stop">Stop (Space)</button>
    <button id="pause" class="stop">Pause</button></div>
  <div class="kv">Automatic: it walks to the target when it sees it and stops on sonar. Overrides: W go, A / D steer, Space stop.</div>
</section>
<section>
  <h2>What starts a crossing</h2>
  <div class="row"><label>Target <select id="target"><option value="walk_signal">Walk signal (Gemini)</option></select></label>
    <label>YOLO26 min conf <input id="min_conf" type="number" min="0.1" max="0.95" step="0.05" style="width:80px"></label></div>
  <div class="row"><label><input id="hazard_stop" type="checkbox"> Stop for traffic in view (car, truck, bus, motorcycle, bicycle)</label></div>
</section>
<section>
  <h2>Voice</h2>
  <div class="row"><button id="voice">Enable voice</button><span id="vstatus" class="kv">off</span></div>
  <ol id="log"></ol>
</section>
</main><script>
const $ = id => document.getElementById(id);
let after = 0, voiceOn = false, busy = false, queue = [], settings = {}, classesLoaded = false;
async function post(patch){ settings = await (await fetch("/settings",{method:"POST",body:JSON.stringify(patch)})).json(); }
$("target").onchange = e => post({target: e.target.value});
$("min_conf").onchange = e => post({min_conf: parseFloat(e.target.value)});
$("hazard_stop").onchange = e => post({hazard_stop: e.target.checked});
$("pause").onclick = () => post({paused: !settings.paused});
const cmd = (c, extra) => fetch("/command", {method:"POST", body: JSON.stringify(Object.assign({cmd: c}, extra || {}))});
$("go").onclick = () => cmd("go");
$("end").onclick = () => cmd("end");
let steerDir = 0, steerTimer = null;
function setSteer(d){ steerDir = d; cmd("steer", {dir: d}); clearInterval(steerTimer);
  if (d) steerTimer = setInterval(() => cmd("steer", {dir: steerDir}), 250); }
const typing = () => ["INPUT","SELECT","TEXTAREA"].includes(document.activeElement && document.activeElement.tagName);
addEventListener("keydown", e => { if (typing() || e.repeat) return; const k = e.key.toLowerCase();
  if (k === "w") cmd("go"); else if (k === "s" || k === " ") { e.preventDefault(); cmd("end"); }
  else if (k === "a") setSteer(1); else if (k === "d") setSteer(-1); });
addEventListener("keyup", e => { const k = e.key.toLowerCase();
  if ((k === "a" && steerDir === 1) || (k === "d" && steerDir === -1)) setSteer(0); });
addEventListener("blur", () => { if (steerDir) setSteer(0); });
function logLine(text, via){ const li=document.createElement("li"); li.textContent=text;
  const s=document.createElement("small"); s.textContent=via; li.appendChild(s);
  const ol=$("log"); ol.prepend(li); while(ol.children.length>20) ol.lastChild.remove(); }
function builtin(text){ return new Promise(r=>{ const u=new SpeechSynthesisUtterance(text); u.onend=u.onerror=r;
  speechSynthesis.speak(u); }); }
async function speak(text){
  try{ const audio = await puter.ai.txt2speech(text,{provider:"elevenlabs",model:"eleven_flash_v2_5",
                                                     voice:"21m00Tcm4TlvDq8ikWAM"});
    await new Promise((res,rej)=>{ audio.onended=res; audio.onerror=rej; audio.play().catch(rej); });
    logLine(text,"ElevenLabs");
  }catch(e){ await builtin(text); logLine(text,"browser voice"); }
}
async function drain(){ if(busy||!queue.length) return; busy=true;
  while(queue.length){ await speak(queue.length>2 ? queue.splice(0).pop() : queue.shift()); } busy=false; }
$("voice").onclick = async () => { voiceOn = true; $("vstatus").textContent = "signing in to Puter...";
  try{ if(!puter.auth.isSignedIn()) await puter.auth.signIn(); $("vstatus").textContent = "on (ElevenLabs)"; }
  catch(e){ $("vstatus").textContent = "on (browser voice; Puter sign-in failed)"; }
  $("voice").disabled = true; queue.push("Voice guidance ready."); drain(); };
async function poll(){
  try{
    const s = await (await fetch("/status?after="+after)).json();
    settings = s.settings;
    if(!classesLoaded && s.classes.length){ const sel=$("target");
      for(const c of s.classes){ const o=document.createElement("option"); o.value=c; o.textContent=c+" (YOLO26)"; sel.appendChild(o); }
      classesLoaded = true; }
    for(const id of ["target","min_conf"]) if(document.activeElement!==$(id)) $(id).value = settings[id];
    $("hazard_stop").checked = settings.hazard_stop;
    $("pause").textContent = settings.paused ? "Resume" : "Pause";
    $("pause").className = settings.paused ? "" : "stop";
    const st = s.status, el = $("state");
    el.textContent = st.label || st.state || "..."; el.className = st.state || "";
    $("sonar").textContent = st.sonar_cm==null ? "-" : st.sonar_cm.toFixed(0)+" cm";
    $("stride").textContent = st.stride==null ? "-" : st.stride;
    $("steer").textContent = !st.angle ? "straight" : st.angle > 0 ? "left " + st.angle + "°" : "right " + (-st.angle) + "°";
    $("dist").textContent = st.est_distance_m==null ? "-" : st.est_distance_m.toFixed(1) + " m";
    $("elapsed").textContent = st.walking_s==null ? "-" : st.walking_s.toFixed(1)+" s";
    $("gemini").textContent = st.gemini || "-"; $("yolo").textContent = st.yolo || "-";
    for(const m of s.lines){ after = m.id; if(voiceOn) queue.push(m.text); } drain();
  }catch(e){ $("state").textContent = "agent offline"; $("state").className = "hazard"; }
  setTimeout(poll, 400);
}
poll();
</script></body></html>"""


class ApproachPanel:
    def __init__(self, port=PANEL_PORT, open_browser=False):
        self.lock = threading.Lock()
        self.settings = dict(DEFAULT_SETTINGS)
        self.status = {}
        self.classes = []
        self.lines = []
        self.next_id = 1
        self.last_said = {}
        self.commands = []
        self.steer_dir, self.steer_at = 0, 0.0
        self.drive_cmd = (0, 0, 0.0)
        self.shot, self.shot_info = None, None  # the last frame Gemini checked (JPEG) and what it said
        panel = self

        class Handler(BaseHTTPRequestHandler):
            def _send(self, body, ctype):
                self.send_response(200)
                self.send_header("Content-Type", ctype)
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self):
                if self.path.startswith("/status"):
                    try:
                        after = int(self.path.split("after=")[1].split("&")[0])
                    except (IndexError, ValueError):
                        after = 0
                    with panel.lock:
                        body = {"status": dict(panel.status), "settings": dict(panel.settings),
                                "shot": panel.shot_info,
                                "classes": panel.classes, "lines": [m for m in panel.lines if m["id"] > after]}
                    self._send(json.dumps(body).encode(), "application/json")
                elif self.path.startswith("/gemini.jpg"):
                    with panel.lock:
                        shot = panel.shot
                    if shot:
                        self._send(shot, "image/jpeg")
                    else:
                        self.send_response(404)
                        self.end_headers()
                else:
                    # The operator UI is SafeRoute's Live Crossing page, which calls this API through its
                    # /api/dog/* routes. The built-in page stays reachable at /legacy for debugging only.
                    if self.path.startswith("/legacy"):
                        self._send(PAGE.encode(), "text/html; charset=utf-8")
                    else:
                        self.send_response(302)
                        self.send_header("Location", CROSSING_UI)
                        self.end_headers()

            def do_POST(self):
                try:
                    body = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)) or b"{}")
                except ValueError:
                    body = {}
                if self.path.startswith("/command"):
                    panel.command(body)
                else:
                    panel.update_settings(body)
                with panel.lock:
                    self._send(json.dumps(panel.settings).encode(), "application/json")

            def log_message(self, *a):
                pass

        self.server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        url = f"http://127.0.0.1:{port}"
        print(f"🎛️  [API] {url} (status / settings / commands) · operator UI: {CROSSING_UI}")
        if open_browser:
            webbrowser.open(CROSSING_UI)

    def update_settings(self, patch):
        with self.lock:
            for key, value in patch.items():
                if key not in DEFAULT_SETTINGS:
                    continue
                if key == "target" and value != "walk_signal" and value not in self.classes:
                    continue
                if key == "min_conf":
                    try:
                        value = min(0.95, max(0.1, float(value)))
                    except (TypeError, ValueError):
                        continue
                if key == "go_timer_s":
                    try:
                        value = min(120.0, max(0.0, float(value or 0)))
                    except (TypeError, ValueError):
                        continue
                if key in ("pitch", "height", "shift", "finish_cm", "cross_max_s", "trim"):
                    lo, hi = {"pitch": (-15, 15), "height": (50, 120), "shift": (-25, 25), "finish_cm": (0, 150),
                              "cross_max_s": (2, 300), "trim": (-15, 15)}[key]
                    try:
                        value = int(min(hi, max(lo, float(value))))
                    except (TypeError, ValueError):
                        continue
                if key == "celebration" and value not in ("paw", "full", "bow", "handshake"):
                    continue
                if key in ("hazard_stop", "paused", "reverse"):
                    value = bool(value)
                self.settings[key] = value
            print(f"🎛️  settings: {self.settings}")

    def command(self, body):
        """{"cmd": "go" | "end"} queued for the control loop; {"cmd": "steer", "dir": -1 | 0 | 1} held until a new
        steer arrives or it goes stale (the page re-sends a held key every 250 ms)."""
        cmd = body.get("cmd")
        with self.lock:
            if cmd in ("go", "end"):
                self.commands.append(cmd)
                print(f"🎛️  command: {cmd}")
            elif cmd == "steer" and body.get("dir") in (-1, 0, 1):
                self.steer_dir, self.steer_at = body["dir"], time.monotonic()
            elif cmd == "drive":
                # Manual WASD driving from the operator page, re-sent while keys are held; goes stale like steer.
                try:
                    self.drive_cmd = (max(-120, min(120, int(body.get("stride", 0)))),
                                      max(-30, min(30, int(body.get("angle", 0)))), time.monotonic())
                except (TypeError, ValueError):
                    pass

    def pop_commands(self):
        with self.lock:
            cmds, self.commands = self.commands, []
            return cmds

    def steer(self, fresh_s):
        """-1 right, 0 straight, 1 left; 0 once the last steer is older than fresh_s."""
        with self.lock:
            return self.steer_dir if time.monotonic() - self.steer_at < fresh_s else 0

    def drive(self, fresh_s):
        """(stride, angle) the operator is driving with, or None once the last drive is older than fresh_s."""
        with self.lock:
            stride, angle, at = self.drive_cmd
            return (stride, angle) if time.monotonic() - at < fresh_s else None

    def set_shot(self, jpeg, info):
        with self.lock:
            self.shot, self.shot_info = jpeg, dict(info, at=time.time())

    def get_settings(self):
        with self.lock:
            return dict(self.settings)

    def set_classes(self, names):
        with self.lock:
            self.classes = list(names)

    def set_status(self, **status):
        with self.lock:
            self.status = status

    def say(self, text):
        now = time.monotonic()
        with self.lock:
            if now - self.last_said.get(text, -1e9) < REPEAT_S:
                return
            self.last_said[text] = now
            self.lines.append({"id": self.next_id, "text": text})
            self.next_id += 1
            del self.lines[:-50]
        print(f"🗣️  {text}")
