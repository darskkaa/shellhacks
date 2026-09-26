"""Local web control panel for the MechDog, over WiFi (preferred) or USB serial (fallback).

Run:  python tools/dog_panel.py [--wifi IP] [--serial COMx] [--usb-only]
Opens http://localhost:8000. The panel starts even if the dog is offline and connects when it shows up:
  WiFi: finds the dog's bridge (main.py, TCP port 5005) by scanning the laptop's local /24 network.
  USB:  used only when WiFi finds nothing; takes over the REPL, which pauses the WiFi bridge until reset.
"""
import argparse
import json
import socket
import threading
import time
import webbrowser
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import serial
from serial.tools import list_ports

ap = argparse.ArgumentParser()
ap.add_argument("--wifi", help="dog IP address (skips the network scan)")
ap.add_argument("--serial", help="COM port (default: auto-detect the CH340)")
ap.add_argument("--usb-only", action="store_true")
ap.add_argument("--wifi-only", action="store_true")
args = ap.parse_args()

HTTP_PORT = 8000
BRIDGE_PORT = 5005
MOVE_TIMEOUT_S = 0.7      # stop if the browser stops sending move commands (tab closed, key released, crash)
OBSTACLE_STOP_CM = 20

ACTIONS = [
    "left_foot_kick", "right_foot_kick", "stand_four_legs", "sit_dowm", "go_prone", "stand_two_legs",
    "handshake", "scrape_a_bow", "nodding_motion", "boxing", "stretch_oneself", "pee", "press_up",
    "rotation_pitch", "rotation_roll", "normal_attitude",
]


class WifiDog:
    """Client for mechdog-bridge (main.py on the dog): newline-delimited JSON over TCP."""
    keepalive = True  # the bridge stops a walk if it hears nothing for 1.5 s, so moves are resent while held

    def __init__(self, ip):
        self.label = "WiFi %s" % ip
        self.sock = socket.create_connection((ip, BRIDGE_PORT), timeout=3)
        self.sock.settimeout(None)
        self.lock = threading.Lock()
        self.alive = True
        self.dist_cm = None
        self.batt_v = None
        self.events = []
        self.rfile = self.sock.makefile("rb")
        hello = json.loads(self.rfile.readline())
        if hello.get("t") != "hello":
            raise ConnectionError("not a MechDog bridge")
        threading.Thread(target=self._reader, daemon=True).start()
        self._send({"t": "sub", "hz": 4})

    def _reader(self):
        try:
            for line in self.rfile:
                try:
                    m = json.loads(line)
                except ValueError:
                    continue
                if m.get("t") == "tel":
                    d = m.get("dist_cm")
                    self.dist_cm = round(d, 1) if isinstance(d, (int, float)) else None
                    self.batt_v = m.get("batt_v")
                elif m.get("t") == "event" and m.get("name") != "watchdog_stop":
                    self.events.append(m["name"].replace("_", " "))
        except OSError:
            pass
        self.alive = False

    def _send(self, msg):
        if not self.alive:
            raise ConnectionError("bridge disconnected")
        with self.lock:
            self.sock.sendall((json.dumps(msg) + "\n").encode())

    def move(self, stride, angle):
        self._send({"t": "move", "stride": stride, "angle": angle})

    def stop(self):
        self._send({"t": "stop"})

    def action(self, i):
        self._send({"t": "action", "id": i})

    def stand(self):
        self._send({"t": "reset"})

    def rgb(self, r, g, b):
        self._send({"t": "rgb", "r": r, "g": g, "b": b})

    def beep(self, freq, ms):
        self._send({"t": "buzzer", "freq": freq, "ms": ms})

    def sensors(self):
        if not self.alive:
            raise ConnectionError("bridge disconnected")
        return self.dist_cm, self.batt_v

    def close(self):
        try:
            self.stop()
        except Exception:
            pass
        self.sock.close()


class SerialDog:
    """Client for mechdog-bridge over USB serial: newline-delimited JSON."""
    keepalive = True

    def __init__(self, port):
        self.label = "USB %s" % port
        self.events = []
        self.lock = threading.Lock()
        self.alive = True
        self.dist_cm = None
        self.batt_v = None
        self.s = serial.Serial()
        self.s.port, self.s.baudrate, self.s.timeout = port, 115200, 0.5
        self.s.dtr = False
        self.s.rts = False
        self.s.open()
        time.sleep(0.1)
        self.s.reset_input_buffer()
        self.s.write(b'{"t":"hello"}\n')
        threading.Thread(target=self._reader, daemon=True).start()
        self._send({"t": "sub", "hz": 4})

    def _reader(self):
        buf = b""
        while self.alive:
            try:
                data = self.s.read(self.s.in_waiting or 1)
            except Exception:
                break
            if not data:
                continue
            buf += data
            while b"\n" in buf:
                line, buf = buf.split(b"\n", 1)
                line = line.strip()
                if not line.startswith(b"{"):
                    continue
                try:
                    m = json.loads(line)
                except ValueError:
                    continue
                if m.get("t") == "tel":
                    d = m.get("dist_cm")
                    self.dist_cm = round(d, 1) if isinstance(d, (int, float)) else None
                    self.batt_v = m.get("batt_v")
                elif m.get("t") == "event" and m.get("name") != "watchdog_stop":
                    self.events.append(m["name"].replace("_", " "))
        self.alive = False

    def _send(self, msg):
        if not self.alive:
            raise ConnectionError("bridge disconnected")
        with self.lock:
            self.s.write((json.dumps(msg) + "\n").encode())

    def move(self, stride, angle):
        self._send({"t": "move", "stride": stride, "angle": angle})

    def stop(self):
        self._send({"t": "stop"})

    def action(self, i):
        self._send({"t": "action", "id": i})

    def stand(self):
        self._send({"t": "reset"})

    def rgb(self, r, g, b):
        self._send({"t": "rgb", "r": r, "g": g, "b": b})

    def beep(self, freq, ms):
        self._send({"t": "buzzer", "freq": freq, "ms": ms})

    def sensors(self):
        if not self.alive:
            raise ConnectionError("bridge disconnected")
        return self.dist_cm, self.batt_v

    def close(self):
        try:
            self.stop()
        except Exception:
            pass
        self.alive = False
        self.s.close()


def local_subnet():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        return s.getsockname()[0].rsplit(".", 1)[0]
    finally:
        s.close()


def find_bridge():
    """Scan the laptop's /24 for something listening on 5005 that answers with a MechDog hello."""
    def probe(ip):
        try:
            with socket.create_connection((ip, BRIDGE_PORT), timeout=0.6) as c:
                c.settimeout(1.5)
                line = c.makefile("rb").readline()
                return ip if b'"hello"' in line and b"MechDog" in line else None
        except OSError:
            return None
    try:
        base = local_subnet()
    except OSError:
        return None
    with ThreadPoolExecutor(64) as ex:
        for ip in ex.map(probe, ["%s.%d" % (base, i) for i in range(1, 255)]):
            if ip:
                return ip
    return None


def find_serial():
    for p in list_ports.comports():
        if p.vid == 0x1A86:  # CH340 on the MechDog controller
            return p.device
    return None


class Controller:
    def __init__(self):
        self.dog = None
        self.known_ip = args.wifi
        self.conn_msg = "looking for the dog..."
        self.walking = False
        self.stride = 0
        self.angle = 0
        self.last_move = 0.0
        self.dist_cm = None
        self.batt_v = None
        self.event = ""
        for fn in (self._connector, self._watchdog, self._sensors):
            threading.Thread(target=fn, daemon=True).start()

    def _connector(self):
        while True:
            if self.dog is None:
                self._try_connect()
            time.sleep(2)

    def _try_connect(self):
        if not args.usb_only:
            ip = self.known_ip
            if ip is None:
                self.conn_msg = "scanning WiFi for the dog..."
                ip = find_bridge()
            if ip:
                try:
                    self.dog = WifiDog(ip)
                    self.known_ip = ip
                    self.conn_msg = "connected over WiFi (%s)" % ip
                    self.event = "dog connected over WiFi"
                    return
                except Exception as e:
                    self.conn_msg = "found %s but could not connect: %s" % (ip, str(e)[:80])
                    if not args.wifi:
                        self.known_ip = None
        if args.wifi_only:
            if self.dog is None:
                self.conn_msg = "dog not found on WiFi. Is it powered on and on the same network?"
            return
        port = args.serial or find_serial()
        if port is None:
            if self.dog is None:
                self.conn_msg = "dog not found on WiFi or USB: power it on (same network as this laptop)"
            return
        try:
            self.conn_msg = "connecting over USB %s..." % port
            self.dog = SerialDog(port)
            self.conn_msg = "connected over USB (%s)" % port
            self.event = "dog connected over USB"
        except Exception as e:
            self.conn_msg = "found %s but could not connect: %s" % (port, str(e)[:80])

    def _call(self, name, *a):
        dog = self.dog
        if dog is None:
            raise RuntimeError("dog offline")
        try:
            return getattr(dog, name)(*a)
        except (serial.SerialException, OSError, TimeoutError, ConnectionError):
            self._drop(dog)
            raise RuntimeError("lost connection to the dog")

    def _drop(self, dog):
        if self.dog is dog:
            self.dog = None
        self.walking = False
        self.dist_cm = self.batt_v = None
        try:
            dog.close()
        except Exception:
            pass

    def move(self, stride, angle):
        stride = max(-100, min(100, int(float(stride))))
        angle = max(-30, min(30, int(float(angle))))
        if stride > 0 and self.dist_cm is not None and self.dist_cm < OBSTACLE_STOP_CM:
            self.stop("obstacle ahead")
            return False
        self.last_move = time.time()
        dog = self.dog
        if (stride, angle) != (self.stride, self.angle) or not self.walking or (dog and dog.keepalive):
            self._call("move", stride, angle)
        self.stride, self.angle, self.walking = stride, angle, bool(stride or angle)
        return True

    def stop(self, why=""):
        self._call("stop")
        self.stride = self.angle = 0
        self.walking = False
        if why:
            self.event = why

    def _watchdog(self):
        while True:
            time.sleep(0.1)
            if self.walking and time.time() - self.last_move > MOVE_TIMEOUT_S:
                try:
                    self.stop()
                except Exception:
                    pass

    def _sensors(self):
        while True:
            dog = self.dog
            if dog is None:
                time.sleep(0.5)
                continue
            try:
                self.dist_cm, self.batt_v = self._call("sensors")
                if dog.events:
                    self.event = dog.events.pop(0)
                    self.walking = False
                if self.walking and self.stride > 0 and self.dist_cm is not None and self.dist_cm < OBSTACLE_STOP_CM:
                    self.stop("obstacle stop")
            except Exception:
                pass
            time.sleep(0.3)

    def handle(self, msg):
        t = msg.get("t")
        if t == "move":
            return {"ok": self.move(msg.get("stride", 0), msg.get("angle", 0))}
        if t == "stop":
            self.stop()
        elif t == "action":
            i = int(msg.get("id", -1))
            if not 0 <= i < len(ACTIONS):
                return {"ok": False, "msg": "bad action"}
            self.stop()
            self._call("action", i)
        elif t == "stand":
            self.stop()
            self._call("stand")
        elif t == "rgb":
            r, g, b = (max(0, min(255, int(float(msg.get(k, 0))))) for k in "rgb")
            self._call("rgb", r, g, b)
        elif t == "beep":
            self._call("beep", int(msg.get("freq", 1500)), int(msg.get("ms", 120)))
        else:
            return {"ok": False, "msg": "unknown command"}
        return {"ok": True}

    def status(self):
        ev, self.event = self.event, ""
        return {"connected": self.dog is not None, "conn": self.conn_msg, "dist_cm": self.dist_cm,
                "batt_v": self.batt_v, "walking": self.walking, "stride": self.stride, "angle": self.angle,
                "event": ev}


PAGE = r"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>MechDog Control</title>
<style>
:root{--bg:#111418;--panel:#1b2027;--text:#e8ecf1;--muted:#8b95a3;--accent:#3d8bfd;--stop:#e5484d;--ok:#30a46c}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);font:15px system-ui,sans-serif;padding:16px}
h1{font-size:20px;margin:0 0 12px}.grid{display:grid;gap:14px;max-width:760px;margin:auto}
.card{background:var(--panel);border-radius:12px;padding:14px}.row{display:flex;gap:8px;flex-wrap:wrap;align-items:center}
.stat{flex:1;min-width:110px}.stat b{display:block;font-size:24px}.stat span{color:var(--muted);font-size:12px}
button{background:#2a313b;color:var(--text);border:0;border-radius:10px;padding:12px 14px;font-size:15px;cursor:pointer;user-select:none;touch-action:none}
button:active,button.on{background:var(--accent)}.pad{display:grid;grid-template-columns:repeat(3,80px);gap:8px;justify-content:center}
.pad button{height:70px;font-size:22px}#stop{background:var(--stop);font-weight:700}
#event{color:var(--stop);min-height:18px;font-weight:600}.muted{color:var(--muted);font-size:13px}
input[type=range]{width:180px}
</style></head><body><div class="grid">
<h1>MechDog Control</h1>
<div id="conn" class="card" style="font-weight:600">connecting...</div>
<div class="card" id="xw" style="display:none">
 <div class="row"><div id="xwstate" style="flex:1;font:700 30px system-ui;padding:10px;border-radius:10px;text-align:center;background:#333">...</div>
 <button id="xwend" style="background:var(--stop);font:700 22px system-ui;padding:16px 26px">END</button></div>
 <div id="xwheld" style="color:#ffd166;font-weight:600;min-height:20px;margin-top:6px"></div>
 <div id="xwd" class="muted"></div></div>
<div class="card"><img id="cam" src="http://127.0.0.1:8002/stream" alt="live camera with detections"
 style="width:100%;border-radius:10px;background:#000;display:block">
 <div id="seen" class="muted" style="margin-top:8px">vision: waiting...</div></div>
<div class="card row">
 <div class="stat"><span>Distance ahead</span><b id="dist">–</b></div>
 <div class="stat"><span>Battery</span><b id="batt">–</b></div>
 <div class="stat"><span>State</span><b id="state">–</b></div>
</div>
<div id="event"></div>
<div class="card">
 <div class="pad">
  <span></span><button data-dir="f">▲</button><span></span>
  <button data-dir="l">◀</button><button id="stop">STOP</button><button data-dir="r">▶</button>
  <span></span><button data-dir="b">▼</button><span></span>
 </div>
 <div class="row" style="justify-content:center;margin-top:12px">
  <label class="muted">Speed <input id="speed" type="range" min="20" max="100" value="50"></label>
  <span id="speedv" class="muted">50</span>
 </div>
 <p class="muted" style="text-align:center">Hold a button or W/A/S/D. Space = stop. Releasing stops the dog.</p>
</div>
<div class="card"><div class="row" id="actions"></div></div>
<div class="card row">
 <button data-rgb="255,0,0">Red</button><button data-rgb="0,255,0">Green</button>
 <button data-rgb="0,0,255">Blue</button><button data-rgb="255,140,0">Orange</button>
 <button data-rgb="0,0,0">Light off</button><button id="beep">Beep</button><button id="stand">Stand</button>
</div>
</div><script>
const ACTIONS=["Left kick","Right kick","Stand 4","Sit","Lie down","Stand 2","Handshake","Bow","Nod","Boxing","Stretch","Pee","Push-up","Pitch roll","Roll","Normal"];
const post=m=>fetch("/cmd",{method:"POST",body:JSON.stringify(m)}).then(r=>r.json()).catch(()=>({}));
const speed=document.getElementById("speed");speed.oninput=()=>speedv.textContent=speed.value;
let timer=null,cur=null;
function vec(d){const s=+speed.value;return{f:[s,0],b:[-s,0],l:[s*0.4,25],r:[s*0.4,-25]}[d]}
function go(d){if(cur===d)return;halt(false);cur=d;const send=()=>{const[st,an]=vec(d);post({t:"move",stride:st,angle:an})};send();timer=setInterval(send,250)}
function halt(send=true){clearInterval(timer);timer=null;cur=null;document.querySelectorAll("[data-dir]").forEach(b=>b.classList.remove("on"));if(send)post({t:"stop"})}
document.querySelectorAll("[data-dir]").forEach(b=>{
 b.onpointerdown=e=>{b.setPointerCapture(e.pointerId);b.classList.add("on");go(b.dataset.dir)};
 b.onpointerup=b.onpointercancel=()=>halt()});
document.getElementById("stop").onclick=()=>halt();
const keys={w:"f",s:"b",a:"l",d:"r",ArrowUp:"f",ArrowDown:"b",ArrowLeft:"l",ArrowRight:"r"};
addEventListener("keydown",e=>{if(e.key===" "){halt();e.preventDefault();return}const d=keys[e.key];if(d){e.preventDefault();go(d)}});
addEventListener("keyup",e=>{if(keys[e.key]===cur)halt()});
addEventListener("blur",()=>{if(cur)halt()});
const box=document.getElementById("actions");
ACTIONS.forEach((n,i)=>{const b=document.createElement("button");b.textContent=n;b.onclick=()=>{halt(false);post({t:"action",id:i})};box.appendChild(b)});
document.querySelectorAll("[data-rgb]").forEach(b=>b.onclick=()=>{const[r,g,bl]=b.dataset.rgb.split(",");post({t:"rgb",r,g,b:bl})});
document.getElementById("beep").onclick=()=>post({t:"beep",freq:1500,ms:150});
document.getElementById("stand").onclick=()=>{halt(false);post({t:"stand"})};
async function poll(){try{const s=await (await fetch("/status")).json();
 dist.textContent=s.dist_cm==null?"–":s.dist_cm+" cm";batt.textContent=s.batt_v==null?"–":s.batt_v+" V";
 conn.textContent=(s.connected?"● ":"○ ")+s.conn;conn.style.color=s.connected?"var(--ok)":"var(--stop)";
 state.textContent=!s.connected?"offline":s.walking?"walking":"idle";if(s.event){event.textContent="⚠ "+s.event;setTimeout(()=>event.textContent="",3000);halt(false)}}
 catch(e){state.textContent="offline";conn.textContent="○ panel server not running";conn.style.color="var(--stop)"}setTimeout(poll,400)}poll();
// Camera source: the crosswalk checker (tools/walk_check.py, :8002) when it runs, else the plain YOLO stream (:8001).
const XW="http://127.0.0.1:8002",XC={"WAIT":"#e67e22","READY TO WALK":"#2ecc71","CROSSING":"#3498db","ARRIVED":"#9b59b6"};
let xwUp=null;const cam=document.getElementById("cam");
function useSrc(up){if(up===xwUp)return;xwUp=up;xw.style.display=up?"":"none";
 cam.src=up?XW+"/stream?"+Date.now():"http://localhost:8001/stream.mjpg?"+Date.now()}
cam.onerror=()=>{cam.alt=xwUp?"crosswalk stream reconnecting...":"no camera stream: run tools/walk_check.py or tools/vision_stream.py"};
xwend.onclick=()=>fetch(XW+"/end",{method:"POST"}).catch(()=>{});
async function crosswalk(){try{const j=await (await fetch(XW+"/state")).json();useSrc(true);
 xwstate.textContent="Crosswalk: "+(j.state||"...");xwstate.style.background=XC[j.state]||"#333";
 xwheld.textContent=j.held?"HELD: signal out of view, assuming the light stays":(j.why?"stopped: "+j.why:"");
 xwd.textContent=`walk ${j.walk} · don't walk ${j.stop} · conflict ${j.conflict} · streak ${j.streak} · ${j.ms} ms · drive ${j.drive?"ON":"off"}`;
 seen.textContent="camera: crosswalk checker (blind-escort model)"}
 catch(e){useSrc(false)}setTimeout(crosswalk,250)}crosswalk();
async function vision(){if(xwUp){setTimeout(vision,400);return}try{const v=await (await fetch("http://localhost:8001/detections")).json();
 seen.textContent="vision "+v.fps.toFixed(1)+" fps · "+(v.dets.length?v.dets.map(o=>o.label+" "+Math.round(o.conf*100)+"%").join(", "):"nothing detected")}
 catch(e){seen.textContent="vision stream offline (run tools/vision_stream.py)"}setTimeout(vision,400)}vision();
</script></body></html>"""


class Handler(BaseHTTPRequestHandler):
    ctl = None

    def _json(self, obj, code=200):
        data = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path == "/status":
            return self._json(self.ctl.status())
        data = PAGE.encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_POST(self):
        try:
            msg = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
            self._json(self.ctl.handle(msg))
        except Exception as e:
            self._json({"ok": False, "msg": str(e)[:200]}, 500)

    def log_message(self, *args):
        pass


def main():
    ctl = Controller()
    Handler.ctl = ctl
    server = ThreadingHTTPServer(("127.0.0.1", HTTP_PORT), Handler)
    url = "http://localhost:%d" % HTTP_PORT
    print("MechDog panel: %s  (Ctrl-C to quit)" % url, flush=True)
    webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        if ctl.dog is not None:
            ctl.dog.close()
            print("Dog stopped.")


if __name__ == "__main__":
    main()
