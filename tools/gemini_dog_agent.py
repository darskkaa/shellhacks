"""
Waymo SafePoint 3D: Gemini Autonomous Escort Agent for MechDog Quadruped.
Connects Gemini Flash-Lite (rate limited, see GeminiClient) to the robot dog bridge (TCP 127.0.0.1:5005 or USB serial via --serial)
and live YOLO vision detections (localhost:8001/detections).

Translates rider voice commands + vision perception into real-time quadruped motor actions:
  - walk(stride, angle)
  - stop()
  - set_speed_mode(normal | brisk | sprint) -> bridge gait cadence
  - speak_guidance(text)
"""

import os
import sys
import ssl
import base64
import time
import json
import http.client
import argparse
from collections import deque
from pathlib import Path

from real_world_escort import (BridgeLink, RealDogController, STOP_CLASSES, VISION_MIN_CONF, det_conf, det_label,
                               fetch_detections, seen_labels)

# Fastest models measured on this key (~0.6-0.8 s per decision); later entries take the overflow.
GEMINI_MODELS = ("gemini-3.5-flash-lite", "gemini-3.1-flash-lite", "gemini-flash-lite-latest")
# Free tier allows 15 requests/min per model per project; stay one under so we never see a 429.
GEMINI_RPM_PER_MODEL = 14
GEMINI_HOST = "generativelanguage.googleapis.com"
# Same scene (labels + coarse position) within this window reuses the last decision instead of calling Gemini.
DECISION_TTL_S = 4.0
# A decision older than this is never replayed while Gemini is unavailable; the dog stops instead.
STALE_DECISION_S = 6.0

GAIT_PRESETS = {
    "normal": RealDogController.GAIT_DEFAULT,
    "brisk": RealDogController.GAIT_FAST,
    "sprint": RealDogController.GAIT_SPRINT,
}
# Faster gaits shorten reaction distance, so any of these in view forces the default gait.
SLOW_DOWN_CLASSES = STOP_CLASSES

# Load GEMINI_API_KEY from environment or repo root .env
def get_api_key():
    key = os.environ.get("GEMINI_API_KEY")
    if key:
        return key
    env_file = Path(__file__).resolve().parent.parent / ".env"
    if env_file.exists():
        with open(env_file) as f:
            for line in f:
                if line.startswith("GEMINI_API_KEY="):
                    return line.strip().split("=", 1)[1].strip("'\"")
    return None


def get_api_keys():
    """Every Gemini key configured: GEMINI_API_KEY, then GEMINI_API_KEY_2 .. GEMINI_API_KEY_9 (environment first, then
    .env). Free-tier quota (15 requests/min, 500/day per model) is per Google Cloud project, so extra keys only add
    capacity when each comes from a different project."""
    names = ["GEMINI_API_KEY"] + [f"GEMINI_API_KEY_{i}" for i in range(2, 10)]
    found = {n: os.environ[n] for n in names if os.environ.get(n)}
    env_file = Path(__file__).resolve().parent.parent / ".env"
    if env_file.exists():
        with open(env_file) as f:
            for line in f:
                name, _, value = line.strip().partition("=")
                if name in names and value and name not in found:
                    found[name] = value.strip("'\"")
    keys = []
    for n in names:
        if n in found and found[n] not in keys:
            keys.append(found[n])
    return keys


def _ssl_context():
    try:
        import certifi
        return ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        return ssl.create_default_context()


def _retry_delay_s(body):
    """Server-requested wait from a 429 body's RetryInfo ("12s"), else None."""
    for detail in (body.get("error") or {}).get("details") or []:
        delay = detail.get("retryDelay")
        if delay:
            try:
                return float(delay.rstrip("s"))
            except ValueError:
                pass
    return None


class GeminiClient:
    """Rate-limited Gemini JSON client over one keep-alive HTTPS connection.
    Calls are spaced at least min_interval apart and each model gets at most rpm_per_model calls in any 60 s
    window, so the fastest model is used until its budget runs out and the next one takes the overflow.
    A 429 or 5xx anyway benches that model (for the server's retryDelay on 429)."""

    def __init__(self, api_key, min_interval=1.0, rpm_per_model=GEMINI_RPM_PER_MODEL, models=GEMINI_MODELS,
                 timeout=5.0):
        self.api_key = api_key
        self.min_interval = min_interval
        self.rpm_per_model = rpm_per_model
        self.sent = {m: deque() for m in models}
        self.last_error = None  # why the last request failed ("daily quota used up", "HTTP 503"); None after a reply
        self.models = models
        self.timeout = timeout
        self.ctx = _ssl_context()
        self.conn = None
        self.next_call = 0.0
        self.benched_until = {}

    def _close(self):
        if self.conn:
            self.conn.close()
            self.conn = None

    def _post(self, model, payload):
        # Retry once on a fresh connection: the server drops idle keep-alive sockets.
        for attempt in range(2):
            if self.conn is None:
                self.conn = http.client.HTTPSConnection(GEMINI_HOST, context=self.ctx, timeout=self.timeout)
            try:
                self.conn.request("POST", f"/v1beta/models/{model}:generateContent", body=payload,
                                  headers={"Content-Type": "application/json", "x-goog-api-key": self.api_key})
                resp = self.conn.getresponse()
                return resp.status, json.loads(resp.read() or b"{}")
            except (OSError, http.client.HTTPException):
                self._close()
                if attempt:
                    raise

    def generate_json(self, prompt, jpeg=None):
        """Parsed JSON reply, or None when throttled locally, every model is benched, or all calls failed.
        jpeg (bytes, or a list of them) is sent alongside the prompt as images, in order."""
        now = time.monotonic()
        if now < self.next_call:
            return None
        self.next_call = now + self.min_interval
        images = [] if jpeg is None else [jpeg] if isinstance(jpeg, (bytes, bytearray)) else list(jpeg)
        payload = json.dumps({
            "contents": [{"parts": [{"inline_data": {"mime_type": "image/jpeg", "data": base64.b64encode(j).decode()}}
                                    for j in images] + [{"text": prompt}]}],
            "generationConfig": {"temperature": 0.2, "maxOutputTokens": 256, "responseMimeType": "application/json"},
        }).encode("utf-8")
        for model in self.models:
            sent = self.sent[model]
            while sent and now - sent[0] > 60.0:
                sent.popleft()
            if len(sent) >= self.rpm_per_model or now < self.benched_until.get(model, 0):
                continue
            sent.append(now)
            try:
                t0 = time.monotonic()
                status, body = self._post(model, payload)
            except (OSError, http.client.HTTPException) as e:
                print(f"[Gemini {model}] connection error: {e}")
                continue
            if status == 200:
                self.last_error = None
                try:
                    text = body["candidates"][0]["content"]["parts"][0]["text"]
                    decision = json.loads(text)
                except (KeyError, IndexError, ValueError) as e:
                    print(f"[Gemini {model}] unparseable reply: {e}")
                    continue
                print(f"[Gemini {model}] {time.monotonic() - t0:.2f}s")
                return decision
            # 402 = the key's prepaid credit is empty: it won't come back in seconds, so stop hammering it.
            wait = (_retry_delay_s(body) or 15.0) if status == 429 else 600.0 if status == 402 else 5.0
            self.benched_until[model] = time.monotonic() + wait
            lines = ((body.get("error") or {}).get("message") or "").strip().splitlines()
            # The quota 429 names the exhausted limit on a "Quota exceeded for metric" line; prefer that.
            reason = next((l for l in lines if "metric" in l), lines[0] if lines else "")
            print(f"[Gemini {model}] HTTP {status}, benched {wait:.0f}s: {reason.strip()[:200]}")
            self.last_error = ("daily quota used up" if status == 429 and "PerDay" in json.dumps(body)
                               or "limit: 500" in reason else "prepaid credit empty (AI Studio billing)"
                               if status == 402 else f"HTTP {status}")
        return None

class MechDogBridgeClient:
    """Client for the MechDog Bridge (simulated or real robot) over TCP or USB serial, auto-reconnecting."""
    def __init__(self, host="127.0.0.1", port=5005, serial_port=None, telemetry_hz=0):
        self.gait = None
        self.rgb = None
        self.dist_cm = None
        self.dist_at = 0.0
        self.dist_history = deque(maxlen=40)  # ~4 s at 10 Hz telemetry (stall detection looks back 2 s)
        self.mode = None            # bridge mode from telemetry: idle / walk / action / safe
        self.imu_ang = None
        self.last_move, self.last_hb = None, 0.0  # move de-duplication (send_cmd)
        self.walking_stride = None
        self.problem, self.problem_at = None, 0.0  # last refusal or safety event from the dog  # (monotonic time, cm or None), newest last
        self.batt_v = None
        self.telemetry_hz = telemetry_hz
        self.link = BridgeLink(host, port, serial_port, on_msg=self._on_msg, on_ready=self._on_ready, tag="DogClient")
        self.link.start()

    def _on_ready(self, first):
        self.gait = self.rgb = self.last_move = None  # bridge may have rebooted or been reset; force resend
        if self.telemetry_hz:
            self.link.send({"t": "sub", "hz": self.telemetry_hz})

    def _on_msg(self, msg):
        if msg.get("t") == "tel":
            d = msg.get("dist_cm")
            self.dist_cm = float(d) if isinstance(d, (int, float)) else None
            self.dist_at = time.monotonic()
            self.dist_history.append((self.dist_at, self.dist_cm))
            b = msg.get("batt_v")
            self.batt_v = float(b) if isinstance(b, (int, float)) else None
            self.mode, self.walking_stride = msg.get("mode"), msg.get("stride")  # what the dog is really doing
            self.imu_ang = msg.get("ang")  # [roll, pitch, ...] degrees from the dog's IMU
        elif msg.get("t") == "ack" and msg.get("ok") is False:
            if msg.get("for") == "move":
                self.last_move = None  # refused (e.g. battery dip): send the move again next tick, don't heartbeat it
            # The bridge refused a command (low_battery, fallen, ...): the walk isn't physically happening.
            self.problem, self.problem_at = f"{msg.get('for')} refused: {msg.get('msg')}", time.monotonic()
            print(f"[DogClient] {self.problem}")
        elif msg.get("t") == "event":
            self.problem, self.problem_at = msg.get("name"), time.monotonic()
            print(f"[DogClient] dog event: {msg.get('name')} {msg.get('reason') or ''}")

    def send_cmd(self, cmd_dict):
        # Hiwonder's examples call move(stride, angle) once and let the gait run. Re-sending the same move 8x/s
        # restarted the step cycle each time, so the dog shuffled in place instead of walking (2026-09-27). A repeat
        # of the current move becomes a heartbeat, which only feeds the bridge's 1.5 s watchdog.
        now = time.monotonic()
        if cmd_dict.get("t") == "move":
            key = (cmd_dict.get("stride"), cmd_dict.get("angle"))
            if key == self.last_move and self.link.ready:
                if now - self.last_hb < 0.4:
                    return True
                self.last_hb = now
                cmd_dict = {"t": "hb"}
            else:
                self.last_move, self.last_hb = key, now
        elif cmd_dict.get("t") in ("stop", "action", "reset", "gait", "posture", "height"):
            self.last_move = None  # anything that changes the legs means the next move must really be sent
        if not self.link.send(cmd_dict):
            self.last_move = None
            print(f"[DogClient Error] Bridge offline, dropped {cmd_dict.get('t')}")
            return False
        return True

    def move(self, stride=50, angle=0):
        # stride: -100..100, angle: -30..30
        stride = max(-100, min(100, stride))
        angle = max(-30, min(30, angle))
        print(f"🐕 [EXECUTE MOVE] Stride: {stride} | Angle: {angle}°")
        self.send_cmd({"t": "move", "stride": stride, "angle": angle})

    def set_gait(self, lift_ms, contact_ms, lift_mm):
        gait = (lift_ms, contact_ms, lift_mm)
        if gait == self.gait:
            return
        print(f"⚙️  [EXECUTE GAIT] lift {lift_ms} ms / contact {contact_ms} ms / lift {lift_mm} mm")
        sent = self.send_cmd({"t": "gait", "lift_ms": lift_ms, "contact_ms": contact_ms, "lift_mm": lift_mm})
        self.gait = gait if sent else None

    def set_speed_mode(self, mode):
        self.set_gait(*GAIT_PRESETS.get(mode, GAIT_PRESETS["normal"]))

    def stop(self):
        print("🛑 [EXECUTE STOP]")
        self.send_cmd({"t": "stop"})

    def heartbeat(self):
        self.send_cmd({"t": "hb"})

    def set_rgb(self, r, g, b):
        # Only on change: each command costs ~130 ms per 20-byte chunk over Bluetooth.
        if (r, g, b) == self.rgb:
            return
        self.rgb = (r, g, b) if self.send_cmd({"t": "rgb", "r": r, "g": g, "b": b}) else None

    def close(self):
        self.link.close()


class GeminiDogAgent:
    def __init__(self, api_key: str = None, dog_host="127.0.0.1", dog_port=5005, vision_url="http://127.0.0.1:8001/detections",
                 dog_serial=None, gemini_interval=1.0, gemini_rpm=GEMINI_RPM_PER_MODEL, telemetry_hz=0):
        self.api_key = api_key or get_api_key()
        self.dog = MechDogBridgeClient(host=dog_host, port=dog_port, serial_port=dog_serial, telemetry_hz=telemetry_hz)
        self.vision_url = vision_url
        self.gemini = GeminiClient(self.api_key, min_interval=gemini_interval, rpm_per_model=gemini_rpm) if self.api_key else None
        self.last_decision = None
        self.last_scene = None
        self.last_decision_at = 0.0

    @staticmethod
    def scene_key(detections):
        """Labels plus left/center/right bucket; decisions are reused while this is unchanged."""
        return tuple(sorted((det_label(d), round(float(d.get("cx", 0.5)) * 2))
                            for d in detections if det_conf(d) >= VISION_MIN_CONF))

    def fetch_vision_detections(self):
        return fetch_detections(self.vision_url)

    def reactive_decision(self, detections):
        """Signal-only rules that skip the LLM round trip; None when no signal is in view."""
        labels = seen_labels(detections)
        if labels & STOP_CLASSES:
            return {"action": "stop", "stride": 0, "angle": 0, "speed_mode": "normal", "led_color": [255, 0, 0],
                    "speech": "Stop. Signal or hazard ahead, holding here."}
        if "ped_signal_walk" in labels:
            return {"action": "walk", "stride": 100, "angle": 0, "speed_mode": "sprint", "led_color": [0, 255, 0],
                    "speech": "Walk signal is on. Crossing now, stay with me."}
        return None

    def decide_action(self, rider_goal: str, detected_objects: list):
        """Query Gemini API or agy CLI with system prompt & vision detections to decide quadruped action."""
        system_instruction = """
You are the AI brain of an autonomous guide dog robot (Hiwonder MechDog) assisting a blind passenger boarding a Waymo autonomous vehicle.
You receive:
1. The passenger's goal (e.g., "Walk me to the car", "Find the curb ramp", "Stop").
2. Real-time vision detections from your camera (classes: curb_ramp_ada, curb_drop_off_hazard, waymo_door_handle, ped_signal_walk, ped_signal_stop, conflict_vehicle_cyclist, etc.).

Safety Rules:
- If curb_drop_off_hazard is directly ahead, STOP immediately and redirect toward curb_ramp_ada.
- If ped_signal_stop or conflict_vehicle_cyclist is detected, STOP immediately.
- If curb_ramp_ada is detected ahead, align dog angle toward ramp center and walk forward (stride 40 to 60).
- If waymo_door_handle is detected, guide passenger directly to handle touchpoint.

Speed Rules (speed_mode sets gait cadence: normal 2.0, brisk 4.0, sprint 5.0 steps/s):
- "sprint": only while crossing a street on ped_signal_walk with no conflict_vehicle_cyclist, stride 80 to 100.
- "brisk": clear, hazard-free path approaching the waymo_door_handle.
- "normal": everything else, including near any curb edge, ramp alignment, turns, or uncertainty.

Respond STRICTLY in JSON format with this exact schema:
{
  "action": "walk" | "stop",
  "stride": integer (-100 to 100),
  "angle": integer (-30 to 30),
  "speed_mode": "normal" | "brisk" | "sprint",
  "led_color": [r, g, b],
  "speech": "Natural spoken sentence guiding the visually impaired rider"
}
"""
        compact = [{"label": det_label(d), "conf": round(det_conf(d), 2), "cx": d.get("cx")} for d in detected_objects]
        prompt = f"""{system_instruction}

RIDER GOAL: "{rider_goal}"
CURRENT VISION DETECTIONS (cx: 0 = far left, 1 = far right): {json.dumps(compact)}

Decide the immediate quadruped motion and spoken escort guidance. Output raw valid JSON only:"""

        # Method 1: Direct Gemini API key (None while rate limited, so the caller can replay or stop)
        if self.gemini:
            return self.gemini.generate_json(prompt)

        # Method 2: Fallback to agy CLI (Local Google Antigravity Auth)
        try:
            import subprocess
            cmd = ["agy", "-p", f"{prompt}\nReturn ONLY JSON."]
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=8.0)
            if proc.returncode == 0:
                raw = proc.stdout.strip()
                # Clean markdown fences or canary
                if "```json" in raw:
                    raw = raw.split("```json")[1].split("```")[0].strip()
                elif "```" in raw:
                    raw = raw.split("```")[1].split("```")[0].strip()
                lines = [l for l in raw.split("\n") if not l.startswith("🌶️")]
                clean_json = "\n".join(lines).strip()
                return json.loads(clean_json)
        except Exception as e:
            print(f"[agy Fallback Error] {e}")

        return None

    def gemini_decision(self, rider_goal, detections):
        """(decision, source). Gemini is only called when the scene changed or the last answer aged out; while it
        is rate limited the last answer is replayed, and once that goes stale the dog stops."""
        now = time.monotonic()
        scene = self.scene_key(detections)
        age = now - self.last_decision_at
        if self.last_decision and scene == self.last_scene and age < DECISION_TTL_S:
            return dict(self.last_decision), "Cached"
        decision = self.decide_action(rider_goal, detections)
        if decision:
            self.last_decision, self.last_scene, self.last_decision_at = decision, scene, now
            return dict(decision), "Gemini"
        if self.last_decision and age < STALE_DECISION_S:
            replay = dict(self.last_decision)
            if seen_labels(detections) & STOP_CLASSES:
                replay.update(action="stop", led_color=[255, 0, 0], speech="Stop. Hazard ahead, holding here.")
            return replay, "Replay"
        return {"action": "stop", "stride": 0, "angle": 0, "speed_mode": "normal", "led_color": [255, 0, 0],
                "speech": "Obstacle scan active. Standing by safely."}, "Fallback"

    def run_escort_step(self, rider_goal: str, reactive=False):
        # 1. Fetch latest YOLO detections
        detections = self.fetch_vision_detections()
        print(f"\n--- [Perception Update] {len(detections)} objects detected ---")
        for d in detections[:3]:
            print(f"  • {det_label(d)} (conf: {det_conf(d):.2f})")

        # 2. Reactive signal rules first, else Gemini autonomous decision
        decision = self.reactive_decision(detections) if reactive else None
        source = "Reactive" if decision else "Gemini"
        if decision is None:
            decision, source = self.gemini_decision(rider_goal, detections)
        speed_mode = decision.get("speed_mode", "normal")
        if speed_mode not in GAIT_PRESETS or seen_labels(detections, min_conf=0) & SLOW_DOWN_CLASSES:
            speed_mode = "normal"
        decision["speed_mode"] = speed_mode
        print(f"🤖 [{source} Decision] Action: {decision.get('action')} | Stride: {decision.get('stride')} | Angle: {decision.get('angle')}° | Speed: {speed_mode}")
        print(f"🗣️  [Voice Guidance] \"{decision.get('speech')}\"")

        # 3. Execute on dog
        if decision.get("action") == "walk":
            self.dog.set_speed_mode(speed_mode)
            self.dog.move(stride=decision.get("stride", 50), angle=decision.get("angle", 0))
        else:
            self.dog.stop()

        rgb = decision.get("led_color", [0, 255, 0])
        self.dog.set_rgb(rgb[0], rgb[1], rgb[2])

        return decision


def main():
    parser = argparse.ArgumentParser(description="Run Gemini Autonomous MechDog Escort Agent")
    parser.add_argument("--key", type=str, default=None, help="Gemini API Key")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="MechDog Bridge IP/Host")
    parser.add_argument("--port", type=int, default=5005, help="MechDog Bridge TCP Port")
    parser.add_argument("--serial", metavar="DEV", help="Drive over USB serial instead of WiFi (e.g. /dev/ttyUSB0, COM3), or Bluetooth with \"ble\" / \"ble:NAME\"")
    parser.add_argument("--goal", type=str, default="Guide me safely to the Waymo passenger door", help="Rider goal prompt")
    parser.add_argument("--once", action="store_true", help="Run a single step instead of continuous loop")
    parser.add_argument("--gemini-interval", type=float, default=1.0, metavar="S",
                        help="Minimum seconds between Gemini requests across all models")
    parser.add_argument("--gemini-rpm", type=int, default=GEMINI_RPM_PER_MODEL, metavar="N",
                        help="Max requests per minute per model (free tier is 15; raise on a paid key)")
    parser.add_argument("--go-signal", action="store_true",
                        help="Wait for a GO / WALK signal (Gemini, with YOLO26 as backup), then walk straight at full "
                             "speed until the sonar says it's there (see signal_approach.py)")
    parser.add_argument("--no-panel", action="store_true",
                        help="--go-signal: don't serve the control API that SafeRoute's crossing page uses")
    parser.add_argument("--reactive", action="store_true",
                        help="Act on ped_signal_walk / stop signals and hazards immediately, skipping Gemini")
    args = parser.parse_args()

    api_key = args.key or get_api_key()
    if not api_key:
        print("\nℹ️  GEMINI_API_KEY not found; using local Google Antigravity (agy CLI) fallback.")

    agent = GeminiDogAgent(api_key=api_key, dog_host=args.host, dog_port=args.port, dog_serial=args.serial,
                           gemini_interval=args.gemini_interval, gemini_rpm=args.gemini_rpm,
                           telemetry_hz=10 if args.go_signal else 0)
    print(f"\n🐾 [MechDog Gemini Brain Active]")
    print(f"Target Bridge: {args.serial or f'{args.host}:{args.port}'}")
    print(f"Active Escort Mission: \"{args.goal}\"\n")

    try:
        if args.go_signal:
            if not agent.gemini:
                sys.exit("--go-signal needs GEMINI_API_KEY")
            from signal_approach import SignalApproach
            panel = None
            if not args.no_panel:
                from approach_panel import ApproachPanel
                panel = ApproachPanel()
            SignalApproach(agent, panel=panel,
                           rpm_per_model=None if args.gemini_rpm == GEMINI_RPM_PER_MODEL else args.gemini_rpm).run()
        elif args.once:
            agent.run_escort_step(args.goal, reactive=args.reactive)
        else:
            while True:
                agent.run_escort_step(args.goal, reactive=args.reactive)
                time.sleep(0.25)  # Gemini calls are paced by GeminiClient, not the loop
    except KeyboardInterrupt:
        print("\nStopping dog and disconnecting...")
        agent.dog.stop()
        agent.dog.close()

if __name__ == "__main__":
    main()
