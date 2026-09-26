"""
Waymo SafePoint 3D: Gemini Autonomous Escort Agent for MechDog Quadruped.
Connects Gemini 2.5 Flash to the robot dog bridge (TCP 127.0.0.1:5005 or USB serial via --serial)
and live YOLO vision detections (localhost:8001/detections).

Translates rider voice commands + vision perception into real-time quadruped motor actions:
  - walk(stride, angle)
  - stop()
  - set_speed_mode(normal | brisk | sprint) -> bridge gait cadence
  - speak_guidance(text)
"""

import os
import sys
import time
import json
import urllib.request
import urllib.error
import argparse
from pathlib import Path

from real_world_escort import BridgeLink, RealDogController

GAIT_PRESETS = {
    "normal": RealDogController.GAIT_DEFAULT,
    "brisk": RealDogController.GAIT_FAST,
    "sprint": RealDogController.GAIT_SPRINT,
}
# Faster gaits shorten reaction distance, so any of these in view forces the default gait.
SLOW_DOWN_CLASSES = {"curb_drop_off_hazard", "ped_signal_stop", "conflict_vehicle_cyclist"}

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

class MechDogBridgeClient:
    """Client for the MechDog Bridge (simulated or real robot) over TCP or USB serial, auto-reconnecting."""
    def __init__(self, host="127.0.0.1", port=5005, serial_port=None):
        self.gait = None
        self.link = BridgeLink(host, port, serial_port, on_ready=self._on_ready, tag="DogClient")
        self.link.start()

    def _on_ready(self, first):
        self.gait = None  # bridge may have rebooted or been reset; force resend

    def send_cmd(self, cmd_dict):
        if not self.link.send(cmd_dict):
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
        self.send_cmd({"t": "rgb", "r": r, "g": g, "b": b})

    def close(self):
        self.link.close()


class GeminiDogAgent:
    def __init__(self, api_key: str = None, dog_host="127.0.0.1", dog_port=5005, vision_url="http://localhost:8001/detections",
                 dog_serial=None):
        self.api_key = api_key or get_api_key()
        self.dog = MechDogBridgeClient(host=dog_host, port=dog_port, serial_port=dog_serial)
        self.vision_url = vision_url
        self.gemini_endpoint = "https://generativelanguage.googleapis.com/v1beta/models/gemini-3.8-flash:generateContent"

    def fetch_vision_detections(self):
        try:
            req = urllib.request.Request(self.vision_url, headers={"User-Agent": "GeminiDogAgent/1.0"})
            with urllib.request.urlopen(req, timeout=1.0) as resp:
                data = json.loads(resp.read().decode())
                return data.get("detections", [])
        except Exception:
            return []

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
        prompt = f"""{system_instruction}

RIDER GOAL: "{rider_goal}"
CURRENT VISION DETECTIONS: {json.dumps(detected_objects)}

Decide the immediate quadruped motion and spoken escort guidance. Output raw valid JSON only:"""

        # Method 1: Direct Gemini API key
        if self.api_key:
            payload = {
                "contents": [{
                    "parts": [{"text": prompt}]
                }],
                "generationConfig": {
                    "temperature": 0.2,
                    "responseMimeType": "application/json"
                }
            }
            try:
                import ssl
                ctx = ssl.create_default_context()
                for ca in ['/etc/ssl/certs/ca-certificates.crt', '/etc/pki/tls/certs/ca-bundle.crt', '/etc/ssl/cert.pem']:
                    if os.path.exists(ca):
                        ctx.load_verify_locations(ca)
                        break
                try:
                    import certifi
                    ctx.load_verify_locations(certifi.where())
                except Exception:
                    pass

                req = urllib.request.Request(
                    self.gemini_endpoint,
                    data=json.dumps(payload).encode("utf-8"),
                    headers={
                        "Content-Type": "application/json",
                        "x-goog-api-key": self.api_key
                    },
                    method="POST"
                )
                with urllib.request.urlopen(req, timeout=8.0, context=ctx) as response:
                    result = json.loads(response.read().decode("utf-8"))
                    text_response = result["candidates"][0]["content"]["parts"][0]["text"]
                    return json.loads(text_response)
            except Exception as e:
                print(f"[Gemini API Error] {e}")

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

        # Fallback deterministic safety rule
        return {
            "action": "stop",
            "stride": 0,
            "angle": 0,
            "speed_mode": "normal",
            "led_color": [255, 0, 0],
            "speech": "Obstacle scan active. Standing by safely."
        }

    def run_escort_step(self, rider_goal: str):
        # 1. Fetch latest YOLO detections
        detections = self.fetch_vision_detections()
        print(f"\n--- [Perception Update] {len(detections)} objects detected ---")
        for d in detections[:3]:
            print(f"  • {d.get('class')} (conf: {d.get('confidence')})")

        # 2. Get Gemini autonomous decision
        decision = self.decide_action(rider_goal, detections)
        speed_mode = decision.get("speed_mode", "normal")
        if speed_mode not in GAIT_PRESETS or any(d.get("class") in SLOW_DOWN_CLASSES for d in detections):
            speed_mode = "normal"
        decision["speed_mode"] = speed_mode
        print(f"🤖 [Gemini Decision] Action: {decision.get('action')} | Stride: {decision.get('stride')} | Angle: {decision.get('angle')}° | Speed: {speed_mode}")
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
    parser.add_argument("--serial", metavar="DEV", help="Drive over USB serial instead of WiFi (e.g. /dev/ttyUSB0, COM3)")
    parser.add_argument("--goal", type=str, default="Guide me safely to the Waymo passenger door", help="Rider goal prompt")
    parser.add_argument("--once", action="store_true", help="Run a single step instead of continuous loop")
    args = parser.parse_args()

    api_key = args.key or get_api_key()
    if not api_key:
        print("\nℹ️  GEMINI_API_KEY not found; using local Google Antigravity (agy CLI) fallback.")

    agent = GeminiDogAgent(api_key=api_key, dog_host=args.host, dog_port=args.port, dog_serial=args.serial)
    print(f"\n🐾 [MechDog Gemini Brain Active]")
    print(f"Target Bridge: {args.serial or f'{args.host}:{args.port}'}")
    print(f"Active Escort Mission: \"{args.goal}\"\n")

    try:
        if args.once:
            agent.run_escort_step(args.goal)
        else:
            while True:
                agent.run_escort_step(args.goal)
                time.sleep(2.0)
    except KeyboardInterrupt:
        print("\nStopping dog and disconnecting...")
        agent.dog.stop()
        agent.dog.close()

if __name__ == "__main__":
    main()
