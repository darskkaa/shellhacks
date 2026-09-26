"""
Waymo SafePoint 3D: Gemini Autonomous Escort Agent for MechDog Quadruped.
Connects Gemini 2.5 Flash to the robot dog TCP bridge (127.0.0.1:5005)
and live YOLO vision detections (localhost:8001/detections).

Translates rider voice commands + vision perception into real-time quadruped motor actions:
  - walk(stride, angle)
  - stop()
  - speak_guidance(text)
"""

import os
import sys
import time
import json
import socket
import urllib.request
import urllib.error
import argparse
from pathlib import Path

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
    """TCP client communicating with MechDog Bridge (simulated or real robot)."""
    def __init__(self, host="127.0.0.1", port=5005):
        self.host = host
        self.port = port
        self.sock = None
        self.connect()

    def connect(self):
        try:
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.sock.settimeout(2.0)
            self.sock.connect((self.host, self.port))
            print(f"[DogClient] Connected to MechDog Bridge at {self.host}:{self.port}")
        except Exception as e:
            print(f"[DogClient Error] Could not connect to {self.host}:{self.port}: {e}")
            self.sock = None

    def send_cmd(self, cmd_dict):
        if not self.sock:
            self.connect()
        if not self.sock:
            return None
        try:
            msg = json.dumps(cmd_dict) + "\n"
            self.sock.sendall(msg.encode())
        except Exception as e:
            print(f"[DogClient Error] Send failed: {e}")
            self.sock = None

    def move(self, stride=50, angle=0):
        # stride: -100..100, angle: -30..30
        stride = max(-100, min(100, stride))
        angle = max(-30, min(30, angle))
        print(f"🐕 [EXECUTE MOVE] Stride: {stride} | Angle: {angle}°")
        self.send_cmd({"t": "move", "stride": stride, "angle": angle})

    def stop(self):
        print("🛑 [EXECUTE STOP]")
        self.send_cmd({"t": "stop"})

    def heartbeat(self):
        self.send_cmd({"t": "hb"})

    def set_rgb(self, r, g, b):
        self.send_cmd({"t": "rgb", "r": r, "g": g, "b": b})

    def close(self):
        if self.sock:
            self.sock.close()
            self.sock = None


class GeminiDogAgent:
    def __init__(self, api_key: str, dog_host="127.0.0.1", dog_port=5005, vision_url="http://localhost:8001/detections"):
        self.api_key = api_key
        self.dog = MechDogBridgeClient(host=dog_host, port=dog_port)
        self.vision_url = vision_url
        self.gemini_endpoint = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={self.api_key}"

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

Respond STRICTLY in JSON format with this exact schema:
{
  "action": "walk" | "stop",
  "stride": integer (-100 to 100),
  "angle": integer (-30 to 30),
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
                req = urllib.request.Request(
                    self.gemini_endpoint,
                    data=json.dumps(payload).encode("utf-8"),
                    headers={"Content-Type": "application/json"},
                    method="POST"
                )
                with urllib.request.urlopen(req, timeout=5.0) as response:
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
        print(f"🤖 [Gemini Decision] Action: {decision.get('action')} | Stride: {decision.get('stride')} | Angle: {decision.get('angle')}°")
        print(f"🗣️  [Voice Guidance] \"{decision.get('speech')}\"")

        # 3. Execute on dog
        if decision.get("action") == "walk":
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
    parser.add_argument("--goal", type=str, default="Guide me safely to the Waymo passenger door", help="Rider goal prompt")
    parser.add_argument("--once", action="store_true", help="Run a single step instead of continuous loop")
    args = parser.parse_args()

    api_key = args.key or get_api_key()
    if not api_key:
        print("\nℹ️  GEMINI_API_KEY not found; using local Google Antigravity (agy CLI) fallback.")

    agent = GeminiDogAgent(api_key=api_key, dog_host=args.host, dog_port=args.port)
    print(f"\n🐾 [MechDog Gemini Brain Active]")
    print(f"Target Bridge: {args.host}:{args.port}")
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
