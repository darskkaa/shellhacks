"""
Run 3 live camera frames through Gemini 3.8 Flash Vision, 2 seconds apart.
Analyzes what the camera is seeing right now (phone screen, pedestrian signal, red light).
"""
import os
import sys
import time
import json
import base64
import urllib.request
from pathlib import Path

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

api_key = get_api_key()
if not api_key:
    print("Error: GEMINI_API_KEY not found in env or .env")
    sys.exit(1)

endpoint = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-3.8-flash:generateContent?key={api_key}"

prompt = """
You are the vision perception system for an autonomous robot dog escorting a blind person.
Look at this live camera frame:
1. What objects or scene are visible? Is there a phone/screen being held up, or a street scene?
2. Is there an illuminated WALK signal (walking person, green walk light, walk symbol)?
3. Is there an illuminated STOP / DONT WALK signal (red hand, orange hand, red traffic light)?
4. What action should the robot dog take?

Respond STRICTLY in valid JSON:
{
  "scene_description": "Brief description of what is in front of the camera",
  "walk_signal_detected": boolean,
  "stop_signal_detected": boolean,
  "signal_type": "WALK" | "STOP" | "NONE",
  "confidence": float,
  "action": "WALK" | "STOP" | "WAIT",
  "guidance_speech": "Short clear spoken guidance for the blind passenger"
}
"""

print("=" * 65)
print("  🐕 LIVE CAMERA GEMINI 3.8 FLASH ESCORT TEST (3x @ 2s interval)")
print("=" * 65)

for i in range(1, 4):
    print(f"\n📸 [Frame {i}/3] Grabbing live snapshot from Brio 105 camera (http://localhost:8001/snapshot.jpg)...")
    try:
        with urllib.request.urlopen("http://localhost:8001/snapshot.jpg", timeout=4) as resp:
            jpg_bytes = resp.read()
    except Exception as e:
        print(f"Error reading snapshot: {e}")
        time.sleep(2)
        continue

    # Save snapshot for inspection
    save_path = f"test_images/live_frame_{i}.jpg"
    with open(save_path, "wb") as f:
        f.write(jpg_bytes)

    b64_img = base64.b64encode(jpg_bytes).decode("utf-8")

    payload = {
        "contents": [{
            "parts": [
                {"text": prompt},
                {"inline_data": {"mime_type": "image/jpeg", "data": b64_img}}
            ]
        }],
        "generationConfig": {
            "temperature": 0.1,
            "responseMimeType": "application/json"
        }
    }

    t0 = time.time()
    req = urllib.request.Request(endpoint, data=json.dumps(payload).encode("utf-8"), headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=20) as r:
            res = json.loads(r.read().decode("utf-8"))
            elapsed = time.time() - t0
            raw_text = res["candidates"][0]["content"]["parts"][0]["text"]
            data = json.loads(raw_text)
            print(f"⏱️ Gemini latency: {elapsed:.2f}s | Saved: {save_path}")
            print(f"🔍 Seen: {data.get('scene_description')}")
            print(f"🚦 Signal: {data.get('signal_type')} (Walk: {data.get('walk_signal_detected')}, Stop: {data.get('stop_signal_detected')}) Conf: {data.get('confidence')}")
            print(f"🐕 Dog Action: [{data.get('action')}]")
            print(f"🗣️ Speech: \"{data.get('guidance_speech')}\"")
    except Exception as e:
        print(f"Gemini API error: {e}")

    if i < 3:
        print("⏳ Waiting 2.0s before next sample...")
        time.sleep(2.0)

print("\n" + "=" * 65)
print("Finished 3x live Gemini vision sequence.")
print("=" * 65)
