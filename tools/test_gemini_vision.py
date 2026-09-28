"""
Test Gemini Vision API on pedestrian crossing scenes for autonomous guide dog decision.
"""
import os
import sys
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

endpoint = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={api_key}"

prompt = """
You are the AI brain of an autonomous guide dog robot for a blind passenger at a street crosswalk.
Analyze this camera image:
1. Is there an active illuminated WALK signal (walking person silhouette / green signal)?
2. Is there an active illuminated STOP / DONT WALK signal (red hand / orange hand / red light)?
3. Is it safe to cross right now?

Respond ONLY in valid JSON matching this schema:
{
  "can_walk": boolean,
  "signal_state": "WALK" | "STOP" | "NONE",
  "confidence": float,
  "reasoning": string,
  "guidance_speech": "Short spoken message for the blind rider"
}
"""

test_files = [
    ("Walk Signal (Green/White)", "test_images/green_walk_crossing.jpg"),
    ("Stop Signal (Don't Walk Hand)", "test_images/dont_walk_hand_real.jpg"),
    ("Red Street Light", "test_images/street_red_light.jpg")
]

for title, path in test_files:
    if not os.path.exists(path):
        continue
    print(f"\n{'='*60}\nQuerying Gemini Vision on: {title} ({path})\n{'='*60}")
    
    with open(path, "rb") as f:
        img_b64 = base64.b64encode(f.read()).decode("utf-8")

    payload = {
        "contents": [{
            "parts": [
                {"text": prompt},
                {"inlineData": {"mimeType": "image/jpeg", "data": img_b64}}
            ]
        }],
        "generationConfig": {
            "temperature": 0.1,
            "responseMimeType": "application/json"
        }
    }

    req = urllib.request.Request(
        endpoint,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"}
    )

    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            text = data["candidates"][0]["content"]["parts"][0]["text"]
            decision = json.loads(text)
            print(json.dumps(decision, indent=2))
    except Exception as e:
        print(f"Gemini API error: {e}")
