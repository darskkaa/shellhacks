"""Operator overrides from the terminal, kept off the Live Crossing screen: force a start and steer.

Run in a second terminal while tools/gemini_dog_agent.py --go-signal is running:
  python tools/dog_keys.py

  W        force go (start walking now, without waiting for the detector)
  A / D    steer left / right while it walks (hold the key; let go to go straight)
  S Space  stop
  R        arm / disarm detection
  Q        quit (the dog keeps its current state)

Talks to the agent's API (tools/approach_panel.py, http://127.0.0.1:8002/command) directly, so it works even if the
SafeRoute page is closed. Windows console only (msvcrt).
"""
import json
import msvcrt
import time
import urllib.request

API = "http://127.0.0.1:8002"
STEER_EVERY_S = 0.2    # the agent drops a steer it hasn't heard for 0.7 s (signal_approach.STEER_FRESH_S)
RELEASE_S = 0.6        # no A / D key-repeat for this long = key released. Covers the console's ~0.5 s delay
                       # before a held key starts repeating.


def post(path, body):
    try:
        req = urllib.request.Request(API + path, data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=1.0) as r:
            return json.loads(r.read())
    except (OSError, ValueError) as e:
        print(f"\n  agent not reachable ({e}); is gemini_dog_agent.py --go-signal running?")
        return None


def main():
    print(__doc__.split("\n\n")[2])
    steer, steer_key_at, steer_sent_at = 0, 0.0, 0.0
    while True:
        now = time.monotonic()
        while msvcrt.kbhit():
            ch = msvcrt.getwch().lower()
            if ch in ("\x00", "\xe0"):  # arrow / function key prefix: swallow the second byte
                msvcrt.getwch()
                continue
            if ch == "w":
                post("/command", {"cmd": "go"})
                print("GO")
            elif ch in ("s", " "):
                post("/command", {"cmd": "end"})
                steer = 0
                print("STOP")
            elif ch in ("a", "d"):
                d = 1 if ch == "a" else -1  # positive turns left
                if d != steer:
                    print("steer " + ("left" if d == 1 else "right"))
                    steer_sent_at = 0.0
                steer, steer_key_at = d, now
            elif ch == "r":
                try:
                    with urllib.request.urlopen(API + "/status", timeout=1.0) as r:
                        paused = json.loads(r.read())["settings"]["paused"]
                    post("/settings", {"paused": not paused})
                    print("detection " + ("ARMED" if paused else "disarmed"))
                except (OSError, ValueError, KeyError):
                    print("  couldn't read the agent's settings")
            elif ch == "q":
                if steer:
                    post("/command", {"cmd": "steer", "dir": 0})
                return
        if steer and now - steer_key_at > RELEASE_S:
            steer = 0
            post("/command", {"cmd": "steer", "dir": 0})
            print("straight")
        elif steer and now - steer_sent_at >= STEER_EVERY_S:
            post("/command", {"cmd": "steer", "dir": steer})
            steer_sent_at = now
        time.sleep(0.03)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        post("/command", {"cmd": "steer", "dir": 0})
