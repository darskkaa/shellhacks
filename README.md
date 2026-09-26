# shellhacks

ShellHacks 2026 project: driving a **Hiwonder MechDog** (ESP32 quadruped robot) from a laptop, with a browser control panel and live YOLO object detection from a USB webcam.

## Layout

| Path | What's in it |
|---|---|
| `robot_dump/` | Files backed up from the dog's ESP32 (MicroPython 1.21). `main.py` is our custom WiFi bridge, a JSON-over-TCP command server on port 5005. |
| `robot_dump/config.example.py` | Bridge config template: WiFi mode, SSID/password, port, safety thresholds. Copy it to `config.py` and fill in your network. |
| `tools/dog_panel.py` | Local web control panel at http://localhost:8000. Talks to the dog over WiFi, falls back to USB serial. |
| `tools/vision_stream.py` | Live YOLO26 detection on the webcam at http://localhost:8001 (MJPEG stream + JSON detections). |
| `models/` | `yolo26n.pt` weights and sample camera snapshots. |
| `hiwonder_mechdog_sdk/` | Vendored copy of Hiwonder's official SDK and examples ([Hiwonder/MechDog](https://github.com/Hiwonder/MechDog)). |
| `docs/` | Hardware research, the SDK/device reference (**read `mechdog-sdk-reference.md` first**), and ShellHacks event notes. |
| `saferoute/` | **SafeRoute Miami** (Waymo track): a separate Node web app that ranks and reroutes Miami drives around crashes, flooding, construction and hurricane surge. See [`saferoute/README.md`](saferoute/README.md). |

## SafeRoute Miami

A second ShellHacks 2026 project in this repo, independent of the MechDog code. Quick start:

```bash
cd saferoute
cp -n .env.example ../.env     # keys live in the repo-root .env (git-ignored)
npm install
npm start                      # http://localhost:3000
```

Full docs, data sources and the demo script are in [`saferoute/README.md`](saferoute/README.md) and [`saferoute/docs/`](saferoute/docs/).

## Hardware

- Hiwonder MechDog (ESP32 main controller, 8 servos, IMU, ultrasonic sonar with RGB LEDs, buzzer). Our unit has **no** onboard camera module.
- Logitech Brio 105 USB webcam on the laptop for vision.
- USB serial to the dog shows up as a CH340 port (was `COM3` for us), 115200 baud.

## Setup

```bash
pip install pyserial mpremote opencv-python ultralytics
```

### 1. Flash the bridge onto the dog

```bash
cp robot_dump/config.example.py robot_dump/config.py   # then edit it
python -m mpremote connect COM3 cp robot_dump/config.py :config.py
python -m mpremote connect COM3 cp robot_dump/main.py :main.py
python -m mpremote connect COM3 reset
```

In `config.py`, set `WIFI_MODE`:
- `"ap"`: the dog hosts its own network `MechDog_wifi` (password `12345678`). Join it from the laptop and the dog is at `192.168.4.1`.
- `"sta"`: the dog joins your network via `STA_SSID` / `STA_PASSWORD`. The panel finds it by scanning your local /24 subnet.

`main.py` blocks the REPL while it runs. Send Ctrl-C a few times over serial to get a prompt back.

> **Do not `import default_test` on the dog.** It runs the factory self-test, which crashes the servo PWM and reboots the robot.

### 2. Run the control panel

```bash
python tools/dog_panel.py                 # auto: WiFi scan, then USB fallback
python tools/dog_panel.py --wifi 192.168.4.1
python tools/dog_panel.py --serial COM3 --usb-only
```

### 3. Run the vision stream

```bash
python tools/vision_stream.py --cam 1     # 1 = Brio, 0 = laptop webcam (can be swapped on your machine)
```

- Viewer: http://localhost:8001
- Stream: http://localhost:8001/stream.mjpg
- Detections: http://localhost:8001/detections (JSON, CORS enabled)

## Bridge protocol (short version)

Newline-delimited JSON over TCP to `<dog-ip>:5005`. The full table is in `docs/mechdog-sdk-reference.md`.

```json
{"t":"move","stride":60,"angle":0}   // walk; stride -100..100, angle -30..30
{"t":"stop"}
{"t":"action","id":6}                // action group 0-15 (6 = handshake)
{"t":"height","mm":90}
{"t":"rgb","r":0,"g":255,"b":0}
{"t":"sub","hz":10}                  // telemetry: distance, battery, IMU
{"t":"hb"}                           // keep-alive
```

Safety built into the bridge:
- A `move` must be repeated, or followed by `hb`, within 1.5 s, or the watchdog stops the dog.
- Walking forward stops automatically when the sonar reads under 20 cm.
- The bridge sends `fall` / `upright` events past 50° of tilt and `low_battery` below 6.6 V.
