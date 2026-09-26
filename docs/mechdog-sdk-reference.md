# MechDog — Device & SDK Reference (pulled live from the robot, 2026-09-26)

## Connection
- USB serial: **COM3** (CH340 USB-serial, 115200). Tool: `python -m mpremote connect COM3 ...`
- Firmware: **MicroPython 1.21.0** ("MechDog with ESP32", build 2025-05-05). Plain ESP32 main controller.
- The running `main.py` blocks the REPL. To get a prompt, send Ctrl-C a few times over serial (see `/tmp/grab.py` pattern), then use `mpremote ... resume <cmd>`. `soft-reset` restarts the bridge.
- **Never `import default_test`**: it runs a factory self-test that crashes the servo PWM and reboots the dog.

## Files on the robot (backed up in `robot_dump/`)
| File | What it is |
|---|---|
| `boot.py` | Stock, empty (webrepl commented out). |
| `config.py` | Custom bridge config: WiFi AP `MechDog_wifi` / `12345678`, TCP port **5005**, watchdog 1500 ms, obstacle stop at 20 cm, low battery 6.6 V, fall at 50°. |
| `main.py` | **Custom "mechdog-bridge-0.1"**: JSON-over-TCP command server, written so an Arduino UNO Q (or any client) can drive the dog. |

## Attached hardware (I2C scan)
- Bus 1: `0x68` IMU (MPU6050/QMI8658), `0x77` glowing ultrasonic sonar (distance + 2 RGB LEDs).
- Bus 2: **empty. No ESP32-S3 camera module attached.** All vision comes from the laptop webcam.
- Also on board: buzzer, battery-voltage ADC.

## PC-side camera
- Logitech **Brio 105** via USB. OpenCV sees 640×480 at index 0 or 1; the other index is the laptop's integrated webcam. Use `cv2.VideoCapture(i, cv2.CAP_DSHOW)`.

## Bridge protocol (what `main.py` accepts)
Newline-delimited JSON over TCP to `192.168.4.1:5005` (laptop must join WiFi `MechDog_wifi`).
On connect the bridge sends: `{"t":"hello","proto":1,"fw":"mechdog-bridge-0.1","caps":{...}}`

| Send | Effect / limits |
|---|---|
| `{"t":"move","stride":S,"angle":A}` | Walk. stride −100..100 (mm, + is forward), angle −30..30 (+ is left). **Must resend or send `hb` within 1.5 s or the watchdog stops it.** |
| `{"t":"stop"}` / `{"t":"reset"}` | Stop. `reset` also restores height 80 and the default gait. |
| `{"t":"action","id":0-15}` | Action group, by index into: left_foot_kick, right_foot_kick, stand_four_legs, sit_dowm, go_prone, stand_two_legs, handshake, scrape_a_bow, nodding_motion, boxing, stretch_oneself, pee, press_up, rotation_pitch, rotation_roll, normal_attitude |
| `{"t":"height","mm":50-120}` | Body height (via `transform`). |
| `{"t":"posture","x":±20,"y":±20}` | Roll/pitch tilt (clamped to ±17°). |
| `{"t":"gait","lift_ms":..,"contact_ms":..,"lift_mm":..}` | Gait tuning. |
| `{"t":"rgb","r":..,"g":..,"b":..}` | Both sonar LEDs. |
| `{"t":"buzzer","freq":..,"ms":..}` | Beep. |
| `{"t":"sub","hz":0-50}` | Telemetry stream: `{"t":"tel","dist_cm","batt_v","imu","ang","mode",...}` |
| `{"t":"ping"}` / `{"t":"hb"}` | Latency check / keep-alive. |

Events pushed to the client: `obstacle_stop` (sonar < 20 cm while walking forward), `fall` / `upright`, `low_battery`, `watchdog_stop` (includes link loss).

## Frozen SDK (inside the firmware; no source on the device)
The Hiwonder Python libraries are compiled into the firmware. Nothing to pip install; they exist only on the robot.
- `HW_MechDog.MechDog()` methods: `move(stride, angle)`, `action_run(name)`, `stop_action`, `wait_action_over`, `action_is_over`, `set_default_pose`, `set_pose`, `transform([x,y,z],[roll,pitch,yaw],ms)`, `set_gait_params(lift_ms, contact_ms, lift_mm)`, `homeostasis` (self-balance), `read_homeostasis_status`, `set_servo`, `read_servo`, `read_all_servo`, offsets (`set_offset`, `read_offset`, `save_offset`, `set_angleoffset`, ...), `leg_set_ik`, `pose`, `run_status`.
- `Hiwonder`: `Battery_power()` (mV), `Buzzer()` with `playTone(freq, ms, block)`, `startMain(fn)` (run a thread), `Button`, `LED`, `Pin`, `ADC`, `I2C`, `UART`, `Digitaltube` (LED matrix), `WonderCam`, and analog sensors (Light/Soil/Raindrop/Knob/Slider).
- `Hiwonder_IIC`: `IIC(bus)`, `MPU()` → `read_angle()`, `I2CSonar(iic)` → `getDistance()` (cm), `setRGB(led 0/1/2, r, g, b)`, `setRGBMode`, `setBreathingCycle`, `startSymphony`; `ESP32S3Cam(iic2)` → `color_follow(color)`, `color_recognition()`, `line_follow`, `face_recognition` (**module not attached**); `QMI8658`; `ASR` / `asr_module` (voice module, not attached); `MP3`.
- Others: `Algo` (IMU pitch/roll fusion), `PWMServo`, `Hiwonder_WIFI`, `Hiwonder_BLE`, `uart_ctl`, `espnow`, `umqtt`, `urequests`, `webrepl`.

## Official repo (cloned to `hiwonder_mechdog_sdk/`)
github.com/Hiwonder/MechDog. Contains Scratch `.sb3` projects, **Python examples** (motion, action groups, touch, light, ASR+MP3, ultrasonic ranging/alarm, self-balance, color recognition/tracking, line following, face recognition), and Arduino C++ sources (`HW_MechDog.cpp/.h`, `Hiwonder.cpp/.h`, servo/offset tools). The Arduino `.cpp` files are the closest thing to readable source for the frozen Python library.

Key example, color tracking: `cam.color_follow(cam.BLUE)` returns `(status, cx, cy, x1, y1, x2, y2)`. `status == 3` means found. Steer ±25 by `cx`, stop once the blob area exceeds 5000.

## PC environment (installed)
Python 3.14.3, `mpremote`, `pyserial`, `opencv-python` 5.0, `numpy` 2.4, `google-genai`, `requests`.
