# 🐕 Real-World Testing Protocol: Hiwonder MechDog Blind Escort

> [!IMPORTANT]
> **SAFETY FIRST**: All real-world testing must be conducted on a **private road, closed driveway, or pedestrian sidewalk** with zero vehicular traffic. The handler must always hold a physical safety leash attached to the dog's chassis/harness.

---

## 🛠️ Hardware Setup

1. **Hiwonder MechDog Quadruped Robot**
   - Main controller: ESP32 with 8 high-torque bus servos.
   - Sensors: Front ultrasonic sonar distance sensor, 6-axis IMU, battery voltage telemetry.
   - Firmware: `robot_dump/main.py` flashed onto the ESP32 (serves TCP port 5005).

2. **Vision Sensor (Webcam)**
   - Logitech Brio 105 (or laptop camera) mounted to the laptop or secured to the dog's upper deck facing forward.
   - Resolution: 640×640 or 1280×720 at 30 FPS.

3. **Battery Safety Checklist**
   - Ensure the 2S LiPo battery is fully charged (7.4V nominal, minimum 7.0V).
   - If voltage drops below 6.8V, the software automatically halts forward motion to prevent brownouts.

---

## ⏱️ Crosswalk Speed & Timing Analysis
Crucial safety consideration for street crossings:

| Mode | Stride Parameter | Gait Preset | Cadence | Ground Speed | 6m Street Cross | Notes |
|---|---|---|---|---|---|---|
| **Sidewalk Escort (Default)** | `stride 40` | `DEFAULT` (200/300/20) | 2.0 steps/s | 0.07 m/s | 88.2 s | Tactile paving & curb scanning |
| **Sidewalk Escort (Brisk)** | `stride 40` | `FAST` (100/150/20) | 4.0 steps/s | 0.14 m/s | 44.1 s | Brisk sidewalk navigation |
| **Crosswalk Transit (Sprint)** | `stride 100` | `SPRINT` (80/120/18) | 5.0 steps/s | **0.43 m/s (~1.0 mph)** | **14.1 s** | ✅ Clears street within standard signal window |

### The Dynamic Gait Speed Architecture
1. **On Sidewalk / Approach**: Use standard stride (`[w]`). Slow, controlled pace for scanning ADA tactile paving, curb drop-offs, and trash obstacles.
2. **Speed Toggle (`[f]`)**: Press `[f]` in `tools/real_world_escort.py` to toggle between Normal and Fast mode (activates `FAST` gait on sidewalks, `SPRINT` gait on crosswalks).
3. **On Crosswalk Street**: Once the dog identifies the `curb_ramp_ada` and pedestrian signal confirms `ped_signal_walk`, press `[c]` (or let Gemini select `speed_mode: "sprint"`) to sprint across the road at 5.0 steps/s (~1 mph)!
4. **Autonomous Safety Guard**: In `gemini_dog_agent.py`, if a `curb_drop_off_hazard`, `ped_signal_stop`, or `conflict_vehicle_cyclist` is detected, speed is automatically demoted to `normal` regardless of rider goals.

---

## 🔒 Built-in Hardware Safety Guards

1. **Hardware Sonar Cutoff (<35 cm)**: If the ultrasonic sensor reads an obstacle within 35 cm, the controller overrides all walk commands and stops immediately.
2. **IMU Tilt / Anti-Fall Guard**: If body tilt (pitch or roll) exceeds 35°, motion halts instantly.
3. **Pacing Cap**: Forward stride is clamped to **stride 40** (approx. 0.8–1.0 m/s), matching natural human walking pace.
4. **Watchdog Heartbeat**: If WiFi or serial communication drops for >1.5 seconds, the ESP32 firmware stops all servos automatically.
5. **Instant Emergency Stop**: Pressing `Spacebar` or `Ctrl+C` in any terminal immediately halts all motion.

---

## 🚀 Step-by-Step Deployment on Private Road

### Step 1: Connect to the Robot Dog's WiFi
1. Power on the MechDog using the battery switch.
2. On your laptop, connect to the WiFi network:
   - **SSID**: `MechDog_wifi`
   - **Password**: `12345678`
   - The dog will be at IP address `192.168.4.1`.

*(Alternatively, if running via USB cable, the CH340 port will appear as `COM3` on Windows or `/dev/ttyUSB0` on Linux).*

---

### Step 2: Launch the 20-Class Blind Escort Vision Pipeline
In **Terminal 1**, start the live camera stream with the custom-trained YOLO model:
```bash
python tools/vision_stream.py --cam 0 --model blind_escort_yolo/weights/yolo11n_blind_escort.pt
```
*Verification: Open `http://localhost:8001` in your browser. You will see real-time bounding boxes detecting curb cuts, drop-offs, door handles, and crosswalks.*

---

### Step 3: Run the Safe Escort Controller
In **Terminal 2**, launch the real-world safety runner:
```bash
python tools/real_world_escort.py --wifi 192.168.4.1 --min-sonar 35 --max-stride 40
```
- It will verify the battery voltage and sonar clear distance before enabling motion.
- Keyboard controls:
  - `w`: Step forward (gentle stride)
  - `s`: Stop immediately
  - `a`: Turn 15° left (re-orient toward curb ramp)
  - `d`: Turn 15° right
  - `q`: Disconnect and power down servos

---

### Step 4: (Optional) Autonomous Gemini AI Escort Brain
In **Terminal 3**, launch the autonomous Gemini 2.5 Flash agent:
```bash
export GEMINI_API_KEY="your-gemini-key"
python tools/gemini_dog_agent.py --host 192.168.4.1 --goal "Guide me safely to the Waymo passenger door"
```
The Gemini agent reads the live camera detections, reasons about curb ramp access and pedestrian obstacles, and sends autonomous walking commands to the dog while speaking audio directions to the rider.
