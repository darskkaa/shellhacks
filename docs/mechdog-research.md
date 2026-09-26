# Hiwonder MechDog — Research Notes

Compiled from live research during ShellHacks 2026 build. All links verified live at time of writing.

## What it is
Hiwonder MechDog: an open-source, ESP32-S3-based educational quadruped robot dog. Programmable via Python, Scratch (WonderCode), and Arduino (C/C++).

## Confirmed hardware (base kit — legs + camera only, no arm/LEGO)
- **Controller:** ESP32-S3
- **Actuation:** 8 coreless servos (legged walking, posture adjustment, self-balancing, height adjustment, standing function)
- **IMU:** posture/tilt/balance sensing
- **Ultrasonic sensor:** distance measurement + built-in RGB light (color-changeable), used for both obstacle avoidance and status/"emotion" feedback
- **Camera:** optional ESP32-S3 vision module on I2C bus 2 (returns color/line/face results, not a video stream to the PC). **Verified 2026-09-26: not attached on our unit.** We use a Logitech Brio 105 USB webcam on the laptop instead. See `mechdog-sdk-reference.md`.

## Confirmed native/built-in features (no custom code required)
- **Color-threshold object tracking** — camera tracks and follows a specific color, native "follow" behavior. Also does **line-following** (tracks colored lines).
- **Ultrasonic obstacle avoidance mode** — toggle in the official app; dog autonomously walks and turns away from obstacles. Confirmed it can run **simultaneously** with camera-based tracking/line-following without logic conflicts.
- **Proximity-based reactive gestures** — documented "shy" backup behavior when something gets close; can also be programmed to react "excited" (arm-wave-style gesture) on approach.
- **Voice control** — advertised feature (MechDog Pro specifically has the AI voice interaction module for two-way voice).
- **App control** — official app: movement, posture, ultrasonic obstacle avoidance toggle, action groups, self-balancing, height adjustment, standing function.

## Kit variants (matters for what's possible)
- **Base/Advanced Kit:** legs, camera, IMU, ultrasonic — no arm, no LEGO expansion, no voice module.
- **Ultimate Kit / MechDog Pro:** adds AI voice interaction module, IoT features, and support for a robot arm attachment.
- **LEGO/micro:bit expansion** (separate add-on, not in any base kit): enables physical manipulation (pick-and-place with a LEGO arm), a LEGO-compatible "horn" for physically pushing obstacles ("Angry Bull" mode), and micro:bit-driven behaviors (temperature-triggered navigation, LED facial expressions).
- **PuppyPi** (different product line, Raspberry Pi 5-based): this is Hiwonder's ROS2/SLAM-capable quadruped — MechDog itself does **not** have ROS/SLAM/LiDAR without swapping to this different product.

**Our build uses base robot + camera only — no arm, no LEGO horn, no micro:bit.**

## Official resources
- **Official GitHub (open-source Python/Arduino/Scratch SDK):** https://github.com/Hiwonder/MechDog
  - `Hiwonder` library — sensors, low-voltage alarms
  - `HW_MechDog` library — movement control
- **Official docs/wiki:** https://docs.hiwonder.com/projects/MechDog/en/latest/ (also mirrored at wiki.hiwonder.com)
  - App Control guide: `docs/2.APP_Control.html`
  - Python Programming Projects: `docs/4.Python_Programming_Projects.html`
  - Arduino Programming Projects: `docs/5.Arduino_Programming_Projects.html`
- **Product pages:** hiwonder.com/products/mechdog, hiwonder.com/products/mechdog-pro
- **Amazon/RobotShop listings** (kit variant comparisons): searchable, confirm Advanced vs Ultimate kit contents before assuming arm/voice availability.

## Community project precedent (Hackster.io, Hiwonder's own account)
- *A Deep Dive into Hacking the Hiwonder MechDog* — general hacking overview.
- *From Angry Bull to Multi-Tasker: Ignite Your MechDog!* — ultrasonic + LEGO horn obstacle-pushing behavior.
- *Exploring the LEGO and micro:bit Expansion of MechDog* — arm pick-and-place, emotional LED/touch interaction, thermal-triggered navigation to a "cool zone."
- *How to Build a Thermo-Smart MechDog* — micro:bit temperature sensor drives autonomous navigation to a target zone.
- *Ball-Chasing Quadruped Robot with ROS 2 & Arduino UNO Q* — different hobbyist build, not official MechDog, shows a ROS2 alternative approach for reference only.

## What we're building on top of this (project plan)
"Autonomous Hazard Patrol" — using only confirmed base-kit features:
1. **Movement:** native ultrasonic obstacle-avoidance mode → autonomous patrol of a small taped-off tabletop area.
2. **Perception:** grab a frame from the Brio 105 USB webcam (`cv2.VideoCapture`), send to Gemini API vision (zero-shot classification, no training) with a hazard-categorization prompt.
3. **Reaction:** map Gemini's returned category to one of the dog's existing native behaviors — turn/avoid, "shy" backup gesture, freeze, or switch into color-tracking follow mode — plus an RGB light-color cue via the ultrasonic sensor's light.

No arm, no LEGO parts, no micro:bit — everything above runs on the base robot + camera + the open-source SDK.

## Next step
Ready to `git clone https://github.com/Hiwonder/MechDog` into this project so the actual SDK source is local and any custom code can be diffed/built against it directly.
