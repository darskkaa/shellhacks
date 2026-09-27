Build the frontend for **SafeRoute Miami + Live Crossing**, a ShellHacks 2026 project. Judges should say "wow" within 3 seconds, and every pixel should still serve a real product for blind and low-vision riders. Frontend only: a real backend already exists (API contract below). Ship a mock mode so the whole site works on Replit with no hardware.

## The story (drives every design decision)
Waymo gets a blind rider door to door, but not across **the last 50 feet**: curb, crosswalk, signal. We built two things:
1. **SafeRoute**: ranks Miami driving and walking routes by real risk. Scores use 34,817 public hazard records (FDOT crashes, Miami-Dade 311 flooding, FDOT construction, live police incidents, NOAA tides and rain, NWS alerts, hurricane surge zones A–E), stored in MongoDB Atlas. Gemini writes a plain-English recommendation from the computed numbers only.
2. **Live Crossing**: a real quadruped guide robot (Hiwonder MechDog) with a camera. It holds at the curb until **Gemini Flash-Lite sees the pedestrian WALK signal** in the live frame. A YOLO26 model runs on-device as a backup detector. Then the robot crosses in a straight line at its fastest gait and stops when its sonar says it has arrived. It speaks every step aloud through **ElevenLabs**.

Hackathon tracks to make visible: Waymo, Best Use of Gemini API, MongoDB Atlas, Best Overall, State Farm (safety), Microsoft.

## Stack
React 18, Vite and TypeScript. Tailwind CSS, Framer Motion, lucide-react icons, and Geist + Geist Mono fonts. react-three-fiber only for the hero if it holds 60 fps; otherwise use a CSS/SVG/canvas equivalent. No component library look: build custom components. Single-page app with routes `/`, `/route`, `/crossing`.

## Art direction: "Miami at night, seen by a machine"
- **Mood:** a wet street at 2 a.m. Deep near-black navy background (`#070a0f`), panels a step lighter (`#0e131a`, `#141b24`), hairline borders (`#212a36`). One signature accent: electric teal `#2ee6d6` fading to blue `#5b8cff`. Two semantic colors carry the whole story: **WALK green `#34d399`** and **STOP red `#ff5a5f`**. Amber `#fbbf24` means caution. Keep all color scarce so the signal colors hit hard.
- **Type:** Geist 800 for huge tight headlines (letter-spacing -0.03em). Geist Mono, uppercase with wide tracking, for every label, eyebrow and number (tabular figures). Headlines can use a subtle gradient from white to teal.
- **Texture:** fine film grain over everything (SVG noise at 3–4% opacity). Soft radial glow behind key elements. Glass panels (backdrop-blur 16px, 8% white border) only over imagery, never over text-heavy areas.
- **Motion:** cinematic but restrained, with spring physics (Framer Motion). Scroll-linked reveals on the landing page. State changes on the crossing dashboard morph within 200–300 ms. Anything decorative respects `prefers-reduced-motion`.
- **References:** the Awwwards Sites of the Day dark collection (restraint, big type, scroll storytelling), Linear's and Vercel's landing pages (precision, glow, grid lines), Apple product pages (one idea per viewport), and the Waymo app's calm confidence. Aim above an Awwwards Honorable Mention.
- **Accessibility is the brand:** WCAG AA contrast minimum, AAA on the crossing state. Full keyboard support, visible focus rings in teal, `aria-live` announcements for state changes, and a working light theme. The product is for blind riders: the UI must be demoably screen-reader-friendly.

## Pages

### 1. Landing `/`
- **Hero (first viewport):** a full-bleed animated scene of a night crosswalk seen top-down or at a low angle. Wet asphalt reflections, zebra stripes, a pedestrian signal head glowing red. Every 6 s the signal flips to the green walking figure, and a small robot silhouette (a stylized quadruped, simple geometric shapes) trots across along a glowing path. A bounding box with a label snaps onto the signal, like the robot's vision. Headline: **"The last 50 feet."** Subhead: "Waymo gets you to the curb. Our guide robot gets you across." Two CTAs: "Watch it cross" (to `/crossing`) and "Plan a safe route" (to `/route`).
- **Live proof strip:** animated counters in Geist Mono: 34,817 hazards scored · 14 public data sources · ~1 s Gemini decision · 0 steps before WALK.
- **How it works:** a pinned scroll story in 4 beats, each a mini animation. (1) Camera sees the curb. (2) Gemini reads the signal: show a JSON chip `{"go": true, "signal": "walk", "confidence": 0.99}` typing out. (3) The robot sprints straight at stride 100. (4) Sonar reads 45 cm, it stops, and ElevenLabs says "We're across."
- **SafeRoute teaser:** a dark stylized Miami map (SVG or canvas) with three route lines. The safest one glows teal and the fastest shows red crash dots, with hover tooltips.
- **Tech and tracks:** a logo grid for Gemini, MongoDB Atlas, Google Maps Platform, ElevenLabs, NOAA/NWS and Waymo, each with one line on what it does here. No filler.
- **Footer:** "Built in 36 hours at ShellHacks 2026" and a team placeholder.

### 2. Route planner `/route`
Rebuild SafeRoute's planner with the new look.
- **Layout:** full-bleed Google Map (dark custom style) with a floating 440px left panel.
- **Panel:** From/To inputs with example chips ("Coconut Grove → Wynwood", "FIU → Brickell", "UM → Brickell"). A Conditions select (Live; Simulate flood warning; Simulate Category 1–5 hurricane). A shimmer "Find the safest route" button.
- **Results:** a verdict card ("Safest route avoids 12 crashes and 3 flood reports"), route cards ranked safest first (risk score, time, crash / flood / work-zone / school-zone counts), a Gemini recommendation block with a typing animation, and an elevation chart with a draggable flood-level slider (0–12 ft).
- **Live Crossing link:** a prominent card linking to `/crossing`: "At the destination? Let the robot cross you."

### 3. Live Crossing `/crossing`: THE DEMO. Make this the most beautiful screen.
Projected in a room, it must read from 10 meters away.
- **Stage:** the robot's live camera fills the viewport (`<img src=STREAM_URL>`, object-fit cover). The ring around the frame reacts to state:
  - wait: desaturated image, thin red inner border
  - walk: green inner glow pulsing at step cadence
  - arrived: teal glow
  - hazard: red flash
  - paused: grey
- **Signal badge**, top center, huge: a glowing lamp plus the word ("DON'T WALK", "WALK", "ACROSS", "STOP", "PAUSED", "OFFLINE") and a small subtitle. It morphs between states; don't cross-fade.
- **Left panel** (glass over the video, 420px):
  1. Brand "Live Crossing" with a live pulse dot, and "robot on COM4 / BLE"
  2. State card with a big state label and a one-line explanation
  3. Sonar gauge: an arc or bar, 0–200 cm, turning red at 45 cm or less, plus the number
  4. Crossing timer and stride ("stride 100 · sprint")
  5. "Gemini sees": the latest verdict sentence plus a confidence bar. When a GO arrives, flash the JSON verdict.
  6. Backup detector (YOLO26, on-device):
     - live detections as chips
     - "Counts as GO" select (populated from `classes`)
     - min-confidence slider
     - mode segmented control ("Only when Gemini is unavailable" / "Always")
     - "Stop on hazards" toggle
  7. Big Pause/Resume button (red when running, teal when paused), plus "Enable voice"
  8. "What the rider hears": a timeline of spoken lines with timestamps, newest on top, each sliding in
- **Bottom-right HUD chips** (Geist Mono): sonar, Gemini GO/no-go, and link status.
- **Offline states:** if the status endpoint returns 503, show an elegant empty state with the command to start the agent. If the camera is down, show a stylized "camera offline" placeholder in place of the video. Never show a broken layout.
- **Demo mode toggle** (visible, top right): runs the mock state machine so the page performs with no hardware.

## Backend API contract (use exactly; base URL from `VITE_API_BASE`, default same origin)
- `GET /api/dog/status?after=<lastLineId>`, polled every 350 ms. 503 `{"offline": true, "error": "..."}` when the agent is down.
  ```json
  {
    "status": {
      "state": "wait | walk | arrived | hazard | paused",
      "label": "Waiting for GO | Walking straight | Arrived | Hazard: curb_drop_off_hazard | Paused | null",
      "sonar_cm": 83.4,
      "stride": 100,
      "walking_s": 4.2,
      "gemini": "GO (a lit green walking-person signal) | no go (living room, no signal) | paused while walking | no fresh verdict",
      "yolo": "traffic light 0.72, person 0.55 | nothing | not loaded",
      "link": "connected | connecting",
      "target": "COM4 | ble"
    },
    "settings": {"backup_class": "traffic light", "backup_mode": "fallback", "backup_conf": 0.5, "hazard_stop": true, "paused": false},
    "classes": ["person", "bicycle", "car", "...80 COCO class names..."],
    "lines": [{"id": 12, "text": "Walk signal is on. Crossing now, stay with me."}]
  }
  ```
  `sonar_cm`, `stride` and `walking_s` may be null. `lines` holds only lines newer than `after`: remember the last id, and speak and log each new line exactly once.
- `POST /api/dog/settings` with a partial settings JSON, e.g. `{"paused": true}` or `{"backup_class": "person"}`. Returns the full settings.
- `GET /api/dog/stream.mjpg`: the live annotated camera as MJPEG, used as an `<img>` src. `GET /api/dog/snapshot.jpg` returns one JPEG; use it as the camera health probe (healthy only when content-type is `image/*`).
- `POST /api/routes` with `{"origin": "...", "destination": "...", "simulate": "storm" | "hurricane-1".."hurricane-5", "waterFt": 0-12}`. Returns `{conditions, routes, explanation}`. Routes come sorted safest first, each with a risk score, crash / flood / work-zone / school-zone counts, `elevationProfile` `[[lat, lng, meters]]` and `hazards` `[[lat, lng, severity, detail]]`. `GET /api/explain?id=` returns `{explanation}` (Gemini's summary, which arrives later).
- The spoken lines the robot produces: "Walk signal is on. Crossing now, stay with me." · "We're across." · "Something is right in front of me. Stopping." · "Stop. Hazard ahead."

## Voice (ElevenLabs, free via Puter)
Load `https://js.puter.com/v2/`. After the user clicks "Enable voice" (browsers need that click before playing audio), call `await puter.auth.signIn()` if not signed in. For each new line:
```js
const audio = await puter.ai.txt2speech(text, { provider: "elevenlabs", model: "eleven_flash_v2_5", voice: "21m00Tcm4TlvDq8ikWAM" });
await audio.play();
```
Queue the lines and never overlap them. If the queue backs up past 2, skip to the newest. Fall back to `speechSynthesis` on any error. Show which engine spoke each line.

## Mock mode (required, and the default when the API is unreachable)
Simulate the real state machine at 350 ms ticks:
- wait 4–6 s, with Gemini verdicts rotating "no go (crosswalk, red hand lit)"
- then GO with confidence 0.97; state walk; sonar falls from 180 to 40 cm at about 43 cm/s; `walking_s` counts up
- then arrived for 8 s, then back to wait
- occasionally a 2 s hazard ("curb_drop_off_hazard")

Emit the matching `lines`. Use a looping night-crosswalk video, or a canvas-drawn scene with a signal that flips red to green in time with the mock, as the fake camera.

## Quality bar
- Lighthouse 95+ on performance, accessibility and best practices. No layout shift. Fonts preloaded.
- Responsive down to 375px wide: on phones the crossing panel becomes a bottom sheet and the signal badge stays on top.
- Clean, typed code: an `api.ts` client with the types above, a `useDogStatus()` polling hook, a `useVoice()` queue hook, and a `mockDog.ts` state machine. No `any`.
- The first paint must already look finished, with no spinners on the landing page.

Deliver the complete, runnable project. Then list what you built and how to point `VITE_API_BASE` at the real backend (for example `http://<laptop-ip>:3000`).
