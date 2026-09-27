# Live Crossing console

Rework `saferoute/public/crossing.html` into the demo screen for the guide robot. Plain HTML in SafeRoute's
design system (same tokens, Geist fonts, light and dark). No build step. Two CDN libraries: Motion (spring motion
on state changes) and canvas-confetti (END celebration). Both are optional: the page works if either fails to load.

## Layout
- **Stage:** the annotated camera stream fills the screen. The frame edge glows green while walking and red on a hazard.
- **Signal pill** (top center): lamp plus word (DON'T WALK, WALK, ACROSS!, STOP, STANDBY, OFFLINE). It pops with
  a spring on every state change.
- **Captions** (bottom center, above the dock): each spoken line appears as a large subtitle and fades after 6 s.
  It is spoken through ElevenLabs via Puter, with the browser voice as fallback, once voice is enabled.
- **Instrument rail** (left): brand and back link, robot link status, state, crossing timer and estimated meters,
  Gemini's latest verdict, and YOLO26 detections.
- **Sonar ruler** (right edge): a vertical 0–200 cm scale with a marker, red at 45 cm or less.
- **Control dock** (bottom center):
  - Arm/Disarm detection, GO (W) and END (Space)
  - W A S D keycaps that light up on keypress and click
  - voice toggle and detection drawer toggle
- **Detection drawer** (right, collapsed by default):
  - YOLO26 "Counts as GO" select and minimum confidence
  - mode (only when Gemini is unavailable / always) and hazard-stop switch
  - "Track a phone" preset (cell phone, always)
- **Celebration:** confetti in the accent and walk colors when the state becomes `celebrate`.

## Data
The page uses the existing routes only: `GET /api/dog/status?after=`, `POST /api/dog/settings`, `POST /api/dog/command`
(`go`, `end`, `steer` with `dir` -1/0/1, a held key re-sent every 250 ms), `/api/dog/stream.mjpg` and
`/api/dog/snapshot.jpg` (camera health). `?demo=1` swaps these for an in-page mock of the same state machine
(wait, walk, celebrate) and a canvas-drawn night crosswalk as the camera.

## States and errors
- **Agent offline** (503): offline pill, plus a banner with the start command.
- **Camera offline:** a placeholder with the start command.
- **Controls:** GO is disabled unless waiting, END is disabled unless walking.

## Style rules
One accent color. Type carries the hierarchy. No gradient blobs, emoji, or glass on text-heavy areas. Motion only
on real state changes, and it respects `prefers-reduced-motion`. Every number shown is live data.

## Verification
Headless Edge screenshots of `?demo=1` in the wait, walk and celebrate states, at 1440×900 and 390×844, plus one
screenshot against the live agent.
