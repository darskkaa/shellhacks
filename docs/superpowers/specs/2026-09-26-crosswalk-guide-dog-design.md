# Crosswalk Guide-Dog Assistant — Design

**Date:** 2026-09-26
**Deadline:** ShellHacks 2026 submission, Sep 27 2026 @ 11:00am EDT

## Goal

MechDog acts as a guide dog for a blind pedestrian at a real crosswalk. A webcam
watches the pedestrian/vehicle traffic light; when it's actually safe (light
color + a vision-language model's confirmation), the dog leads the person
forward and Meta glasses speak the decision over Bluetooth audio (ElevenLabs
TTS). If it's unsafe, the dog stops and the glasses say so.

## Non-goals

- No Google Maps/Street View integration — out of scope for this build.
- No indoor/simulated crosswalk fallback — testing happens at a real crosswalk.
- No pedestrian WALK/DON'T-WALK icon detection — using the vehicle traffic
  light (red/yellow/green) as the primary signal, since it's a standard
  detectable class and more reliable under time pressure.
- No glasses camera integration — no official live-video API exists for Meta
  glasses; the glasses are used only for Bluetooth audio output.

## Architecture

```
webcam → YOLO (traffic light bbox) → HSV color crop → debounced light state
                                                              │
                                        GREEN? ──► Claude vision API
                                                    (frame + light state)
                                                    → verdict + one-line reason
                                                              │
                                                      state machine
                                                (WAIT / CHECKING / SAFE / CROSSING)
                                                  │                        │
                                          dog bridge (TCP JSON)     ElevenLabs audio
                                          move / stop / rgb         (cached "Stop/Wait/
                                                                     Walk now" clips +
                                                                     live reasoning line)
                                                                          │
                                                                Bluetooth → glasses speaker
```

## Components

1. **Light-state detector** — extends the existing `tools/vision_stream.py`
   YOLO pipeline. Crops each detected `traffic light` box, splits it into
   thirds, and classifies which third is lit via HSV thresholding. Debounces
   over ~5 consecutive frames before accepting a state change (kills flicker).
   No `traffic light` box detected → state is `UNKNOWN`.

2. **VLM safety check** (`tools/vlm_check.py`) — fires when the light state
   transitions to GREEN, and periodically (~every 2s) while it stays GREEN.
   Sends the current frame + light state to Claude's vision API with a prompt
   asking for a SAFE/UNSAFE verdict and a one-sentence reason (e.g. "car is
   turning right into the crosswalk"). Never called while `CROSSING` — the
   fast light path alone governs mid-crossing safety to avoid added latency.

3. **State machine** (`tools/safety_state.py`) — states: `WAIT` (light
   red/unknown) → `CHECKING` (light green, no VLM confirmation yet) → `SAFE`
   (VLM confirmed) → `CROSSING` (dog is walking). Any transition back to
   red/unknown light while `CROSSING` immediately forces `WAIT` and a stop
   command.

4. **Robot control** — reuses the existing bridge client from
   `tools/dog_panel.py` (JSON-over-TCP to the dog on port 5005). `SAFE` sends
   a `move` command (forward, moderate stride) plus green RGB; `WAIT`/`STOP`
   sends `stop` plus red RGB and a buzzer chirp. The existing bridge watchdog
   (dog auto-stops if `move`/`hb` isn't repeated within 1.5s) is relied on as
   a fail-safe for any orchestrator hang or crash.

5. **Audio output** (`tools/voice_out.py`) — "Stop", "Wait", and "Walk now,
   let's go" are pre-generated with ElevenLabs before the demo and cached
   locally as audio files for instant, latency-free playback on state change.
   The VLM's contextual reason sentence is synthesized live via ElevenLabs
   and played as a short follow-up. Playback targets the laptop's active
   audio output device, which must be set to the Bluetooth-paired glasses
   ahead of time at the OS level.

6. **Orchestrator** (`tools/crosswalk_assistant.py`) — single process: runs
   the capture loop, calls the light detector every frame, triggers the VLM
   check per the rules above, drives the state machine, and dispatches robot
   commands and audio output on every state transition.

## Data flow / fail-safes

- No traffic light detected, or the VLM call errors or times out → treated as
  `UNSAFE` (fail closed). The dog never crosses on missing information.
- Bluetooth audio device unavailable at runtime → falls back to the laptop's
  default speaker with a logged warning; this never blocks or crashes the
  safety loop.
- Robot bridge disconnects → orchestrator stops sending commands; the dog's
  own watchdog halts it within 1.5s regardless of orchestrator state.

## Secrets / config

`ANTHROPIC_API_KEY` and `ELEVENLABS_API_KEY` go in a new gitignored local
config file (`tools/secrets.env`, mirroring the existing
`robot_dump/config.example.py` → `robot_dump/config.py` pattern), with a
`tools/secrets.example.env` template committed instead.

## Testing given the deadline

- Dry-run the state machine and light classifier against the existing sample
  frames in `models/` — no camera or robot required, catches logic bugs fast.
- One on-location rehearsal at the actual crosswalk before the submission
  deadline to tune HSV thresholds for real lighting and confirm Bluetooth
  audio + robot behavior end-to-end.
