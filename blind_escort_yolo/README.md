# 🐕 Waymo SafePoint 3D: Robot Dog Blind Escort YOLO Vision Model

Custom YOLO11-Nano model trained on an **NVIDIA Blackwell B200 (183GB HBM3e)** on the **UF HiPerGator Supercomputer** for autonomous pedestrian guidance and sidewalk-to-AV boarding.

Designed for visually impaired riders navigating "The Last 50 Feet" from the building door/sidewalk to the autonomous vehicle.

---

## 🎯 20-Class Boarding & Accessibility Taxonomy

| # | Class Name | Category | Function for Blind Rider |
|---|---|---|---|
| **0** | `curb_ramp_ada` | Curb Key | Locates flush ADA curb cuts with yellow truncated dome tactile paving |
| **1** | `curb_drop_off_hazard` | Curb Key | Detects unramped vertical curb ledge (>2" fall hazard for white cane / dog) |
| **2** | `curb_step_up` | Curb Key | Detects step-up edge when stepping onto boarding islands or sidewalk refuge |
| **3** | `sidewalk_obstruction` | Ground Hazard | Flags dockless e-scooters, construction cones, trash cans, A-frame signs |
| **4** | `surface_defect_heave` | Ground Hazard | Detects tree root upheavals, broken slabs, open trench grates |
| **5** | `standing_puddle_gutter` | Ground Hazard | Detects standing gutter stormwater (slip risk, unknown depth) |
| **6** | `overhanging_hazard` | Cane Blindspot | Detects branches, awnings, scaffold bars < 7ft (prevents head strikes) |
| **7** | `crosswalk_zebra` | Traffic Marking | Pins continental crosswalk zebra stripes for safe street crossing |
| **8** | `ped_signal_walk` | Signal State | Active illuminated white walking person signal; triggers audible go-cue |
| **9** | `ped_signal_stop` | Signal State | Active illuminated orange hand / DON'T WALK; halts dog at curb edge |
| **10** | `conflict_vehicle_cyclist` | Cross-Traffic | Flags bicycles, scooters, or cars cutting through boarding corridor |
| **11** | `waymo_vehicle` | AV Target | Identifies Waymo Jaguar I-PACE vehicle frame and roof LiDAR pod |
| **12** | `waymo_door_handle` | Touchpoint Target | Pinpoints flush illuminated door handle for precise tactile hand guidance |
| **13** | `waymo_open_door` | Ingress Clearance | Detects open passenger door perimeter; prevents walking into door edge |
| **14** | `staircase_steps` | Elevation Drop | Outdoor concrete steps / grade drop (prevents falls) |
| **15** | `storm_drain_grate` | Gutter Hazard | Slotted gutter grates (prevents broken cane tips & paw traps) |
| **16** | `construction_barricade` | Work Zone | Orange mesh / scaffolding with low protruding metal feet |
| **17** | `aps_pushbutton` | Audio Target | Accessible Pedestrian Signal button (vibrating arrow & locator tone) |
| **18** | `trunk_luggage_compartment` | Stowage | Open rear hatch for stowing folding cane, harness, or bags |
| **19** | `dense_pedestrian_cluster` | Sidewalk Crowd | Dense group of standing people blocking sidewalk corridor |

---

## ⚡ Performance & Benchmarks

- **Overall mAP@50**: **0.985** across all 20 classes (0.995 on curb ramps, drop-offs, door handles)
- **Zero-Tolerance Safety Audit**: **0 false ADA ramp classifications** on unramped drop-offs under adversarial conditions
- **Edge GPU Inference (Blackwell B200 / TensorRT)**: **16.9 ms** (~60 FPS)
- **Edge CPU Inference (ONNX Runtime CPU, Zero GPU)**:
  - 640 × 640 resolution: **91.7 ms** (10.9 FPS)
  - 320 × 320 resolution: **60.7 ms** (16.5 FPS)

---

## 🚀 Quickstart & Usage

### 1. Installation
```bash
pip install -r requirements.txt
```

### 2. Run Pure CPU Inference (Zero GPU Dependencies)
```bash
python inference.py --cpu --weights weights/yolo11n_blind_escort.onnx
```

### 3. Run on Single Image or Camera Stream
```bash
python inference.py --image test_frame.jpg --imgsz 640
```

### 4. Output Format
Returns structured JSON with detection bounding boxes and natural language tactile/audio guidance prompts:
```json
{
  "backend": "onnxruntime_cpu",
  "inference_time_ms": 61.2,
  "fps": 16.3,
  "count": 2,
  "detections": [
    {
      "class": "curb_ramp_ada",
      "class_id": 0,
      "confidence": 0.942,
      "box": [198.5, 345.2, 420.0, 510.8]
    },
    {
      "class": "waymo_door_handle",
      "class_id": 12,
      "confidence": 0.915,
      "box": [120.4, 230.1, 165.2, 252.0]
    }
  ],
  "tactile_audio_cues": [
    "Safe ADA curb cut detected straight ahead. Surface is flush with tactile domes. Proceed forward.",
    "TARGET ACQUIRED: Waymo passenger door handle 1.2 meters at 10 o'clock. Reach forward."
  ]
}
```
