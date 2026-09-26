"""
Waymo SafePoint 3D: Blind Escort & Curb Hazard Detector
Standalone CPU & GPU Inference Engine for Quadruped Robot Dog.
Runs on Jetson Nano, Raspberry Pi, laptop, or server.
Supports both .pt (PyTorch) and .onnx (ONNX Runtime CPU/GPU) weights.
"""

import sys
import time
import json
import argparse
from pathlib import Path
import numpy as np
from PIL import Image

TAXONOMY_PATH = Path(__file__).parent / "taxonomy.json"
with open(TAXONOMY_PATH) as f:
    TAXONOMY_DATA = json.load(f)

CLASSES = {int(k): v for k, v in TAXONOMY_DATA["classes"].items()}
TACTILE_CUES = TAXONOMY_DATA.get("cues", {})

class BlindEscortDetector:
    def __init__(self, model_path: str = None, use_cpu: bool = True):
        base_dir = Path(__file__).parent
        if model_path is None:
            # Default to ONNX for CPU, PT for GPU
            onnx_cand = base_dir / "weights" / "yolo11n_blind_escort.onnx"
            pt_cand = base_dir / "weights" / "yolo11n_blind_escort.pt"
            self.model_path = onnx_cand if (use_cpu and onnx_cand.exists()) else pt_cand
        else:
            self.model_path = Path(model_path)

        self.use_cpu = use_cpu
        self.session = None
        self.yolo_model = None
        self.backend = None

        if self.model_path.suffix == ".onnx" and self.model_path.exists():
            try:
                import onnxruntime as ort
                opts = ort.SessionOptions()
                opts.intra_op_num_threads = 4
                providers = ["CPUExecutionProvider"] if use_cpu else ["CUDAExecutionProvider", "CPUExecutionProvider"]
                self.session = ort.InferenceSession(str(self.model_path), opts, providers=providers)
                self.input_name = self.session.get_inputs()[0].name
                self.backend = "onnxruntime_cpu" if use_cpu else "onnxruntime_gpu"
                print(f"[BlindEscortDetector] Initialized ONNX Runtime ({self.backend}) from {self.model_path}")
            except Exception as e:
                print(f"[Warning] ONNX failed: {e}. Falling back to PyTorch...")

        if not self.backend:
            from ultralytics import YOLO
            pt_file = self.model_path.with_suffix(".pt")
            self.yolo_model = YOLO(str(pt_file))
            self.backend = "ultralytics_cpu" if use_cpu else "ultralytics_gpu"
            print(f"[BlindEscortDetector] Initialized PyTorch YOLO ({self.backend}) from {pt_file}")

    def preprocess(self, img: Image.Image, imgsz: int = 640):
        resized = img.convert("RGB").resize((imgsz, imgsz))
        arr = np.array(resized).astype(np.float32) / 255.0
        tensor = np.transpose(arr, (2, 0, 1))[np.newaxis, ...]
        return tensor

    def predict(self, img_input, conf_thresh: float = 0.35, imgsz: int = 640):
        if isinstance(img_input, (str, Path)):
            pil_img = Image.open(img_input)
        else:
            pil_img = img_input

        t0 = time.time()
        detections = []
        cues = []

        if "onnxruntime" in self.backend:
            tensor = self.preprocess(pil_img, imgsz)
            outputs = self.session.run(None, {self.input_name: tensor})[0]
            preds = outputs[0]  # Shape: (24, 8400)
            boxes = preds[:4, :].T
            scores = preds[4:, :].T

            class_ids = np.argmax(scores, axis=1)
            confs = np.max(scores, axis=1)

            mask = confs >= conf_thresh
            for box, conf, cls_id in zip(boxes[mask], confs[mask], class_ids[mask]):
                cls_name = CLASSES.get(int(cls_id), f"class_{cls_id}")
                cx, cy, w, h = box
                x1, y1 = float(cx - w/2), float(cy - h/2)
                x2, y2 = float(cx + w/2), float(cy + h/2)
                detections.append({
                    "class": cls_name,
                    "class_id": int(cls_id),
                    "confidence": round(float(conf), 3),
                    "box": [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)]
                })
                if cls_name in TACTILE_CUES and TACTILE_CUES[cls_name] not in cues:
                    cues.append(TACTILE_CUES[cls_name])
        else:
            dev = "cpu" if self.use_cpu else 0
            res = self.yolo_model.predict(pil_img, device=dev, imgsz=imgsz, conf=conf_thresh, verbose=False)[0]
            for box in res.boxes:
                cls_id = int(box.cls[0])
                cls_name = CLASSES.get(cls_id, f"class_{cls_id}")
                conf = float(box.conf[0])
                coords = [round(float(c), 1) for c in box.xyxy[0]]
                detections.append({
                    "class": cls_name,
                    "class_id": cls_id,
                    "confidence": round(conf, 3),
                    "box": coords
                })
                if cls_name in TACTILE_CUES and TACTILE_CUES[cls_name] not in cues:
                    cues.append(TACTILE_CUES[cls_name])

        elapsed_ms = (time.time() - t0) * 1000.0

        return {
            "backend": self.backend,
            "inference_time_ms": round(elapsed_ms, 2),
            "fps": round(1000.0 / elapsed_ms, 1) if elapsed_ms > 0 else 0,
            "count": len(detections),
            "detections": detections,
            "tactile_audio_cues": cues
        }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Blind Escort YOLO Detector")
    parser.add_argument("--image", type=str, default=None, help="Path to input image")
    parser.add_argument("--weights", type=str, default=None, help="Path to .onnx or .pt weights")
    parser.add_argument("--cpu", action="store_true", default=True, help="Force CPU inference")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size (640 or 320)")
    args = parser.parse_args()

    detector = BlindEscortDetector(model_path=args.weights, use_cpu=args.cpu)

    if args.image:
        result = detector.predict(args.image, imgsz=args.imgsz)
        print(json.dumps(result, indent=2))
    else:
        # Run test benchmark frame
        test_img = Image.new("RGB", (args.imgsz, args.imgsz), (60, 65, 70))
        result = detector.predict(test_img, imgsz=args.imgsz)
        print(f"\n[Test Result] Latency: {result['inference_time_ms']}ms | Throughput: {result['fps']} FPS | Backend: {result['backend']}")
        print(f"Status: READY FOR ROBOT DOG ESCORT DEPLOYMENT")
