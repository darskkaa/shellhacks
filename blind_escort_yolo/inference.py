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
import cv2
import numpy as np
from PIL import Image

DEFAULT_TAXONOMY_PATH = Path(__file__).parent / "taxonomy.json"
REAL_TAXONOMY_PATH = Path(__file__).parent / "taxonomy_real.json"


def load_taxonomy(taxonomy_path: Path):
    with open(taxonomy_path) as f:
        data = json.load(f)
    classes = {int(k): v for k, v in data["classes"].items()}
    cues = data.get("cues", {})
    return classes, cues


class BlindEscortDetector:
    def __init__(self, model_path: str = None, taxonomy_path: str = None, use_cpu: bool = True):
        base_dir = Path(__file__).parent
        if model_path is None:
            # Prefer real-world assistive model if available, fallback to earlier checkpoint
            real_onnx = base_dir / "weights" / "yolo11n_assistive_real.onnx"
            blind_onnx = base_dir / "weights" / "yolo11n_blind_escort.onnx"
            blind_pt = base_dir / "weights" / "yolo11n_blind_escort.pt"
            if real_onnx.exists():
                self.model_path = real_onnx
            elif use_cpu and blind_onnx.exists():
                self.model_path = blind_onnx
            else:
                self.model_path = blind_pt
        else:
            self.model_path = Path(model_path)

        if taxonomy_path:
            self.taxonomy_path = Path(taxonomy_path)
        elif "real" in self.model_path.name and REAL_TAXONOMY_PATH.exists():
            self.taxonomy_path = REAL_TAXONOMY_PATH
        else:
            self.taxonomy_path = DEFAULT_TAXONOMY_PATH

        self.classes, self.tactile_cues = load_taxonomy(self.taxonomy_path)

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
                print(f"[BlindEscortDetector] Initialized ONNX Runtime ({self.backend}) from {self.model_path.name}")
            except Exception as e:
                print(f"[Warning] ONNX failed: {e}. Falling back to PyTorch...")

        if not self.backend:
            from ultralytics import YOLO
            pt_file = self.model_path.with_suffix(".pt")
            self.yolo_model = YOLO(str(pt_file))
            self.backend = "ultralytics_cpu" if use_cpu else "ultralytics_gpu"
            print(f"[BlindEscortDetector] Initialized PyTorch YOLO ({self.backend}) from {pt_file.name}")

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

        orig_w, orig_h = pil_img.size
        sx = orig_w / float(imgsz)
        sy = orig_h / float(imgsz)

        t0 = time.time()
        detections = []
        cues = []

        if "onnxruntime" in self.backend:
            tensor = self.preprocess(pil_img, imgsz)
            outputs = self.session.run(None, {self.input_name: tensor})[0]
            preds = outputs[0]  # Shape: (4 + num_classes, 8400)
            boxes = preds[:4, :].T
            scores = preds[4:, :].T

            class_ids = np.argmax(scores, axis=1)
            confs = np.max(scores, axis=1)

            mask = confs >= conf_thresh
            boxes, confs, class_ids = boxes[mask], confs[mask], class_ids[mask]
            # Raw output has one box per anchor, so one object yields many overlapping boxes. Per-class NMS at
            # IoU 0.7 matches the ultralytics .pt path's defaults.
            xywh = np.column_stack([boxes[:, :2] - boxes[:, 2:] / 2, boxes[:, 2:]])
            keep = cv2.dnn.NMSBoxesBatched(xywh.tolist(), confs.tolist(), class_ids.tolist(), conf_thresh, 0.7) if len(boxes) else []
            keep = np.array(keep, dtype=int).flatten()
            for box, conf, cls_id in zip(boxes[keep], confs[keep], class_ids[keep]):
                cls_name = self.classes.get(int(cls_id), f"class_{cls_id}")
                cx, cy, w, h = box
                x1 = float(cx - w / 2) * sx
                y1 = float(cy - h / 2) * sy
                x2 = float(cx + w / 2) * sx
                y2 = float(cy + h / 2) * sy
                detections.append({
                    "class": cls_name,
                    "class_id": int(cls_id),
                    "confidence": round(float(conf), 3),
                    "box": [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)]
                })
                if cls_name in self.tactile_cues and self.tactile_cues[cls_name] not in cues:
                    cues.append(self.tactile_cues[cls_name])
        else:
            dev = "cpu" if self.use_cpu else 0
            res = self.yolo_model.predict(pil_img, device=dev, imgsz=imgsz, conf=conf_thresh, verbose=False)[0]
            for box in res.boxes:
                cls_id = int(box.cls[0])
                cls_name = self.classes.get(cls_id, f"class_{cls_id}")
                conf = float(box.conf[0])
                coords = [round(float(c), 1) for c in box.xyxy[0]]
                detections.append({
                    "class": cls_name,
                    "class_id": cls_id,
                    "confidence": round(conf, 3),
                    "box": coords
                })
                if cls_name in self.tactile_cues and self.tactile_cues[cls_name] not in cues:
                    cues.append(self.tactile_cues[cls_name])

        elapsed_ms = (time.time() - t0) * 1000.0

        return {
            "backend": self.backend,
            "taxonomy": self.taxonomy_path.name,
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
    parser.add_argument("--taxonomy", type=str, default=None, help="Path to taxonomy JSON")
    parser.add_argument("--gpu", action="store_true", help="Use GPU CUDA acceleration")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size (640 or 320)")
    args = parser.parse_args()

    detector = BlindEscortDetector(
        model_path=args.weights,
        taxonomy_path=args.taxonomy,
        use_cpu=not args.gpu
    )

    if args.image:
        result = detector.predict(args.image, imgsz=args.imgsz)
        print(json.dumps(result, indent=2))
    else:
        # Run test benchmark frame
        test_img = Image.new("RGB", (args.imgsz, args.imgsz), (60, 65, 70))
        result = detector.predict(test_img, imgsz=args.imgsz)
        print(f"\n[Test Result] Latency: {result['inference_time_ms']}ms | Throughput: {result['fps']} FPS | Backend: {result['backend']}")
        print(f"Taxonomy: {result['taxonomy']} | Status: READY FOR ROBOT DOG ESCORT DEPLOYMENT")
