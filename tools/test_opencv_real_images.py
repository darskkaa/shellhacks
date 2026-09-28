"""
Test OpenCV DNN inference with YOLO11-Nano on real street images.
Tests real pedestrian walk signal, real don't walk signal, and real street red light.
"""
import os
import sys
import json
import time
import cv2
import numpy as np

TAXONOMY_PATH = "blind_escort_yolo/taxonomy.json"
MODEL_PATH = "blind_escort_yolo/weights/yolo11n_blind_escort.onnx"

with open(TAXONOMY_PATH) as f:
    tax = json.load(f)
    CLASS_NAMES = {int(k): v for k, v in tax["classes"].items()}

print(f"Loading ONNX model via OpenCV DNN: {MODEL_PATH}")
net = cv2.dnn.readNetFromONNX(MODEL_PATH)

test_files = [
    ("Real Pedestrian WALK Signal (Manhattan)", "test_images/ped_walk_manhattan.jpg"),
    ("Real Pedestrian DONT WALK / Stop Signal", "test_images/ped_dont_walk.jpg"),
    ("Real Street Red Light", "test_images/street_red_light.jpg"),
]

conf_threshold = 0.25
nms_threshold = 0.45

for title, img_path in test_files:
    if not os.path.exists(img_path):
        print(f"Skipping {img_path} (not found)")
        continue

    print(f"\n{'='*60}")
    print(f"Testing: {title}")
    print(f"File: {img_path}")
    print(f"{'='*60}")

    frame = cv2.imread(img_path)
    if frame is None:
        print(f"Error loading {img_path}")
        continue

    h, w = frame.shape[:2]
    print(f"Original resolution: {w}x{h}")

    t0 = time.time()
    blob = cv2.dnn.blobFromImage(frame, 1/255.0, (640, 640), swapRB=True, crop=False)
    net.setInput(blob)
    out = net.forward()
    infer_ms = (time.time() - t0) * 1000

    # Output shape (1, 24, 8400) -> transpose to (8400, 24)
    pred = out[0].T
    boxes_raw = pred[:, :4]
    scores_raw = pred[:, 4:]

    class_ids = np.argmax(scores_raw, axis=1)
    confs = np.max(scores_raw, axis=1)

    valid_mask = confs >= conf_threshold
    valid_boxes = boxes_raw[valid_mask]
    valid_confs = confs[valid_mask]
    valid_ids = class_ids[valid_mask]

    print(f"Inference latency: {infer_ms:.1f} ms | Raw candidates above {conf_threshold*100:.0f}%: {len(valid_confs)}")

    annotated = frame.copy()
    dets = []

    if len(valid_confs) > 0:
        scale_x = w / 640.0
        scale_y = h / 640.0

        cv_boxes = []
        for b in valid_boxes:
            cx, cy, bw, bh = b
            bx1 = int((cx - bw / 2.0) * scale_x)
            by1 = int((cy - bh / 2.0) * scale_y)
            bw_px = int(bw * scale_x)
            bh_px = int(bh * scale_y)
            cv_boxes.append([max(0, bx1), max(0, by1), max(1, bw_px), max(1, bh_px)])

        indices = cv2.dnn.NMSBoxes(cv_boxes, valid_confs.tolist(), conf_threshold, nms_threshold)
        if len(indices) > 0:
            for idx in indices.flatten():
                bx, by, bw_px, bh_px = cv_boxes[idx]
                bx2, by2 = min(w, bx + bw_px), min(h, by + bh_px)
                cls_id = int(valid_ids[idx])
                label = CLASS_NAMES.get(cls_id, f"class_{cls_id}")
                c = float(valid_confs[idx])

                dets.append({
                    "class_id": cls_id,
                    "label": label,
                    "conf": round(c, 3),
                    "box": [bx, by, bx2, by2]
                })

                color = (0, 255, 0) if "walk" in label else (0, 0, 255)
                cv2.rectangle(annotated, (bx, by), (bx2, by2), color, max(2, int(w / 400)))
                text = f"{label}: {c*100:.1f}%"
                cv2.putText(annotated, text, (bx, max(30, by - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, max(0.6, w / 1200), color, 2)

    print(f"Detections after NMS ({len(dets)}):")
    if dets:
        for d in dets:
            print(f"  • [{d['label']}] (ID {d['class_id']}) - Confidence: {d['conf']*100:.1f}% - Box: {d['box']}")
    else:
        # Check top 3 raw scores to inspect what the model saw
        top_indices = np.argsort(confs)[-3:][::-1]
        print("  (No detections passed confidence threshold. Top raw predictions:)")
        for idx in top_indices:
            cls_id = int(class_ids[idx])
            print(f"    - Class: {CLASS_NAMES.get(cls_id, cls_id)} (ID {cls_id}) with max conf {confs[idx]*100:.1f}%")

    out_name = f"test_images/annotated_{os.path.basename(img_path)}"
    cv2.imwrite(out_name, annotated)
    print(f"Saved annotated result to: {out_name}")

print("\nDone testing OpenCV DNN on real street images.")
