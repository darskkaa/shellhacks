"""
Waymo SafePoint 3D / ShellHacks 2026: Real-World Assistive Vision YOLO11-Nano Training.
Trained on NVIDIA Blackwell B200 (183GB HBM3e) on UF HiPerGator Supercomputer.

Dataset: Abtinzandi/Obstacle-Detection-Dataset-YOLO (24,326 real outdoor smartphone photos).
Classes: Pedestrian crosswalk, Stairs, Traffic Cone, Manhole, Car, Person, Traffic sign, etc.
"""

import os
import shutil
import time
from pathlib import Path
from ultralytics import YOLO

def main():
    work_dir = Path("/blue/cqu.fgcu/speppers5329.fgcu/waymo_curbrisk")
    yaml_path = (work_dir / "datasets" / "real_world_assistive" / "data.yaml").resolve()

    print("\n=== Fine-Tuning YOLO11-Nano on Real-World Assistive Dataset (Blackwell B200) ===")
    print(f"Data config: {yaml_path}")
    model_pt = work_dir / "yolo11n.pt"
    model = YOLO(str(model_pt))

    t0 = time.time()
    results = model.train(
        data=str(yaml_path),
        epochs=15,
        imgsz=640,
        batch=64,
        device=0,
        workers=2,
        half=True,
        project=str(work_dir / "runs"),
        name="yolo11n_b200_real",
        exist_ok=True
    )
    elapsed = time.time() - t0
    print(f"\n[Training Finished] 15 epochs completed in {elapsed:.2f}s on NVIDIA B200!")

    checkpoints_dir = work_dir / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    best_pt = work_dir / "runs" / "yolo11n_b200_real" / "weights" / "best.pt"
    dest_pt = checkpoints_dir / "yolo11n_assistive_real.pt"

    if best_pt.exists():
        shutil.copy(best_pt, dest_pt)
        print(f"[Checkpoint Saved] {dest_pt}")

        print("[Exporting] Exporting to ONNX for Robot Dog & Meta Glasses Edge Deployment...")
        best_model = YOLO(str(dest_pt))
        onnx_file = best_model.export(format="onnx", imgsz=640, dynamic=False, half=False)
        dest_onnx = checkpoints_dir / "yolo11n_assistive_real.onnx"
        if Path(onnx_file).exists():
            shutil.copy(onnx_file, dest_onnx)
            print(f"[Edge ONNX Model Saved] {dest_onnx}")

if __name__ == "__main__":
    main()
