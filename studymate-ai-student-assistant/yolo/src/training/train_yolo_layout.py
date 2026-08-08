"""
train_yolo_layout.py
----------------------
Goal: train a YOLOv8 model on our sampled PubLayNet data (layout detection:
text, title, list, table, figure).

We use transfer learning: start from a YOLOv8 model pretrained on COCO,
then fine-tune it on our document layout dataset. This converges much
faster than training from scratch, especially with a small (800 image)
sample.

Usage:
    python src/training/train_yolo_layout.py
"""

from pathlib import Path
from ultralytics import YOLO

# ============ SETTINGS ============
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_YAML = PROJECT_ROOT / "data" / "processed" / "publaynet_yolo" / "data.yaml"

BASE_MODEL = "yolov8n.pt"   # nano version: fastest, good for a first experiment
EPOCHS = 50
IMAGE_SIZE = 640             # standard YOLO input size
BATCH_SIZE = 16              # lower this if you get an out-of-memory error

RUN_NAME = "layout_detection_v1"
# ====================================


def main():
    print(f"Loading base model: {BASE_MODEL}")
    # This automatically downloads the pretrained COCO weights the first time it's run
    model = YOLO(BASE_MODEL)

    print("Starting training...")
    results = model.train(
        data=str(DATA_YAML),
        epochs=EPOCHS,
        imgsz=IMAGE_SIZE,
        batch=BATCH_SIZE,
        name=RUN_NAME,
        project=str(PROJECT_ROOT / "models" / "layout_detection"),
        device=0,   # use GPU 0 (your RTX 2000 Ada)
    )

    print("\nTraining finished.")
    print(f"Results and weights saved under: models/layout_detection/{RUN_NAME}/")
    print("Best weights file: weights/best.pt")


if __name__ == "__main__":
    main()
