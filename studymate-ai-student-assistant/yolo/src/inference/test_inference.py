"""
test_inference.py
-------------------
Goal: run the trained YOLO model on new images it has never seen during
training, and save the predictions drawn on top of the images.

Two modes:
1. RANDOM_FROM_RAW: automatically pick a few random images from the full
   PubLayNet raw folder that were NOT part of our 800-image sample.
2. CUSTOM_IMAGE: test on a specific image you provide (e.g. a screenshot
   of a lecture slide or PDF page).

Usage:
    python src/inference/test_inference.py
"""

import json
import random
from pathlib import Path

from ultralytics import YOLO

# ============ SETTINGS ============
MODE = "RANDOM_FROM_RAW"   # or "CUSTOM_IMAGE"
NUM_RANDOM_IMAGES = 6

CUSTOM_IMAGE_PATH = None   # e.g. r"C:\Users\Test\Desktop\my_slide.jpg"

CONFIDENCE_THRESHOLD = 0.5   # only show predictions the model is at least this confident about

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_WEIGHTS = PROJECT_ROOT / "models" / "layout_detection" / "layout_detection_v1" / "weights" / "best.pt"

RAW_IMAGES_DIR = PROJECT_ROOT / "data" / "raw" / "PubLayNet" / "train-0" / "publaynet" / "train"
SAMPLE_ANNOTATIONS_FILE = PROJECT_ROOT / "data" / "processed" / "publaynet_sample" / "annotations_sample.json"

OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "inference_test"
# ====================================


def get_unseen_images():
    """Return image paths that were NOT part of our 800-image training sample."""
    with open(SAMPLE_ANNOTATIONS_FILE, "r", encoding="utf-8") as f:
        sample_data = json.load(f)

    used_filenames = {img["file_name"] for img in sample_data["images"]}

    all_available = list(RAW_IMAGES_DIR.glob("*.jpg"))
    unseen = [p for p in all_available if p.name not in used_filenames]

    return unseen


def main():
    print(f"Loading trained model from: {MODEL_WEIGHTS}")
    model = YOLO(str(MODEL_WEIGHTS))

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if MODE == "CUSTOM_IMAGE":
        if not CUSTOM_IMAGE_PATH:
            raise ValueError("Set CUSTOM_IMAGE_PATH to test a specific image.")
        images_to_test = [Path(CUSTOM_IMAGE_PATH)]
    else:
        print("Selecting random images the model has never seen...")
        unseen_images = get_unseen_images()
        images_to_test = random.sample(unseen_images, NUM_RANDOM_IMAGES)

    print(f"Running inference on {len(images_to_test)} image(s)...")

    for img_path in images_to_test:
        results = model.predict(
            source=str(img_path),
            conf=CONFIDENCE_THRESHOLD,
            save=False,
            verbose=False,
        )

        result = results[0]
        annotated_frame = result.plot()  # draws boxes + labels + confidence scores

        output_path = OUTPUT_DIR / f"pred_{img_path.name}"

        import cv2
        cv2.imwrite(str(output_path), annotated_frame)
        print(f"Saved: {output_path}  ({len(result.boxes)} objects detected)")

    print(f"\nDone. Check results in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
