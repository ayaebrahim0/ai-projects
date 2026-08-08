"""
visualize_yolo_labels.py
--------------------------
Goal: sanity check -- draw the YOLO-format bounding boxes on top of a few
sample images, to visually confirm the COCO -> YOLO conversion is correct
before we start training.

Usage:
    python src/data_prep/visualize_yolo_labels.py
"""

import random
from pathlib import Path

import cv2

# ============ SETTINGS ============
NUM_SAMPLES_TO_CHECK = 6   # how many random images to visualize
RANDOM_SEED = 7

PROJECT_ROOT = Path(__file__).resolve().parents[2]
YOLO_DATASET_ROOT = PROJECT_ROOT / "data" / "processed" / "publaynet_yolo"
IMAGES_DIR = YOLO_DATASET_ROOT / "images" / "train"
LABELS_DIR = YOLO_DATASET_ROOT / "labels" / "train"

OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "yolo_sanity_check"

CLASS_NAMES = ["text", "title", "list", "table", "figure"]
CLASS_COLORS = [
    (0, 255, 0),     # text -> green
    (0, 0, 255),     # title -> red
    (255, 0, 0),     # list -> blue
    (0, 255, 255),   # table -> yellow
    (255, 0, 255),   # figure -> magenta
]
# ====================================


def yolo_to_pixel_box(x_center, y_center, w, h, img_width, img_height):
    """Convert normalized YOLO coords back to pixel coordinates (for drawing)."""
    x_center_px = x_center * img_width
    y_center_px = y_center * img_height
    w_px = w * img_width
    h_px = h * img_height

    x_min = int(x_center_px - w_px / 2)
    y_min = int(y_center_px - h_px / 2)
    x_max = int(x_center_px + w_px / 2)
    y_max = int(y_center_px + h_px / 2)

    return x_min, y_min, x_max, y_max


def main():
    random.seed(RANDOM_SEED)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    image_files = list(IMAGES_DIR.glob("*.jpg"))
    sample_files = random.sample(image_files, NUM_SAMPLES_TO_CHECK)

    for img_path in sample_files:
        label_path = LABELS_DIR / (img_path.stem + ".txt")

        image = cv2.imread(str(img_path))
        img_height, img_width = image.shape[:2]

        if not label_path.exists() or label_path.stat().st_size == 0:
            print(f"Warning: no labels found for {img_path.name}")
            continue

        with open(label_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        for line in lines:
            parts = line.strip().split()
            class_id = int(parts[0])
            x_c, y_c, w, h = map(float, parts[1:])

            x_min, y_min, x_max, y_max = yolo_to_pixel_box(x_c, y_c, w, h, img_width, img_height)

            color = CLASS_COLORS[class_id]
            label_text = CLASS_NAMES[class_id]

            cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color, 2)
            cv2.putText(image, label_text, (x_min, max(y_min - 5, 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        output_path = OUTPUT_DIR / img_path.name
        cv2.imwrite(str(output_path), image)
        print(f"Saved visualization: {output_path}")

    print(f"\nDone. Check the images in: {OUTPUT_DIR}")
    print("Open them and confirm the boxes correctly surround the text/title/table/figure regions.")


if __name__ == "__main__":
    main()
