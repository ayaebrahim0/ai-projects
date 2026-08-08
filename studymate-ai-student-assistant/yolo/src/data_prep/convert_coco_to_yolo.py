"""
convert_coco_to_yolo.py
------------------------
Goal: convert the sampled COCO-format annotations (annotations_sample.json)
into YOLOv8's expected format, and split the data into train/val sets.

COCO bbox format: [x_min, y_min, width, height] in pixels
YOLO bbox format: [x_center, y_center, width, height] normalized (0 to 1)

Output structure created:
    data/processed/publaynet_yolo/
        images/train/*.jpg
        images/val/*.jpg
        labels/train/*.txt
        labels/val/*.txt
        data.yaml

Usage:
    python src/data_prep/convert_coco_to_yolo.py
"""

import json
import random
import shutil
from pathlib import Path

# ============ SETTINGS ============
VAL_SPLIT_RATIO = 0.2   # 20% of images go to validation
RANDOM_SEED = 42

PROJECT_ROOT = Path(__file__).resolve().parents[2]

SAMPLE_IMAGES_DIR = PROJECT_ROOT / "data" / "processed" / "publaynet_sample" / "images"
SAMPLE_ANNOTATIONS_FILE = PROJECT_ROOT / "data" / "processed" / "publaynet_sample" / "annotations_sample.json"

OUTPUT_ROOT = PROJECT_ROOT / "data" / "processed" / "publaynet_yolo"
OUTPUT_IMAGES_TRAIN = OUTPUT_ROOT / "images" / "train"
OUTPUT_IMAGES_VAL = OUTPUT_ROOT / "images" / "val"
OUTPUT_LABELS_TRAIN = OUTPUT_ROOT / "labels" / "train"
OUTPUT_LABELS_VAL = OUTPUT_ROOT / "labels" / "val"
DATA_YAML_PATH = OUTPUT_ROOT / "data.yaml"
# ====================================


def coco_bbox_to_yolo(bbox, img_width, img_height):
    """
    Convert a COCO bbox [x_min, y_min, box_width, box_height] in pixels
    to YOLO format [x_center, y_center, box_width, box_height] normalized to 0-1.
    """
    x_min, y_min, box_w, box_h = bbox

    x_center = x_min + box_w / 2
    y_center = y_min + box_h / 2

    # normalize by image dimensions
    x_center /= img_width
    y_center /= img_height
    box_w /= img_width
    box_h /= img_height

    return x_center, y_center, box_w, box_h


def main():
    random.seed(RANDOM_SEED)

    print(f"Reading sampled annotations from: {SAMPLE_ANNOTATIONS_FILE}")
    with open(SAMPLE_ANNOTATIONS_FILE, "r", encoding="utf-8") as f:
        coco_data = json.load(f)

    images = coco_data["images"]
    annotations = coco_data["annotations"]
    categories = coco_data["categories"]

    # COCO category ids are not guaranteed to start at 0 or be contiguous,
    # so we build a mapping: coco_category_id -> yolo_class_index (0-based)
    sorted_categories = sorted(categories, key=lambda c: c["id"])
    coco_id_to_yolo_idx = {cat["id"]: idx for idx, cat in enumerate(sorted_categories)}
    class_names = [cat["name"] for cat in sorted_categories]

    print(f"Classes found: {class_names}")

    # group annotations by image_id for fast lookup
    annotations_by_image = {}
    for ann in annotations:
        annotations_by_image.setdefault(ann["image_id"], []).append(ann)

    # shuffle and split images into train/val
    images_shuffled = images.copy()
    random.shuffle(images_shuffled)

    val_count = int(len(images_shuffled) * VAL_SPLIT_RATIO)
    val_images = images_shuffled[:val_count]
    train_images = images_shuffled[val_count:]

    print(f"Total images: {len(images_shuffled)} -> train: {len(train_images)}, val: {len(val_images)}")

    # create output folders
    for folder in [OUTPUT_IMAGES_TRAIN, OUTPUT_IMAGES_VAL, OUTPUT_LABELS_TRAIN, OUTPUT_LABELS_VAL]:
        folder.mkdir(parents=True, exist_ok=True)

    def process_split(image_list, images_out_dir, labels_out_dir, split_name):
        print(f"\nProcessing {split_name} split ({len(image_list)} images)...")
        for idx, img in enumerate(image_list, start=1):
            file_name = img["file_name"]
            img_width = img["width"]
            img_height = img["height"]

            # copy image
            src_img_path = SAMPLE_IMAGES_DIR / file_name
            dst_img_path = images_out_dir / file_name
            shutil.copy2(src_img_path, dst_img_path)

            # build YOLO label file (same name, .txt extension)
            label_file_name = Path(file_name).stem + ".txt"
            label_path = labels_out_dir / label_file_name

            img_annotations = annotations_by_image.get(img["id"], [])

            lines = []
            for ann in img_annotations:
                yolo_class_idx = coco_id_to_yolo_idx[ann["category_id"]]
                x_c, y_c, w, h = coco_bbox_to_yolo(ann["bbox"], img_width, img_height)
                lines.append(f"{yolo_class_idx} {x_c:.6f} {y_c:.6f} {w:.6f} {h:.6f}")

            with open(label_path, "w", encoding="utf-8") as f:
                f.write("\n".join(lines))

            if idx % 100 == 0:
                print(f"   ...processed {idx}/{len(image_list)}")

    process_split(train_images, OUTPUT_IMAGES_TRAIN, OUTPUT_LABELS_TRAIN, "train")
    process_split(val_images, OUTPUT_IMAGES_VAL, OUTPUT_LABELS_VAL, "val")

    # write data.yaml
    yaml_content = (
        f"path: {OUTPUT_ROOT.as_posix()}\n"
        f"train: images/train\n"
        f"val: images/val\n\n"
        f"nc: {len(class_names)}\n"
        f"names: {class_names}\n"
    )
    with open(DATA_YAML_PATH, "w", encoding="utf-8") as f:
        f.write(yaml_content)

    print(f"\nDone! data.yaml written to: {DATA_YAML_PATH}")
    print("YOLO-ready dataset structure created at:", OUTPUT_ROOT)


if __name__ == "__main__":
    main()
