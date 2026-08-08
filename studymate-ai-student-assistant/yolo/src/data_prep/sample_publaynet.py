"""
sample_publaynet.py
--------------------
Goal: take a small random sample of PubLayNet (images + annotations)
instead of working with the full dataset from the start.

Fix: only samples from images that actually exist on disk, since this
Kaggle download is only part 0/7 of the full PubLayNet dataset --
train.json references all 335k images across all 7 parts, but only
a subset of those image files are actually present locally.

Usage:
    python src/data_prep/sample_publaynet.py
"""

import json
import random
import shutil
from pathlib import Path

# ============ SETTINGS (edit here if needed) ============
SAMPLE_SIZE = 800   # number of images to sample
RANDOM_SEED = 42     # for reproducibility

PROJECT_ROOT = Path(__file__).resolve().parents[2]  # -> student assistant/
RAW_IMAGES_DIR = PROJECT_ROOT / "data" / "raw" / "PubLayNet" / "train-0" / "publaynet" / "train"
RAW_ANNOTATIONS_FILE = PROJECT_ROOT / "data" / "raw" / "PubLayNet" / "labels" / "publaynet" / "train.json"

OUTPUT_IMAGES_DIR = PROJECT_ROOT / "data" / "processed" / "publaynet_sample" / "images"
OUTPUT_ANNOTATIONS_FILE = PROJECT_ROOT / "data" / "processed" / "publaynet_sample" / "annotations_sample.json"
# ==========================================================


def main():
    random.seed(RANDOM_SEED)

    print(f"Reading annotations file from: {RAW_ANNOTATIONS_FILE}")
    if not RAW_ANNOTATIONS_FILE.exists():
        raise FileNotFoundError(
            f"Annotations file not found at: {RAW_ANNOTATIONS_FILE}\n"
            "Check the path or update RAW_ANNOTATIONS_FILE above."
        )

    if not RAW_IMAGES_DIR.exists():
        raise FileNotFoundError(f"Images folder not found at: {RAW_IMAGES_DIR}")

    with open(RAW_ANNOTATIONS_FILE, "r", encoding="utf-8") as f:
        coco_data = json.load(f)

    all_images = coco_data["images"]
    all_annotations = coco_data["annotations"]
    categories = coco_data["categories"]

    print(f"Full JSON contains {len(all_images)} images and {len(all_annotations)} annotations")

    # --- KEY FIX: only keep images that actually exist on disk ---
    print("Checking which image files actually exist locally (this may take a moment)...")
    existing_filenames = {p.name for p in RAW_IMAGES_DIR.glob("*.jpg")}
    print(f"Found {len(existing_filenames)} image files physically present in {RAW_IMAGES_DIR}")

    available_images = [img for img in all_images if img["file_name"] in existing_filenames]
    print(f"{len(available_images)} images from the JSON match files that actually exist")

    if SAMPLE_SIZE > len(available_images):
        raise ValueError(
            f"SAMPLE_SIZE ({SAMPLE_SIZE}) is larger than the number of available images "
            f"({len(available_images)}). Lower SAMPLE_SIZE or download more data."
        )

    # sample only from images that exist
    sampled_images = random.sample(available_images, SAMPLE_SIZE)
    sampled_image_ids = {img["id"] for img in sampled_images}

    print(f"Selected {len(sampled_images)} images randomly (all guaranteed to exist)")

    sampled_annotations = [
        ann for ann in all_annotations if ann["image_id"] in sampled_image_ids
    ]
    print(f"Filtered {len(sampled_annotations)} matching annotations")

    OUTPUT_IMAGES_DIR.mkdir(parents=True, exist_ok=True)

    print("Copying images...")
    for idx, img in enumerate(sampled_images, start=1):
        src_path = RAW_IMAGES_DIR / img["file_name"]
        dst_path = OUTPUT_IMAGES_DIR / img["file_name"]
        shutil.copy2(src_path, dst_path)

        if idx % 100 == 0:
            print(f"   ...copied {idx}/{len(sampled_images)}")

    sample_coco = {
        "images": sampled_images,
        "annotations": sampled_annotations,
        "categories": categories,
    }

    with open(OUTPUT_ANNOTATIONS_FILE, "w", encoding="utf-8") as f:
        json.dump(sample_coco, f, ensure_ascii=False, indent=2)

    print("\nDone! Summary:")
    print(f"   - Images copied to: {OUTPUT_IMAGES_DIR}")
    print(f"   - New annotations file: {OUTPUT_ANNOTATIONS_FILE}")
    print(f"   - Actual images copied: {SAMPLE_SIZE}")


if __name__ == "__main__":
    main()
