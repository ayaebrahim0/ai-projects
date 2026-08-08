"""
ocr_pipeline.py
-----------------
Goal: combine our Phase 1 YOLO layout model with EasyOCR to extract
actual text from a document image.

Pipeline:
    1. Run the trained YOLO model to find "text" regions in the page
    2. Crop each detected text region out of the original image
    3. Run EasyOCR on each crop to get the actual recognized text
    4. Print / save the results

Usage:
    python src/inference/ocr_pipeline.py
"""

from pathlib import Path

import cv2
import easyocr
from ultralytics import YOLO

# ============ SETTINGS ============
PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_WEIGHTS = PROJECT_ROOT / "models" / "layout_detection" / "layout_detection_v1" / "weights" / "best.pt"

# Use a CLEAN image (no boxes drawn on it already) so the model can detect normally.
# Any filename that exists in this folder works:
#   data/processed/publaynet_sample/images/
CLEAN_IMAGES_DIR = PROJECT_ROOT / "data" / "processed" / "publaynet_sample" / "images"
TEST_IMAGE_PATH = CLEAN_IMAGES_DIR / "PMC4076111_00006.jpg"

CONFIDENCE_THRESHOLD = 0.5
TEXT_CLASS_NAME = "text"   # we only OCR regions the YOLO model labeled as "text"

OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "ocr_test"
# ====================================


def main():
    print(f"Loading YOLO layout model from: {MODEL_WEIGHTS}")
    yolo_model = YOLO(str(MODEL_WEIGHTS))

    print("Loading EasyOCR reader...")
    # DEBUG STEP: test with English only first, since this PubLayNet image
    # is a scientific paper written entirely in English. Mixing "ar"+"en"
    # can cause the model to misread Latin characters as Arabic ones.
    OCR_LANGUAGES = ["en"]   # change to ["ar", "en"] later once English-only is confirmed clean
    ocr_reader = easyocr.Reader(OCR_LANGUAGES, gpu=True)

    print(f"Running YOLO on: {TEST_IMAGE_PATH}")
    results = yolo_model.predict(
        source=str(TEST_IMAGE_PATH),
        conf=CONFIDENCE_THRESHOLD,
        verbose=False,
    )
    result = results[0]

    image = cv2.imread(str(TEST_IMAGE_PATH))
    if image is None:
        raise FileNotFoundError(f"Could not load image at: {TEST_IMAGE_PATH}")

    class_names = result.names  # maps class_id -> class name, e.g. {0: 'text', 1: 'title', ...}

    # DEBUG: show everything the model detected, regardless of class,
    # so we can tell if the model saw nothing at all vs. saw boxes but none were "text"
    print(f"\n[DEBUG] Total boxes detected (any class): {len(result.boxes)}")
    for box in result.boxes:
        cid = int(box.cls[0])
        conf = float(box.conf[0])
        print(f"[DEBUG]   class={class_names[cid]}  confidence={conf:.2f}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    text_region_count = 0
    extracted_texts = []

    for box in result.boxes:
        class_id = int(box.cls[0])
        class_name = class_names[class_id]

        if class_name != TEXT_CLASS_NAME:
            continue  # skip titles, tables, figures, lists -- only OCR "text" regions

        text_region_count += 1

        # get pixel coordinates of this box
        x_min, y_min, x_max, y_max = map(int, box.xyxy[0])

        # crop that region out of the original image
        cropped_region = image[y_min:y_max, x_min:x_max]

        # upscale only genuinely tiny crops -- aggressive upscaling made results worse
        crop_height = cropped_region.shape[0]
        if crop_height < 100:
            scale_factor = 100 / crop_height
            new_width = int(cropped_region.shape[1] * scale_factor)
            cropped_region = cv2.resize(
                cropped_region, (new_width, 100), interpolation=cv2.INTER_CUBIC
            )

        # save the crop for inspection
        crop_path = OUTPUT_DIR / f"crop_{text_region_count}.jpg"
        cv2.imwrite(str(crop_path), cropped_region)

        # run OCR on the cropped region (default greedy decoder -- beamsearch
        # tested worse for this font/quality)
        ocr_result = ocr_reader.readtext(cropped_region, detail=0)
        recognized_text = " ".join(ocr_result)

        extracted_texts.append(recognized_text)

        print(f"\n--- Text region {text_region_count} ---")
        print(f"Saved crop: {crop_path}")
        print(f"Recognized text: {recognized_text}")

    # save all extracted text to a single file
    output_txt_path = OUTPUT_DIR / "extracted_text.txt"
    with open(output_txt_path, "w", encoding="utf-8") as f:
        f.write("\n\n".join(extracted_texts))

    print(f"\nDone. Found {text_region_count} text regions.")
    print(f"All extracted text saved to: {output_txt_path}")


if __name__ == "__main__":
    main()
