# 🦷 Teeth Segmentation with U-Net (ResNet50 Encoder)

A binary segmentation model that takes a photo of teeth and predicts a pixel-level mask of the teeth region — built on a self-collected, hand-annotated dataset.

**Notebook:** [`TeethSegmentation_UNet.ipynb`](./TeethSegmentation_UNet.ipynb)

## Overview

- **Dataset:** self-collected teeth images (291 image/mask pairs), with masks drawn by hand — not a public dataset.
- **Architecture:** a U-Net decoder on top of a **ResNet50** encoder pretrained on ImageNet. The encoder is frozen (transfer learning) so training only updates the decoder, which is far more data-efficient on a small (291-image) dataset than training an encoder from scratch. Skip connections are taken from four ResNet50 feature maps (`conv1_relu`, `conv2_block3_out`, `conv3_block4_out`, and the bottleneck `conv4_block6_out`) and concatenated into the decoder path.
- **Loss / metric:** Dice loss / Dice coefficient — the standard choice for binary segmentation where the foreground (teeth) covers a small, imbalanced fraction of the image, unlike plain pixel accuracy.
- **Output:** a 256×256 binary mask (sigmoid activation, thresholded at 0.5).
- **Training:** 20 epochs, batch size 8, Adam (lr 1e-4), best weights checkpointed on `val_loss`.

## Pipeline

```
Teeth photo + hand-drawn mask (291 pairs)
        │
        ▼
Resize to 256×256, normalize image to [0,1], binarize mask
        │
        ▼
ResNet50 encoder (frozen, ImageNet weights) ──► skip connections at 4 scales
        │
        ▼
U-Net decoder (Conv2DTranspose + concat + Conv2D blocks)
        │
        ▼
Sigmoid output ──► binary teeth mask (256×256×1)
```

## Files

- [`TeethSegmentation_UNet.ipynb`](./TeethSegmentation_UNet.ipynb) — data loading, model definition, training, and an inference cell that loads the saved model and runs it on a new uploaded image.
- [`dataset/`](./dataset) — the 291 hand-annotated image/mask pairs (`Img/` + `masks_output/`).
- **Trained weights are not included in this repo** (file size). The notebook reproduces them by running training end-to-end; if you'd like the pretrained `.h5` weights directly, reach out to me (**aya8eb@gmail.com**).

## Running the notebook

Built and run in **Google Colab**. To reproduce:

1. Upload `dataset/` (or a zipped `Dataset-u-net.zip` with `Img/` and `masks_output/` subfolders) to the Colab environment.
2. Run the notebook top to bottom — training takes ~20 epochs on the 291-image dataset.
3. The inference cell at the end mounts Google Drive to load a saved model and lets you upload a new teeth photo to see the predicted mask.

```bash
pip install opencv-python matplotlib scikit-learn tensorflow
```

## Notes

- Since the encoder is frozen, most of the trainable parameters are in the decoder — this keeps training fast and reduces overfitting risk on a 291-image dataset.
- The dataset and masks are self-collected and hand-annotated, so there's no external dataset license to worry about, but also no independent train/test split beyond the 80/20 hold-out used here.
