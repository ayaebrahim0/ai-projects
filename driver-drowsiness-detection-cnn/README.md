# 😴 Driver Drowsiness Detection — Eye State Classification (CNN)

A CNN that classifies a cropped eye image as **Open** or **Closed** — the core perception signal for a driver-drowsiness alert system (in a full system, several consecutive "closed" predictions over time would trigger an alert).

**Notebook:** [`DriverDrowsinessDetection_CNN.ipynb`](./DriverDrowsinessDetection_CNN.ipynb)

## Dataset

MRL Eye Dataset (via Kaggle, `mrl-dataset`) — grayscale 64×64 eye crops, labeled `Open_Eyes` / `Closed_Eyes` (4,000 images used here). Not included in this repo (large image dataset) — the notebook expects it at `/kaggle/input/mrl-dataset/train/{Open_Eyes,Closed_Eyes}/`, i.e. it's set up to run directly on Kaggle.

## Model

A compact CNN:
- 2 convolutional blocks (Conv2D → Conv2D → BatchNorm → MaxPooling → Dropout), with 32 then 64 filters
- Flatten → Dense(256) → BatchNorm → Dense(128) → Dense(84) → BatchNorm → Dropout
- Sigmoid output for binary classification (Open vs. Closed)

Trained with `ModelCheckpoint` keeping the best weights by validation loss (30 epochs, batch size 32).

## Results

~99.9% test accuracy, with the confusion matrix at the end of the notebook showing essentially no confusion between the two classes on the held-out test set.

**Trained weights (`bestModel.h5`) are not included in this repo.** Reach out (**aya8eb@gmail.com**) if you'd like them directly, or retrain from the notebook.

## Running the notebook

```bash
pip install tensorflow numpy pandas pillow matplotlib seaborn scikit-learn tqdm
```

Best run on [Kaggle](https://www.kaggle.com) with the MRL Eye Dataset attached as an input, or adjust the data paths to point at a local copy.
