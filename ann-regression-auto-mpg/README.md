# 🚗 Fuel Efficiency (MPG) Regression with a Feed-Forward Neural Network

Predicts a car's fuel efficiency (miles per gallon) from its specs, using a small feed-forward neural network (Keras/TensorFlow).

**Notebook:** [`ANN_Regression_AutoMPG.ipynb`](./ANN_Regression_AutoMPG.ipynb)

## Dataset

[Auto MPG](https://www.kaggle.com/datasets/uciml/autompg-dataset) (the classic UCI Auto MPG dataset, also on Kaggle) — 398 cars with `mpg`, `cylinders`, `displacement`, `horsepower`, `weight`, `acceleration`, `model year`, `origin`, and `car name`. Not included in this repo — download `auto-mpg.csv` from the link above and place it next to the notebook.

## Approach

- **Cleaning:** fixes the `horsepower` column (it contains `'?'` for missing values), drops the `car name` text column, and one-hot encodes the `origin` country (UK / Europe / USA).
- **Scaling:** z-score (standard) normalization on all features and on the target (`mpg`), since the raw columns are on very different scales (e.g. `weight` in the thousands vs. `acceleration` around 10-25).
- **Model:** a feed-forward network with 3 hidden layers (16 ReLU units each) and a single linear output unit, trained with the Adam optimizer on MSE loss.
- **Training:** `EarlyStopping`, `ReduceLROnPlateau`, and `ModelCheckpoint` (keeps the best model by validation loss) — up to 5000 epochs, but early stopping typically ends training much sooner.
- **Evaluation:** predictions are inverse-transformed back to real MPG units for interpretable comparison against the true values.

## Running the notebook

```bash
pip install tensorflow scikit-learn pandas numpy matplotlib seaborn
```

Place `auto-mpg.csv` next to the notebook (or update the read path) and run all cells.
