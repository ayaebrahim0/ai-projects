# 🐦 Arabic Twitter Sentiment Analysis (Bidirectional LSTM)

Binary sentiment classification (positive / negative) of Arabic tweets, comparing a classical baseline against two deep learning approaches.

**Notebook:** [`Arabic_Sentiment_LSTM.ipynb`](./Arabic_Sentiment_LSTM.ipynb)

## Dataset

[Arabic Sentiment Twitter Corpus](https://www.kaggle.com/datasets/mksaad/arabic-sentiment-twitter-corpus) (Kaggle) — ~45k labeled training tweets, ~11.5k test tweets. Downloaded automatically inside the notebook via `kagglehub`, no manual download needed.

## Text cleaning

A custom `clean_arabic_text` function: strips URLs, mentions, and hashtags; removes Arabic diacritics; normalizes Alef/Taa Marbouta/Alef Maksoura character variants; collapses repeated-character elongation (e.g. `هههههههه` → `هه`); and strips punctuation, digits, and emoji.

## Approaches compared

1. **Baseline — TF-IDF + Logistic Regression:** unigrams + bigrams, 50k max features. ~79% test accuracy.
2. **First deep learning attempt — plain (unidirectional) LSTM:** failed to converge (training accuracy plateaued ~63%, validation/test accuracy collapsed to near-random, ~50%).
3. **Final model — Tokenizer + Embedding + Bidirectional LSTM:** ~77% test accuracy.

| Model | Test Accuracy |
|---|---|
| TF-IDF + Logistic Regression (baseline) | ~79% |
| Plain LSTM (unidirectional) | ~50% (failed to converge) |
| **Bidirectional LSTM** (final) | **~77%** |

The simple TF-IDF + Logistic Regression baseline turned out to be competitive with the Bidirectional LSTM — a good reminder to always check a classical baseline before reaching for a deep learning model.

## Running the notebook

```bash
pip install kagglehub pandas scikit-learn tensorflow
```

Run all cells — the dataset downloads automatically on the first cell.
