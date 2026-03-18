# Who Said It?

A Naive Bayes text classifier trained on 6 years of personal Discord message logs to
predict which of 8 friends sent a given message — complete with a calibrated
confidence score.

```
$ python predict.py "lets gooo"
Predicted: Person C (confidence: 0.45)
```

---

## Overview

This project builds an end-to-end NLP classification pipeline from scratch using only
`pandas`, `numpy`, `scikit-learn`, and `matplotlib`. The input is a raw CSV export of
54,000+ Discord messages spanning 6 years. The output is a trained model that, given
any message string, predicts the author and how confident it is.

The dataset was collected using a custom Discord scraper bot I built —
[github.com/Xander-Murray/scraper](https://github.com/Xander-Murray/scraper) — which
exports server message history to CSV.

The dataset is intentionally challenging:

- **8 classes** with a **10x class imbalance** (Person A: 11,751 messages vs. Person H: 1,330)
- **Short messages** — median length is under 30 characters; most messages are a few words
- **Heavy vocabulary overlap** — 8 friends in the same server using the same slang

All real names are anonymised (mapped to Person A–H) and excluded from the repository.
The raw data never leaves the local machine.

---

## Results

| Metric | Value |
|---|---|
| Test accuracy | **40.1%** |
| Majority-class baseline | 22.7% |
| Improvement over baseline | +17.4 pp |
| Best CV accuracy (5-fold) | 38.2% |
| Training samples | 32,604 |
| Test samples | 8,152 |

40% accuracy may look modest in isolation, but given that random guessing across 8
classes yields 12.5% and always predicting the majority class yields 22.7%, the model
is extracting genuine signal from noisy, short, informal text.

---

## Technical Decisions

**ComplementNB over MultinomialNB**

Standard Multinomial Naive Bayes estimates P(word | class). With a 10x class
imbalance, estimates for minority classes are noisy. Complement Naive Bayes instead
estimates P(word | NOT class), which leverages all other classes' data to produce
more stable minority-class estimates — consistently outperforming MultinomialNB on
imbalanced text datasets.

**TF-IDF with bigrams**

`TfidfVectorizer(max_features=50000, ngram_range=(1,2))` captures both single words
and two-word phrases. Bigrams pick up phrasing patterns (e.g. "my broda", "lets gooo")
that disappear when treating words independently. IDF downweights terms shared across
all authors, automatically amplifying the discriminative vocabulary.

**Calibrated confidence scores**

Raw Naive Bayes `predict_proba` outputs cluster near `1/N` in multi-class settings —
the model's probability estimates are systematically compressed. A
`CalibratedClassifierCV` wrapper with isotonic regression maps the raw outputs to
calibrated probabilities using held-out predictions, making the confidence score
interpretable: a score of 0.7 means the model is right ~70% of the time at that
confidence level.

**Stratified train/test split**

`train_test_split(..., stratify=y)` preserves the class distribution in both splits.
Without this, random chance could under-represent minority classes in the test set,
producing misleadingly high reported accuracy.

---

## Pipeline

```
Raw CSV (54,240 messages)  ← collected via github.com/Xander-Murray/scraper
    │
    ▼
Filter authors (≥200 messages) → 8 authors, 53,963 messages
    │
    ▼
Anonymise (real names → Person A–H)
    │
    ▼
Clean text
  • Remove URLs, Discord emoji, @mentions
  • Strip non-ASCII (Unicode emoji)
  • Lowercase + collapse whitespace
    │
    ▼
Drop empty rows after cleaning → 40,756 messages
    │
    ▼
TF-IDF vectorisation (50k features, unigrams + bigrams)
Stratified 80/20 train/test split
    │
    ▼
GridSearchCV — ComplementNB, alpha ∈ [0.01, 0.1, 0.5, 1.0], 5-fold CV
    │
    ▼
CalibratedClassifierCV (isotonic regression, cv=5)
    │
    ▼
models/nb_pipeline.pkl
    │
    ▼
predict.py — CLI inference
```

---

## Project Structure

```
who_said/
├── notebooks/
│   └── eda_preprocessing.ipynb   # Full pipeline: EDA → preprocessing → training → evaluation
├── models/
│   └── nb_pipeline.pkl            # Serialised vectoriser + calibrated model
├── train.py                       # Standalone training script
├── predict.py                     # Inference CLI
└── data/                          # Gitignored — contains real names
    ├── messages.csv               # Raw Discord export
    └── messages_clean.csv         # Cleaned, anonymised
```

---

## Usage

**Train**

```bash
python train.py
# Loading data...
#   40756 messages, 8 authors
# Vectorising...
# Running GridSearchCV ...
#   Best alpha: 1.0
#   Best CV accuracy: 0.3817
# Calibrating model (isotonic, cv=5)...
#   Test accuracy: 0.4008
# Saved models/nb_pipeline.pkl
```

**Predict**

```bash
python predict.py "my shower smell like lucky charms"
# Predicted: Person D (confidence: 0.61)
```

**Notebook**

```bash
jupyter notebook notebooks/eda_preprocessing.ipynb
```

---

## Stack

- **Python 3.13**
- **pandas** — data loading, cleaning, manipulation
- **scikit-learn** — TF-IDF, ComplementNB, GridSearchCV, CalibratedClassifierCV, metrics
- **matplotlib** — all visualisations (no seaborn)
- **numpy** — numerical operations

No deep learning frameworks, no external NLP libraries. The goal was to build the
most effective pipeline possible within the constraints of classical ML.
