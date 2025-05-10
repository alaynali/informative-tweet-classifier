# Informative Tweet Classifier

This project fine-tunes a Transformer-based model to classify English COVID-19-related tweets as either **INFORMATIVE** or **UNINFORMATIVE**. It is based on the WNUT-2020 shared task dataset and developed as part of an ECE 364 project.

## Features

- Binary classification using `transformers` and HuggingFace models
- Evaluation on test set with accuracy and F1 score
- Outputs Kaggle-compatible `prediction.csv`

## Setup

```bash
pip install -r requirements.txt
```

## Usage

- Loads and tokenizes `train.tsv`, `valid.tsv`, and `test.tsv`
- Trains on the combined train+valid data
- Evaluates on the test set
- Generates predictions and saves `prediction.csv`

### Option 1: Python Script
Train and evaluate the model:

```bash
python main.py
```

### Option 2: Jupyter Notebook
Run all cells in ```informative_tweet_classifier.ipynb``` from top to bottom.



