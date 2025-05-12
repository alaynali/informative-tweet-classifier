# Informative Tweet Classifier

This project fine-tunes a Transformer-based model to classify English COVID-19-related tweets as either **INFORMATIVE** or **UNINFORMATIVE**. It is based on the WNUT-2020 shared task dataset and developed as part of an ECE 364 project.

## Features

- Binary classification using `transformers` and HuggingFace models
- Evaluation on test set with accuracy and F1 score
- Outputs Kaggle-compatible `prediction.csv`

## Setup

First set up the virtual environment. Go to the root directory of the project
and based on your machine follow one of two instructions below:

If you use a mac or linux machine:

```bash
python -m venv tweetenv
source tweetenv/bin/activate
```

If you use a windows machine:
```bash
python -m venv tweetenv
.\tweetenv\Scripts\activate
```

Following either of the two processes above should set up and activate a virtual environment. Now, anytime you install libraries, they get installed within this environment and not in your global Python installation. This isolation makes package management more organized and prevents conflicts with system-wide packages.

Next, let's install the libraries, run:

```bash
pip install -r requirements.txt
```
You should be good to go now! Now follow and run the informative_tweet_classifier.ipynb notebook to run the model.

## Usage

Run all cells in ```informative_tweet_classifier.ipynb``` from top to bottom.
- Loads and tokenizes `train.tsv`, `valid.tsv`, and `test.tsv`
- Trains on the combined train+valid data
- Evaluates on the test set
- Generates predictions and saves `prediction.csv`

NOTE:

- If you want to train only on `train.tsv` and validate on `valid.tsv` simply change the flags in trainer constructor from:

```bash
    train_dataset=full_train_encoded,
    eval_dataset=test_encoded,
```

to 

```bash
    train_dataset=train_encoded,
    eval_dataset=valid_encoded,
```

And that is all! 
