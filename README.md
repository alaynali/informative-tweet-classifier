# Tweet Classification with Transformers and Augmentation

This repository fine-tunes a HuggingFace Transformer model on a custom text classification dataset using augmentation techniques and Optuna hyperparameter search.

## Features

- Text classification using `transformers`
- Data augmentation with `nlpaug`
- Hyperparameter tuning with `optuna`
- Evaluation with accuracy and F1 score
- TensorBoard logging support

## Setup

```bash
pip install -r requirements.txt
```

## Usage

Run the main script:

```bash
python main.py
```

Make sure to place your dataset in the script or adjust the file paths accordingly.

## Notes

- Tokenizer and model are loaded using HuggingFace Hub.
- Augmentation uses WordNet-based synonym replacement.
- Tuning uses Optuna for learning rate, batch size, and weight decay.