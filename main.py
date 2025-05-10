# === imports ===
import os, random, numpy as np, torch
from collections import Counter

# data
import pandas as pd
from datasets import load_dataset, concatenate_datasets, Dataset

# tokenization & modeling
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    AutoConfig,
    set_seed,
    TrainerCallback,
    EarlyStoppingCallback
)

# metrics & visualiztion
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns

# logging
from torch.utils.tensorboard import SummaryWriter
from transformers.integrations import TensorBoardCallback

# === version check & device setup ===
print(transformers.__version__)
print(torch.__version__)
print(torch.cuda.is_available())

device = torch.device("cuda")
print(device)

# load tsv ciles as huggingface datasets
def load_tsv(path):
    df = pd.read_csv(path, sep='\t')
    df['label'] = df['Label'].map({'INFORMATIVE': 1, 'UNINFORMATIVE': 0})
    return Dataset.from_pandas(df[['Text', 'label']])

train_dataset = load_tsv("./data/train.tsv")
valid_dataset = load_tsv("./data/valid.tsv")
test_dataset = load_tsv("./data/test.tsv")

print(Counter(train_dataset["label"]))
print(Counter(valid_dataset["label"]))
print(Counter(test_dataset["label"]))

# === tokenizer & model ===
checkpoint = "albert/albert-base-v2"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2)
model.to(device)

# 30 million parameters limit
print("Trainable params:", sum(p.numel() for p in model.parameters() if p.requires_grad))

def tokenize(batch):
    return tokenizer(batch["Text"], padding=True, truncation=True)

# tokenize datasets
train_encoded = train_dataset.map(tokenize, batched=True) # CHANGED: Use original train_dataset
valid_encoded = valid_dataset.map(tokenize, batched=True) # CHANGED: Use original valid_dataset
test_encoded = test_dataset.map(tokenize, batched=True)

# combine train and valid datsets - for final run after hyperparameter tuning
full_train_encoded = concatenate_datasets([train_encoded, valid_encoded])

# === set random seed (for reproducibility) ===
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# metrics
def compute_metrics(pred):
    preds = pred.predictions.argmax(-1)
    return {
        "accuracy": accuracy_score(pred.label_ids, preds),
        "f1": f1_score(pred.label_ids, preds)
    }

# === training ! ===
training_args = TrainingArguments(
    output_dir="./results",
    seed=SEED,
    report_to=["tensorboard"],
    logging_dir="./logs",
    logging_strategy="steps",
    logging_steps=100, 
    eval_strategy="steps",
    eval_steps=200,  
    save_strategy="steps",
    save_steps=200,  
    per_device_train_batch_size=32,  
    per_device_eval_batch_size=64,  
    num_train_epochs=8, 
    weight_decay=0.1,   
    learning_rate=3e-5,  
    warmup_ratio=0.1,   
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    fp16=True,  
    gradient_accumulation_steps=2, 
    lr_scheduler_type="cosine_with_restarts",
    max_grad_norm=1.0,
    save_total_limit=2, # for disk size on colab
)

# trainer setup
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=full_train_encoded,
    eval_dataset=test_encoded,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3, early_stopping_threshold=0.005)]
)

# train the model
trainer.train()

# evaluate model on text set
eval_results = trainer.evaluate()
print(f"Final evaluation results: {eval_results}")

#save final model
trainer.save_model("./final_model")
print("Training complete. Final model saved to './final_model'")

# === prediction and evaluation ===
test_df = pd.read_csv("./data/test.tsv", sep="\t")
test_df["label"] = test_df["Label"].map({"UNINFORMATIVE": 0, "INFORMATIVE": 1})
test_ds = Dataset.from_pandas(test_df)

# tokenize test data
test_encoded = test_ds.map(
    lambda b: tokenizer(b["Text"], padding="max_length", truncation=True, max_length=128),
    batched=True
)
test_encoded.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

# predict
predictions = trainer.predict(test_encoded)
preds = predictions.predictions.argmax(-1)
labels = predictions.label_ids

# print metrics
cm = confusion_matrix(labels, preds)
print("Confusion Matrix:")
print(cm)

acc = accuracy_score(labels, preds)
f1 = f1_score(labels, preds)

print(f"Test Accuracy: {acc:.4f}")
print(f"Test F1 Score: {f1:.4f}")

# classification report
report = classification_report(labels, preds, target_names=["UNINFORMATIVE", "INFORMATIVE"])
print("\nClassification Report:")
print(report)

# === plot confusion matrix === 
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["UNINFORMATIVE", "INFORMATIVE"], yticklabels=["UNINFORMATIVE", "INFORMATIVE"])
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.show()

# === generate prediction.csv ===
label_map = {1: "INFORMATIVE", 0: "UNINFORMATIVE"}
submission = pd.DataFrame({
    "Id": test_df["Id"],
    "Label": [label_map[p] for p in preds]
})
submission.to_csv("prediction.csv", index=False)