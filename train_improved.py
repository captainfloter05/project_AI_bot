import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback
)
from datasets import Dataset
from augment_data import augment_dataset  # используем аугментацию

# ====================== КОНФИГУРАЦИЯ ======================
MODEL_NAME = "DeepPavlov/rubert-base-cased"
OUTPUT_DIR = "intent_model_improved"
DATASET_FILE = "dataset.csv"
AUGMENTED_FILE = "dataset_augmented.csv"
NUM_EPOCHS = 10
BATCH_SIZE = 8
LEARNING_RATE = 2e-5
MAX_LENGTH = 128
EARLY_STOPPING_PATIENCE = 4
WARMUP_RATIO = 0.15
WEIGHT_DECAY = 0.01

def load_or_augment():
    if os.path.exists(AUGMENTED_FILE):
        print("Загрузка аугментированного датасета...")
        return pd.read_csv(AUGMENTED_FILE)
    else:
        print("Аугментированный датасет не найден. Создаём...")
        return augment_dataset(DATASET_FILE, AUGMENTED_FILE, augment_factor=4)

def train():
    df = load_or_augment()
    label_list = sorted(df["intent"].unique())
    label2id = {label: i for i, label in enumerate(label_list)}
    id2label = {i: label for label, i in label2id.items()}
    df["label"] = df["intent"].map(label2id)

    train_df, val_df = train_test_split(
        df, test_size=0.2, random_state=42, stratify=df["label"]
    )
    print(f"Train примеров: {len(train_df)}, Val: {len(val_df)}")

    train_dataset = Dataset.from_pandas(train_df[["text", "label"]])
    val_dataset = Dataset.from_pandas(val_df[["text", "label"]])

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(label_list),
        id2label=id2label,
        label2id=label2id,
        hidden_dropout_prob=0.35,
        attention_probs_dropout_prob=0.35
    )

    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=MAX_LENGTH
        )

    train_dataset = train_dataset.map(tokenize_function, batched=True)
    val_dataset = val_dataset.map(tokenize_function, batched=True)

    train_dataset = train_dataset.remove_columns(["text"])
    val_dataset = val_dataset.remove_columns(["text"])
    if "__index_level_0__" in train_dataset.column_names:
        train_dataset = train_dataset.remove_columns("__index_level_0__")
    if "__index_level_0__" in val_dataset.column_names:
        val_dataset = val_dataset.remove_columns("__index_level_0__")

    train_dataset.set_format("torch")
    val_dataset.set_format("torch")

    training_args = TrainingArguments(
        output_dir="./results_improved",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        num_train_epochs=NUM_EPOCHS,
        weight_decay=WEIGHT_DECAY,
        logging_dir="./logs_improved",
        logging_steps=20,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        save_total_limit=2,
        fp16=torch.cuda.is_available(),
        lr_scheduler_type="linear",
        warmup_ratio=WARMUP_RATIO,
    )

    def compute_metrics(eval_pred):
        if hasattr(eval_pred, 'predictions'):
            preds = np.argmax(eval_pred.predictions, axis=1)
            labels = eval_pred.label_ids
        else:
            preds = np.argmax(eval_pred[0], axis=1)
            labels = eval_pred[1]
        return {"accuracy": accuracy_score(labels, preds)}

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=EARLY_STOPPING_PATIENCE)]
    )

    trainer.train()
    eval_results = trainer.evaluate()
    print(f"Validation accuracy: {eval_results['eval_accuracy']:.4f}")

    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"Модель сохранена в {OUTPUT_DIR}")

    predictions = trainer.predict(val_dataset)
    pred_labels = np.argmax(predictions.predictions, axis=1)
    true_labels = predictions.label_ids
    print("\nClassification Report (validation):")
    print(classification_report(true_labels, pred_labels, target_names=label_list))

if __name__ == "__main__":
    train()