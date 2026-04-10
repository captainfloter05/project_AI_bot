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

# ====================== КОНФИГУРАЦИЯ ======================
MODEL_NAME = "DeepPavlov/rubert-base-cased"
OUTPUT_DIR = "intent_model"
DATASET_FILE = "dataset.csv"
NUM_EPOCHS = 5
BATCH_SIZE = 8
LEARNING_RATE = 2e-5
MAX_LENGTH = 64
EARLY_STOPPING_PATIENCE = 2

# Расширенные примеры интентов (25–30+ фраз на каждый)
INTENT_EXAMPLES = {
    "greeting": [
        "привет", "здравствуйте", "добрый день", "доброе утро", "добрый вечер",
        "приветствую", "здравия желаю", "приветик", "хай", "здорово",
        "салют", "доброго времени суток", "привет, как дела", "здравствуйте, как жизнь",
        "добрый день, чем занимаетесь", "привет, я новенький", "здравствуйте, это бот?",
        "приветствую вас", "здравствуйте, очень приятно", "моё почтение",
        "доброго здоровья", "привет, давно не виделись", "здравствуйте, извините",
        "добрый вечер, что нового", "привет, как тебя зовут", "здравствуйте, я здесь впервые",
        "привет, погода", "здравствуйте, помогите", "доброе утро, бот"
    ],
    "goodbye": [
        "пока", "до свидания", "всего хорошего", "удачи", "до встречи",
        "прощай", "пока пока", "бывай", "счастливо", "до скорого",
        "пока, бот", "до свидания, спасибо", "всего доброго", "до связи",
        "увидимся", "прощайте", "до завтра", "все, я ухожу", "счастливо оставаться",
        "пока, удачи", "до встречи, бот", "всего наилучшего", "прощай, до новых встреч",
        "до свидания, пока", "удачи, бот", "бывайте здоровы", "разрешите откланяться",
        "пока, спасибо за общение", "до свидания, приятно было пообщаться"
    ],
    "addition": [
        "сумма 5 10", "сложи 2 и 3", "5 плюс 7", "сколько будет 12 + 15",
        "прибавь 8 к 3", "10 + 20", "сложи 4 и 6", "сумма чисел 9 и 1",
        "1 + 1", "100 + 200", "сложи 5.5 и 2.5", "15 + 15", "прибавь 7 к 8",
        "сколько получится 3 + 4", "сумма 8 и 9", "сложи 12 и 7", "5 плюс 5",
        "10 + 5", "прибавь 2 к 2", "сумма 1000 и 500", "сложи 1.2 и 3.4",
        "5 + 3", "7 + 2", "сложи 9 и 8", "прибавь 10 к 10", "сумма 25 и 25",
        "сложи 0.5 и 0.5", "сколько будет 4 + 4", "сумма 3 и 7", "сложи 6 и 4",
        "8 + 2", "прибавь 1 к 1", "сумма 12 и 8", "сложи 15 и 5", "20 + 30",
        "сколько будет 2 + 2", "вычисли 123 + 456", "прибавь 7.5 к 2.3",
        "найди сумму 40 и 60", "сколько будет 1000 плюс 2000"
    ],
    "time": [
        "сколько времени", "который час", "текущее время", "дата и время",
        "какой сегодня день", "какая дата", "какое сегодня число", "покажи время",
        "часы", "время", "сейчас сколько", "точное время", "который сейчас час",
        "дата", "сегодняшняя дата", "число", "день недели", "какой день",
        "сколько сейчас времени", "время московское", "который час в москве",
        "дата сегодня", "текущая дата", "часы и минуты", "покажи дату",
        "сколько часов", "который час сейчас", "текущее время и дата",
        "какое число сегодня", "какой сегодня день недели", "сколько минут",
        "текущий час", "дата и время сейчас", "покажи текущее время", "который час, бот",
        "какая сегодня дата", "время в россии", "подскажите точное время"
    ],
    "weather": [
        "погода в москве", "какая погода", "будет ли дождь", "прогноз погоды",
        "температура на улице", "погода сегодня", "что с погодой", "сколько градусов",
        "погода в спб", "погода в новосибирске", "прогноз на завтра", "температура в москве",
        "какая погода в лондоне", "идет ли дождь", "сильный ветер", "солнечно ли",
        "погода в екатеринбурге", "прогноз погоды на сегодня", "температура сейчас",
        "какая температура", "будет ли снег", "погода в казани", "погода в сочи",
        "прогноз на выходные", "температура воздуха", "что за погода", "погода на неделю",
        "какая погода завтра", "погода в россии", "температура в спб", "погода в киеве",
        "прогноз погоды на завтра", "будет ли дождь сегодня", "снег ли", "ветер какой",
        "погода в краснодаре", "погода в воронеже", "погода в самаре", "погода в омске",
        "узнай погоду в рязани", "какая сегодня погода, интересно?", "не подскажете погоду?",
        "что там на улице? тепло?", "стоит ли брать зонт?", "прогноз на сегодня в москве"
    ]
}

# ====================== ПОДГОТОВКА ДАТАСЕТА ======================
def create_dataset():
    if os.path.exists(DATASET_FILE):
        print(f"Файл {DATASET_FILE} уже существует. Загружаем его.")
        return pd.read_csv(DATASET_FILE)

    data = []
    for intent, phrases in INTENT_EXAMPLES.items():
        for phrase in phrases:
            data.append({"text": phrase, "intent": intent})

    df = pd.DataFrame(data)
    df.to_csv(DATASET_FILE, index=False, encoding="utf-8")
    print(f"Создан датасет: {len(df)} строк, классы: {df['intent'].unique()}")
    return df

# ====================== FINE-TUNING ======================
def tokenize_function(examples, tokenizer):
    return tokenizer(
        examples["text"],
        padding="max_length",
        truncation=True,
        max_length=MAX_LENGTH
    )

def train():
    # Загрузка датасета
    df = create_dataset()
    label_list = sorted(df["intent"].unique())
    label2id = {label: i for i, label in enumerate(label_list)}
    id2label = {i: label for label, i in label2id.items()}
    df["label"] = df["intent"].map(label2id)

    # Разделение на train/val
    train_df, val_df = train_test_split(
        df, test_size=0.2, random_state=42, stratify=df["label"]
    )
    print(f"Train: {len(train_df)}, Val: {len(val_df)}")

    # Конвертация в datasets.Dataset
    train_dataset = Dataset.from_pandas(train_df[["text", "label"]])
    val_dataset = Dataset.from_pandas(val_df[["text", "label"]])

    # Загрузка токенизатора и модели
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(label_list),
        id2label=id2label,
        label2id=label2id
    )

    # Токенизация
    train_dataset = train_dataset.map(
        lambda x: tokenize_function(x, tokenizer), batched=True
    )
    val_dataset = val_dataset.map(
        lambda x: tokenize_function(x, tokenizer), batched=True
    )

    # Удаляем ненужные колонки
    train_dataset = train_dataset.remove_columns(["text", "__index_level_0__"])
    val_dataset = val_dataset.remove_columns(["text", "__index_level_0__"])
    train_dataset.set_format("torch")
    val_dataset.set_format("torch")

    # Аргументы обучения
    training_args = TrainingArguments(
        output_dir="./results",
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        num_train_epochs=NUM_EPOCHS,
        weight_decay=0.01,
        logging_dir="./logs",
        logging_steps=10,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        save_total_limit=2,
        fp16=torch.cuda.is_available(),
    )

    # Функция метрик
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        preds = np.argmax(predictions, axis=1)
        acc = accuracy_score(labels, preds)
        return {"accuracy": acc}

    # Trainer с early stopping
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        processing_class=tokenizer,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=EARLY_STOPPING_PATIENCE)]
    )

    # Обучение
    print("Начинаем fine-tuning...")
    trainer.train()

    # Оценка на валидации
    eval_results = trainer.evaluate()
    print(f"Validation accuracy: {eval_results['eval_accuracy']:.4f}")

    # Сохранение модели и токенизатора
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"Модель сохранена в {OUTPUT_DIR}")

    # Дополнительно: классификационный отчёт на валидации
    predictions = trainer.predict(val_dataset)
    pred_labels = np.argmax(predictions.predictions, axis=1)
    true_labels = predictions.label_ids
    print("\nClassification Report (validation):")
    print(classification_report(true_labels, pred_labels, target_names=label_list))

if __name__ == "__main__":
    train()