import pandas as pd
import random
import re
from num2words import num2words

def number_to_words(text):
    """Заменяет цифры на слова (5 -> пять)"""
    def replace(match):
        num = match.group(0)
        try:
            return num2words(int(num), lang='ru')
        except:
            return num
    return re.sub(r'\b\d+\b', replace, text)

def words_to_number(text):
    """Заменяет слова-числа на цифры (пять -> 5)"""
    mapping = {
        'один': '1', 'два': '2', 'три': '3', 'четыре': '4', 'пять': '5',
        'шесть': '6', 'семь': '7', 'восемь': '8', 'девять': '9', 'десять': '10'
    }
    for word, digit in mapping.items():
        text = re.sub(rf'\b{word}\b', digit, text)
    return text

def add_asr_noise(text):
    """Имитирует типичные ошибки ASR"""
    noise_map = {
        'привет': 'превед',
        'здравствуйте': 'здрасти',
        'погода': 'пагода',
        'москва': 'масква',
        'сколько': 'сколька',
        'сейчас': 'щас',
        'пожалуйста': 'пажалуйста',
    }
    for correct, noisy in noise_map.items():
        if random.random() < 0.3:
            text = text.replace(correct, noisy)
    return text

def augment_dataset(input_csv="dataset.csv", output_csv="dataset_augmented.csv", augment_factor=3):
    df = pd.read_csv(input_csv)
    augmented_rows = []
    for _, row in df.iterrows():
        original_text = row['text']
        intent = row['intent']
        # добавляем оригинал
        augmented_rows.append({'text': original_text, 'intent': intent})
        for _ in range(augment_factor):
            new_text = original_text
            # 1. Замена цифр на слова и обратно
            if random.random() < 0.5:
                new_text = number_to_words(new_text)
            else:
                new_text = words_to_number(new_text)
            # 2. ASR шум
            if random.random() < 0.4:
                new_text = add_asr_noise(new_text)
            # 3. Перестановка слов (для длинных фраз)
            if len(new_text.split()) >= 3 and random.random() < 0.2:
                words = new_text.split()
                random.shuffle(words)
                new_text = ' '.join(words)
            augmented_rows.append({'text': new_text, 'intent': intent})
    new_df = pd.DataFrame(augmented_rows)
    new_df.to_csv(output_csv, index=False, encoding="utf-8")
    print(f"Аугментация завершена. Исходный размер: {len(df)}, новый: {len(new_df)}")
    return new_df

if __name__ == "__main__":
    augment_dataset()