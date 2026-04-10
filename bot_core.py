import re
import torch
import spacy
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from weather_api import get_weather
from database import init_db

class DialogState:
    START = "start"
    WAIT_CITY = "wait_city"
    WAIT_DATE = "wait_date"

class ChatBot:
    def __init__(self, model_path="intent_model"):
        # Загрузка BERT модели для классификации интентов
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.model.to(self.device)
        self.model.eval()

        # Загрузка spaCy для NER (город)
        self.nlp = spacy.load("ru_core_news_sm")

        # Состояния пользователей
        self.user_states = {}
        self.user_data = {}

        # Последние данные для логирования
        self.last_intent = None
        self.last_city = None

        init_db()

    def _predict_intent(self, text: str) -> tuple:
        """Возвращает (intent_name, confidence)."""
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=64
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=1)
            confidence, pred_idx = torch.max(probs, dim=1)
            pred_idx = pred_idx.item()
            confidence = confidence.item()

        id2label = self.model.config.id2label
        intent = id2label[pred_idx]
        return intent, confidence

    def _extract_city(self, text: str):
        doc = self.nlp(text)
        for ent in doc.ents:
            if ent.label_ in ("LOC", "GPE"):
                lemmas = [token.lemma_ for token in ent]
                return " ".join(lemmas).strip()
        return None

    def _extract_numbers(self, text: str):
        return [float(x) for x in re.findall(r"\d+\.?\d*", text)]

    def greet(self):
        return "Здравствуйте! Чем могу помочь?"

    def farewell(self):
        return "До свидания!"

    def addition(self, text: str):
        nums = self._extract_numbers(text)
        if len(nums) >= 2:
            return f"Результат: {nums[0] + nums[1]}"
        return "Не удалось распознать числа. Напишите, например: сумма 5 10"

    def time(self):
        from datetime import datetime
        return datetime.now().strftime("Сейчас время %H:%M:%S, %d.%m.%Y")

    def unknown(self):
        return "Извините, я не понимаю ваш запрос."

    def _get_state(self, user_id):
        return self.user_states.get(user_id, DialogState.START)

    def _set_state(self, user_id, state):
        self.user_states[user_id] = state

    def _get_user_data(self, user_id):
        if user_id not in self.user_data:
            self.user_data[user_id] = {}
        return self.user_data[user_id]

    def process(self, user_id: str, message: str) -> str:
        state = self._get_state(user_id)
        data = self._get_user_data(user_id)
        self.last_intent = None
        self.last_city = None

        # Состояние ожидания города
        if state == DialogState.WAIT_CITY:
            city = message.strip()
            if city:
                data['city'] = city
                self._set_state(user_id, DialogState.WAIT_DATE)
                self.last_intent = "weather_ask_city"
                return "На какую дату? (например: завтра, 2025-05-20)"
            else:
                return "Пожалуйста, укажите город."

        # Состояние ожидания даты
        if state == DialogState.WAIT_DATE:
            date = message.strip()
            city = data.get('city')
            if city and date:
                response = get_weather(city, date)
                self.last_intent = "weather_with_date"
                self.last_city = city
                # Очищаем данные пользователя
                del self.user_data[user_id]
                self._set_state(user_id, DialogState.START)
                return response
            else:
                self._set_state(user_id, DialogState.START)
                return "Произошла ошибка. Попробуйте сначала."

        # Основной START – определяем интент
        intent, conf = self._predict_intent(message)
        self.last_intent = intent

        if conf < 0.5:
            return "Не уверен в ответе. Переформулируйте, пожалуйста."

        if intent == "greeting":
            return self.greet()
        elif intent == "goodbye":
            return self.farewell()
        elif intent == "addition":
            return self.addition(message)
        elif intent == "time":
            return self.time()
        elif intent == "weather":
            city = self._extract_city(message)
            if city:
                data['city'] = city
                self._set_state(user_id, DialogState.WAIT_DATE)
                self.last_intent = "weather_ask_date"
                return "На какую дату?"
            else:
                self._set_state(user_id, DialogState.WAIT_CITY)
                self.last_intent = "weather_ask_city"
                return "В каком городе?"
        else:
            return self.unknown()