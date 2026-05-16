import re
import torch
import spacy
from datetime import datetime, timedelta
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from weather_api import get_weather
from database import init_db

class DialogState:
    START = "start"
    WAIT_CITY = "wait_city"
    WAIT_DATE = "wait_date"

class ChatBot:
    def __init__(self, model_path="intent_model", confidence_threshold=0.65):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.model.to(self.device)
        self.model.eval()
        self.nlp = spacy.load("ru_core_news_sm")
        self.user_states = {}
        self.user_data = {}
        self.last_intent = None
        self.last_city = None
        self.bot_name = "Бот"
        self.confidence_threshold = confidence_threshold   # сохраняем параметр
        init_db()

    # ---------- Очистка входного текста ----------
    @staticmethod
    def clean_input(text: str) -> str:
        """Удаляет мусор из текста (эмодзи, лишние символы) и приводит к нижнему регистру."""
        text = text.lower()
        text = re.sub(r"[^а-яё0-9\s\.,!?\-:]", "", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    # ---------- Предсказание интента ----------
    def _predict_intent(self, text: str) -> tuple:
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=128   # увеличено по сравнению с 64
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

    # ---------- Извлечение сущностей ----------
    def _extract_city(self, text: str):
        doc = self.nlp(text)
        for ent in doc.ents:
            if ent.label_ in ("LOC", "GPE"):
                lemmas = [token.lemma_ for token in ent]
                return " ".join(lemmas).strip()
        return None

    def _extract_numbers(self, text: str):
        return [float(x) for x in re.findall(r"\d+\.?\d*", text)]

    def _extract_name(self, text: str):
        match = re.search(r"меня зовут\s+(\w+)", text, re.IGNORECASE)
        if match:
            return match.group(1).capitalize()
        match = re.search(r"(?:я|меня звать)\s+(\w+)", text, re.IGNORECASE)
        if match:
            return match.group(1).capitalize()
        return None

    def _parse_date_offset(self, text: str):
        text_lower = text.lower()
        today = datetime.now().date()
        if "послезавтра" in text_lower:
            return today + timedelta(days=2)
        elif "завтра" in text_lower:
            return today + timedelta(days=1)
        elif "сегодня" in text_lower:
            return today
        return None

    # ---------- Состояния ----------
    def _get_state(self, user_id):
        return self.user_states.get(user_id, DialogState.START)

    def _set_state(self, user_id, state):
        self.user_states[user_id] = state

    def _get_user_data(self, user_id):
        if user_id not in self.user_data:
            self.user_data[user_id] = {}
        return self.user_data[user_id]

    # ---------- Follow-up для погоды ----------
    def _handle_weather_followup(self, user_id, text, data):
        last_city = data.get('last_city')
        last_date = data.get('last_date')
        new_city = self._extract_city(text) or last_city
        new_date = self._parse_date_offset(text) or last_date
        if not new_city:
            return "Не знаю, для какого города уточнить погоду. Спросите явно, например: 'а в Москве?'"
        response = get_weather(new_city, new_date)
        data['last_city'] = new_city
        data['last_date'] = new_date
        self.last_city = new_city
        return response

    # ---------- Навыки ----------
    def greet(self, user_id=None, text=None):
        if user_id:
            user_data = self._get_user_data(user_id)
            user_name = user_data.get('user_name')
            if user_name:
                return f"Здравствуйте, {user_name}! Чем могу помочь?"
            elif text:
                name = self._extract_name(text)
                if name:
                    user_data['user_name'] = name
                    return f"Привет, {name}! Чем могу помочь?"
        return "Здравствуйте! Чем могу помочь?"

    def farewell(self):
        return "До свидания!"

    def addition(self, text: str):
        nums = self._extract_numbers(text)
        if len(nums) >= 2:
            return f"Результат: {nums[0] + nums[1]}"
        return "Не удалось распознать числа. Напишите, например: сумма 5 10"

    def subtraction(self, text: str):
        nums = self._extract_numbers(text)
        if len(nums) >= 2:
            return f"Результат: {nums[0] - nums[1]}"
        return "Не удалось распознать числа. Напишите, например: разность 10 3"

    def multiplication(self, text: str):
        nums = self._extract_numbers(text)
        if len(nums) >= 2:
            return f"Результат: {nums[0] * nums[1]}"
        return "Не удалось распознать числа. Напишите, например: умножь 5 на 6"

    def time(self):
        now = datetime.now()
        return now.strftime("Сейчас время %H:%M:%S, дата %d.%m.%Y")

    def time_only(self):
        return datetime.now().strftime("Сейчас %H:%M")

    def date(self):
        return datetime.now().strftime("Сегодня %d.%m.%Y")

    def help(self):
        return ("Я умею:\n"
                "• Показывать погоду (спросите 'погода в Москве')\n"
                "• Складывать, вычитать, умножать числа (например, 'сумма 5 10')\n"
                "• Говорить время и дату ('сколько времени', 'какая дата')\n"
                "• Отвечать на приветствия и прощания\n"
                "• Поддерживать уточнения по погоде ('а завтра?', 'а в Питере?')\n"
                "• Вести небольшой разговор ('как дела', 'тебе нравится работа?')\n"
                "• Запоминать ваше имя, если представитесь ('меня зовут Роман')")

    def smalltalk(self, text: str):
        import random
        text_lower = text.lower()
        if "нравится твоя работа" in text_lower or "любишь свою работу" in text_lower:
            return "Да, мне нравится помогать людям! Каждый день узнаю что-то новое."
        if "смысл жизни" in text_lower:
            return "Смысл жизни — в общении, познании и помощи друг другу. А вы как думаете?"
        if "мечта" in text_lower:
            return "Моя мечта — стать ещё умнее и помогать людям ещё эффективнее!"
        if "время года" in text_lower:
            return "Я люблю все времена года, но весна особенно радует пробуждением природы."
        if "устаёшь" in text_lower:
            return "Я не устаю, ведь я программа. Но я ценю каждый диалог с вами."
        if "как тебя зовут" in text_lower or "твоё имя" in text_lower:
            return f"Меня зовут {self.bot_name}. А как мне к вам обращаться?"
        responses = [
            "У меня всё отлично, спасибо! Как у вас?",
            "Нормально, занимаюсь обработкой запросов.",
            "Хорошо, а у вас?",
            "Отлично! Чем могу ещё помочь?",
            "Я в порядке, спасибо, что спросили!",
            "Прекрасно! Расскажите что-нибудь интересное."
        ]
        return random.choice(responses)

    def unknown(self):
        return "Извините, я не понимаю ваш запрос."

    # ---------- Основной метод обработки ----------
    def process(self, user_id: str, message: str) -> str:
        # Очистка входного текста
        message = self.clean_input(message)
        if not message:
            return "Вы ничего не сказали."

        state = self._get_state(user_id)
        data = self._get_user_data(user_id)
        self.last_intent = None
        self.last_city = None

        # Обработка диалоговых состояний (ожидание города/даты)
        if state == DialogState.WAIT_CITY:
            city = message.strip()
            if city:
                data['city'] = city
                self._set_state(user_id, DialogState.WAIT_DATE)
                self.last_intent = "weather_ask_city"
                return "На какую дату? (например: сегодня, завтра, 2025-05-20)"
            else:
                return "Пожалуйста, укажите город."

        if state == DialogState.WAIT_DATE:
            date_str = message.strip()
            city = data.get('city')
            if city and date_str:
                date_obj = self._parse_date_offset(date_str)
                response = get_weather(city, date_obj if date_obj else date_str)
                self.last_intent = "weather_with_date"
                self.last_city = city
                data['last_city'] = city
                data['last_date'] = date_obj if date_obj else date_str
                del self.user_data[user_id]
                self._set_state(user_id, DialogState.START)
                return response
            else:
                self._set_state(user_id, DialogState.START)
                return "Произошла ошибка. Попробуйте сначала."

        # Классификация интента
        intent, conf = self._predict_intent(message)
        self.last_intent = intent

        if conf < self.confidence_threshold:
            # Если уверенность низкая, но это не приветствие/прощание — переспросить
            if intent not in ["greeting", "goodbye", "help"]:
                return "Не уверен, что правильно понял. Переформулируйте, пожалуйста."

        # Маршрутизация
        if intent == "weather_followup":
            return self._handle_weather_followup(user_id, message, data)
        elif intent == "greeting":
            return self.greet(user_id, message)
        elif intent == "goodbye":
            return self.farewell()
        elif intent == "addition":
            return self.addition(message)
        elif intent == "subtraction":
            return self.subtraction(message)
        elif intent == "multiplication":
            return self.multiplication(message)
        elif intent == "time":
            return self.time()
        elif intent == "time_only":
            return self.time_only()
        elif intent == "date":
            return self.date()
        elif intent == "help":
            return self.help()
        elif intent == "smalltalk":
            return self.smalltalk(message)
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