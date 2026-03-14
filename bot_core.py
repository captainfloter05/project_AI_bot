import re
from datetime import datetime
import spacy
from weather_api import get_weather
from database import init_db

# Определяем состояния диалога
class DialogState:
    START = "start"
    WAIT_CITY = "wait_city"
    WAIT_DATE = "wait_date"          # для дополнительного задания

class ChatBot:
    def __init__(self):
        # Загружаем русскую модель spaCy
        self.nlp = spacy.load("ru_core_news_sm")
        self.patterns = []
        self._register_patterns()
        
        # Хранилище состояний и данных пользователей
        self.user_states = {}          # user_id -> состояние
        self.user_data = {}             # user_id -> dict с временными данными (город, дата)
        
        # Для логирования (последние intent и city)
        self.last_intent = None
        self.last_city = None
        
        init_db()

    def _register_patterns(self):
        self.patterns.append((re.compile(r"^(привет|здравствуйте)", re.IGNORECASE), self.greet))
        self.patterns.append((re.compile(r"^(пока|до свидания)", re.IGNORECASE), self.farewell))
        self.patterns.append((re.compile(r"^сумма\s+(\d+\.?\d*)\s+(\d+\.?\d*)", re.IGNORECASE), self.addition))
        self.patterns.append((re.compile(
            r"(сколько времени|который час|текущее время|дата и время|какой сегодня день|какая дата|какое сегодня число)",
            re.IGNORECASE), self.time))
        self.default_handler = self.unknown

    # --- Управление состояниями ---
    def _get_state(self, user_id):
        """Возвращает текущее состояние пользователя (по умолчанию START)."""
        return self.user_states.get(user_id, DialogState.START)

    def _set_state(self, user_id, state):
        """Устанавливает состояние пользователя."""
        self.user_states[user_id] = state

    def _get_user_data(self, user_id):
        """Возвращает словарь с данными пользователя (создаёт при необходимости)."""
        if user_id not in self.user_data:
            self.user_data[user_id] = {}
        return self.user_data[user_id]

    # --- NLP методы ---
    def _extract_city(self, text):
        """Извлекает город через NER и приводит к именительному падежу."""
        doc = self.nlp(text)
        for ent in doc.ents:
            if ent.label_ in ("LOC", "GPE"):
                lemmas = [token.lemma_ for token in ent]
                return " ".join(lemmas).strip()
        return None

    def _is_weather_query(self, text):
        keywords = ["погода", "прогноз", "температура", "градус", "потепление", "похолодание"]
        return any(kw in text.lower() for kw in keywords)

    # --- Обработчики команд (для состояния START) ---
    def greet(self, match): 
        return "Здравствуйте! Чем могу помочь?"
    
    def farewell(self, match): 
        return "До свидания!"
    
    def addition(self, match):
        try:
            a, b = float(match.group(1)), float(match.group(2))
            return f"Результат: {a + b}"
        except ValueError:
            return "Ошибка: введите два числа."
    
    def time(self, match):
        return datetime.now().strftime("Сейчас время %H:%M:%S, %d.%m.%Y")
    
    def unknown(self, match=None):
        return "Извините, я не понимаю ваш запрос."

    # --- Основной метод обработки сообщения ---
    def process(self, user_id: str, message: str) -> str:
        state = self._get_state(user_id)
        data = self._get_user_data(user_id)
        
        # Логируемые параметры (будут сохранены в БД)
        self.last_intent = None
        self.last_city = None

        # --- Обработка в зависимости от состояния ---
        if state == DialogState.START:
            # Проверяем, является ли сообщение запросом погоды
            if self._is_weather_query(message):
                city = self._extract_city(message)
                if city:
                    # Город найден – сразу отвечаем (без даты)
                    response = get_weather(city)
                    self.last_intent = "weather"
                    self.last_city = city
                    return response
                else:
                    # Город не указан – переходим в WAIT_CITY
                    self._set_state(user_id, DialogState.WAIT_CITY)
                    self.last_intent = "ask_city"
                    return "В каком городе вас интересует погода?"
            
            # Если не погода – проверяем паттерны команд
            for pattern, handler in self.patterns:
                if match := pattern.search(message):
                    response = handler(match)
                    # Определяем интент для лога
                    if handler == self.greet:
                        self.last_intent = "greet"
                    elif handler == self.farewell:
                        self.last_intent = "farewell"
                    elif handler == self.addition:
                        self.last_intent = "addition"
                    elif handler == self.time:
                        self.last_intent = "time"
                    else:
                        self.last_intent = "unknown"
                    return response
            
            # Ничего не подошло
            self.last_intent = "unknown"
            return self.unknown()

        elif state == DialogState.WAIT_CITY:
            # Ожидаем название города
            city = message.strip()
            if city:
                data['city'] = city
                # Переходим к запросу даты (доп. задание)
                self._set_state(user_id, DialogState.WAIT_DATE)
                self.last_intent = "ask_date"
                return "На какую дату?"
            else:
                return "Пожалуйста, укажите город."

        elif state == DialogState.WAIT_DATE:
            # Ожидаем дату
            date = message.strip()
            city = data.get('city')
            if city and date:
                # Вызываем погоду с датой (get_weather теперь принимает дату)
                response = get_weather(city, date)
                # Очищаем данные и возвращаемся в START
                del self.user_data[user_id]
                self._set_state(user_id, DialogState.START)
                self.last_intent = "weather_with_date"
                self.last_city = city
                return response
            else:
                # Если вдруг city пропало (ошибка) – перезапускаем
                self._set_state(user_id, DialogState.START)
                return "Произошла ошибка. Попробуйте сначала."

        # На всякий случай (не должно достигаться)
        return self.unknown()