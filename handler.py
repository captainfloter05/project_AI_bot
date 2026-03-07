import requests
from datetime import datetime

def handle_greeting(match):
    return "Здравствуйте! Чем могу помочь?"

def handle_farewell(match):
    return "До свидания!"

def handle_weather(match):
    city = match.group(1)
    # Имитация ответа о погоде
    return f"Погода в {city}: +15°C, облачно."

def handle_addition(match):
    try:
        a = float(match.group(1))
        b = float(match.group(2))
        return f"Результат: {a + b}"
    except ValueError:
        return "Ошибка: введите два числа."

def handle_time(match):
    now = datetime.now()
    formatted = now.strftime("Сейчас время %H:%M:%S, %d.%m.%Y")
    return formatted

def handle_unknown(match):
    return "Извините, я не понимаю ваш запрос."