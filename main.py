from bot_core import ChatBot
from database import save_log
from tts_manager import tts_manager
from voice import VoiceInput

import os
print("Текущая директория:", os.getcwd())
print("Содержимое:", os.listdir())

def main():
    bot = ChatBot(model_path="intent_model", confidence_threshold=0.2)  # порог уверенности
    voice = VoiceInput(model_size="base")     # можно "small" для точности
    user_id = "default"

    print("Привет! Я бот. Теперь я понимаю голос и речь.")
    print("Выберите режим ввода: 'v' — голос, 't' — текст, 'q' — выход.\n")

    while True:
        mode = input("Режим (v/t/q): ").strip().lower()
        if mode == 'q':
            response = bot.farewell()
            print("Бот:", response)
            tts_manager.speak(response, async_mode=False)
            break
        elif mode == 'v':
            user_text = voice.listen(duration=5, silence_timeout=2.0)
            if not user_text:
                print("Ничего не распознано. Попробуйте ещё раз.")
                continue
            print(f"Вы сказали: {user_text}")
        elif mode == 't':
            user_text = input("Вы: ").strip()
            if not user_text:
                continue
        else:
            print("Неверный режим. Введите v, t или q.")
            continue

        response = bot.process(user_id, user_text)
        print("Бот:", response)
        tts_manager.speak(response, async_mode=True)
        save_log(user_text, response, bot.last_intent, bot.last_city)

        # Если бот попрощался — завершаем
        if response == "До свидания!":
            break

if __name__ == "__main__":
    main()