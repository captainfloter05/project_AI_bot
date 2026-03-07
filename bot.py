import re
from patterns import patterns, default_pattern
from logger import log_message

class ChatBot:
    def __init__(self):
        self.patterns = patterns + [default_pattern]  # Добавляем обработчик по умолчанию

    def process(self, message: str) -> str:
        """Обрабатывает сообщение и возвращает ответ бота."""
        for pattern, handler in self.patterns:
            match = pattern.search(message)
            if match:
                return handler(match)
        # Этот код никогда не выполнится, т.к. default_pattern всегда сработает
        return "Не понимаю запрос."

def main():
    bot = ChatBot()
    print("Привет! Я бот, я умею складывать числа и давать прогноз погоды.\n")

    while True:
        user_input = input(" Вы: ").strip()
        if not user_input:
            continue

        response = bot.process(user_input)
        print("Бот:", response)

        # Логируем диалог
        log_message(user_input, response)

        # Выход при прощании
        if response == "До свидания!":
            break

if __name__ == "__main__":
    main()