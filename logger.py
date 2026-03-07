import datetime

LOG_FILE = "chat_log.txt"

def log_message(user_message: str, bot_response: str):
    """Записывает сообщение пользователя и ответ бота в файл с временной меткой."""
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(f"[{timestamp}] User: {user_message}\n")
        f.write(f"[{timestamp}] Bot: {bot_response}\n")
        f.write("-" * 50 + "\n")