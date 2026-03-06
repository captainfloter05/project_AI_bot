import sqlite3
import datetime

DB_NAME = "chatbot.db"

def init_db():
    """Создаёт таблицу logs, если она ещё не существует."""
    conn = sqlite3.connect(DB_NAME)
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            user_message TEXT NOT NULL,
            bot_response TEXT NOT NULL
        )
    """)
    conn.commit()
    conn.close()

def save_log(user_message: str, bot_response: str):
    """Сохраняет одно сообщение пользователя и ответ бота в БД."""
    conn = sqlite3.connect(DB_NAME)
    cur = conn.cursor()
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cur.execute(
        "INSERT INTO logs (timestamp, user_message, bot_response) VALUES (?, ?, ?)",
        (timestamp, user_message, bot_response)
    )
    conn.commit()
    conn.close()