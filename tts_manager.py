# tts_manager.py
import pyttsx3
import threading

class TTSManager:
    def __init__(self):
        # Ничего не инициализируем глобально
        pass

    def speak(self, text: str, async_mode: bool = True):
        if not text:
            return

        def _speak():
            # Каждый поток создаёт свой движок
            engine = pyttsx3.init()
            
            # Настраиваем русский голос (если есть)
            voices = engine.getProperty('voices')
            for voice in voices:
                if 'russian' in voice.name.lower():
                    engine.setProperty('voice', voice.id)
                    break
            engine.setProperty('rate', 150)   # скорость речи
            
            engine.say(text)
            engine.runAndWait()
            engine.stop()   # явно останавливаем движок

        if async_mode:
            threading.Thread(target=_speak, daemon=True).start()
        else:
            _speak()

tts_manager = TTSManager()