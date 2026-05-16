import whisper
import sounddevice as sd
from scipy.io.wavfile import write
import re
import numpy as np
import tempfile
import os

class VoiceInput:
    def __init__(self, model_size="base", sample_rate=16000):
        """
        model_size: "tiny", "base", "small", "medium", "large"
        sample_rate: частота дискретизации (Гц)
        """
        self.sample_rate = sample_rate
        self.model = whisper.load_model(model_size)
        
    def record_audio(self, duration=4, silence_timeout=2.0):
        """
        Записывает речь до наступления тишины или по истечении duration.
        Возвращает путь к временному WAV-файлу или None, если ничего не записано.
        """
        print("🎤 Говорите... (окончание по тишине или через {} сек)".format(duration))
        fs = self.sample_rate
        chunk_duration = 0.5
        chunk_samples = int(fs * chunk_duration)
        silent_chunks = 0
        required_silent = int(silence_timeout / chunk_duration)
        max_chunks = int(duration / chunk_duration)
        blocks = []
        
        for _ in range(max_chunks):
            chunk = sd.rec(chunk_samples, samplerate=fs, channels=1, dtype='int16')
            sd.wait()
            blocks.append(chunk)
            if np.max(np.abs(chunk)) < 500:
                silent_chunks += 1
            else:
                silent_chunks = 0
            if silent_chunks >= required_silent:
                break
        
        if not blocks:
            return None
        
        audio = np.concatenate(blocks, axis=0)
        temp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        write(temp_file.name, fs, audio)
        return temp_file.name
    
    def transcribe(self, audio_path, language="ru"):
        """Распознаёт аудиофайл и возвращает сырой текст."""
        if audio_path is None:
            return ""
        result = self.model.transcribe(audio_path, language=language, task="transcribe")
        return result["text"].strip()
    
    def listen(self, duration=4, silence_timeout=2.0):
        """Записывает, распознаёт и возвращает очищенный текст."""
        audio_file = self.record_audio(duration, silence_timeout)
        raw_text = self.transcribe(audio_file) if audio_file else ""
        if audio_file:
            os.unlink(audio_file)
        return self.clean_text(raw_text)
    
    @staticmethod
    def clean_text(text):
        """Очистка распознанного текста: нижний регистр, удаление шума, исправление частых ошибок."""
        if not text:
            return ""
        text = text.lower()
        # Исправление типичных ошибок Whisper
        corrections = {
            "масква": "москва",
            "питр": "питер",
            "спб": "санкт-петербург",
            "пагода": "погода",
            "скелько": "сколько",
            "привет бот": "привет, бот",
            "скока": "сколько",
            "щас": "сейчас",
        }
        for wrong, right in corrections.items():
            text = text.replace(wrong, right)
        # Оставляем только буквы, цифры, пробелы и знаки препинания
        text = re.sub(r"[^а-яё0-9\s\.,!?\-:]", "", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text