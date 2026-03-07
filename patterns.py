import re
from handler import (
    handle_greeting,
    handle_farewell,
    handle_weather,
    handle_addition,
    handle_time,
    handle_unknown
)

patterns = [
    (re.compile(r"^(привет|здравствуйте)", re.IGNORECASE), handle_greeting),
    (re.compile(r"^(пока|до свидания)", re.IGNORECASE), handle_farewell),
    (re.compile(r"^погода\s+(.+)", re.IGNORECASE), handle_weather),
    (re.compile(r"^сумма\s+(\d+\.?\d*)\s+(\d+\.?\d*)", re.IGNORECASE), handle_addition),
    (re.compile(r"(сколько времени|который час|текущее время|дата и время|какой сегодня день|какая дата|какое сегодня число)", re.IGNORECASE), handle_time),
]

default_pattern = (re.compile(r".*"), handle_unknown)