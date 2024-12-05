import os
from enum import IntEnum
from typing import Literal


class Color(IntEnum):
    BLACK = 30
    RED = 31
    GREEN = 32
    YELLOW = 33
    BLUE = 34
    PURPLE = 35
    CYAN = 36
    WHITE = 37


class Style(IntEnum):
    NORMAL = 0
    BOLD = 1
    ITALIC = 3
    URL = 4
    BLINK = 5
    SELECTED = 7


class LogLevel(IntEnum):
    DEBUG = 10
    INFO = 20
    WARNING = 30
    ERROR = 40
    CRITICAL = 50


def colorize(text, color=Color.WHITE, style=Style.NORMAL):
    return f"\033[{style};{color}m{text}\033[0m"


def set_log_level(level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]):
    os.environ["LOGLEVEL"] = LogLevel[level].name


def get_log_level() -> int:
    level = os.environ.get("LOGLEVEL", "INFO")
    return LogLevel[level].value


def info(text="", end="\n"):
    if get_log_level() <= LogLevel.INFO.value:
        print(colorize(text, Color.GREEN), end=end)


def debug(text="", end="\n"):
    if get_log_level() <= LogLevel.DEBUG.value:
        print(colorize(text, Color.BLUE), end=end)


def warning(text="", end="\n"):
    if get_log_level() <= LogLevel.WARNING.value:
        print(colorize(text, Color.YELLOW), end=end)


def error(text="", end="\n"):
    if get_log_level() <= LogLevel.ERROR.value:
        print(colorize(text, Color.RED), end=end)


def critical(text="", end="\n"):
    if get_log_level() <= LogLevel.CRITICAL.value:
        print(colorize(text, Color.PURPLE, Style.BOLD), end=end)
