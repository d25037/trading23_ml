from json import dumps

from loguru import logger
from tqdm import tqdm
from typer import echo

import constants


def notify(message: str, level: str = "info"):
    """
    ユーザーにメッセージを表示するとともに、指定したログレベルで記録する。
    levelには "info", "error", "debug" などを指定できる。
    """
    echo(message)
    # 指定されたログレベルに対応するloggerメソッドを取得
    log_func = getattr(logger, level.lower(), logger.info)
    log_func(message)


def format_json(data: dict) -> str:
    """辞書を整形済みのJSON文字列に変換するヘルパー関数"""
    return dumps(data, ensure_ascii=False, indent=2)


def create_progress_bar(iterable, desc: str = "Processing"):
    return tqdm(iterable, bar_format=constants.SHORT_PROGRESS_BAR, desc=desc)
