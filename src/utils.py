from json import dumps
from typing import Iterable, TypeVar

import polars as pl
from loguru import logger
from tqdm import tqdm
from typer import echo

import constants


def notify(message: str, level: str = "info") -> None:
    """
    ユーザーにメッセージを表示するとともに、指定したログレベルで記録する。
    levelには "info", "error", "debug" などを指定できる。
    """
    echo(message)
    # 指定されたログレベルに対応するloggerメソッドを取得
    log_func = getattr(logger, level.lower(), logger.info)
    log_func(message)
    return


def debug_df_lf(df_lf: pl.DataFrame | pl.LazyFrame) -> None:
    if isinstance(df_lf, pl.LazyFrame):
        df_lf = df_lf.collect()
    logger.debug(f"len: {len(df_lf)}")
    logger.debug(df_lf.head())
    logger.debug(df_lf.tail())
    return


def format_json(data: dict["str", "str"]) -> str:
    """辞書を整形済みのJSON文字列に変換するヘルパー関数"""
    return dumps(data, ensure_ascii=False, indent=2)


T = TypeVar("T")


def create_progress_bar(iterable: Iterable[T], desc: str = "Processing"):
    return tqdm(iterable, bar_format=constants.SHORT_PROGRESS_BAR, desc=desc)


def calculate_close_ratio(
    df: pl.DataFrame, today: int = -1, yesterday: int = -2
) -> float:
    # 今日の終値と昨日の終値の比率を計算
    return round(
        (df["close"].item(today) - df["close"].item(yesterday))
        / df["close"].item(yesterday)
        * 100,
        2,
    )
