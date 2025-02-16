from datetime import datetime
from sys import stderr
from typing import Optional

import polars as pl
from loguru import logger
from tqdm import tqdm
from typer import Typer

import constants
import fetcher
import schemas

app = Typer(no_args_is_help=True)


@app.command()
def analyze_weekly_stock_candidates():
    """
    日経225の各銘柄について週足データを作成し、
    SMAの条件に応じてロング候補とショート候補を判定・追加し、
    結果をファイルに保存する。
    """
    logger.remove()
    logger.add(stderr, level="INFO")

    # CSVから日経225のコード一覧を取得
    nikkei225 = fetcher.load_csv(schemas.CsvFile.NIKKEI225)
    code_list = nikkei225.get_column("code").to_list()

    long_candidates, short_candidates = [], []

    pbar = tqdm(code_list, bar_format=constants.SHORT_PROGRESS_BAR, desc="Processing")

    for code in pbar:
        pbar.set_description(f"Processing: {code}")

        df = make_ohlc_weekly(code)
        if df is None:
            pbar.set_postfix(
                {"message": f"データが存在しないため {code} はスキップします。"}
            )
            continue

        stock_status = check_stock_weekly(df)
        message = None
        if stock_status == schemas.StockStatus.BULLISH:
            add_candidate_weekly(long_candidates, code)
            message = f"ロング候補に追加 {code}"
        elif stock_status == schemas.StockStatus.BEARISH:
            add_candidate_weekly(short_candidates, code)
            message = f"ショート候補に追加 {code}"

        if message:
            pbar.set_postfix({"message": message})

    # 現在日時を付与して結果を保存
    now_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_candidates(long_candidates, is_long=True, now_str=now_str)
    save_candidates(short_candidates, is_long=False, now_str=now_str)


def check_stock_weekly(df: pl.DataFrame) -> schemas.StockStatus:
    """
    週足データから最新の株価とSMAの状態、出来高の推移を用いて、
    ロング候補、ショート候補、または中立状態を判定する。

    Parameters:
        df (pl.DataFrame): 週足の株価データ

    Returns:
        schemas.StockStatus:
            BULLISH（ロング候補）,
            BEARISH（ショート候補）,
            NEUTRAL（中立）のいずれか
    """
    # 最新週の終値とSMA値を取得
    close_price = df["close"].item(-1)
    sma25 = df["SMA25"].item(-1)
    sma75 = df["SMA75"].item(-1)
    previous_sma75 = df["SMA75"].item(-2)

    # 直近13週とその前の13週の出来高合計を取得
    recent_volume_sum = df["volume"][-13:].sum()
    previous_volume_sum = df["volume"][-26:-13].sum()

    # ロング候補の判定条件
    is_bullish = (
        close_price > sma25 > sma75
        and sma75 > previous_sma75
        and recent_volume_sum >= previous_volume_sum * 0.7
    )
    if is_bullish:
        return schemas.StockStatus.BULLISH

    # ショート候補の判定条件
    is_bearish = (
        close_price < sma25 < sma75
        and sma75 < previous_sma75
        and recent_volume_sum <= previous_volume_sum * 1.3
    )
    if is_bearish:
        return schemas.StockStatus.BEARISH

    return schemas.StockStatus.NEUTRAL


def add_candidate_weekly(candidates: list[str], code: str) -> None:
    candidate = f"TSE:{code}"
    return candidates.append(candidate)


def save_candidates(candidates: list[str], is_long: bool, now_str: str):
    long_or_short = "long" if is_long else "short"
    filename = f"{long_or_short}_candidates_{now_str}"
    candidates_str = ",".join(candidates)
    with open(f"outputs/{filename}.txt", "w") as file:
        file.write(candidates_str)
        logger.info(f"outputs/{filename}.txt に保存しました。")


@app.command()
def make_ohlc_weekly(code: str) -> Optional[pl.DataFrame]:
    """
    指定された銘柄コードの株価データ（日足）を取得し、
    週足のOHLCデータに集約、さらに25週・75週のSMAを計算して返す。

    Parameters:
        code (str): 銘柄コード

    Returns:
        Optional[pl.DataFrame]: 週足に集約されたデータ。データが取得できない場合は None を返す。
    """
    # 日足データを取得
    daily_df = fetcher.fetch_daily_quotes(code)
    if daily_df is None:
        return None

    # 日付を Date 型に変換してソート
    daily_df = daily_df.with_columns(
        pl.col("date").str.strptime(pl.Date, "%Y-%m-%d")
    ).sort("date")

    # 週足に集約し、OHLCと出来高の集計を行い、その後SMA計算
    weekly_df = (
        daily_df.group_by_dynamic(
            index_column="date",
            every="1w",
            period="1w",  # 1週間ごとのウィンドウ
        )
        .agg(
            [
                pl.col("open").first().alias("open"),
                pl.col("high").max().alias("high"),
                pl.col("low").min().alias("low"),
                pl.col("close").last().alias("close"),
                pl.col("volume").sum().alias("volume"),
            ]
        )
        .sort("date")
        .with_columns(
            [
                pl.col("close").rolling_mean(window_size=25).alias("SMA25"),
                pl.col("close").rolling_mean(window_size=75).alias("SMA75"),
            ]
        )
    )

    logger.debug(weekly_df)
    logger.debug(weekly_df["SMA25"].item(-1))

    return weekly_df
