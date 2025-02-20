import os
from datetime import datetime
from typing import Optional

import polars as pl
from loguru import logger
from typer import Argument, Option, Typer

import fetcher
import schemas
from utils import create_progress_bar, format_json, notify

app = Typer(no_args_is_help=True)


@app.command()
def daily_backtest(
    to: Optional[str] = Option(
        None, help="JQuants APIからfetchしてくる最新の日付(YYYYMMDD形式)"
    ),
):
    """
    デイリーバックテストを実行し、結果をファイルに保存する。
    """

    # ロング候補とショート候補の銘柄コードを取得
    long_candidates = open_latest_file(schemas.StockStatus.BULLISH)
    if long_candidates is None:
        return
    short_candidates = open_latest_file(schemas.StockStatus.BEARISH)
    if short_candidates is None:
        return

    # 当日のリターンを計算
    long_results = calculate_daily_returns(long_candidates, to=to)
    short_results = calculate_daily_returns(short_candidates, to=to)

    # 結果をサマライズ
    long_summary = summarize_results(long_results)
    short_summary = summarize_results(short_results)

    notify("ロング候補の結果")
    notify(format_json(long_summary))
    notify("ショート候補の結果")
    notify(format_json(short_summary))


def summarize_results(daily_results: list[schemas.DailyResults]) -> dict:
    """
    バックテスト結果のリストを受け取り、
    結果を集計して返す。

    Parameters:
        results (list[dict]): バックテスト結果のリスト

    Returns:
        dict: 集計結果
    """
    latest_total = 0.0
    two_days_ago_total = 0.0
    three_days_ago_total = 0.0
    four_days_ago_total = 0.0
    five_days_ago_total = 0.0

    latest_positive_count = 0
    two_days_ago_positive_count = 0
    three_days_ago_positive_count = 0
    four_days_ago_positive_count = 0
    five_days_ago_positive_count = 0
    for daily_result in daily_results:
        latest_total += daily_result.latest_result
        two_days_ago_total += daily_result.two_days_ago_result
        three_days_ago_total += daily_result.three_days_ago_result
        four_days_ago_total += daily_result.four_days_ago_result
        five_days_ago_total += daily_result.five_days_ago_result

        if daily_result.latest_result > 0:
            latest_positive_count += 1
        if daily_result.two_days_ago_result > 0:
            two_days_ago_positive_count += 1
        if daily_result.three_days_ago_result > 0:
            three_days_ago_positive_count += 1
        if daily_result.four_days_ago_result > 0:
            four_days_ago_positive_count += 1
        if daily_result.five_days_ago_result > 0:
            five_days_ago_positive_count += 1

    # 平均リターンを計算
    latest_return_average = round(latest_total / len(daily_results), 2)
    two_days_ago_return_average = round(two_days_ago_total / len(daily_results), 2)
    three_days_ago_return_average = round(three_days_ago_total / len(daily_results), 2)
    four_days_ago_return_average = round(four_days_ago_total / len(daily_results), 2)
    five_days_ago_return_average = round(five_days_ago_total / len(daily_results), 2)
    # 上昇した銘柄の割合を計算
    latest_win_ratio = round(latest_positive_count / len(daily_results), 2)
    two_days_ago_win_ratio = round(two_days_ago_positive_count / len(daily_results), 2)
    three_days_ago_win_ratio = round(
        three_days_ago_positive_count / len(daily_results), 2
    )
    four_days_ago_win_ratio = round(
        four_days_ago_positive_count / len(daily_results), 2
    )
    five_days_ago_win_ratio = round(
        five_days_ago_positive_count / len(daily_results), 2
    )

    return {
        "latest_return_average": latest_return_average,
        "two_days_ago_return_average": two_days_ago_return_average,
        "three_days_ago_return_average": three_days_ago_return_average,
        "four_days_ago_return_average": four_days_ago_return_average,
        "five_days_ago_return_average": five_days_ago_return_average,
        "latest_win_ratio": latest_win_ratio,
        "two_days_ago_win_ratio": two_days_ago_win_ratio,
        "three_days_ago_win_ratio": three_days_ago_win_ratio,
        "four_days_ago_win_ratio": four_days_ago_win_ratio,
        "five_days_ago_win_ratio": five_days_ago_win_ratio,
    }


def calculate_daily_returns(
    code_list: list[str], to: Optional[str] = None
) -> list[schemas.DailyResults]:
    """
    指定された複数の銘柄コードについてデイリーバックテストを実行し、
    結果のリストを返す。
    """
    results = []
    pbar = create_progress_bar(code_list)

    for code in pbar:
        pbar.set_description(f"Processing: {code}")
        df = fetcher.fetch_daily_quotes(code, to=to)

        latest_close = df["close"].item(-1)
        one_day_ago_close = df["close"].item(-2)
        two_days_ago_close = df["close"].item(-3)
        three_days_ago_close = df["close"].item(-4)
        four_days_ago_close = df["close"].item(-5)
        five_days_ago_close = df["close"].item(-6)

        latest_result = round(
            100 * (latest_close - one_day_ago_close) / one_day_ago_close, 2
        )
        two_days_ago_result = round(
            100 * (one_day_ago_close - two_days_ago_close) / two_days_ago_close, 2
        )
        three_days_ago_result = round(
            100 * (two_days_ago_close - three_days_ago_close) / three_days_ago_close, 2
        )
        four_days_ago_result = round(
            100 * (three_days_ago_close - four_days_ago_close) / four_days_ago_close, 2
        )
        five_days_ago_result = round(
            100 * (four_days_ago_close - five_days_ago_close) / five_days_ago_close, 2
        )

        daily_result = schemas.DailyResults(
            code=str(code),
            latest_result=latest_result,
            two_days_ago_result=two_days_ago_result,
            three_days_ago_result=three_days_ago_result,
            four_days_ago_result=four_days_ago_result,
            five_days_ago_result=five_days_ago_result,
        )
        results.append(daily_result)

    return results


def open_latest_file(status: schemas.StockStatus):
    """
    outputs/jp ディレクトリ内のファイルを読み込み、
    最新のロング候補またはショート候補の銘柄コードを取得する。
    """

    path = "./outputs/jp"
    files = os.listdir(path)

    # files を降順にソート
    files.sort(reverse=True)

    # status に応じてファイルを選択
    if status == schemas.StockStatus.BULLISH:
        selected_files = [file for file in files if "long" in file]
    elif status == schemas.StockStatus.BEARISH:
        selected_files = [file for file in files if "short" in file]
    else:
        return

    logger.debug(selected_files)

    latest_file = os.path.join(path, selected_files[0])
    with open(latest_file, "r") as file:
        data = file.read().strip()

    # カンマで区切ってリストにする
    tokens = [token.strip() for token in data.split(",") if token.strip()]

    # "TSE:" を除去して整数に変換
    candidate_ids = []
    for token in tokens:
        if token.startswith("TSE:"):
            num_part = token[len("TSE:") :]
            try:
                candidate_ids.append(int(num_part))
            except ValueError:
                # 数字に変換できなかった場合はスキップ
                pass
    logger.debug(candidate_ids)

    return candidate_ids


@app.command()
def analyze_weekly_stock_candidates(
    to: Optional[str] = Option(
        None, help="JQuants APIからfetchしてくる最新の日付(YYYYMMDD形式)"
    ),
):
    """
    日経225の各銘柄について週足データを作成し、
    SMAの条件に応じてロング候補とショート候補を判定・追加し、
    結果をファイルに保存する。
    """

    date = to if to else datetime.now().strftime("%Y%m%d")
    message = f"基準日{date} 週足データの作成を開始します。"
    notify(message)

    # CSVから日経225のコード一覧を取得
    nikkei225 = fetcher.load_csv(schemas.CsvFile.NIKKEI225)
    code_list = nikkei225.get_column("code").to_list()

    long_candidates, short_candidates = [], []

    pbar = create_progress_bar(code_list)

    for code in pbar:
        pbar.set_description(f"Processing: {code}")

        df = make_ohlc_weekly(code, to=to)

        stock_status = check_stock_weekly(df)
        pbar_message: Optional[str] = None
        if stock_status == schemas.StockStatus.BULLISH:
            add_candidate_weekly(long_candidates, code)
            pbar_message = f"ロング候補に追加 {code}"
        elif stock_status == schemas.StockStatus.BEARISH:
            add_candidate_weekly(short_candidates, code)
            pbar_message = f"ショート候補に追加 {code}"

        if pbar_message:
            pbar.set_postfix({"message": pbar_message})

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

    # # ショート候補の判定条件
    # is_bearish = (
    #     close_price < sma25 < sma75
    #     and sma75 < previous_sma75
    #     and recent_volume_sum <= previous_volume_sum * 1.3
    # )

    # ショート候補の判定条件
    is_bearish = (
        close_price > sma75
        and close_price < sma25
        and recent_volume_sum <= previous_volume_sum
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
    with open(f"outputs/jp/{filename}.txt", "w") as file:
        file.write(candidates_str)
        message = f"outputs/jp/{filename}.txt に保存しました。"
        notify(message)


@app.command()
def make_ohlc_weekly(
    code: str = Argument(..., help="Jquants APIからfetchしてくる銘柄コード"),
    to: Optional[str] = Option(
        None, help="JQuants APIからfetchしてくる最新の日付(YYYYMMDD形式)"
    ),
) -> pl.DataFrame:
    """
    指定された銘柄コードの株価データ（日足）を取得し、
    週足のOHLCデータに集約、さらに25週・75週のSMAを計算して返す。

    Parameters:
        code (str): 銘柄コード
        to (Optional[str]): 取得するデータの最新日付。指定しない場合は最新のデータを取得する。

    Returns:
        Optional[pl.DataFrame]: 週足に集約されたデータ。データが取得できない場合は None を返す。
    """
    # 日足データを取得
    daily_df = fetcher.fetch_daily_quotes(code, to=to)

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
