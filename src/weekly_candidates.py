import os
from datetime import datetime
from typing import Optional, cast

import polars as pl
from loguru import logger
from typer import Argument, Option, Typer, echo

import fetcher
import schemas
import utils
from utils import create_progress_bar, notify

app = Typer(no_args_is_help=True)


@app.command()
def daily_backtest(
    to: Optional[str] = Option(
        None, help="JQuants APIからfetchしてくる最新の日付(YYYYMMDD形式)"
    ),
    force_fetch: bool = Option(False, help="最新のデータを取得するかどうか"),
):
    """
    デイリーバックテストを実行し、結果をファイルに保存する。
    """

    # ロング候補とショート候補の銘柄コードを取得
    bullish_candidates = open_latest_file(schemas.StockStatus.BULLISH)
    if bullish_candidates is None:
        return
    bearish_candidates = open_latest_file(schemas.StockStatus.BEARISH)
    if bearish_candidates is None:
        return

    latest_market_date = fetcher.get_latest_market_date()

    # 当日のリターンを計算
    bullish_results = calculate_daily_returns(
        bullish_candidates,
        latest_market_date=latest_market_date,
        to=to,
        force_fetch=force_fetch,
    )
    bearish_results = calculate_daily_returns(
        bearish_candidates,
        latest_market_date=latest_market_date,
        to=to,
        force_fetch=force_fetch,
    )

    # 結果をサマライズ
    bullish_summary = summarize_results(bullish_results)
    bearish_summary = summarize_results(bearish_results)

    # topix のデータを取得
    topix_summary = summarize_topix(to)

    # bullish_summary、bearish_summary、topix_summaryの結果をDataFrameに変換
    df = pl.DataFrame(
        {
            "Return Average": [
                "Latest",
                "2days ago",
                "3days ago",
                "4days ago",
                "5days ago",
            ],
            "Bullish": bullish_summary,
            "Bearish": bearish_summary,
            "Topix": topix_summary,
        }
    )
    df = df.with_columns(
        [
            (pl.col("Bullish") - pl.col("Topix")).alias("Bull-Tx"),
            (pl.col("Bearish") - pl.col("Topix")).alias("Bear-Tx"),
        ]
    )
    notify("デイリーバックテストの結果")
    logger.debug(df)
    echo(df)

    # notify("ロング候補の結果")
    # notify(format_json(bullish_summary))
    # notify("ショート候補の結果")
    # notify(format_json(bearish_summary))
    # notify("TOPIXの結果")
    # notify(format_json(topix_summary))


def summarize_topix(to: Optional[str]):
    # topix のデータを取得
    topix_df = fetcher.fetch_topix(to=to)
    latest_return = utils.calculate_close_ratio(topix_df)
    two_days_ago_return = utils.calculate_close_ratio(topix_df, -2, -3)
    three_days_ago_return = utils.calculate_close_ratio(topix_df, -3, -4)
    four_days_ago_return = utils.calculate_close_ratio(topix_df, -4, -5)
    five_days_ago_return = utils.calculate_close_ratio(topix_df, -5, -6)

    return [
        latest_return,
        two_days_ago_return,
        three_days_ago_return,
        four_days_ago_return,
        five_days_ago_return,
    ]
    # return {
    #     "latest_return": latest_return,
    #     "two_days_ago_return": two_days_ago_return,
    #     "three_days_ago_return": three_days_ago_return,
    #     "four_days_ago_return": four_days_ago_return,
    #     "five_days_ago_return": five_days_ago_return,
    # }


def summarize_results(daily_results: list[schemas.DailyResults]) -> list[float]:
    """
    バックテスト結果のリストを受け取り、
    結果を集計して返す。

    Parameters:
        results (list[float]): バックテスト結果のリスト

    Returns:
        dict: 集計結果
    """
    if len(daily_results) == 0:
        return [0.0, 0.0, 0.0, 0.0, 0.0]

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

    # # 上昇した銘柄の割合を計算
    # latest_win_ratio = round(latest_positive_count / len(daily_results), 2)
    # two_days_ago_win_ratio = round(two_days_ago_positive_count / len(daily_results), 2)
    # three_days_ago_win_ratio = round(
    #     three_days_ago_positive_count / len(daily_results), 2
    # )
    # four_days_ago_win_ratio = round(
    #     four_days_ago_positive_count / len(daily_results), 2
    # )
    # five_days_ago_win_ratio = round(
    #     five_days_ago_positive_count / len(daily_results), 2
    # )

    return [
        latest_return_average,
        two_days_ago_return_average,
        three_days_ago_return_average,
        four_days_ago_return_average,
        five_days_ago_return_average,
        # latest_win_ratio,
        # two_days_ago_win_ratio,
        # three_days_ago_win_ratio,
        # four_days_ago_win_ratio,
        # five_days_ago_win_ratio,
    ]

    # return {
    #     "latest_return_average": latest_return_average,
    #     "two_days_ago_return_average": two_days_ago_return_average,
    #     "three_days_ago_return_average": three_days_ago_return_average,
    #     "four_days_ago_return_average": four_days_ago_return_average,
    #     "five_days_ago_return_average": five_days_ago_return_average,
    #     "latest_win_ratio": latest_win_ratio,
    #     "two_days_ago_win_ratio": two_days_ago_win_ratio,
    #     "three_days_ago_win_ratio": three_days_ago_win_ratio,
    #     "four_days_ago_win_ratio": four_days_ago_win_ratio,
    #     "five_days_ago_win_ratio": five_days_ago_win_ratio,
    # }


def calculate_daily_returns(
    code_list: list[str],
    latest_market_date: str,
    to: Optional[str] = None,
    force_fetch: bool = False,
) -> list[schemas.DailyResults]:
    """
    指定された複数の銘柄コードについてデイリーバックテストを実行し、
    結果のリストを返す。
    """
    results: list[schemas.DailyResults] = []
    pbar = create_progress_bar(code_list)
    latest_market_date = fetcher.get_latest_market_date()

    for code in pbar:
        pbar.set_description(f"Processing: {code}")
        df = fetcher.get_daily_quotes(
            code, latest_market_date=latest_market_date, force_fetch=force_fetch
        )
        if to:
            # "to" を文字列から日付型に変換（YYYY-MM-DD）
            to_date = datetime.strptime(to, "%Y-%m-%d").date()
            df = df.filter(pl.col("date") <= to_date)

        latest_close = cast(float, df["close"].item(-1))
        one_day_ago_close = cast(float, df["close"].item(-2))
        two_days_ago_close = cast(float, df["close"].item(-3))
        three_days_ago_close = cast(float, df["close"].item(-4))
        four_days_ago_close = cast(float, df["close"].item(-5))
        five_days_ago_close = cast(float, df["close"].item(-6))

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
        selected_files = [file for file in files if "bull" in file]
    elif status == schemas.StockStatus.BEARISH:
        selected_files = [file for file in files if "bear" in file]
    else:
        return

    logger.debug(selected_files)

    latest_file = os.path.join(path, selected_files[0])
    with open(latest_file, "r") as file:
        data = file.read().strip()

    # カンマで区切ってリストにする
    tokens = [token.strip() for token in data.split(",") if token.strip()]

    # "TSE:" を除去して整数に変換
    candidate_ids: list[str] = []
    for token in tokens:
        if token.startswith("TSE:"):
            num_part = token[len("TSE:") :]
            candidate_ids.append(num_part)
    logger.debug(candidate_ids)

    return candidate_ids


@app.command()
def create_weekly_stock_candidates(
    to: Optional[str] = Option(None, help="基準となる最新の日付(YYYY-MM-DD形式)"),
    force_fetch: bool = Option(False, help="最新のデータを取得するかどうか"),
):
    """
    日経225の各銘柄について週足データを作成し、
    SMAの条件に応じてロング候補とショート候補を判定・追加し、
    結果をファイルに保存する。
    """

    date = to if to else datetime.now().strftime("%Y-%m-%d")
    message = f"基準日{date} 週足データの作成を開始します。"
    notify(message)

    # CSVから日経225のコード一覧を取得
    nikkei225 = fetcher.load_csv(schemas.CsvFile.NIKKEI225)
    code_list: list[str] = nikkei225.get_column("code").to_list()

    # 最新の取引日を取得
    latest_market_date = fetcher.get_latest_market_date()

    bullish_candidates: list[str] = []
    bearish_candidates: list[str] = []

    pbar = create_progress_bar(code_list)

    for code in pbar:
        pbar.set_description(f"Processing: {code}")

        df = make_ohlc_weekly(
            code, to=to, force_fetch=force_fetch, latest_market_date=latest_market_date
        )

        stock_status = check_stock_weekly(df)
        pbar_message: Optional[str] = None
        if stock_status == schemas.StockStatus.BULLISH:
            add_candidate_weekly(bullish_candidates, code)
            pbar_message = f"Bullish銘柄に追加 {code}"
        elif stock_status == schemas.StockStatus.BEARISH:
            add_candidate_weekly(bearish_candidates, code)
            pbar_message = f"Bearish銘柄に追加 {code}"

        if pbar_message:
            pbar.set_postfix({"message": pbar_message})

    # 現在日時を付与して結果を保存
    now_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_candidates(bullish_candidates, is_bull=True, now_str=now_str)
    save_candidates(bearish_candidates, is_bull=False, now_str=now_str)


def check_stock_weekly(df: pl.DataFrame) -> schemas.StockStatus:
    """
    週足データから最新の株価とSMAの状態、出来高の推移を用いて、
    bullish、bearish、neutralを判定する。

    Parameters:
        df (pl.DataFrame): 週足の株価データ

    Returns:
        schemas.StockStatus:
            BULLISH,
            BEARISH,
            NEUTRALのいずれか
    """
    # 最新週の終値とSMA値を取得
    close_price: float = df["close"].item(-1)
    sma25: float = df["SMA25"].item(-1)
    sma75: float = df["SMA75"].item(-1)
    prev_sma75: float = df["SMA75"].item(-2)
    histogram: float = df["Histogram"].item(-1)
    prev_histogram: float = df["Histogram"].item(-2)
    prev2_histogram: float = df["Histogram"].item(-3)
    prev3_histogram: float = df["Histogram"].item(-4)

    # 直近13週とその前の13週の出来高合計を取得
    recent_volume_sum: float = df["volume"][-13:].sum()
    previous_volume_sum: float = df["volume"][-26:-13].sum()

    # 共通の判定条件
    # 出来高が増加傾向
    vol_cond = recent_volume_sum >= previous_volume_sum
    # MACDヒストグラムが0より大きい
    macd_cond1 = histogram > 0
    # histogramが直近4週で最小ではない
    macd_cond2 = histogram > min(prev_histogram, prev2_histogram, prev3_histogram)

    # ロング候補の判定条件
    bull_cond1 = close_price > sma25 > sma75
    bull_cond2 = sma75 > prev_sma75

    is_bullish = bull_cond1 and bull_cond2 and vol_cond and macd_cond1 and macd_cond2
    if is_bullish:
        return schemas.StockStatus.BULLISH

    # ショート候補の判定条件
    bear_cond1 = close_price < sma25 < sma75
    bear_cond2 = sma75 < prev_sma75

    is_bearish = bear_cond1 and bear_cond2 and vol_cond and macd_cond1 and macd_cond2

    # # ショート候補の判定条件
    # is_bearish = (
    #     close_price > sma75
    #     and close_price < sma25
    #     and recent_volume_sum <= previous_volume_sum
    # )
    if is_bearish:
        return schemas.StockStatus.BEARISH

    return schemas.StockStatus.NEUTRAL


def add_candidate_weekly(candidates: list[str], code: str) -> None:
    candidate = f"TSE:{code}"
    return candidates.append(candidate)


def save_candidates(candidates: list[str], is_bull: bool, now_str: str):
    bull_or_bear = "bull" if is_bull else "bear"
    filename = f"{bull_or_bear}_candidates_{now_str}"
    candidates_str = ",".join(candidates)
    with open(f"outputs/jp/{filename}.txt", "w") as file:
        file.write(candidates_str)
        message = f"outputs/jp/{filename}.txt に保存しました。"
        notify(message)


@app.command()
def make_ohlc_weekly(
    code: str = Argument(..., help="Jquants APIからfetchしてくる銘柄コード"),
    to: Optional[str] = Option(
        None, help="JQuants APIからfetchしてくる最新の日付(YYYY-MM-DD形式)"
    ),
    force_fetch: bool = Option(False, help="最新のデータを取得するかどうか"),
    latest_market_date: Optional[str] = None,
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

    # 最新の日付を取得
    if not latest_market_date:
        latest_market_date = fetcher.get_latest_market_date()

    # 日足データを取得
    daily_df = fetcher.get_daily_quotes(
        code, latest_market_date=latest_market_date, force_fetch=force_fetch
    )

    # もし日付がstrなら、Date 型に変換してソート
    if isinstance(daily_df["date"].item(0), str):
        daily_df = daily_df.with_columns(
            pl.col("date").str.strptime(pl.Date, "%Y-%m-%d")
        ).sort("date")

    # 指定された日付までのデータを取得
    if to:
        # "to" を文字列から日付型に変換（YYYY-MM-DD）
        to_date = datetime.strptime(to, "%Y-%m-%d").date()
        daily_df = daily_df.filter(pl.col("date") <= to_date)

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
                pl.col("close").rolling_mean(window_size=5).alias("SMA5"),
                pl.col("close").rolling_mean(window_size=25).alias("SMA25"),
                pl.col("close").rolling_mean(window_size=75).alias("SMA75"),
            ]
        )
    )

    # 終値を取得して EMA12, EMA26 を計算
    close_prices: list[float] = weekly_df["close"].to_list()
    ema12: list[float] = calculate_ema(close_prices, 12)
    ema26: list[float] = calculate_ema(close_prices, 26)

    # EMA の結果を DataFrame に追加
    weekly_df = weekly_df.with_columns(
        [pl.Series("EMA12", ema12), pl.Series("EMA26", ema26)]
    )

    # MACDライン = EMA12 - EMA26
    weekly_df = weekly_df.with_columns(
        (pl.col("EMA12") - pl.col("EMA26")).alias("MACD")
    )

    # シグナルラインは MACD の9日 EMA
    macd_line: list[float] = weekly_df["MACD"].to_list()
    signal = calculate_ema(macd_line, 9)
    weekly_df = weekly_df.with_columns(pl.Series("Signal", signal))

    # ヒストグラム = MACD - シグナル
    weekly_df = weekly_df.with_columns(
        (pl.col("MACD") - pl.col("Signal")).alias("Histogram")
    )

    logger.debug(weekly_df.head())
    logger.debug(weekly_df.tail())

    return weekly_df


# EMA（指数移動平均）を計算する関数
def calculate_ema(prices: list[float], span: int) -> list[float]:
    """
    prices: 数値のリスト（例：終値のリスト）
    span: EMAの期間（例：12, 26, 9 など）
    """
    alpha = 2 / (span + 1)
    ema_values: list[float] = []
    for i, price in enumerate(prices):
        if i == 0:
            ema_values.append(price)
        else:
            ema_values.append(round(alpha * price + (1 - alpha) * ema_values[-1], 2))
    return ema_values
