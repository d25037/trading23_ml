from sys import stderr

import pandas as pd
from loguru import logger
from typer import Typer

import fetcher
import schemas

app = Typer(no_args_is_help=True)


@app.command()
def run():
    logger.remove()
    logger.add(stderr, level="INFO")

    nikkei225 = fetcher.load_csv(schemas.CsvFile.NIKKEI225)
    code_list = nikkei225.get_column("code").to_list()

    for code in code_list:
        analyze_stock(code)


@app.command("dataset")
def create_dataset(code: str) -> pd.DataFrame | None:
    df_pl = fetcher.fetch_daily_quotes(code)
    if df_pl is None:
        return logger.error("Failed to fetch data.")

    df_pd: pd.DataFrame = df_pl.to_pandas()
    df_pd_last = df_pd.tail(100)
    logger.debug(df_pd_last)

    # df_pdの最後の100行を取得
    return df_pd_last


# RSIの計算関数（期間14日）
def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    # 平均値の計算（単純移動平均）
    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


# ゴールデンクロスの検出：直前の日と比較して5日MAが25日MAを上抜けたか
def detect_golden_cross(short_ma, long_ma):
    cross = (short_ma > long_ma) & (short_ma.shift(1) <= long_ma.shift(1))
    return cross


# デッドクロスの検出
def detect_death_cross(short_ma, long_ma):
    cross = (short_ma < long_ma) & (short_ma.shift(1) >= long_ma.shift(1))
    return cross


@app.command("analyze")
def analyze_stock(code: str) -> pd.DataFrame | None:
    logger.remove()
    logger.add(stderr, level="INFO")

    df = create_dataset(code)
    if df is None:
        return logger.error("Failed to fetch data.")

    df.sort_values("date", inplace=True)

    # 移動平均線の計算
    df["ma5"] = df["close"].rolling(window=5).mean()
    df["ma25"] = df["close"].rolling(window=25).mean()
    df["ma75"] = df["close"].rolling(window=75).mean()

    # 25日移動平均線を基にしたボリンジャーバンドの計算（±2σ）
    df["std25"] = df["close"].rolling(window=25).std()
    df["upper_band"] = df["ma25"] + 2 * df["std25"]
    df["lower_band"] = df["ma25"] - 2 * df["std25"]

    # RSI(14)の計算
    df["RSI"] = calculate_rsi(df["close"], period=14)

    # 出来高の移動平均（直近5営業日と直近60営業日）
    df["vol_5"] = df["volume"].rolling(window=5).mean()
    df["vol_60"] = df["volume"].rolling(window=60).mean()

    # 5日MAと25日MA間のゴールデンクロス／デッドクロスの検出
    df["golden_cross"] = detect_golden_cross(df["ma5"], df["ma25"])
    df["death_cross"] = detect_death_cross(df["ma5"], df["ma25"])

    # 分析対象は最新のデータ（最終行）および直近10営業日でのクロスの有無
    last_idx = df.index[-1]
    recent_golden = df.loc[df.index[-10:], "golden_cross"].any()
    recent_death = df.loc[df.index[-10:], "death_cross"].any()

    # ロング候補の条件
    # 1. 現在のCloseが25日MAより上、かつ25日MAが75日MAより上
    long_cond1 = pd.to_numeric(
        df.loc[last_idx, "close"], errors="coerce"
    ) > pd.to_numeric(df.loc[last_idx, "ma25"], errors="coerce") and pd.to_numeric(
        df.loc[last_idx, "ma25"], errors="coerce"
    ) > pd.to_numeric(df.loc[last_idx, "ma75"], errors="coerce")
    # 2. 直近10日間にゴールデンクロスが発生している
    long_cond2 = recent_golden
    # 3. 上昇局面だが、RSIは60以下である
    long_cond3 = pd.to_numeric(df.loc[last_idx, "RSI"], errors="coerce") < 60
    # 4. 現在の株価がボリンジャーバンド下限付近（たとえば下限から3%以内）
    long_cond4 = (
        abs(
            pd.to_numeric(df.loc[last_idx, "close"], errors="coerce")
            - pd.to_numeric(df.loc[last_idx, "lower_band"], errors="coerce")
        )
        / pd.to_numeric(df.loc[last_idx, "lower_band"], errors="coerce")
        < 0.03
    )
    # 5. 直近5日平均出来高が60日平均の1.5倍以上、または前日比で大幅増加（200%超）している
    vol_condition = pd.to_numeric(
        df.loc[last_idx, "vol_5"], errors="coerce"
    ) > 1.5 * pd.to_numeric(df.loc[last_idx, "vol_60"], errors="coerce") or (
        pd.to_numeric(df.loc[last_idx, "volume"], errors="coerce")
        > 2 * pd.to_numeric(df.loc[last_idx - 1, "volume"], errors="coerce")
    )

    long_candidate = long_cond1 and long_cond3 and vol_condition

    # ショート候補の条件
    # 1. 現在のCloseが25日MAより下、かつ25日MAが75日MAより下
    short_cond1 = pd.to_numeric(
        df.loc[last_idx, "close"], errors="coerce"
    ) < pd.to_numeric(df.loc[last_idx, "ma25"], errors="coerce") and pd.to_numeric(
        df.loc[last_idx, "ma25"], errors="coerce"
    ) < pd.to_numeric(df.loc[last_idx, "ma75"], errors="coerce")
    # 2. 直近10日間にデッドクロスが発生している
    short_cond2 = recent_death
    # 3. 下落局面だが、RSIは40以上である
    short_cond3 = pd.to_numeric(df.loc[last_idx, "RSI"], errors="coerce") > 40
    # 4. 株価がボリンジャーバンド上限付近（上限から3%以内）
    short_cond4 = (
        abs(
            pd.to_numeric(df.loc[last_idx, "close"], errors="coerce")
            - pd.to_numeric(df.loc[last_idx, "upper_band"], errors="coerce")
        )
        / pd.to_numeric(df.loc[last_idx, "upper_band"], errors="coerce")
        < 0.03
    )
    # 5. 出来高の条件はロングと同様
    short_vol_condition = vol_condition

    short_candidate = (
        short_cond1
        # and short_cond2
        and short_cond3
        # and short_cond4
        and short_vol_condition
    )

    # 結果の表示
    logger.debug("=== 分析対象日:", df.loc[last_idx, "date"], "===")
    logger.debug("最新のClose:", df.loc[last_idx, "close"])
    logger.debug("MA5:", df.loc[last_idx, "ma5"])
    logger.debug("MA25:", df.loc[last_idx, "ma25"])
    logger.debug("MA75:", df.loc[last_idx, "ma75"])
    logger.debug("RSI(14):", df.loc[last_idx, "RSI"])
    logger.debug(
        "出来高条件 (vol_5 > 1.5*vol_60):",
        pd.to_numeric(df.loc[last_idx, "vol_5"], errors="coerce")
        > 1.5 * pd.to_numeric(df.loc[last_idx, "vol_60"], errors="coerce"),
    )

    if long_candidate:
        logger.info("【ロング候補シグナル】が検出されました！")
    else:
        logger.debug("ロング候補シグナルは検出されませんでした。")

    if short_candidate:
        logger.info("【ショート候補シグナル】が検出されました！")
    else:
        logger.debug("ショート候補シグナルは検出されませんでした。")

    # 必要に応じて、計算結果のDataFrameを返す
    return df


# 使用例（メイン処理）
if __name__ == "__main__":
    # ここにCSVファイルのパスを指定してください
    csv_file = "stock_data.csv"
    df_result = analyze_stock(csv_file)
