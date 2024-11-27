import io
import os
import random
from sys import stderr

import matplotlib.pyplot
import mplfinance as mpf  # type: ignore
import pandas as pd
import polars as pl
import psutil
from loguru import logger
from PIL import Image
from tqdm import tqdm
from typer import Typer

import constants
import database
import fetcher
import schemas

app = Typer(no_args_is_help=True)


def slice_df_for_candlestick(df: pl.DataFrame, number: int):
    date = df[number - 1]["date"].item()
    # close = df[number - 1]["close"].item()
    code = df[number - 1]["code"].item()
    result_1 = df[number - 1]["result_1"].item()
    result_3 = df[number - 1]["result_3"].item()
    result_5 = df[number - 1]["result_5"].item()

    # day1 = df[number]
    # day1_close = round(100 * (df[number - 1]["day1_close"].item() - close) / close, 2)
    # day3 = df[number + 3]
    # day3_close = round(100 * (df[number - 1]["day3_close"].item() - close) / close, 2)
    # day5 = df[number + 5]
    # day5_close = round(100 * (df[number - 1]["day5_close"].item() - close) / close, 2)

    stock_sliced = df.slice(number - 21, 21)
    df_sampled = stock_sliced.with_columns(pl.col("date").str.to_date().alias("date"))
    img = create_candlestick_chart_from_df(df_sampled, code, date, write=True)
    return (img, date, result_1, result_3, result_5)


def create_candlestick_chart_from_df(
    df: pl.DataFrame,
    code: str,
    file_name: str,
    with_volume: bool = False,
    write: bool = False,
):
    # DataFrameをmplfinanceの形式に変換
    df_pd: pd.DataFrame = df.to_pandas()
    ohlc_data = df_pd[["date", "open", "high", "low", "close", "volume"]].set_index(
        "date"
    )

    # ローソク足チャートの描画
    fig, ax_list = mpf.plot(
        ohlc_data,
        type="candle",
        volume=with_volume,
        style="yahoo",
        figsize=(3, 3),
        returnfig=True,
        mav=(5),
    )

    # 横軸と縦軸の目盛りを非表示にする
    for ax in ax_list:
        ax.set_xticks([])
        ax.set_yticks([])

    # チャートを画像として保存
    if write:
        dir_name = code[:-1]
        dir_path = f"./data/img/{dir_name}"
        os.makedirs(dir_path, exist_ok=True)
        fig.savefig(f"{dir_path}/{file_name}.png")

    # figをPNG形式に変換
    buf = io.BytesIO()
    fig.savefig(buf, format="png")

    # PNGデータをBLOB型に変換
    # image = buf.getvalue()

    buf.seek(0)
    image = buf.getvalue()
    buf.close()
    matplotlib.pyplot.close(fig)

    return image


@app.command()
def create_dataset():
    logger.remove()
    logger.add(stderr, level="INFO")

    conn = database.open_db()

    topix400 = fetcher.load_csv(schemas.CsvFile.TOPIX400)
    numbers = random.sample(range(50, 1200), 1000)

    for code in tqdm(
        topix400.get_column("code"), bar_format=constants.SHORT_PROGRESS_BAR
    ):
        logger.info(f"Processing {code}...")
        df = database.select_ohlc_or_topix_with_future_close(code, conn)

        if len(df) < 1210:
            logger.info(f"{code} Data length is not enough: {len(df)}")
            continue

        for number in tqdm(numbers, bar_format=constants.SHORT_PROGRESS_BAR):
            (img, date, result_1, result_3, result_5) = slice_df_for_candlestick(
                df=df, number=number
            )

            result = schemas.Result3(
                **{
                    "code": str(code),
                    "date": date,
                    "result_1": result_1,
                    "result_3": result_3,
                    "result_5": result_5,
                    "image": img,
                }
            )

            database.insert_result(conn, result)

            # メモリリーク対策
            matplotlib.pyplot.close("all")
            # mem = psutil.virtual_memory().used / 1e9
            # mem = round(mem, 1)
            # logger.info(f"Memory Usage: {mem} GB")

    return


@app.command()
def img_reader(write: bool = False):
    conn = database.open_db()
    cursor = conn.cursor()

    cursor.execute("SELECT image, nextday_close FROM result LIMIT 1")
    image_data, label = cursor.fetchone()

    img = Image.open(io.BytesIO(image_data)).convert("RGB")

    logger.info(f"img.format: {img.format}")  # PNG
    logger.info(f"img.size  : {img.size}")  # (300, 300)
    logger.info(f"img.mode  : {img.mode}")  # RGBA
    logger.info(f"label    : {label}")

    # Save the image as PNG
    if write:
        path = "./data/output.png"
        img.save(f"{path}", "PNG")
        logger.info(f"Image has been saved as PNG to {path}")

    return
