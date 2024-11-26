import sqlite3

import pandas as pd
import polars as pl
from loguru import logger
from typer import Typer

import schemas
from constants import DB_PATH

app = Typer(no_args_is_help=True)


def open_db():
    file_path = DB_PATH
    conn = sqlite3.connect(file_path)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS ohlc (
            code TEXT NOT NULL,
            date TEXT NOT NULL,
            open FLOAT,
            high FLOAT,
            low FLOAT,
            close FLOAT,
            volume FLOAT
        )
    """)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS topix (
            date TEXT NOT NULL,
            open FLOAT,
            high FLOAT,
            low FLOAT,
            close FLOAT
        )
    """)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS result (
            code TEXT NOT NULL,
            date TEXT NOT NULL,
            day1_close FLOAT,
            day3_close FLOAT,
            day5_close FLOAT,
            image BLOB
        )
    """)

    # conn.execute("""
    #     CREATE TABLE IF NOT EXISTS result_wo_volume (
    #         code TEXT NOT NULL,
    #         date TEXT NOT NULL
    #         nextday_open FLOAT,
    #         nextday_close FLOAT,
    #         image BLOB
    #     )
    # """)

    return conn


def insert_ohlc(df: pl.DataFrame, table_name: str = "ohlc"):
    conn = open_db()
    df_pd: pd.DataFrame = df.to_pandas()

    df_pd.to_sql(
        table_name,
        conn,
        if_exists="append",
        index=False,
        method="multi",
        chunksize=5000,
    )

    cursor = conn.cursor()
    cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
    record_count = cursor.fetchone()[0]
    logger.debug(f"number of records: {record_count}")
    return


def select_ohlc_by_code(conn: sqlite3.Connection, code: str | int):
    if isinstance(code, int):
        code = str(code)

    code = code + "0"
    return pl.read_database(
        query=f"SELECT * FROM ohlc WHERE code={code}", connection=conn
    )


def select_topix(conn: sqlite3.Connection):
    return pl.read_database(query="SELECT * FROM topix", connection=conn).lazy()


def insert_result(conn: sqlite3.Connection, result: schemas.Result2):
    logger.debug(f"date: {result.date}")
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO result (code, date, day1_close, day3_close, day5_close, image) values (?, ?, ?, ?, ?, ?)",
        (
            result.code,
            result.date,
            result.day1_close,
            result.day3_close,
            result.day5_close,
            result.image,
        ),
    )

    conn.commit()
    cur.close()
    return


@app.command("select-result")
def select_result_by_outlook(
    outlook: schemas.Outlook, target: str, quartile: bool = False
):
    logger.info(f"outlook: {outlook}")
    conn = open_db()
    lf = pl.read_database(query="SELECT * FROM result", connection=conn).lazy()

    if outlook != schemas.Outlook.ALL:
        expr = {
            schemas.Outlook.BULLISH: pl.col(target) >= 0.75,
            schemas.Outlook.BEARISH: pl.col(target) <= 0.25,
        }.get(outlook, (pl.col(target) < 0.75) & (pl.col(target) > 0.25))

        df_date = (
            lf.group_by("date", maintain_order=True)
            .mean()
            .filter(expr)
            .collect()
            .sort("date")
        )

        df = lf.collect().filter(pl.col("date").is_in(df_date["date"]))
    else:
        df = lf.collect().sort("date")

    logger.debug(df)

    if not quartile:
        # df["result_1"]のカウント
        logger.debug(
            df.group_by("result_1")
            .agg(count=pl.col("result_1").count())
            .sort("result_1")
        )
        return df

    df_describe = df.describe()
    threshold_25 = float(
        df_describe.filter(pl.col("statistic") == "25%")[target].item()
    )
    threshold_50 = float(
        df_describe.filter(pl.col("statistic") == "50%")[target].item()
    )
    threshold_75 = float(
        df_describe.filter(pl.col("statistic") == "75%")[target].item()
    )

    def categorize_by_nextday_close(value):
        if value < threshold_25:
            return 0
        elif value < threshold_50:
            return 1
        elif value < threshold_75:
            return 2
        else:
            return 3

    # result_quartile カラムを作成
    df_new = df.with_columns(
        pl.col(target)
        .map_elements(
            lambda value: categorize_by_nextday_close(value), return_dtype=pl.Int64
        )
        .alias("result_quartile")
    )

    logger.debug(
        df_new.group_by("result_quartile")
        .agg(count=pl.col("result_quartile").count())
        .sort("result_quartile")
    )

    return df_new
