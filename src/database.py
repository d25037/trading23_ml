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
            result_1 FLOAT,
            result_3 FLOAT,
            result_5 FLOAT,
            image BLOB
        )
    """)

    return conn


def insert_ohlc(df: pl.DataFrame, table_name: str = "ohlc"):
    with open_db() as conn:
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

    logger.info(f"Inserted {len(df)} records into {table_name}")
    return


def select_ohlc_all_by_code(conn: sqlite3.Connection, code: str | int):
    if isinstance(code, int):
        code = str(code)

    code = code + "0"
    df = pl.read_database(
        query=f"SELECT * FROM ohlc WHERE code={code}", connection=conn
    ).drop_nulls()

    # polars DataFrame を辞書のリストに変換してから pydantic で検証
    records = df.to_dicts()
    [schemas.Ohlc(**row) for row in records]

    # ["date"]をdatetime型に変換 (with_columnsを用いて)
    df = df.with_columns([pl.col("date").cast(pl.Date).alias("date")])
    return df


def select_ohlc_one_by_code_latest_date(conn: sqlite3.Connection, code: str | int):
    if isinstance(code, int):
        code = str(code)

    code = code + "0"
    return pl.read_database(
        query=f"SELECT * FROM ohlc WHERE code={code} ORDER BY date DESC LIMIT 1",
        connection=conn,
    )


def delete_ohlc_all_by_code(conn: sqlite3.Connection, code: str | int):
    if isinstance(code, int):
        code = str(code)

    code = code + "0"
    conn.execute(f"DELETE FROM ohlc WHERE code={code}")
    conn.commit()
    logger.info(f"Deleted all records with code: {code}")
    return


def select_topix(conn: sqlite3.Connection):
    return pl.read_database(query="SELECT * FROM topix", connection=conn).lazy()


def insert_result(conn: sqlite3.Connection, result: schemas.Result3):
    logger.debug(f"date: {result.date}")
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO result (code, date, result_1, result_3, result_5, image) values (?, ?, ?, ?, ?, ?)",
        (
            result.code,
            result.date,
            result.result_1,
            result.result_3,
            result.result_5,
            result.image,
        ),
    )

    conn.commit()
    cur.close()
    return


@app.command("select-result")
def select_result_by_outlook(
    outlook: schemas.Outlook, target: str, quartile: bool = False
) -> pl.DataFrame:
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


def select_ohlc_or_topix_with_future_close(
    code_or_topix: str, conn: sqlite3.Connection | None
) -> pl.DataFrame:
    if code_or_topix == "topix":
        query = "SELECT * FROM topix"
        column_name = "topix"
    else:
        query = f"SELECT * FROM ohlc WHERE code={code_or_topix}0"
        column_name = "result"

    if conn is None:
        conn = open_db()

    lf = (
        pl.read_database(query=query, connection=conn)
        .lazy()
        .sort("date")
        .drop_nulls()
        .with_columns(
            [
                pl.col("close").shift(-1).alias("day1_close"),
                pl.col("close").shift(-3).alias("day3_close"),
                pl.col("close").shift(-5).alias("day5_close"),
            ]
        )
        .drop_nulls()
        .with_columns(
            [
                (pl.col("day1_close") / pl.col("close"))
                .round(4)
                .alias(f"{column_name}_1"),
                (pl.col("day3_close") / pl.col("close"))
                .round(4)
                .alias(f"{column_name}_3"),
                (pl.col("day5_close") / pl.col("close"))
                .round(4)
                .alias(f"{column_name}_5"),
            ]
        )
    )
    df = lf.collect()
    logger.debug(df)
    return df


@app.command()
def select_ohlc_with_result_binary(code: str):
    conn = open_db()

    df1 = select_ohlc_or_topix_with_future_close(code, conn)
    topix_df = select_ohlc_or_topix_with_future_close("topix", conn)

    df3 = (
        df1.join(topix_df, on="date", how="left")
        .drop_nulls()
        .drop(
            [
                "day1_close_right",
                "day3_close_right",
                "day5_close_right",
                "close_right",
            ]
        )
    )

    logger.debug(df3)

    df4 = df3.with_columns(
        [
            pl.when(pl.col("result_1") > pl.col("topix_1"))
            .then(1)
            .otherwise(0)
            .alias("result_1_binary"),
            pl.when(pl.col("result_3") > pl.col("topix_3"))
            .then(1)
            .otherwise(0)
            .alias("result_3_binary"),
            pl.when(pl.col("result_5") > pl.col("topix_5"))
            .then(1)
            .otherwise(0)
            .alias("result_5_binary"),
        ]
    )
    df4 = df4.drop(["topix_1", "topix_3", "topix_5"])
    logger.debug(df4)

    return df4


@app.command()
def select_results_binary() -> pl.DataFrame:
    conn = open_db()
    df = pl.read_database(query="SELECT * FROM result", connection=conn)

    topix_df = select_ohlc_or_topix_with_future_close("topix", conn)

    df_joined = df.join(topix_df, on="date", how="left").drop_nulls()

    logger.debug(df_joined)

    df_compared = df_joined.with_columns(
        [
            pl.when(pl.col("result_1") > pl.col("topix_1"))
            .then(1)
            .otherwise(0)
            .alias("result_1_binary"),
            pl.when(pl.col("result_3") > pl.col("topix_3"))
            .then(1)
            .otherwise(0)
            .alias("result_3_binary"),
            pl.when(pl.col("result_5") > pl.col("topix_5"))
            .then(1)
            .otherwise(0)
            .alias("result_5_binary"),
        ]
    )
    logger.debug(df_compared)

    return df_compared
