import sqlite3

import pandas as pd
import polars as pl
from loguru import logger
from typer import Typer

import models
from constants import DB_PATH

app = Typer(no_args_is_help=True)


def open_db():
    file_path = DB_PATH
    conn = sqlite3.connect(file_path)

    # conn.execute("""
    #     CREATE TABLE IF NOT EXISTS ohlc (
    #         code TEXT NOT NULL,
    #         date TEXT NOT NULL,
    #         open FLOAT,
    #         high FLOAT,
    #         low FLOAT,
    #         close FLOAT,
    #         volume FLOAT,
    #     )
    # """)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS result (
            code TEXT NOT NULL,
            date TEXT NOT NULL,
            nextday_open FLOAT,
            nextday_close FLOAT,
            result_0 FLOAT,
            result_1 FLOAT,
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


# def insert_ohlc(df: pd.DataFrame):
#     conn = open_db()

#     df.to_sql(
#         "ohlc",
#         conn,
#         if_exists="append",
#         index=False,
#         method="multi",
#         chunksize=5000,
#     )

#     cursor = conn.cursor()
#     cursor.execute("SELECT COUNT(*) FROM ohlc")
#     record_count = cursor.fetchone()[0]
#     print(f"number of records: {record_count}")
#     return


def select_ohlc_by_code(conn: sqlite3.Connection, code: str):
    code = code + "0"

    sql_query = f"SELECT * FROM ohlc WHERE code={code}"

    dfs = []
    for chunk in pd.read_sql_query(sql_query, conn, chunksize=5000):
        dfs.append(chunk)
    df = pd.concat(dfs)

    return df


def insert_result(conn: sqlite3.Connection, result: models.Result):
    logger.debug(f"date: {result.date}")
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO result (code, date, nextday_open, nextday_close, image) values (?, ?, ?, ?, ?)",
        (
            result.code,
            result.date,
            result.nextday_open,
            result.nextday_close,
            result.image,
        ),
    )

    conn.commit()
    cur.close()
    return


@app.command()
def test():
    conn = open_db()
    lf = pl.read_database(query="SELECT * FROM result", connection=conn).lazy()
    logger.debug(f"len: {len(lf.collect())}")

    df_1 = lf.group_by("date", maintain_order=True).mean().collect().sort("date")
    print(df_1)

    df_2 = (
        lf.group_by("date", maintain_order=True)
        .mean()
        .filter(pl.col("result_0") < 0.25)
        .collect()
        .sort("date")
    )
    logger.debug(df_2)

    df_3 = (
        lf.group_by("date", maintain_order=True)
        .mean()
        .filter(pl.col("result_0") > 0.75)
        .collect()
        .sort("date")
    )
    logger.debug(df_3)

    # len()
    df_4 = lf.group_by("date", maintain_order=True).count().collect().sort("date")
    logger.debug(df_4)

    # cur = conn.cursor()
    # for date in df_4["date"]:
    #     cur.execute(f"DELETE FROM result WHERE date='{date}'")
    # conn.commit()
    # cur.close()

    df_5 = lf.collect().filter(pl.col("date").is_in(df_2["date"])).sort("date")
    logger.debug(df_5)

    return
