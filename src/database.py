import sqlite3

import pandas as pd

import models


def open_db():
    file_path = "/data/trading23_ml.sqlite"
    conn = sqlite3.connect(file_path)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS ohlc (
            code TEXT NOT NULL,
            date TEXT NOT NULL,
            open FLOAT,
            high FLOAT,
            low FLOAT,
            close FLOAT,
            volume FLOAT,
            morning_close FLOAT,
            afternoon_open FLOAT
        )
    """)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS result (
            code TEXT NOT NULL,
            date TEXT NOT NULL,
            day1_morning FLOAT,
            day1_allday FLOAT,
            day5 FLOAT,
            day20 FLOAT,
            image BLOB
        )
    """)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS result_wo_volume (
            code TEXT NOT NULL,
            date TEXT NOT NULL,
            standardized_diff FLOAT,
            day1_morning FLOAT,
            day1_allday FLOAT,
            day5 FLOAT,
            day20 FLOAT,
            image BLOB
        )
    """)

    return conn


def insert_ohlc(df: pd.DataFrame):
    conn = open_db()

    df.to_sql(
        "ohlc",
        conn,
        if_exists="append",
        index=False,
        method="multi",
        chunksize=5000,
    )

    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM ohlc")
    record_count = cursor.fetchone()[0]
    print(f"number of records: {record_count}")
    return


def select_ohlc_by_code(conn: sqlite3.Connection, code: str):
    code = code + "0"

    sql_query = f"SELECT * FROM ohlc WHERE code={code}"

    dfs = []
    for chunk in pd.read_sql_query(sql_query, conn, chunksize=5000):
        dfs.append(chunk)
    df = pd.concat(dfs)

    return df


def insert_result(conn: sqlite3.Connection, result: models.Result):
    print(f"date: {result.date}")
    cur = conn.cursor()
    cur.execute(
        f"INSERT INTO result (code, date, day1_morning, day1_allday, day5, day20) values ({result.code}, '{result.date}', {result.day1_morning}, {result.day1_allday}, {result.day5}, {result.day20})"
    )
    conn.commit()
    cur.close()
    return


def insert_result_wo_volume(conn: sqlite3.Connection, result: models.ResultWoVolume):
    print(f"date: {result.date}")
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO result_wo_volume (code, date, standardized_diff, day1_morning, day1_allday, day5, day20, image) values (?, ?, ?, ?, ?, ?, ?, ?)",
        (
            result.code,
            result.date,
            result.standardized_diff,
            result.day1_morning,
            result.day1_allday,
            result.day5,
            result.day20,
            result.image,
        ),
    )
    conn.commit()
    cur.close()
    return
