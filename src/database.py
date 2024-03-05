import sqlite3

import pandas as pd


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

    return conn


def insert_db(df: pd.DataFrame):
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


def select_all():
    conn = open_db()
