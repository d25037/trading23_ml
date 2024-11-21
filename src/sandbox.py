import polars as pl
import torch
from loguru import logger
from torch.utils.data import DataLoader, Dataset, random_split
from typer import Typer

import analyzer
import database
import fetcher

app = Typer()


@app.command()
def tensor():
    tensor = torch.randn(3, 3)
    predict = torch.argmax(tensor, dim=1)
    probability_b = torch.max(tensor, dim=1)

    print(tensor)
    print("---")
    print(tensor.shape)
    print("---")
    print(predict)
    print("---")
    print(predict.shape)
    print("---")
    print(probability_b.values)
    print("---")
    for x, y in zip(predict.tolist(), probability_b.values.tolist()):
        print(x, y)

    return


@app.command()
def candle_stick(code: str):
    stock = fetcher.fetch_daily_quotes(code)
    if stock is None:
        return

    j = 920

    date = stock[j - 1]["date"].item()
    close = stock[j - 1]["close"].item()

    nextday = stock[j]
    nextday_open = nextday["open"].item()
    nextday_close = nextday["close"].item()
    result_open = round(100 * (nextday_open - close) / close, 2)
    result_close = round(100 * (nextday_close - close) / close, 2)

    print(nextday)
    print(f"date: {date}")
    print(f"close: {close}")
    print(f"result_open: {result_open}")
    print(f"result_close: {result_close}")

    stock_sliced = stock.slice(j - 10, 10)
    df = stock_sliced.with_columns(pl.col("date").str.to_date().alias("date"))
    print(df)
    analyzer.create_candlestick_chart(df, code, "test", write=True)


@app.command()
def dataset(label: str):
    batch_size = 128

    dataset = analyzer.SQLiteDataset(label)
    n_samples = len(dataset)
    n_train = int(0.75 * n_samples)
    n_test = n_samples - n_train
    train_dataset, test_dataset = random_split(dataset, [n_train, n_test])

    # ランダムサンプリング
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    # test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # CUDAが使える場合は使う
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for images, labels in train_dataloader:
        images, labels = images.to(device), labels.to(device)
        print(images.shape)
        print(labels.shape)
        print(labels)
        break


@app.command()
def db():
    conn = database.open_db()
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

    df_bearish = lf.collect().filter(pl.col("date").is_in(df_2["date"])).sort("date")
    logger.debug(df_bearish)
    df_bearish_describe = df_bearish.describe()

    threshold_25 = float(
        df_bearish_describe.filter(pl.col("statistic") == "25%")["nextday_close"].item()
    )
    logger.debug(threshold_25)

    threshold_50 = float(
        df_bearish_describe.filter(pl.col("statistic") == "50%")["nextday_close"].item()
    )
    logger.debug(threshold_50)

    threshold_75 = float(
        df_bearish_describe.filter(pl.col("statistic") == "75%")["nextday_close"].item()
    )
    logger.debug(threshold_75)

    def categorize_by_nextday_close(value):
        if value < threshold_25:
            return 0
        elif value < threshold_50:
            return 1
        elif value < threshold_75:
            return 2
        else:
            return 3

    # result_new カラムを作成
    df_bearish_new = df_bearish.with_columns(
        pl.col("nextday_close")
        .map_elements(
            lambda value: categorize_by_nextday_close(value), return_dtype=pl.Int64
        )
        .alias("result_new")
    )
    logger.debug(df_bearish_new)

    logger.debug(
        df_bearish_new.group_by("result_new")
        .agg(count=pl.col("result_new").count())
        .sort("result_new")
    )

    return df_bearish_new
