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
    tensor = torch.randn(1, 10)
    predict = torch.argmax(tensor, dim=1)
    probability_b = torch.max(tensor, dim=1)

    print(tensor)
    print("---")
    print(predict.item())
    print("---")
    print(probability_b.values.item())

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
def num():
    number = 0.1234
    print(f"{number:.2f}")
    print(f"{round(number, 2)}")
