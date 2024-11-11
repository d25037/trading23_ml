import io
import os
import random
import time
from sys import stderr
from time import sleep

import mplfinance as mpf
import pandas as pd
import polars as pl
import torch
import torch.utils.data
import torchvision
from loguru import logger
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset, random_split
from typer import Typer

import database
import fetcher
import models

app = Typer(no_args_is_help=True)


@app.command("cudable")
def is_cudable() -> None:
    if torch.cuda.is_available():
        print("CUDA is available")
        print(f"Torch CUDA version: {torch.__version__}")
        print(f"Device Name: {torch.cuda.get_device_name()}")
    else:
        print("ERROR: CUDA is unavailable")


@app.command()
def test(code: str):
    stock = fetcher.fetch_daily_quotes(code)
    if stock is None:
        return

    j = 675

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
    create_candlestick_chart(df, code, "test")


def create_candlestick_chart(df: pl.DataFrame, code: str, file_name: str):
    dir_path = f"./data/img/{code}"
    os.makedirs(dir_path, exist_ok=True)

    # DataFrameをmplfinanceの形式に変換
    df_pd: pd.DataFrame = df.to_pandas()
    ohlc_data = df_pd[["date", "open", "high", "low", "close", "volume"]].set_index(
        "date"
    )

    # ローソク足チャートの描画
    fig, axlist = mpf.plot(
        ohlc_data,
        type="candle",
        volume=True,
        style="yahoo",
        figsize=(3, 3),
        returnfig=True,
    )

    # 横軸と縦軸の目盛りを非表示にする
    for ax in axlist:
        ax.set_xticks([])
        ax.set_yticks([])

    # チャートを画像として保存
    fig.savefig(f"{dir_path}/{file_name}.png")

    # figをPNG形式に変換
    buf = io.BytesIO()
    fig.savefig(buf, format="png")

    # PNGデータをBLOB型に変換
    image = buf.getvalue()

    return image


def create_candlestick_chart_to_bytes(df: pl.DataFrame):
    # DataFrameをmplfinanceの形式に変換
    df_pd: pd.DataFrame = df.to_pandas()
    ohlc_data = df_pd[["date", "open", "high", "low", "close", "volume"]].set_index(
        "date"
    )

    # ローソク足チャートの描画
    fig, axlist = mpf.plot(
        ohlc_data,
        type="candle",
        style="yahoo",
        figsize=(3, 3),
        returnfig=True,
    )

    # 横軸と縦軸の目盛りを非表示にする
    ax = axlist[0]  # axlistはリストなので、最初の要素を取得
    ax.set_xticks([])
    ax.set_yticks([])

    # figをPNG形式に変換
    buf = io.BytesIO()
    fig.savefig(buf, format="png")

    # PNGデータをBLOB型に変換
    image = buf.getvalue()

    return image


@app.command()
def create_data_set():
    logger.remove()
    logger.add(stderr, level="INFO")

    conn = database.open_db()

    nikkei225 = fetcher.load_nikkei225_csv()
    numbers = random.sample(range(20, 1100), 100)

    i = 1
    for code in nikkei225.get_column("code"):
        df = fetcher.fetch_daily_quotes(code)
        if df is None:
            continue

        for number in numbers:
            file_name = df[number - 1]["date"].item()
            close = df[number - 1]["close"].item()

            nextday = df[number]
            nextday_open = nextday["open"].item()
            nextday_close = nextday["close"].item()
            result_open = round(100 * (nextday_open - close) / close, 2)
            result_close = round(100 * (nextday_close - close) / close, 2)

            stock_sliced = df.slice(number - 10, 10)
            df_sampled = stock_sliced.with_columns(
                pl.col("date").str.to_date().alias("date")
            )
            img = create_candlestick_chart(df_sampled, code, file_name)

            result = models.Result(
                **{
                    "code": str(code),
                    "date": file_name,
                    "nextday_open": result_open,
                    "nextday_close": result_close,
                    "image": img,
                }
            )

            database.insert_result(conn, result)

        if i % 10 == 0:
            logger.info(f"{i}/225 has been processed.")
        i += 1

    return


def get_atr(df: pd.DataFrame):
    df["high-low"] = df["high"] - df["low"]
    sum = df["high-low"].sum()
    atr = sum / len(df)
    # print(df)
    # print(atr)
    return atr


def get_standardized_diff(df: pd.DataFrame):
    df["high-low"] = df["high"] - df["low"]
    highetst_high = df["high"].max()
    lowest_low = df["low"].min()
    average_diff = df["high-low"].mean()

    standardized_diff = average_diff / (highetst_high - lowest_low)

    return round(standardized_diff, 3)


def img_reader():
    conn = database.open_db()
    cursor = conn.cursor()

    cursor.execute("SELECT image, day1_allday FROM result LIMIT 1")
    image_data, label = cursor.fetchone()

    img = Image.open(io.BytesIO(image_data))

    print(f"img.format: {img.format}")
    print(f"img.size  : {img.size}")
    print(f"img.mode  : {img.mode}")
    print(f"label    : {label}")


def dataset_reader():
    dataset = SQLiteDataset("day1_morning")
    item = dataset.__getitem__(0)

    print(item)
    print(f"item[0].shape: {dataset.__getitem__(0)[0].shape}")
    print(f"item[1].shape: {dataset.__getitem__(0)[1]}")


class SQLiteDataset(Dataset):
    def __init__(self, label):
        self.conn = database.open_db()
        self.cursor = self.conn.cursor()
        self.label = label

    def __getitem__(self, index):
        self.cursor.execute(
            f"SELECT image, {self.label} FROM result LIMIT 1 OFFSET ?",
            (index,),
        )
        image_data, label = self.cursor.fetchone()
        image = Image.open(io.BytesIO(image_data)).convert("L")
        image = torchvision.transforms.ToTensor()(image)
        label = torch.tensor(label, dtype=torch.float)
        return image, label

    def __len__(self):
        self.cursor.execute("SELECT COUNT(*) FROM result")
        return self.cursor.fetchone()[0]


class CNNModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn5 = nn.BatchNorm2d(512)
        self.pool = nn.MaxPool2d(2, stride=2)
        self.fc1 = nn.Linear(512 * 9 * 9, 128)  # 出力サイズ要確認
        self.fc2 = nn.Linear(128, 1)  # 回帰タスクのため出力は1

    def forward(self, x):
        x = self.pool(nn.functional.relu(self.conv1(x)))
        # x = self.bn1(x)
        x = self.pool(nn.functional.relu(self.conv2(x)))
        x = nn.functional.dropout(x, p=0.5)
        x = self.bn2(x)
        x = self.pool(nn.functional.relu(self.conv3(x)))
        x = self.pool(nn.functional.relu(self.conv4(x)))
        x = nn.functional.dropout(x, p=0.5)
        x = self.pool(nn.functional.relu(self.conv5(x)))
        x = self.bn5(x)
        # 入力画像のサイズを取得
        b, c, h, w = x.shape
        # `view`関数の引数を動的に計算
        x = x.view(-1, c * h * w)
        x = nn.functional.relu(self.fc1(x))
        x = self.fc2(x)
        return x.view(-1, 1)  # 次元を削減


def analyzer(label: str):
    # 経過時間
    start = time.time()

    batch_size = 128
    max_epoch = 32

    print(f"label: {label}")
    print(f"batch_size: {batch_size}")
    print(f"max_epoch: {max_epoch}")
    print("--------------------")

    dataset = SQLiteDataset(label)
    n_samples = len(dataset)
    n_train = int(0.75 * n_samples)
    n_test = n_samples - n_train
    train_dataset, test_dataset = random_split(dataset, [n_train, n_test])

    # ランダムサンプリング
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # モデルのインスタンス化
    model = CNNModel()
    loss_fn = nn.MSELoss()  # 回帰なので平均二乗誤差
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CNNModel().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=0.0001)

    for epoch in range(max_epoch):
        for images, labels in train_dataloader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = loss_fn(outputs, labels.float().view(-1, 1))
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch + 1}, Loss: {loss.item():.4f}")

        # テストデータでの評価
        if epoch % 3 == 0:
            with torch.no_grad():
                for images, labels in test_dataloader:
                    images, labels = images.to(device), labels.to(device)
                    outputs = model(images)
                    test_loss = loss_fn(outputs, labels.float().view(-1, 1))

            print(f"Epoch {epoch + 1}(Test), Loss: {test_loss.item():.4f}")

    torch.save(model.state_dict(), "model.pth")

    # 経過時間
    elapsed_time = time.time() - start
    print(f"elapsed_time: {elapsed_time:.2f}")

    return
