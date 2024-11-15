import io
import os
import random
import time
from sys import stderr

import matplotlib.pyplot
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
    create_candlestick_chart(df, code, "test", write=True)


@app.command()
def test2(label: str):
    batch_size = 128

    dataset = SQLiteDataset(label)
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


def create_candlestick_chart(
    df: pl.DataFrame, code: str, file_name: str, write: bool = False
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
        volume=True,
        style="yahoo",
        figsize=(3, 3),
        returnfig=True,
    )

    # 横軸と縦軸の目盛りを非表示にする
    for ax in ax_list:
        ax.set_xticks([])
        ax.set_yticks([])

    # チャートを画像として保存
    if write:
        dir_path = f"./data/img/{code}"
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
        if len(df) < 1100:
            logger.info(f"{code} Data length is not enough: {len(df)}")
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


@app.command()
def img_reader(write: bool = False):
    conn = database.open_db()
    cursor = conn.cursor()

    cursor.execute("SELECT image, nextday_close FROM result LIMIT 1")
    image_data, label = cursor.fetchone()

    img = Image.open(io.BytesIO(image_data)).convert("L")

    logger.info(f"img.format: {img.format}")  # PNG
    logger.info(f"img.size  : {img.size}")  # (300, 300)
    logger.info(f"img.mode  : {img.mode}")  # RGBA
    logger.info(f"label    : {label}")

    # Convert RGBA to RGB if the image has an alpha channel
    # if img.mode == "RGBA":
    #     img = img.convert("RGB")

    # Save the image as PNG
    path = "./data/output.png"
    img.save(f"{path}", "PNG")
    logger.info(f"Image has been saved as PNG to {path}")
    return


def dataset_reader():
    dataset = SQLiteDataset("day1_morning")
    item = dataset.__getitem__(0)

    print(item)
    print(f"item[0].shape: {dataset.__getitem__(0)[0].shape}")
    print(f"item[1].shape: {dataset.__getitem__(0)[1]}")


class PolarsDataset(Dataset):
    def __init__(self, df: pl.DataFrame, image_transform=None):
        """
        Polars DataFrame を受け取り、画像とラベルのペアを返すデータセット。

        Args:
            df (polars.DataFrame): 画像パスとラベルを含む DataFrame。
                "image" 列には画像へのパス、"label" 列にはラベルが含まれている必要があります。
            image_transform (callable, optional): 画像に適用する変換。Defaults to None.
        """
        self.df = df
        self.image_transform = image_transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        """
        インデックスに対応する画像とラベルのペアを返す。

        Args:
            idx (int): インデックス。

        Returns:
            tuple: (image, label) のペア。
        """
        image_data = self.df["image"][idx]
        label = self.df["result_1"][idx]
        image = Image.open(io.BytesIO(image_data)).convert("L")
        image = torchvision.transforms.ToTensor()(image)
        label = torch.tensor(label, dtype=torch.long)
        return image, label


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
        label = torch.tensor(label, dtype=torch.long)
        # label = torch.tensor(label, dtype=torch.float)
        return image, label

    def __len__(self):
        self.cursor.execute("SELECT COUNT(*) FROM result")
        return self.cursor.fetchone()[0]


class CNNModel(nn.Module):
    # 利用するレイヤーや初期設定したい内容の記述
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(2, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # 出力サイズを (1, 1) に指定
        self.fc1 = nn.Linear(64, 32)  # 出力サイズ要確認
        self.fc2 = nn.Linear(32, 3)

    def forward(self, x):
        x = self.pool(nn.functional.relu(self.conv1(x)))
        x = self.bn1(x)
        x = self.pool(nn.functional.relu(self.conv2(x)))
        x = self.bn2(x)
        x = self.avgpool(x)  # AdaptiveAvgPool2d を適用
        x = torch.flatten(x, 1)  # 1次元に変換
        x = nn.functional.relu(self.fc1(x))
        x = self.fc2(x)

        # # 入力画像のサイズを取得
        # b, c, h, w = x.shape
        # # `view`関数の引数を動的に計算
        # x = x.view(-1, c * h * w)
        # x = nn.functional.relu(self.fc1(x))
        # x = self.fc2(x)
        return x


@app.command()
def training(label: str):
    # 経過時間
    start = time.time()

    BATCH_SIZE = 128
    MAX_EPOCH = 32

    print(f"label: {label}")
    print(f"batch size: {BATCH_SIZE}")
    print(f"max epoch: {MAX_EPOCH}")
    print("--------------------")

    conn = database.open_db()
    lf = pl.read_database(query="SELECT * FROM result", connection=conn).lazy()

    df_negarive_date = (
        lf.group_by("date", maintain_order=True)
        .mean()
        .filter((pl.col("result_0") > 0.75))
        .collect()
        .sort("date")
    )
    logger.debug(df_negarive_date)
    df = lf.collect().filter(pl.col("date").is_in(df_negarive_date["date"]))

    dataset = PolarsDataset(df)

    n_samples = len(dataset)
    n_train = int(0.75 * n_samples)
    n_test = n_samples - n_train
    train_dataset, test_dataset = random_split(dataset, [n_train, n_test])

    # ランダムサンプリング
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # CUDAが使えない場合はエラーを出力して終了
    if not torch.cuda.is_available():
        logger.error("CUDA is not available")
        return

    device = torch.device("cuda")

    # モデルのインスタンス化
    model = CNNModel().to(device)

    loss_fn = nn.CrossEntropyLoss()  # 分類なのでクロスエントロピー
    # loss_fn = nn.MSELoss()  # 回帰なので平均二乗誤差
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=0.0001)

    logger.info("start training")
    for epoch in range(MAX_EPOCH):
        for images, labels in train_dataloader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            # loss = loss_fn(outputs, labels.float().view(-1, 1))
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()

        logger.info(f"Epoch {epoch + 1}, Loss: {loss.item():.4f}")

        # テストデータでの評価
        if epoch % 3 == 0:
            with torch.no_grad():
                for images, labels in test_dataloader:
                    images, labels = images.to(device), labels.to(device)
                    outputs = model(images)
                    test_loss = loss_fn(outputs, labels)

            logger.info(f"Epoch {epoch + 1}(Test), Loss: {test_loss.item():.4f}")

    torch.save(model.state_dict(), "./model/model_1.pth")

    # 経過時間
    elapsed_time = time.time() - start
    logger.info(f"elapsed_time: {elapsed_time:.2f}")

    return
