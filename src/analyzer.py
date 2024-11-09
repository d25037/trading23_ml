import io
import os
import random
import time

import mplfinance as mpf
import pandas as pd
import torch
import torch.utils.data
import torchvision
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset, random_split
from typer import Typer

import database
import fetcher
import models

app = Typer()


@app.command("cudable")
def is_cudable() -> None:
    if torch.cuda.is_available():
        print("CUDA is available")
        print(f"Torch CUDA version: {torch.__version__}")
        print(f"Device Name: {torch.cuda.get_device_name()}")
    else:
        print("ERROR: CUDA is unavailable")


def create_candlestick_chart_to_png(df: pd.DataFrame, code: str, file_name: str):
    dir_path = f"/data/data_set/{code}"
    os.makedirs(dir_path, exist_ok=True)

    df.loc[:, "date"] = pd.to_datetime(df["date"]).infer_objects()

    # DataFrameをmplfinanceの形式に変換
    ohlc_data = df[["date", "open", "high", "low", "close", "volume"]].set_index("date")

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
    ax = axlist[0]  # axlistはリストなので、最初の要素を取得
    ax.set_xticks([])
    ax.set_yticks([])

    # # 出来高軸を取得
    # ax_volume = axlist[1]
    # # 出来高軸のラベルを非表示にする
    # ax_volume.set_xticks([])
    # ax_volume.set_yticks([])

    # チャートを画像として保存
    fig.savefig(f"{dir_path}/{file_name}.png")

    return


def create_candlestick_chart_to_bytes(df: pd.DataFrame):
    df.loc[:, "date"] = pd.to_datetime(df["date"]).infer_objects()

    # DataFrameをmplfinanceの形式に変換
    ohlc_data = df[["date", "open", "high", "low", "close"]].set_index("date")

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


def create_data_set():
    conn = database.open_db()

    nikkei225 = fetcher.load_nikkei225_csv()
    for i in range(len(nikkei225)):
        code = str(nikkei225.iloc[i, 0])
        df = database.select_ohlc_by_code(conn, code)
        df = df.dropna()

        limit = len(df)
        numbers = random.sample(range(70, limit - 30), 20)
        for j in numbers:
            file_name = str(df.iloc[j]["date"])
            df_sampled = df[j - 60 : j]
            atr = get_atr(df_sampled.tail())
            create_candlestick_chart_to_png(df_sampled, code, file_name)

            row_1 = df.iloc[j + 1]
            day1_morning = round((row_1["morning_close"] - row_1["open"]) / atr, 2)
            day1_allday = round((row_1["close"] - row_1["open"]) / atr, 2)

            row_5 = df.iloc[j + 5]
            day5 = round((row_5["close"] - row_1["open"]) / atr, 2)

            row_20 = df.iloc[j + 20]
            day20 = round((row_20["close"] - row_1["open"]) / atr, 2)

            result = models.Result(
                **{
                    "code": code,
                    "date": file_name,
                    "day1_morning": day1_morning,
                    "day1_allday": day1_allday,
                    "day5": day5,
                    "day20": day20,
                }
            )

            database.insert_result(conn, result)

        if i % 10 == 0:
            print(f"i: {i}")

    return


def create_data_set_wo_volume():
    conn = database.open_db()

    nikkei225 = fetcher.load_nikkei225_csv()
    for i in range(len(nikkei225)):
        code = str(nikkei225.iloc[i, 0])
        df = database.select_ohlc_by_code(conn, code)
        df = df.dropna()

        limit = len(df)
        numbers = random.sample(range(70, limit - 30), 100)
        for j in numbers:
            file_name = str(df.iloc[j]["date"])
            df_sampled = df[j - 60 : j]

            standardized_diff = get_standardized_diff(df_sampled)

            atr = get_atr(df_sampled.tail())
            image = create_candlestick_chart_to_bytes(df_sampled)

            row_1 = df.iloc[j + 1]
            day1_morning = round((row_1["morning_close"] - row_1["open"]) / atr, 2)
            day1_allday = round((row_1["close"] - row_1["open"]) / atr, 2)

            row_5 = df.iloc[j + 5]
            day5 = round((row_5["close"] - row_1["open"]) / atr, 2)

            row_20 = df.iloc[j + 20]
            day20 = round((row_20["close"] - row_1["open"]) / atr, 2)

            result = models.ResultWoVolume(
                **{
                    "code": code,
                    "date": file_name,
                    "standardized_diff": standardized_diff,
                    "day1_morning": day1_morning,
                    "day1_allday": day1_allday,
                    "day5": day5,
                    "day20": day20,
                    "image": image,
                }
            )

            database.insert_result_wo_volume(conn, result)

        if i % 5 == 0:
            print(f"i: {i}")

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
            f"SELECT image, {self.label} FROM result_wo_volume WHERE standardized_diff<0.12 LIMIT 1 OFFSET ?",
            (index,),
        )
        image_data, label = self.cursor.fetchone()
        image = Image.open(io.BytesIO(image_data)).convert("L")
        image = torchvision.transforms.ToTensor()(image)
        label = torch.tensor(label, dtype=torch.float)
        return image, label

    def __len__(self):
        self.cursor.execute(
            "SELECT COUNT(*) FROM result_wo_volume WHERE standardized_diff<0.12"
        )
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
