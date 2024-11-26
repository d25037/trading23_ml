import io
import os
import random
import time
from sys import stderr

import matplotlib.pyplot
import mplfinance as mpf  # type: ignore
import pandas as pd
import polars as pl
import timm
import torch
import torch.utils.data
import torchvision  # type: ignore
import torchvision.transforms as transforms
from loguru import logger
from PIL import Image
from sklearn.metrics import confusion_matrix
from torch import nn
from torch.utils.data import DataLoader, Dataset, random_split
from tqdm import tqdm
from typer import Typer

import database
import fetcher
import schemas

app = Typer(no_args_is_help=True)


@app.command("cudable")
def is_cudable() -> None:
    if torch.cuda.is_available():
        print("CUDA is available")
        print(f"Torch CUDA version: {torch.__version__}")
        print(f"Device Name: {torch.cuda.get_device_name()}")
    else:
        print("ERROR: CUDA is unavailable")


def slice_df_for_candlestick(df: pl.DataFrame, number: int):
    date = df[number - 1]["date"].item()
    close = df[number - 1]["close"].item()
    code = df[number - 1]["code"].item()

    day1 = df[number]
    day1_close = round(100 * (day1["close"].item() - close) / close, 2)
    day3 = df[number + 3]
    day3_close = round(100 * (day3["close"].item() - close) / close, 2)
    day5 = df[number + 5]
    day5_close = round(100 * (day5["close"].item() - close) / close, 2)

    stock_sliced = df.slice(number - 35, 35)
    df_sampled = stock_sliced.with_columns(pl.col("date").str.to_date().alias("date"))
    img = create_candlestick_chart_from_df(df_sampled, code, date, write=True)
    return (img, date, day1_close, day3_close, day5_close)


def create_candlestick_chart_from_df(
    df: pl.DataFrame,
    code: str,
    file_name: str,
    volume: bool = False,
    write: bool = False,
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
        volume=volume,
        style="yahoo",
        figsize=(3, 3),
        returnfig=True,
        mav=(5, 25),
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
def create_dataset():
    logger.remove()
    logger.add(stderr, level="INFO")

    conn = database.open_db()

    nikkei225 = fetcher.load_nikkei225_csv()
    numbers = random.sample(range(50, 1200), 300)

    for i, code in enumerate(nikkei225.get_column("code")):
        logger.info(f"Processing {code}...")
        df = database.select_ohlc_by_code(conn, code=code).drop_nulls()

        if len(df) < 1210:
            logger.info(f"{code} Data length is not enough: {len(df)}")
            continue

        for number in numbers:
            (img, date, day1_close, day3_close, day5_close) = slice_df_for_candlestick(
                df=df, number=number
            )
            # file_name = df[number - 1]["date"].item()
            # close = df[number - 1]["close"].item()

            # nextday = df[number]
            # nextday_open = nextday["open"].item()
            # nextday_close = nextday["close"].item()
            # result_open = round(100 * (nextday_open - close) / close, 2)
            # result_close = round(100 * (nextday_close - close) / close, 2)

            # stock_sliced = df.slice(number - 10, 10)
            # df_sampled = stock_sliced.with_columns(
            #     pl.col("date").str.to_date().alias("date")
            # )
            # img = create_candlestick_chart_from_df(df_sampled, code, file_name)

            result = schemas.Result2(
                **{
                    "code": str(code),
                    "date": date,
                    "day1_close": day1_close,
                    "day3_close": day3_close,
                    "day5_close": day5_close,
                    "image": img,
                }
            )

            database.insert_result(conn, result)

        if (i + 1) % 10 == 0:
            logger.info(f"{i+1}/225 has been processed.")

    return


@app.command()
def img_reader(write: bool = False):
    conn = database.open_db()
    cursor = conn.cursor()

    cursor.execute("SELECT image, nextday_close FROM result LIMIT 1")
    image_data, label = cursor.fetchone()

    img = Image.open(io.BytesIO(image_data)).convert("RGB")

    logger.info(f"img.format: {img.format}")  # PNG
    logger.info(f"img.size  : {img.size}")  # (300, 300)
    logger.info(f"img.mode  : {img.mode}")  # RGBA
    logger.info(f"label    : {label}")

    # Save the image as PNG
    if write:
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
    def __init__(
        self,
        df: pl.DataFrame,
        label: str,
        image_transform: transforms.Compose | None = None,
    ):
        self.df = df
        # self.label = label
        self.image_transform = image_transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # image
        image_data = self.df["image"][idx]
        image = Image.open(io.BytesIO(image_data)).convert("RGB")
        if self.image_transform:
            image = self.image_transform(image)
        else:
            image = transforms.ToTensor()(image)

        # label
        label = self.df["result_quartile"][idx]
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
        image = Image.open(io.BytesIO(image_data)).convert("RGB")
        image = torchvision.transforms.ToTensor()(image)
        label = torch.tensor(label, dtype=torch.long)
        return image, label

    def __len__(self):
        self.cursor.execute("SELECT COUNT(*) FROM result")
        return self.cursor.fetchone()[0]


# class CNNModel(nn.Module):
#     # 利用するレイヤーや初期設定したい内容の記述
#     def __init__(self):
#         super().__init__()
#         self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
#         self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
#         self.bn1 = nn.BatchNorm2d(32)
#         self.bn2 = nn.BatchNorm2d(64)
#         self.pool = nn.MaxPool2d(2, stride=2)
#         self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # 出力サイズを (1, 1) に指定
#         self.fc1 = nn.Linear(64, 32)  # 出力サイズ要確認
#         self.fc2 = nn.Linear(32, 4)

#     def forward(self, x):
#         x = self.pool(nn.functional.relu(self.conv1(x)))
#         x = self.bn1(x)
#         x = self.pool(nn.functional.relu(self.conv2(x)))
#         x = self.bn2(x)
#         x = self.avgpool(x)  # AdaptiveAvgPool2d を適用
#         x = torch.flatten(x, 1)  # 1次元に変換
#         x = nn.functional.relu(self.fc1(x))
#         x = self.fc2(x)

#         # # 入力画像のサイズを取得
#         # b, c, h, w = x.shape
#         # # `view`関数の引数を動的に計算
#         # x = x.view(-1, c * h * w)
#         # x = nn.functional.relu(self.fc1(x))
#         # x = self.fc2(x)
#         return x


@app.command()
def training(label: str, outlook: schemas.Outlook):
    # 経過時間
    start = time.time()

    BATCH_SIZE = 128
    MAX_EPOCH = 1000
    LABEL_COUNT = 4
    SHORT_PROGRESS_BAR = "{l_bar}{bar:10}{r_bar}{bar:-10b}"

    logger.info(f"label: {label}")
    logger.info(f"label count: {LABEL_COUNT}")
    logger.info(f"batch size: {BATCH_SIZE}")
    logger.info(f"max epoch: {MAX_EPOCH}")
    print("--------------------")

    df = database.select_result_by_outlook(outlook, target=label, quartile=True)

    # データの前処理
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    dataset = PolarsDataset(df, label=label, image_transform=transform)

    n_samples = len(dataset)
    n_train = int(0.75 * n_samples)
    n_test = n_samples - n_train
    train_dataset, test_dataset = random_split(dataset, [n_train, n_test])

    # ランダムサンプリング
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    logger.info(f"train: {len(train_dataset)}, test: {len(test_dataset)}")

    # CUDAが使えない場合はエラーを出力して終了
    if not torch.cuda.is_available():
        logger.error("CUDA is not available")
        return

    device = torch.device("cuda")

    # 事前学習済みのViTをロード
    model = timm.create_model(
        "vit_small_patch16_224", pretrained=True, num_classes=LABEL_COUNT
    )

    # 事前学習済みのResNet18をロード
    # model = torchvision.models.resnet152(pretrained=True)

    # 入力画像のサイズに合わせて最初の畳み込み層を修正
    # model.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)

    # ラベル数に合わせて最後の全結合層を修正
    # num_ftrs = model.fc.in_features
    # model.fc = nn.Linear(num_ftrs, LABEL_COUNT)

    model = model.to(device)

    criterion = nn.CrossEntropyLoss()  # 分類なのでクロスエントロピー
    optimizer = torch.optim.AdamW(model.parameters())
    # optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    logger.info("start training")
    for epoch in range(MAX_EPOCH):
        logger.info(f"Epoch {epoch + 1}")
        for images, labels in tqdm(train_dataloader, bar_format=SHORT_PROGRESS_BAR):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        logger.info(f"Training -- Loss: {round(loss.item(), 3)}")

        # テストデータでの評価
        pred_list = []
        true_list = []

        if (epoch + 1) % 2 == 0:
            with torch.no_grad():
                for images, labels in tqdm(
                    test_dataloader, bar_format=SHORT_PROGRESS_BAR
                ):
                    images, labels = images.to(device), labels.to(device)
                    outputs = model(images)
                    test_loss = criterion(outputs, labels)
                    # probability = round(torch.max(outputs, dim=1).values.item(), 2)
                    # logger.info(
                    #     f"predicted: {predicted}[確率: {probability}], labels: {labels}"
                    # )
                    pred = torch.argmax(outputs, dim=1)

                    pred_list += pred.detach().cpu().numpy().tolist()
                    true_list += labels.detach().cpu().numpy().tolist()
            # logger.info(f"outputs: {outputs}")
            # logger.info(f"outputs.shape: {outputs.shape}")
            # logger.info(f"labels: {labels}")
            # logger.info(f"labels.shape: {labels.shape}")

            logger.info(f"Test -- Loss: {round(test_loss.item(), 3)}")

            # Confusion matrixの生成
            cm = confusion_matrix(
                y_true=true_list,
                y_pred=pred_list,
            )
            print(cm)
            # logger.info(f"predicted: {predicted_list}")

        if (epoch + 1) % 100 == 0:
            # モデルの保存
            final_test_loss = (round(test_loss.item(), 3)) * 1000
            file_name = f"./model/{outlook.value}_{final_test_loss}.pth"
            torch.save(model.state_dict(), file_name)

    # 経過時間
    elapsed_time = time.time() - start
    logger.info(f"elapsed_time: {round(elapsed_time, 2)}")

    return


# @app.command()
# def predict():
#     logger.remove()
#     logger.add(stderr, level="INFO")

#     nikkei225 = fetcher.load_nikkei225_csv()

#     model_bearish = CNNModel()
#     model_bearish.load_state_dict(torch.load("./model/Bearish_0.813.pth"))

#     model_neutral = CNNModel()
#     model_neutral.load_state_dict(torch.load("./model/Neutral_0.864.pth"))

#     model_bullish = CNNModel()
#     model_bullish.load_state_dict(torch.load("./model/Bullish_0.767.pth"))

#     i = 1
#     for code in nikkei225.get_column("code"):
#         df = fetcher.fetch_daily_quotes(code)
#         if df is None:
#             continue

#         stock_sliced = df.tail(10)
#         df_sampled = stock_sliced.with_columns(
#             pl.col("date").str.to_date().alias("date")
#         )
#         img = create_candlestick_chart(df_sampled, code, "predict")

#         image = Image.open(io.BytesIO(img)).convert("RGB")
#         image = torchvision.transforms.ToTensor()(image)
#         image = image.unsqueeze(0)

#         model_bullish.eval()
#         with torch.no_grad():
#             output = model_bullish(image)
#             predicted = torch.argmax(output, dim=1)
#             probability = torch.max(output, dim=1)

#             logger.info(f"{code} Bullish: {predicted} [確率: {probability.values}]")

#         model_neutral.eval()
#         with torch.no_grad():
#             output = model_neutral(image)
#             predicted = torch.argmax(output, dim=1)
#             probability = torch.max(output, dim=1)

#             logger.info(f"{code} Neutral: {predicted} [確率: {probability.values}]")

#         model_bearish.eval()
#         with torch.no_grad():
#             output = model_bearish(image)
#             predicted = torch.argmax(output, dim=1)
#             probability = torch.max(output, dim=1)

#             logger.info(f"{code} Bearish: {predicted} [確率: {probability.values}]")

#         if i % 10 == 0:
#             logger.info(f"{i}/225 has been processed.")
#         i += 1
