import time

import matplotlib.pyplot as plt
import polars as pl
import torch
from loguru import logger
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from typer import Typer, echo

import analyzer
import constants
import database
import fetcher
import schemas
import train

app = Typer()


@app.command()
def calc_nikkei(today: float, yesterday: float):
    nikkei = round((today - yesterday) / yesterday * 100, 2)
    echo(nikkei)


@app.command()
def test_tqdm():
    nikkei225 = fetcher.load_csv(schemas.CsvFile.NIKKEI225)
    code_list = nikkei225.get_column("code").to_list()

    pbar = tqdm(code_list, bar_format=constants.SHORT_PROGRESS_BAR, desc="Processing")

    for code in pbar:
        pbar.set_postfix({"message": f"ロング候補に追加: {code}"})
        pbar.set_description(f"Processing {code}")
        time.sleep(0.1)


@app.command()
def history():
    # CSV を LazyFrame として読み込み
    df = pl.scan_csv("data/trade_history/DOMESTIC_STOCK_20250212150310.csv")

    # 約定日ごとに約定金額を合計
    df_daily_sum = df.group_by("約定日", maintain_order=True).agg(
        pl.col("実現損益").sum().alias("daily_sum")
    )

    # 実際の計算を行い、DataFrame に変換
    df_daily_sum = df_daily_sum.collect()

    # 約定日でソート（文字列型の場合の例：必要に応じて修正）
    df_daily_sum = df_daily_sum.sort("約定日")

    # 累計カラムを追加
    df_daily_sum = df_daily_sum.with_columns(
        (pl.col("daily_sum").cum_sum().alias("cumulative_sum"))
    )

    # Filter out rows where the date is None
    df_daily_sum = df_daily_sum.filter(pl.col("約定日").is_not_null())

    # matplotlib で折れ線グラフとしてプロット
    x = df_daily_sum["約定日"]
    y = df_daily_sum["cumulative_sum"]

    plt.figure(figsize=(8, 4))
    plt.plot(x, y, marker="o", label="Cumulative Sum")
    plt.title("日別 約定金額の累計")
    plt.xlabel("約定日")
    plt.ylabel("累計約定金額")
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.show()


# @app.command()
# def timm():
#     import timm

#     model = timm.create_model("resnet18", pretrained=True)
#     print(model)


@app.command()
def suji():
    code = "13010"
    if len(code) > 4:
        code = code[:-1]

    print(code)


@app.command()
def tensor():
    tensor = torch.randn(100, 4)
    predict = torch.argmax(tensor, dim=1)
    label = torch.randint(0, 4, (100,))

    print(tensor)
    print("---")
    print(tensor.shape)
    print("---")
    print("predict")
    print(predict)
    print("---")
    print(predict.shape)
    print("---")
    print("label")
    print(label)
    print("---")
    print(label.shape)
    print(torch.sum(predict == label).item())

    # Confusion matrixの生成
    cm = confusion_matrix(
        y_true=label,
        y_pred=predict,
    )
    print(cm)

    # # Confusion matrixの表示
    # disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    # disp.plot(cmap=plt.cm.Blues)

    # # X軸ラベル、Y軸ラベル、タイトルの追加
    # plt.xlabel("Predicted")
    # plt.ylabel("Actual")
    # plt.title("Confusion Matrix")

    # # Confusion matrixの表示
    # plt.show()

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
    analyzer.create_candlestick_chart_from_df(df, code, "test", write=True)


def dataset_reader():
    dataset = train.SQLiteDataset("day1_morning")
    item = dataset.__getitem__(0)

    print(item)
    print(f"item[0].shape: {dataset.__getitem__(0)[0].shape}")
    print(f"item[1].shape: {dataset.__getitem__(0)[1]}")


@app.command()
def dataset(label: str):
    batch_size = 256

    dataset = train.SQLiteDataset(label)
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
