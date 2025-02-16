import io
import time
from datetime import datetime

import polars as pl
import timm
import torch
import torch.nn as nn
import torchvision
from loguru import logger
from PIL import Image
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms
from tqdm import tqdm
from typer import Typer

import constants
import database

app = Typer(no_args_is_help=True)


@app.command("cudable")
def is_cudable() -> None:
    if torch.cuda.is_available():
        logger.info("CUDA is available")
        logger.info(f"Torch CUDA version: {torch.__version__}")
        logger.info(f"Device Name: {torch.cuda.get_device_name()}")
    else:
        logger.error("ERROR: CUDA is unavailable")


@app.command()
def timm_list(model_name: str) -> None:
    model_name = f"*{model_name}*"
    models_list = timm.list_models(model_name, pretrained=True)
    for model in models_list:
        logger.info(model)


class PolarsDataset(Dataset):
    def __init__(
        self,
        df: pl.DataFrame,
        label: str,
        image_transform: transforms.Compose | None = None,
    ):
        self.df = df
        self.label = label
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
        label = self.df[f"{self.label}_binary"][idx]
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
#         x = self.pool(F.relu(self.conv1(x)))
#         x = self.bn1(x)
#         x = self.pool(F.relu(self.conv2(x)))
#         x = self.bn2(x)
#         x = self.avgpool(x)  # AdaptiveAvgPool2d を適用
#         x = torch.flatten(x, 1)  # 1次元に変換
#         x = F.relu(self.fc1(x))
#         x = self.fc2(x)

#         # # 入力画像のサイズを取得
#         # b, c, h, w = x.shape
#         # # `view`関数の引数を動的に計算
#         # x = x.view(-1, c * h * w)
#         # x = F.relu(self.fc1(x))
#         # x = self.fc2(x)
#         return x


@app.command()
def run(label: str):
    # 経過時間
    start = time.time()

    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    logger.add(f"./logs/{now}.log", level="DEBUG")

    BATCH_SIZE = 64
    MAX_EPOCH = 1000
    LABEL_COUNT = 2

    logger.info(f"label: {label}")
    logger.info(f"label count: {LABEL_COUNT}")
    logger.info(f"batch size: {BATCH_SIZE}")
    logger.info(f"max epoch: {MAX_EPOCH}")
    print("--------------------")

    # df = database.select_result_by_outlook(outlook, target=label, quartile=True)
    df = database.select_results_binary()

    # データの前処理
    transform = transforms.Compose(
        [
            # transforms.Resize((300, 300)),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    dataset = PolarsDataset(df, label=label, image_transform=transform)

    n_samples = len(dataset)
    n_train = int(0.75 * n_samples)
    n_test = n_samples - n_train

    # 固定されたシード値を設定
    torch.manual_seed(42)  # 例：シード値を42に設定
    # 乱数ジェネレータを作成
    generator = torch.Generator().manual_seed(42)

    train_dataset, test_dataset = random_split(
        dataset, [n_train, n_test], generator=generator
    )

    # ランダムサンプリング
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    logger.info(f"train: {len(train_dataset)}, test: {len(test_dataset)}")

    # CUDAが使えない場合はエラーを出力して終了
    if not torch.cuda.is_available():
        logger.error("CUDA is unavailable")
        return

    device = torch.device("cuda")

    # 事前学習済みのViTをロード
    model = timm.create_model("resnet50", pretrained=True, num_classes=LABEL_COUNT)

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
        for images, labels in tqdm(
            train_dataloader, bar_format=constants.SHORT_PROGRESS_BAR
        ):
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
                    test_dataloader, bar_format=constants.SHORT_PROGRESS_BAR
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

        if (epoch + 1) % 10 == 0:
            # モデルの保存
            final_test_loss = (round(test_loss.item(), 3)) * 1000
            file_name = f"./model/efficientnet_{final_test_loss}.pth"
            torch.save(model.state_dict(), file_name)

    # 経過時間
    elapsed_time = time.time() - start
    logger.info(f"elapsed_time: {round(elapsed_time, 2)}")

    return
