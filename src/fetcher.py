import datetime
import json
from typing import Optional

import polars as pl
import requests
from loguru import logger
from tqdm import tqdm
from typer import Exit, Option, Typer, echo

import constants
import custom_error
import database
import schemas
from database import insert_ohlc
from utils import debug_df_lf, notify

app = Typer(no_args_is_help=True)


@app.command("settings")
def load_settings() -> schemas.AppSettings:
    with open(f"{constants.APP_SETTINGS_PATH}", "r") as f:
        items = schemas.AppSettings(**json.load(f))
        logger.debug(items)
        return items


@app.command("csv")
def load_csv(index: schemas.CsvFile) -> pl.DataFrame:
    if index == schemas.CsvFile.NIKKEI225:
        file_name = constants.NIKKEI225_PATH
    elif index == schemas.CsvFile.TOPIX400:
        file_name = constants.TOPIX400_PATH

    with open(f"{file_name}") as f:
        df = (
            pl.scan_csv(f)
            .with_columns(pl.col("code").cast(str).alias("code"))
            .collect()
        )
        # 各行ごとの辞書リストを取得
        rows = df.to_dicts()
        # 各行をpydanticモデルで検証
        [schemas.StockListCsv(**row) for row in rows]
        debug_df_lf(df)

    return df


@app.command("refresh-token")
def fetch_refresh_token() -> None:
    app_settings = load_settings()
    body = {"mailaddress": app_settings.mailaddress, "password": app_settings.password}

    notify("Fetch Refresh token...")

    r_post = requests.post(
        "https://api.jquants.com/v1/token/auth_user", data=json.dumps(body)
    )
    if r_post.status_code == 200:
        notify("Refresh token was successfully fetched.")
    else:
        notify(f"Failed to fetch Refresh token: {r_post.status_code}", "error")
        return

    refresh_token = schemas.RefreshToken(**r_post.json()).refresh_token
    app_settings.refresh_token = refresh_token
    with open(f"{constants.APP_SETTINGS_PATH}", "w") as f:
        json.dump(app_settings.model_dump(), f)

    notify(f"Refresh token was saved to {constants.APP_SETTINGS_PATH}")


@app.command("id-token")
def fetch_id_token() -> None:
    app_settings = load_settings()
    params = {"refreshtoken": app_settings.refresh_token}

    notify("Fetch ID token...")

    r_post = requests.post(
        "https://api.jquants.com/v1/token/auth_refresh", params=params
    )
    if r_post.status_code == 200:
        notify("ID token was successfully fetched.")
    else:
        notify(f"Failed to fetch ID token: {r_post.status_code}", "error")
        notify(r_post.json(), "error")
        return

    id_token = schemas.IdToken(**r_post.json()).id_token
    app_settings.id_token = id_token
    with open(f"{constants.APP_SETTINGS_PATH}", "w") as f:
        json.dump(app_settings.model_dump(), f)

    notify(f"ID token was saved to {constants.APP_SETTINGS_PATH}")


@app.command("tokens")
def fetch_tokens() -> None:
    fetch_refresh_token()
    fetch_id_token()


@app.command("daily-quotes")
def fetch_daily_quotes(
    code: str,
    insert_db: bool = False,
) -> pl.DataFrame:
    app_settings: schemas.AppSettings = load_settings()
    headers: dict[str, str] = {
        "Authorization": "Bearer {}".format(app_settings.id_token)
    }
    params: dict[str, str] = {"code": code}

    logger.info("Fetch Daily Quotes")

    r = requests.get(
        "https://api.jquants.com/v1/prices/daily_quotes",
        headers=headers,
        params=params,
    )
    if r.status_code == 401:
        raise custom_error.TokenExpiredError("ID Token has expired.")

    if r.status_code != 200:
        message = f"Failed to fetch Daily Quotes: {r.status_code} {r.json()}"
        echo(message)
        logger.error(message)
        logger.error(r.json())
        raise Exit(1)

    logger.info(f"Successfully fetched Daily Quotes of {code}")
    daily_quotes = schemas.DailyQuotes(**r.json())

    df = pl.DataFrame(daily_quotes.model_dump()["daily_quotes"])
    debug_df_lf(df)

    if insert_db:
        insert_ohlc(df)

    return df


@app.command()
def get_daily_quotes(
    code: str, latest_market_date: str, force_fetch: bool = False
) -> pl.DataFrame:
    with database.open_db() as conn:
        if force_fetch:
            # DBから削除して、再取得
            database.delete_ohlc_all_by_code(conn, code)
            return fetch_daily_quotes(code, insert_db=True)

        # DBから最新の日付のデータを取得
        latest_ohlc = database.select_ohlc_one_by_code_latest_date(conn, code)

        if len(latest_ohlc) == 0:
            # DBにデータがない場合は取得
            return fetch_daily_quotes(code, insert_db=True)

        if latest_ohlc["date"][0] == latest_market_date:
            # 最新の日付が最新の市場日付と一致する場合はDBから取得
            return database.select_ohlc_all_by_code(conn, code)
        else:
            # DBから削除して、再取得
            database.delete_ohlc_all_by_code(conn, code)

    return fetch_daily_quotes(code, insert_db=True)


@app.command("nikkei225")
def fetch_nikkei225() -> None:
    nikkei225 = load_csv(schemas.CsvFile.NIKKEI225)
    code_list = nikkei225.get_column("code").to_list()
    for code in tqdm(code_list):
        fetch_daily_quotes(code, insert_db=True)


@app.command("topix400")
def fetch_topix400() -> None:
    topix400 = load_csv(schemas.CsvFile.TOPIX400)
    code_list = topix400.get_column("code").to_list()
    for code in tqdm(code_list, bar_format=constants.SHORT_PROGRESS_BAR):
        fetch_daily_quotes(code, insert_db=True)


@app.command("topix")
def fetch_topix(
    insert_db: bool = False, to: Optional[str] = Option(None)
) -> pl.DataFrame:
    app_settings = load_settings()
    headers = {"Authorization": "Bearer {}".format(app_settings.id_token)}
    params: dict[str, str] = {}
    if type(to) is str and len(to) > 0:
        params["to"] = to

    logger.info("Fetch Topix")

    r = requests.get(
        "https://api.jquants.com/v1/indices/topix",
        headers=headers,
        params=params,
    )
    if r.status_code != 200:
        notify(f"Failed to fetch Topix: {r.status_code}", "error")
        notify(r.json(), "error")
        raise Exit(1)

    logger.info("Successfully fetched Topix")
    topix = schemas.Topix(**r.json())

    df = pl.DataFrame(topix.model_dump()["topix"])
    debug_df_lf(df)

    if insert_db:
        insert_ohlc(df, table_name="topix")

    return df


@app.command("trading-calendar")
def fetch_trading_calender() -> pl.LazyFrame:
    app_settings = load_settings()
    today = datetime.date.today()
    yesterday = today - datetime.timedelta(days=1)
    day_before_1600 = today - datetime.timedelta(days=1600)
    # today -> YYYY-MM-DD
    yesterday_str = yesterday.strftime("%Y-%m-%d")
    day_before_1600_str = day_before_1600.strftime("%Y-%m-%d")
    headers = {"Authorization": "Bearer {}".format(app_settings.id_token)}
    params = {
        "holidaydivision": "1",
        "from": f"{day_before_1600_str}",
        "to": f"{yesterday_str}",
    }

    r = requests.get(
        "https://api.jquants.com/v1/markets/trading_calendar",
        headers=headers,
        params=params,
    )
    lf = pl.LazyFrame(r.json().get("trading_calendar"))

    debug_df_lf(lf)

    return lf


@app.command("latest-market-date")
def get_latest_market_date() -> str:
    lf = fetch_trading_calender()
    # ["HolidayDivision"] == "1" が市場日
    latest_date = (
        lf.filter(pl.col("HolidayDivision") == "1").collect().sort("Date")["Date"][-1]
    )
    notify(f"最新の取引日: {latest_date}")
    return latest_date
