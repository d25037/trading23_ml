import datetime
import json
from sys import stderr

import polars as pl
import requests
from loguru import logger
from tqdm import tqdm
from typer import Typer

import constants
import schemas
from database import insert_ohlc

app = Typer(no_args_is_help=True)


@app.command("settings")
def load_settings():
    with open(f"{constants.APP_SETTINGS_PATH}", "r") as f:
        items = schemas.AppSettings(**json.load(f))
        logger.debug(items)
        return items


@app.command("csv-nikkei225")
def load_nikkei225_csv() -> pl.DataFrame:
    with open(f"{constants.NIKKEI225_PATH}") as f:
        df = pl.read_csv(f)
        logger.debug(df)
    return df


@app.command("csv-topix400")
def load_topix400_csv() -> pl.DataFrame:
    with open(f"{constants.TOPIX400_PATH}") as f:
        df = pl.read_csv(f)
        logger.debug(df)
    return df


@app.command("refresh-token")
def fetch_refresh_token():
    app_settings = load_settings()
    body = {"mailaddress": app_settings.mailaddress, "password": app_settings.password}

    logger.info("Fetch Refresh token...")

    r_post = requests.post(
        "https://api.jquants.com/v1/token/auth_user", data=json.dumps(body)
    )
    if r_post.status_code == 200:
        logger.info("Refresh token was successfully fetched.")
    else:
        logger.error(f"Failed to fetch Refresh token: {r_post.status_code}")
        return

    refresh_token = schemas.RefreshToken(**r_post.json()).refresh_token
    app_settings.refresh_token = refresh_token
    with open(f"{constants.APP_SETTINGS_PATH}", "w") as f:
        json.dump(app_settings.model_dump(), f)

    logger.info(f"Refresh token was saved to {constants.APP_SETTINGS_PATH}")


@app.command("id-token")
def fetch_id_token():
    app_settings = load_settings()
    params = {"refreshtoken": app_settings.refresh_token}

    logger.info("Fetch ID token...")

    r_post = requests.post(
        "https://api.jquants.com/v1/token/auth_refresh", params=params
    )
    if r_post.status_code == 200:
        logger.info("ID token was successfully fetched.")
    else:
        logger.error(f"Failed to fetch ID token: {r_post.status_code}")
        logger.error(r_post.json())
        return

    id_token = schemas.IdToken(**r_post.json()).id_token
    app_settings.id_token = id_token
    with open(f"{constants.APP_SETTINGS_PATH}", "w") as f:
        json.dump(app_settings.model_dump(), f)

    logger.info(f"ID token was saved to {constants.APP_SETTINGS_PATH}")


@app.command("tokens")
def fetch_tokens():
    logger.remove()
    logger.add(stderr, level="INFO")
    fetch_refresh_token()
    fetch_id_token()


@app.command("daily-quotes")
def fetch_daily_quotes(code: str, insert_db: bool = False):
    app_settings = load_settings()
    headers = {"Authorization": "Bearer {}".format(app_settings.id_token)}
    params = {"code": code}

    logger.info("Fetch Daily Quotes")

    r = requests.get(
        "https://api.jquants.com/v1/prices/daily_quotes",
        headers=headers,
        params=params,
    )
    if r.status_code != 200:
        logger.error(f"Failed to fetch Daily Quotes: {r.status_code}")
        logger.error(r.json())
        return

    logger.info(f"Successfully fetched Daily Quotes of {code}")
    daily_quotes = schemas.DailyQuotes(**r.json())
    logger.debug(daily_quotes.daily_quotes)
    logger.debug(f"len: {len(daily_quotes.daily_quotes)}")

    df = pl.DataFrame(daily_quotes.model_dump()["daily_quotes"])

    if insert_db:
        insert_ohlc(df)

    return df


@app.command("nikkei225")
def fetch_nikkei225():
    logger.remove()
    logger.add(stderr, level="INFO")

    nikkei225 = load_nikkei225_csv()
    code_list = nikkei225.get_column("code").to_list()
    for code in tqdm(code_list):
        fetch_daily_quotes(code, insert_db=True)


@app.command("topix400")
def fetch_topix400():
    logger.remove()
    logger.add(stderr, level="INFO")

    topix400 = load_topix400_csv()
    code_list = topix400.get_column("code").to_list()
    for code in tqdm(code_list, bar_format=constants.SHORT_PROGRESS_BAR):
        fetch_daily_quotes(code, insert_db=True)


@app.command("topix")
def fetch_topix(insert_db: bool = False):
    app_settings = load_settings()
    headers = {"Authorization": "Bearer {}".format(app_settings.id_token)}

    logger.info("Fetch Topix")

    r = requests.get(
        "https://api.jquants.com/v1/indices/topix",
        headers=headers,
    )
    if r.status_code != 200:
        logger.error(f"Failed to fetch Topix: {r.status_code}")
        logger.error(r.json())
        return

    logger.info("Successfully fetched Topix")
    topix = schemas.Topix(**r.json())
    logger.debug(topix.topix)
    logger.debug(f"len: {len(topix.topix)}")

    df = pl.DataFrame(topix.model_dump()["topix"])
    logger.debug(df)

    if insert_db:
        insert_ohlc(df, table_name="topix")

    return df


@app.command("training-calendar")
def fetch_trading_calender():
    app_settings = load_settings()
    today = datetime.date.today()
    yesterday = today - datetime.timedelta(days=1)
    day_before_1600 = today - datetime.timedelta(days=1600)
    # today -> YYYY-MM-DD
    yesterday = yesterday.strftime("%Y-%m-%d")
    day_before_1600 = day_before_1600.strftime("%Y-%m-%d")
    headers = {"Authorization": "Bearer {}".format(app_settings.id_token)}
    params = {"holidaydivision": 1, "from": f"{day_before_1600}", "to": f"{yesterday}"}

    r = requests.get(
        "https://api.jquants.com/v1/markets/trading_calendar",
        headers=headers,
        params=params,
    )
    lf = pl.LazyFrame(r.json().get("trading_calendar"))
    logger.debug(f"len: {len(lf.collect())}")
    df = lf.select("Date").collect().sample(10).sort("Date")
    logger.debug(f"len: {df}")

    # logger.debug(df)
    # logger.debug(f"len: {df}")
    return
