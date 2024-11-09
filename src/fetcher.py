import json
from time import sleep

import pandas as pd
import polars as pl
import requests
from loguru import logger
from pydantic import BaseModel, Field
from typer import Typer

import database

APP_SETTINGS_PATH = "./app_settings.json"
NIKKEI225_PATH = "./data/nikkei225.csv"

app = Typer(no_args_is_help=True)


@app.command()
def test():
    nikkei225 = load_nikkei225_csv_as_pl()
    print(nikkei225)


class AppSettings(BaseModel):
    mailaddress: str
    password: str
    refresh_token: str
    id_token: str


@app.command()
def load_settings():
    with open(f"{APP_SETTINGS_PATH}", "r") as f:
        items = AppSettings(**json.load(f))
        logger.debug(items)
        return items


@app.command("load-nikkei225")
def load_nikkei225_csv_as_pl():
    with open(f"{NIKKEI225_PATH}") as f:
        df = pl.read_csv(f)
        logger.debug(df)
    return df


def load_nikkei225_csv():
    with open(f"{NIKKEI225_PATH}") as f:
        df = pd.read_csv(f)
    return df


class RefreshToken(BaseModel):
    refresh_token: str = Field(alias="refreshToken")


@app.command()
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

    refresh_token = RefreshToken(**r_post.json()).refresh_token
    app_settings.refresh_token = refresh_token
    with open(f"{APP_SETTINGS_PATH}", "w") as f:
        json.dump(app_settings.model_dump(), f)

    logger.info(f"Refresh token was saved to {APP_SETTINGS_PATH}")


class IdToken(BaseModel):
    id_token: str = Field(alias="idToken")


@app.command()
def fetch_id_token():
    app_settings = load_settings()
    params = {"refreshtoken": app_settings.refresh_token}

    print("Fetch ID token...")

    r_post = requests.post(
        "https://api.jquants.com/v1/token/auth_refresh", params=params
    )
    if r_post.status_code == 200:
        logger.info("ID token was successfully fetched.")
    else:
        logger.error(f"Failed to fetch ID token: {r_post.status_code}")
        logger.error(r_post.json())
        return

    id_token = IdToken(**r_post.json()).id_token
    app_settings.id_token = id_token
    with open(f"{APP_SETTINGS_PATH}", "w") as f:
        json.dump(app_settings.model_dump(), f)

    logger.info(f"ID token was saved to {APP_SETTINGS_PATH}")


class Ohlc(BaseModel):
    code: str = Field(alias="Code")
    date: str = Field(alias="Date")
    open: float | None = Field(alias="AdjustmentOpen")
    high: float | None = Field(alias="AdjustmentHigh")
    low: float | None = Field(alias="AdjustmentLow")
    close: float | None = Field(alias="AdjustmentClose")
    volume: float | None = Field(alias="AdjustmentVolume")


class DailyQuotes(BaseModel):
    daily_quotes: list[Ohlc]


@app.command()
def fetch_daily_quotes(code: str):
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
    daily_quotes = DailyQuotes(**r.json())
    logger.debug(daily_quotes.daily_quotes)

    return pl.DataFrame(daily_quotes.model_dump()["daily_quotes"])


def fetch_daily_quotes_of_nikkei225_to_db():
    nikkei225 = load_nikkei225_csv_as_pl()
    for i in range(len(nikkei225)):
        code = str(nikkei225.iloc[i, 0])
        df = fetch_daily_quotes(code)
        database.insert_db(df)
        print(f"Inserted {code} to database.")
        sleep(1)


def fetch_trading_calender():
    app_settings = load_settings()
    headers = {"Authorization": "Bearer {}".format(app_settings.id_token)}
    params = {"holidaydivision": 1, "from": "2022-01-01", "to": "2022-12-31"}

    r = requests.get(
        "https://api.jquants.com/v1/markets/trading_calendar",
        headers=headers,
        params=params,
    )
    print(r.json())
