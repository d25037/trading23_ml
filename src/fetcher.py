import json
from time import sleep

import database
import pandas as pd
import requests
from pydantic import BaseModel, Field


class AppSettings(BaseModel):
    mailaddress: str
    password: str
    refresh_token: str
    id_token: str


def load_settings():
    with open("/data/app_settings.json", "r") as f:
        return AppSettings(**json.load(f))


def load_nikkei225_csv():
    with open("/data/nikkei225.csv") as f:
        df = pd.read_csv(f)
    return df


class RefreshToken(BaseModel):
    refresh_token: str = Field(alias="refreshToken")


def fetch_refresh_token():
    app_settings = load_settings()
    body = {"mailaddress": app_settings.mailaddress, "password": app_settings.password}

    print("Fetch Refresh token...")

    r_post = requests.post(
        "https://api.jquants.com/v1/token/auth_user", data=json.dumps(body)
    )
    if r_post.status_code == 200:
        print("Refresh token was successfully fetched.")
    else:
        print(f"Failed to fetch Refresh token: {r_post.status_code}")
        return

    refresh_token = RefreshToken(**r_post.json()).refresh_token
    app_settings.refresh_token = refresh_token
    with open("/data/app_settings.json", "w") as f:
        json.dump(app_settings.model_dump(), f)

    print("Refresh token was saved to /data/app_settings.json")


class IdToken(BaseModel):
    id_token: str = Field(alias="idToken")


def fetch_id_token():
    app_settings = load_settings()
    params = {"refreshtoken": app_settings.refresh_token}

    print("Fetch ID token...")

    r_post = requests.post(
        "https://api.jquants.com/v1/token/auth_refresh", params=params
    )
    if r_post.status_code == 200:
        print("ID token was successfully fetched.")
    else:
        print(f"Failed to fetch ID token: {r_post.status_code}")
        print(r_post.json())
        return

    id_token = IdToken(**r_post.json()).id_token
    app_settings.id_token = id_token
    with open("/data/app_settings.json", "w") as f:
        json.dump(app_settings.model_dump(), f)

    print("ID token was saved to /data/app_settings.json")


class OhlcPremium(BaseModel):
    code: str = Field(alias="Code")
    date: str = Field(alias="Date")
    open: float | None = Field(alias="AdjustmentOpen")
    high: float | None = Field(alias="AdjustmentHigh")
    low: float | None = Field(alias="AdjustmentLow")
    close: float | None = Field(alias="AdjustmentClose")
    volume: float | None = Field(alias="AdjustmentVolume")
    morning_close: float | None = Field(alias="MorningAdjustmentClose")
    afternoon_open: float | None = Field(alias="AfternoonAdjustmentOpen")


class DailyQuotes(BaseModel):
    daily_quotes: list[OhlcPremium]


def fetch_daily_quotes(code: str):
    app_settings = load_settings()
    headers = {"Authorization": "Bearer {}".format(app_settings.id_token)}
    params = {"code": code}

    print("Fetch Daily Quotes")

    r = requests.get(
        "https://api.jquants.com/v1/prices/daily_quotes",
        headers=headers,
        params=params,
    )

    daily_quotes = DailyQuotes(**r.json())

    return pd.DataFrame(daily_quotes.model_dump()["daily_quotes"])


def fetch_daily_quotes_of_nikkei225_to_db():
    nikkei225 = load_nikkei225_csv()
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
