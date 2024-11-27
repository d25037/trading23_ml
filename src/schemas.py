from enum import Enum

from pydantic import BaseModel, Field


class AppSettings(BaseModel):
    mailaddress: str
    password: str
    refresh_token: str
    id_token: str


class RefreshToken(BaseModel):
    refresh_token: str = Field(alias="refreshToken")


class IdToken(BaseModel):
    id_token: str = Field(alias="idToken")


class Ohlc(BaseModel):
    code: str = Field(alias="Code")
    date: str = Field(alias="Date")
    open: float | None = Field(alias="AdjustmentOpen")
    high: float | None = Field(alias="AdjustmentHigh")
    low: float | None = Field(alias="AdjustmentLow")
    close: float | None = Field(alias="AdjustmentClose")
    volume: float | None = Field(alias="AdjustmentVolume")


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
    daily_quotes: list[Ohlc]


class TopixOhlc(BaseModel):
    date: str = Field(alias="Date")
    open: float | None = Field(alias="Open")
    high: float | None = Field(alias="High")
    low: float | None = Field(alias="Low")
    close: float | None = Field(alias="Close")


class Topix(BaseModel):
    topix: list[TopixOhlc]


class Result(BaseModel):
    code: str
    date: str
    nextday_open: float
    nextday_close: float
    image: bytes


class Result2(BaseModel):
    code: str
    date: str
    day1_close: float
    day3_close: float
    day5_close: float
    image: bytes


class Result3(BaseModel):
    code: str
    date: str
    result_1: float
    result_3: float
    result_5: float
    image: bytes


class Outlook(Enum):
    BULLISH = "Bullish"
    BEARISH = "Bearish"
    NEUTRAL = "Neutral"
    ALL = "All"


class CsvFile(Enum):
    NIKKEI225 = "Nikkei225"
    TOPIX400 = "Topix400"
