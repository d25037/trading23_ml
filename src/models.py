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


class Result(BaseModel):
    code: str
    date: str
    day1_morning: float
    day1_allday: float
    day5: float
    day20: float


class ResultWoVolume(BaseModel):
    code: str
    date: str
    standardized_diff: float
    day1_morning: float
    day1_allday: float
    day5: float
    day20: float
    image: bytes
