import os
import re
import pandas as pd
from ta import add_all_ta_features
from datetime import datetime, timedelta
from ta.trend import SMAIndicator, EMAIndicator
from scipy.signal import savgol_filter
from .utils import timeframe_to_timedelta


def add_ta_features(df):
    dd = add_all_ta_features(
        df, open="open", high="high", low="low", close="close", volume="volume"
    )
    df = dd.copy()
    return df


def add_calendar_features(df):
    date = df["datetime"]
    df["hour"] = date.dt.hour
    df["day"] = date.dt.day
    df["day_of_week"] = date.dt.dayofweek
    df["month"] = date.dt.month
    df["week_of_year"] = date.dt.isocalendar().week
    df["year"] = date.dt.year
    return df


def add_time_idx(df):
    # infer timeframe
    df = df.sort_values("datetime")
    deltas_frq = df.datetime.diff().value_counts(normalize=True)
    if deltas_frq.max() < 0.99:
        raise ValueError("Mismatch in dataset timeframe")
    delta = deltas_frq.idxmax()

    # add time idx
    df["time_idx"] = ((df.datetime - datetime(2000, 1, 1)) / delta).astype(int)
    # df["time_idx"] -= df["time_idx"].min()

    assert df["time_idx"].duplicated().sum() == 0
    return df


def add_price_moving_features(df, windows=[50, 100, 200]):
    for w in windows:
        df[f"ma_{w}"] = SMAIndicator(
            close=df["close"], window=w, fillna=False
        ).sma_indicator()

        df[f"ema_{w}"] = EMAIndicator(
            close=df["close"], window=w, fillna=False
        ).ema_indicator()
    return df


def process_dtypes(df):
    df["datetime"] = pd.to_datetime(df["datetime"])
    return df


def process_common(df, n_beginning_days_skip=90):
    # skip the beginnings
    min_date = df["datetime"].min()
    split = min_date + timedelta(days=n_beginning_days_skip)
    df = df[df.datetime >= split]
    # remove duplicates
    df = df[~df.datetime.duplicated(keep="last")]
    # remove garbage features
    df = df.loc[:, (df.notnull().mean() > 0.85)]
    # remove NaNs
    df = df.dropna()
    return df


def smooth_values(
    df,
    window=21,
    order=4,
    exclude=[
        "datetime",
        "symbol",
        "hour",
        "day",
        "day_of_week",
        "month",
        "week_of_year",
        "year",
        "time_idx",
    ],
):
    features = df.columns[~df.columns.isin(exclude)]
    for col in features:
        df[col] = savgol_filter(df[col], window_length=window, polyorder=order)
    return df


def resample_ohlcv(df, timeframe, date_col=None, drop_ends=False):
    if date_col:
        df[date_col] = pd.to_datetime(df[date_col])
        df = df.set_index(date_col)

    delta = timeframe_to_timedelta(timeframe)
    offset = pd.offsets.Second(delta.total_seconds())
    rule = offset.freqstr

    ohlc_dict = {
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    }

    df = df.resample(rule, closed="right", label="right").apply(ohlc_dict)

    if drop_ends:
        df = df[1:-1]
    if date_col:
        df = df.reset_index()
    return df
