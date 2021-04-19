import os
import sys
import re
import ccxt
import pytz
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from core.config.defaults import DATA_DIR
from .utils import timeframe_to_timedelta, historical_to_dataframe


def get_historical_data(
    symbol, exchange, timeframe, start_date=None, total=None, max_per_page=500, timezone=None,
):
    """Get historical OHLCV for a symbol pair

    Decorators:
        retry

    Args:
        symbol (str): Contains the symbol pair to operate on i.e. BURST/BTC
        exchange (str): Contains the exchange to fetch the historical data from.
        timeframe (str): A string specifying the ccxt time unit i.e. 5m or 1d.
        start_date (datetime, optional)
        max_periods (int, optional): Defaults to 100. Maximum number of time periods
          back to fetch data for.

    Returns:
        list: Contains a list of lists which contain timestamp, open, high, low, close, volume.
    """

    try:
        if timeframe not in exchange.timeframes:
            raise ValueError(
                "{} does not support {} timeframe for OHLCV data. Possible values are: {}".format(
                    exchange, timeframe, list(exchange.timeframes)
                )
            )
    except AttributeError:
        print(
            "%s interface does not support timeframe queries! We are unable to fetch data!", exchange,
        )
        raise AttributeError(sys.exc_info())

    if not start_date:
        single_frame = timeframe_to_timedelta(timeframe)
        start_date = datetime.now() - (total * single_frame)

    start_date = int(start_date.timestamp() * 1000)
    stop_limit = total or np.inf
    historical_data = []

    try:
        cursor = start_date
        while True:
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, since=cursor, limit=max_per_page)
            if (not ohlcv) or len(ohlcv) == 0:
                break
            historical_data += ohlcv
            if len(historical_data) >= stop_limit:
                historical_data = historical_data[:stop_limit]
                break
            cursor = ohlcv[-1][0] + 1
    except Exception as e:
        raise e

    if not historical_data:
        return

    # Sort by timestamp in ascending order
    historical_data.sort(key=lambda d: d[0])
    historical_data = historical_to_dataframe(historical_data, timezone)

    return historical_data


def get_crypto_dataset(
    exchange,
    symbol,
    timeframe,
    data_fp=None,
    data_frame=None,
    start_date=datetime(2017, 1, 1),
    retry_skip_delta=timedelta(days=10),
    **kwargs,
):
    if data_fp is None:
        save_dir = os.path.join(DATA_DIR, exchange.name.lower())
        data_fp = os.path.join(save_dir, f"{symbol.replace('/', '-')}_{timeframe}.csv")
        os.makedirs(save_dir, exist_ok=True)

    if (data_frame is not None) or os.path.isfile(data_fp):  # resume from current
        if data_frame is not None:
            old_data = data_frame
        else:
            old_data = pd.read_csv(data_fp)

        old_data["datetime"] = pd.to_datetime(old_data["datetime"])
        return old_data
        last_date = old_data.datetime.iloc[-2]
        data = get_historical_data(
            symbol=symbol, exchange=exchange, timeframe=timeframe, start_date=last_date, **kwargs
        )
        data = data.reset_index()
        data["datetime"] = pd.to_datetime(data["datetime"])

        combined = pd.concat([old_data, data])
        combined = combined[~combined.datetime.duplicated(keep="last")]
        result = combined.reset_index(drop=True)

    else:
        while True:
            data = get_historical_data(symbol=symbol, exchange=exchange, timeframe=timeframe, start_date=start_date,)
            if data is None:
                start_date += retry_skip_delta
            else:
                data = data.reset_index()
                data["datetime"] = pd.to_datetime(data["datetime"])
                result = data
                break

    result.to_csv(data_fp, index=False)
    return result
