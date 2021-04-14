import pandas as pd
import re
import pytz
from datetime import datetime, timedelta


def timeframe_to_timedelta(timeframe):
    timeframe_regex = re.compile("([0-9]+)([a-zA-Z])")
    timeframe_matches = timeframe_regex.match(timeframe)
    time_quantity = timeframe_matches.group(1)
    time_period = timeframe_matches.group(2)
    timedelta_values = {
        "m": "minutes",
        "h": "hours",
        "d": "days",
        "w": "weeks",
        "M": "months",
        "y": "years",
    }
    timedelta_args = {timedelta_values[time_period]: int(time_quantity)}
    tfdelta = timedelta(**timedelta_args)
    return tfdelta


def timestamp_to_datetime(timestamp, timezone=None, to_str=False):
    time = datetime.fromtimestamp(timestamp, timezone)
    if to_str:
        time = time.strftime("%Y-%m-%d %H:%M:%S")
    return time


def historical_to_dataframe(historical_data, timezone=None):
    """Converts historical data matrix to a pandas dataframe.

    Args:
        historical_data (list): A matrix of historical OHCLV data.

    Returns:
        pandas.DataFrame: Contains the historical data in a pandas dataframe.
    """

    dataframe = pd.DataFrame(historical_data)
    dataframe.transpose()

    dataframe.columns = ["timestamp", "open", "high", "low", "close", "volume"]
    dataframe["datetime"] = dataframe.timestamp.apply(
        lambda x: timestamp_to_datetime(x / 1000, timezone=timezone, to_str=True)
    )

    dataframe.set_index("datetime", inplace=True, drop=True)
    dataframe.drop("timestamp", axis=1, inplace=True)

    return dataframe
