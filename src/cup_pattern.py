#!/usr/bin/env python3
"""
Description: This module has classes and functions that ...

Created: 2025
Author: Tom Jordan
"""
from dataclasses import dataclass
import logging
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.subplots as psp

import data_core as dc

logger = logging.getLogger(Path(__file__).stem)

DATETIME = 'Datetime'
OPEN = 'Open'
LOW = 'Low'
HIGH = 'High'
CLOSE = 'Close'
SMA50 = 'SMA50'
SMA150 = 'SMA150'
SMA200 = 'SMA200'
VOLUME = 'Volume'

HANDLE_Y = 'HandleY'

@dataclass
class Config:
    mrkt_symbol = 'SPY'
    # symbol = 'TSM'
    # symbol = 'AAPL'
    # symbol = 'TSLA'
    symbol = 'PLTR'
    # symbol = 'MSFT'
    # symbol = 'NVDA'
    interval = "1d"
    period = "5y"
    # dtime1 = '2021-02-01'
    # dtime2 = '2021-06-01'
    dtime1 = '2024-02-01'
    dtime2 = '2025-02-01'
    cupdepth_pct = 20.0
    volume_period = 20
    rs_period = 21

    def __post_init__(self) -> None:
        # self.dtime1 = datetime.strptime(self.dtime1, '%Y-%m-%d')
        # self.dtime2 = datetime.strptime(self.dtime2, '%Y-%m-%d')
        self.volume_smacol = f'VMA{self.volume_period}'
        self.rs_smacol = f'RS{self.rs_period}'

# Helper function to find local minima and maxima
# def append_local_minmax(df:pd.DataFrame) -> None:
#     local_min = (df[LOW].shift(1) > df[LOW]) & (df[LOW].shift(-1) > df[LOW])
#     local_max = (df[HIGH].shift(1) < df[HIGH]) & (df[HIGH].shift(-1) < df[HIGH])
#     df.loc[:, 'LocalMin'] = local_min
#     df.loc[:, 'LocalMax'] = local_max

# def append_handle_columns(df: pd.DataFrame) -> None:
#     """
#     # Initialize the 'HandleDt' column with '1970-01-01 00:00:00' and HANDLE_Y with 0
#     # Iterate through the DataFrame to find the required rows
#     # Find the local maxima and minima
#     # Find the required depth
#     # Find the handle

#     :param df: A DataFrame containing the OHLC data with columns 'Open', 'High',
#         'Low', 'Close', and 'Date'.
#     """
#     df.loc[:, 'HandleDt'] = datetime(1970, 1, 1)
#     df.loc[:, HANDLE_Y] = 0.0

#     # Iterate through the DataFrame to find the required rows
#     for i in range(len(df)):
#         start_max = 0
#         if df.loc[i, 'LocalMax']:
#             start_max = df.loc[i, HIGH]
#             value_min = start_max
#             for j in range(i + 1, len(df)):
#                 if df.loc[j, 'LocalMin'] and df.loc[j, LOW] < value_min:
#                     value_min = df.loc[j, LOW]
#                 elif df.loc[j, HIGH] >= start_max:
#                     depth_pct = ((start_max - value_min) / start_max) * 100
#                     # NOTE: The depth_pct will be 0 if the local minima occurs on
#                     # the start or end maxima
#                     if depth_pct > 0:
#                         df.loc[i, 'DepthPct'] = depth_pct
#                         df.loc[i, 'HandleDt'] = df.loc[j, DATETIME]
#                         df.loc[i, HANDLE_Y] = df.loc[j, HIGH]

#                     break
#     # Convert 'HandleDt' to datetime format
#     df['HandleDt'] = pd.to_datetime(df['HandleDt'], utc=True)
#     df['HandleDt'] = df['HandleDt'].dt.strftime('%Y-%m-%d')

def append_local_minmax(df:pd.DataFrame) -> None:
    local_min = (df[LOW].shift(1) > df[LOW]) & (df[LOW].shift(-1) > df[LOW])
    local_max = (df[HIGH].shift(1) < df[HIGH]) & (df[HIGH].shift(-1) < df[HIGH])
    df.loc[:, 'LocalMin'] = local_min
    df.loc[:, 'LocalMax'] = local_max

def append_handle_columns(df: pd.DataFrame) -> None:
    """
    # Initialize the 'HandleDt' column with '1970-01-01 00:00:00' and HANDLE_Y with 0
    # Iterate through the DataFrame to find the required rows
    # Find the local maxima and minima
    # Find the required depth
    # Find the handle

    :param df: A DataFrame containing the OHLC data with columns 'Open', 'High',
        'Low', 'Close', and 'Date'.
    """
    df.loc[:, 'HandleDt'] = datetime(1970, 1, 1)
    df.loc[:, HANDLE_Y] = 0.0

    # Iterate through the DataFrame to find the required rows
    for i in range(len(df)):
        current_dt = df.index[i]
        start_max = 0
        if df.loc[current_dt, 'LocalMax']:
            start_max = df.loc[current_dt, HIGH]
            value_min = start_max

            for j in range(i + 1, len(df)):
                next_dt = df.index[j]
                if df.loc[next_dt, 'LocalMin'] and df.loc[next_dt, LOW] < value_min:
                    value_min = df.loc[next_dt, LOW]
                elif df.loc[next_dt, HIGH] >= start_max:
                    depth_pct = ((start_max - value_min) / start_max) * 100
                    # NOTE: The depth_pct will be 0 if the local minima occurs on
                    # the start or end maxima
                    if depth_pct > 0:
                        df.loc[current_dt, 'DepthPct'] = depth_pct
                        # df.loc[current_dt, 'HandleDt'] = df.loc[next_dt, DATETIME]
                        df.loc[current_dt, 'HandleDt'] = next_dt
                        df.loc[current_dt, HANDLE_Y] = df.loc[next_dt, HIGH]
                    break
    # Convert 'HandleDt' to datetime format
    df['HandleDt'] = pd.to_datetime(df['HandleDt'], utc=True)
    df['HandleDt'] = df['HandleDt'].dt.strftime('%Y-%m-%d')


def plot_candlesticks_and_cups(df:pd.DataFrame, cf:Config) -> None:
    """_summary_

    :param df: A DataFrame containing the OHLC data with columns 'Open', 'High',
        'Low', 'Close', and 'Date'.
    :param required_depth: The required depth of the cup pattern as a percentage of
        local maxima (0 to 100 inclusive). For example, a value of 20 requires
        a depth of 20% or more from the local maxima to the local minima.
    :param symbol: Ticker symbol
    :param interval: Time interval between rows
    :param period: Length of time between first and last row
    """
    # Create the candlestick figure
    subtitle = f"{cf.interval}, {cf.period}, {cf.cupdepth_pct}% (interval, period, min cup depth)"
    title = f"Cup & Handle Chart"
    title = f"<b>{title}</b><br>{cf.symbol} {subtitle}"

    fig = psp.make_subplots(rows=2, cols=1,
                            shared_xaxes=True,
                            specs=[[{"secondary_y": True}], [{}]],
                            vertical_spacing=0.07,
                            row_heights=[0.65, 0.35])
    fig.add_traces(
        data=[
            go.Candlestick(
                    x=df.index,
                    open=df['Open'],
                    high=df[HIGH],
                    low=df[LOW],
                    close=df[CLOSE],
                    name='Candlesticks'),
            go.Scatter(
                    x=df.index,
                    y=df[SMA50],
                    mode='lines',
                    line=dict(color='purple'),
                    name=SMA50),
            go.Scatter(x=df.index,
                    y=df[SMA150],
                    mode='lines',
                    line=dict(color='blue'),
                    name=SMA150),
            go.Scatter(x=df.index,
                    y=df[SMA200],
                    mode='lines',
                    line=dict(color='green'),
                    name=SMA200),
            # Create a trace for the 'RS' column
            go.Scatter(
                x=df.index,
                y=df['RS'],
                mode='lines',
                line=dict(color='red'),
                name='RS'
            ),
            # go.Scatter(
            #     x=df.index,
            #     y=df[cf.rs_smacol],
            #     mode='lines',
            #     line=dict(color='orange'),
            #     name=cf.rs_smacol)
        ],
        rows=[1,1,1,1,1],
        cols=[1,1,1,1,1],
        secondary_ys=[False, False, False, False, True]
    )
    fig.add_traces(
        data = [
            go.Bar(x=df.index,
                    y=df[VOLUME],
                    marker_color=df['VColor'],
                    marker_opacity=0.7,
                    name=VOLUME),
            go.Scatter(x=df.index,
                    y=df[cf.volume_smacol],
                    mode='lines',
                    line=dict(color='blue',
                                width=1.0),
                    name=cf.volume_smacol),
        ],
        rows=[2, 2],
        cols=[1, 1]
    )
    # Add line segments for 'HandleDt' where HANDLE_Y != 0 and 'DepthPct' >= required_depth
    for i in range(len(df)):
        current_dt = df.index[i]
        if df.loc[current_dt, HANDLE_Y] != 0 and df.loc[current_dt, 'DepthPct'] >= cf.cupdepth_pct:
            fig.add_trace(
                go.Scatter(
                    # x=[df.loc[i, DATETIME], df.loc[i, 'HandleDt']],
                    x=[current_dt, df.loc[current_dt, 'HandleDt']],
                    y=[df.loc[current_dt, HIGH], df.loc[current_dt, HANDLE_Y]],
                    mode='lines',
                    line=dict(color='blue', dash='dot'),
                    # name='Handle Line'
                    name=f"{df.loc[current_dt, 'HandleDt']}, {df.loc[current_dt, HANDLE_Y]:0.2f}, {df.loc[current_dt, 'DepthPct']:0.1f}%",
                    yaxis='y1'
                    ),
                row=1, col=1)

    fig.update_xaxes(showspikes=True, spikemode="across", spikethickness=1, matches='x', showticklabels=True)
    fig.update_yaxes(showspikes=True, spikemode="across", spikethickness=1)

    # Update layout
    fig.update_layout(
        title=title,
        # xaxis_title=DATETIME,  # Don't set this if xaxis_title is True in update_layout
        yaxis_title='Price, $',
        yaxis2=dict(title='RS', side='right', overlaying='y', range=[0,100]),
        yaxis3_title=VOLUME,
        hovermode='x',
        xaxis_rangeslider_visible=True, # if True, adds candlesticks from volume plot
        barmode='overlay'
    )
    fig.show()


def main() -> None:
    cf = Config()

    mini_file = dc.get_jsonpath(cf.symbol, cf.interval, cf.period)
    mini_file = mini_file.parent / f"{mini_file.stem}_{cf.dtime1}_{cf.dtime2}.json"
    # if mini_file.exists():
    if False:
        print(f"Reading JSON file '{mini_file}'")
        mini_df = pd.read_json(mini_file)
        print(f"{len(mini_df)=}")
    else:
        fetcher = dc.StockDataFetcher()

        print(f"Fetching data for {cf.symbol=}, {cf.interval=}, {cf.period=}")
        mrkt_df = fetcher.history(cf.mrkt_symbol, cf.interval, cf.period)
        mrkt_df = dc.clean_nans(mrkt_df)
        dc.indexto_datetime(mrkt_df)
        # dc.plot_candlesticks_ohlc(mrkt_df, cf.mrkt_symbol, cf.interval, cf.period)
        # print(f"{mrkt_df.tail()=}")

        # if not isinstance(mrkt_df.index, pd.DatetimeIndex):
        #     raise ValueError("mrkt_df DataFrame must have a DatetimeIndex.")
        # else:
        #     # print(f"DatetimeIndex:\n{df.index}")
        #     print(f"DatetimeIndex type: {type(mrkt_df.index)}")
        #     print(f"DatetimeIndex min: {mrkt_df.index.min()}")
        #     print(f"DatetimeIndex max: {mrkt_df.index.max()}")


        df = fetcher.history(cf.symbol, cf.interval, cf.period)
        df = dc.clean_nans(df)
        dc.indexto_datetime(df)
        # if not isinstance(df.index, pd.DatetimeIndex):
        #     raise ValueError("DataFrame must have a DatetimeIndex.")
        # else:
        #     # print(f"DatetimeIndex:\n{df.index}")
        #     print(f"DatetimeIndex type: {type(df.index)}")
        #     print(f"DatetimeIndex min: {df.index.min()}")
        #     print(f"DatetimeIndex max: {df.index.max()}")

        # dc.verify_dates(df[DATETIME], mrkt_df[DATETIME])

        # # Use inner join to assure the dataframes have the same date range
        # common_dates = df.index.intersection(mrkt_df.index)
        # print(f"common_dates: {common_dates}")
        # # Slice both DataFrames to the *exact* common date range
        # start_date = common_dates.min()  # Get the earliest date in the intersection
        # end_date = common_dates.max()    # Get the latest date in the intersection

        # df = df.loc[start_date:end_date]
        # mrkt_df = mrkt_df.loc[start_date:end_date]

        df, mrkt_df = dc.intersectby_date(df, mrkt_df)

        # df = df.loc[common_dates]
        # mrkt_df = mrkt_df.loc[common_dates]

        print(f"After common dates: {len(df)=}, {len(mrkt_df)=}")
        # dc.verify_dates(df[DATETIME], mrkt_df[DATETIME])
        dc.verify_dates(df.index, mrkt_df.index)

        # dc.plot_candlesticks_ohlc(mrkt_df, cf.mrkt_symbol, cf.interval, cf.period)

        # print(f"Starting debug:")
        # print(df.index)
        # print(mrkt_df.index)
        # print(df.index.dtype)
        # print(mrkt_df.index.dtype)
        # print(df.index[:5])
        # print(mrkt_df.index[:5])

        # df['RS'] = dc.compute_ibd_relative_strength(df, mrkt_df)
        # df['RS'] = dc.compute_ibd_rs_gemini(df[CLOSE], mrkt_df[CLOSE])
        df['RS'] = dc.calculate_ibd_rs_perplexity(df[CLOSE], mrkt_df[CLOSE])

        # print('RS ChatGPT +++++++++++++++++++++++')
        # df['RS'] = dc.compute_ibd_rs_chatgpt(df, mrkt_df)


        # stock_ticker = 'TSLA'
        # # universe_tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'TSLA', 'NVDA', 'SPY']
        # universe_tickers = ['TSLA', 'SPY']
        # df['RS'] = dc.compute_ibd_rs_chatgpt(stock_ticker, universe_tickers)


        # df[cf.rs_smacol] = df['RS'].rolling(window=rs_period).mean()

        df[SMA50] = df[CLOSE].rolling(window=50).mean()
        df[SMA150] = df[CLOSE].rolling(window=150).mean()
        df[SMA200] = df[CLOSE].rolling(window=200).mean()
        df[cf.volume_smacol] = df[VOLUME].rolling(window=cf.volume_period).mean()
        df['VColor'] = np.where(df[CLOSE] > df[OPEN], 'green', 'red')
        append_local_minmax(df)
        # print("Local minima and maxima appended:")
        # print(df)
        append_handle_columns(df)
        print(f"{len(df)=}")

        # filter by datetime range
        mini_df = dc.filterby_datetime(df, cf.dtime1, cf.dtime2)
        # mini_df.reset_index(drop=True, inplace=True)
        print(f"{len(mini_df)=}")

        dc.save_json(mini_df, mini_file)
        print(f"Wrote JSON file '{mini_file}'")

    mini_df = mini_df.round(2)
    print(mini_df.tail(10))

    # Print local min and max rows only
    mask = (mini_df['LocalMin']) | (mini_df['LocalMax'])
    minmax_df = mini_df.loc[mask]
    # selected_columns = [DATETIME, CLOSE, 'LocalMin', 'LocalMax', 'HandleDt', 'HandleY', 'DepthPct']
    selected_columns = [CLOSE, 'LocalMin', 'LocalMax', 'HandleDt', 'HandleY', 'DepthPct']
    print("minmax_df:\n")

    print(minmax_df.loc[:, selected_columns])

    # plot_cups(mini_df, symbol, interval, period)
    # plot_candlesticks_with_handles(mini_df, symbol, interval, period)
    plot_candlesticks_and_cups(mini_df, cf)


    # dc.plot_candlesticks_ohlc(mrkt_df, cf.mrkt_symbol, cf.interval, cf.period)

    # print(f"{len(mini_df)=}, # NaNs: {mini_df.isnull().sum()}")

    # print(f"{len(df)=}, # NaNs: {df.isnull().sum()}")
    # print(f"{len(mrkt_df)=}, # NaNs: {mrkt_df.isnull().sum()}")

    # dc.verify_dates(df[DATETIME], mrkt_df[DATETIME])
    print("Verifying dates after plot:")
    dc.verify_dates(df.index, mrkt_df.index)

    # filter by datetime range
    # start= '2024-06-24'
    # end = '2024-07-03'
    start= '2024-05-02'
    end = '2024-05-13'
    test_df = dc.filterby_datetime(df, start, end)
    # test_df.reset_index(drop=True, inplace=True)
    print(f"{cf.symbol} RS GAP filtered by datetime range: {start=}, {end=}")
    print(test_df)
    print(f"    {len(test_df)=}, # NaNs: {test_df.isnull().sum()}")

    test2_df = dc.filterby_datetime(mrkt_df, start, end)
    # test2_df.reset_index(drop=True, inplace=True)
    print(f"{cf.mrkt_symbol} RS GAP filtered by datetime range: {start=}, {end=}")
    print(test2_df)
    print(f"    {len(test2_df)=}, # NaNs: {test2_df.isnull().sum()}")
    # dc.verify_dates(test_df[DATETIME], test2_df[DATETIME])
    dc.verify_dates(test_df.index, test2_df.index)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(name)s | %(levelname)s | %(message)s")
    np.set_printoptions(precision=6, linewidth=150, threshold=10000)
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.width', 1000)

    main()


