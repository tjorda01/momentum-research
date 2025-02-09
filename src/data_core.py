
from pathlib import Path
from typing import List

import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
import plotly.subplots as psp

DATETIME = 'Datetime'
OPEN = 'Open'
LOW = 'Low'
HIGH = 'High'
CLOSE = 'Close'


class StockDataFetcher:
    def __init__(self) -> None:
        pass

    def ohlc_candles(self, symbol:str) -> pd.DataFrame:
        # Fetch historical data for the last 2 years
        end_date = pd.Timestamp.today()
        start_date = end_date - pd.DateOffset(years=2)

        # Use yfinance to download the data
        df = yf.download(symbol, start=start_date, end=end_date, interval='1d')
        print(df.head())
        print(f"Dtypes:\n{df.dtypes}")
        print(f"Columns:\n{df.columns}")
        print(f"Describe:\n{df.columns}")

        df.reset_index(inplace=True)
        df['Date'] = pd.to_datetime(df['Date'])
        # df.set_index('Date', inplace=True)
        # df.index = pd.to_datetime(df[df.index])
        print("After reset:")
        print(f"  Dtypes:\n{df.dtypes}")
        print(f"  Columns:\n{df.columns}")
        print(f"  Describe:\n{df.describe()}")

        # Check if the data fetch was successful
        if df.empty:
            raise Exception('Error fetching data')

        return df

    def history(self, symbol:str, interval:str='1d', period:str='1mo', refresh:bool=False) -> pd.DataFrame:
        """ Returns a DataFrame of historical data for the given symbol.

        :param symbol: Stock or ETF symbol
        :param interval: Valid intervals: [1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d,
            5d, 1wk, 1mo, 3mo]
        :param period: Valid periods:  ['1d', '5d', '1mo', '3mo', '6mo', '1y',
            '2y', '5y', '10y', 'ytd', 'max']
        :return: DataFrame
        """
        filepath = get_jsonpath(symbol, interval, period)
        if filepath.exists():
            print(f"Reading from JSON file '{filepath}'")
            df = pd.read_json(filepath)
        else:
            ticker = yf.Ticker(symbol)
            # ticker.info
            # ticker.calendar
            # ticker.analyst_price_targets
            # ticker.quarterly_income_stmt
            df = ticker.history(interval=interval, period=period)
            outpath = save_json(df, filepath)
            print(f"Wrote JSON file '{outpath}'")

        return df

    def stock_splits(self, symbol:str) -> pd.DataFrame:
        """
        Get stock split data including dates and split values.

        Parameters:
        symbol (str): The stock symbol to retrieve split data for.

        Returns:
        pd.DataFrame: A DataFrame containing the stock split dates and values.
        """
        # Fetch the stock data
        stock = yf.Ticker(symbol)

        # Retrieve the splits data
        splits = stock.splits

        # Convert to DataFrame
        splits_df = splits.reset_index()
        splits_df.columns = ['Date', 'Stock Split Value']

        return splits_df


    def get_sp500_tickers(self) -> List[str]:
        # Download the S&P 500 data
        sp500 = yf.Ticker("^GSPC")

        # Get the list of tickers
        tickers:List[str] = sp500.constituents
        print(f"Number of tickers: {len(tickers)}, {type(tickers)}")
        return tickers


def plot_candlesticks_ohlc(df:pd.DataFrame, symbol:str, interval:str, period:str) -> None:
    """
    Display a Plotly candlestick plot of the OHLC DataFrame.

    :param df: A DataFrame containing the OHLC data with columns:
        'Datetime', 'Open', 'High', 'Low', 'Close'.
    :param symbol: Ticker symbol
    :param interval: Time interval between rows
    :param period: Length of time between first and last row
    """
    min_date = df[DATETIME].min().strftime('%Y-%m-%d')
    max_date = df[DATETIME].max().strftime('%Y-%m-%d')

    suptitle = f"Candlestick Chart: {symbol}, {interval}, {period} (interval, period)"
    title = f"<b>{suptitle}</b><br>{min_date} - {max_date}"

    fig = go.Figure(data=[go.Candlestick(x=df.index,
                                         open=df[OPEN],
                                         high=df[HIGH],
                                         low=df[LOW],
                                         close=df[CLOSE],
                                         name='Candles'),
                   ])
    fig.update_layout(title=title,
                      xaxis_title=DATETIME,
                      yaxis_title='Price',
                      xaxis_rangeslider_visible=False)
    fig.show()

def plot_candlesticks(df:pd.DataFrame, symbol:str, interval:str, period:str) -> None:
    """
    Display a Plotly candlestick plot of the OHLC DataFrame.

    :param df: A DataFrame containing the OHLC data with columns:
        'Datetime', 'Open', 'High', 'Low', 'Close', 'SMA50', 'SMA150', and 'SMA200.
    :param symbol: Ticker symbol
    :param interval: Time interval between rows
    :param period: Length of time between first and last row
    """
    title = f"Candlestick Chart: {symbol}, {interval}, {period} (interval, period)"
    fig = go.Figure(data=[go.Candlestick(x=df.index,
                                         open=df[OPEN],
                                         high=df[HIGH],
                                         low=df[LOW],
                                         close=df[CLOSE],
                                         name='Candles'),
                          go.Scatter(x=df.index,
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
                    ])
    fig.update_layout(title=title,
                      xaxis_title=DATETIME,
                      yaxis_title='Price',
                      xaxis_rangeslider_visible=False)
    fig.show()

def plot_minervini(df:pd.DataFrame, symbol:str, interval:str, period:str) -> None:
    """
    Display a Plotly candlestick plot of the OHLC DataFrame.

    :param df: A DataFrame containing the OHLC data with columns 'Open', 'High',
        'Low', 'Close', and 'Date'.
    :param symbol: Ticker symbol
    :param interval: Time interval between rows
    :param period: Length of time between first and last row
    """
    title = f"Candlestick Chart: {symbol}, {interval}, {period} (interval, period)"
    fig = psp.make_subplots(rows=2, cols=1,
                            shared_xaxes=True,
                            vertical_spacing=0.07,
                            row_heights=[0.65, 0.35])
    fig.add_traces(data=[go.Candlestick(x=df.index,
                                        open=df[OPEN],
                                        high=df[HIGH],
                                        low=df[LOW],
                                        close=df[CLOSE],
                                        name='Candles'),
                         go.Scatter(x=df.index,
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
                        ],
                        rows=[1,1,1,1],
                        cols=[1,1,1,1])
    fig.add_traces([go.Scatter(x=df.index,
                               y=df[SEPA],
                               mode='lines',
                               line=dict(color='Blue'),
                               name=SEPA),
                    go.Scatter(x=df.index,
                               y=df[STAGE2_CRITERIA],
                               mode='lines',
                               line=dict(color='Green'),
                               name=STAGE2_CRITERIA)
                   ],
                   rows=[2, 2],
                   cols=[1, 1])
    fig.update_xaxes(showspikes=True, spikemode="across", spikethickness=1, matches='x', showticklabels=True)
    fig.update_yaxes(showspikes=True, spikemode="across", spikethickness=1)
    # fig.for_each_yaxis(lambda y: y.update(showticklabels=True,matches='y'))
    # fig.for_each_xaxis(lambda x: x.update(showticklabels=True,matches='x'))

    fig.update_layout(title=title,
                      hovermode= 'x',
                      xaxis_title=DATETIME,
                      yaxis_title='Price, $',
                      xaxis_rangeslider_visible=False)
    fig.show()

def get_jsonpath(symbol:str, interval:str, period:str) -> Path:
    path = Path(__file__)
    return path.parent.parent / f"data/{symbol}-{interval}-{period}.json"

def save_json(df:pd.DataFrame, outpath:Path) -> Path:
    df.to_json(outpath)
    return outpath

def indexto_datetime(df:pd.DataFrame) -> None:
    df.reset_index(inplace=True)
    df.rename(columns={'index': DATETIME, 'Date': DATETIME}, inplace=True)
    # The crucial step: Convert the DATETIME column to datetime objects
    df[DATETIME] = pd.to_datetime(df[DATETIME])
    df.set_index(DATETIME, inplace=True)
    # df.index = pd.to_datetime(df.index)  # Ensure it's a DatetimeIndex

    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame must have a DatetimeIndex.")
    else:
        # print(f"DatetimeIndex:\n{df.index}")
        print(f"DatetimeIndex type: {type(df.index)}")
        print(f"DatetimeIndex min: {df.index.min()}")
        print(f"DatetimeIndex max: {df.index.max()}")

def intersectby_date(df1: pd.DataFrame, df2: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """ Finds the common dates between two DataFrames and aligns them.

    :param df1: The first DataFrame (with a DateTimeIndex).
    :param df2: The second DataFrame (with a DateTimeIndex).
    :return:
        A tuple containing the two aligned DataFrames.  Returns empty DataFrames
        if there are no common dates or if there's an error.
    """
    # try:
    #     common_dates = df1.index.intersection(df2.index)
    #     if common_dates.empty:
    #         ValueError("No common dates found between DataFrames.")

    #     aligned_df1 = df1.loc[common_dates]
    #     aligned_df2 = df2.loc[common_dates]
    #     return aligned_df1, aligned_df2

    # except Exception as e:
    #     raise ValueError(f"An error occurred during date alignment") from e
   # Ensure that the indices are DatetimeIndex
    if not isinstance(df1.index, pd.DatetimeIndex) or not isinstance(df2.index, pd.DatetimeIndex):
        raise ValueError("Both DataFrames must have a DatetimeIndex.")

    # Find the intersection of the datetime indices
    common_datetimes = df1.index.intersection(df2.index)

    # If no common datetimes, return empty DataFrames with the original columns
    if common_datetimes.empty:
        return pd.DataFrame(columns=df1.columns), pd.DataFrame(columns=df2.columns)

    # Select only the rows with common datetimes
    df1_aligned = df1.loc[common_datetimes]
    df2_aligned = df2.loc[common_datetimes]

    return df1_aligned, df2_aligned


def verify_dates(index1: pd.DatetimeIndex, index2: pd.DatetimeIndex, chatty:bool=True) -> None:
    """Verifies that two DatetimeIndexes have the same minimum and maximum dates.

    :param index1: The first DatetimeIndex.
    :param index2: The second DatetimeIndex.
    :raises ValueError: If the minimum or maximum dates do not match.
                        The error message will include the mismatched values.
    """
    min1 = index1.min()
    min2 = index2.min()
    max1 = index1.max()
    max2 = index2.max()
    if chatty or min1 != min2 or max1 != max2:
        DTFORMAT = '%Y-%m-%d'
        error_message = ""
        delim = ""
        min1str = min1.strftime(DTFORMAT)
        min2str = min2.strftime(DTFORMAT)
        if min1 != min2:
            error_message += f"Minimum dates do not match: {min1str} != {min2str} (index1 min vs index2 min)"
            delim = f" and\n{' ' * len(ValueError.__name__)}  "
        else:
            print(f"Minimum dates match: {min1str} == {min2str} (index1 min vs index2 min)")

        max1str = max1.strftime(DTFORMAT)
        max2str = max2.strftime(DTFORMAT)
        if max1 != max2:
            # error_message += f"Maximum dates do not match: index1 max = {max1str}, index2 max = {max2str}\n"
            error_message += f"{delim}Maximum dates do not match: {max1str} != {max2str} (index1 max vs index2 max)"
        else:
            print(f"Maximum dates match: {max1str} == {max2str} (index1 max vs index2 max)")

        if error_message:
            raise ValueError(error_message)

def has_nans(df:pd.DataFrame) -> int:
    s_count = df.isna().sum()
    total = 0
    # Loop to print each index and the value at that index
    for idx, value in s_count.items():
        if value > 0:
            total += value
            print(f"Column {idx} has {value} NaNs")

    return total

def clean_nans(df:pd.DataFrame) -> pd.DataFrame:
    """ Interpolate missing values using linear interpolation.

    :param df: DataFrame to clean
    :return: cleaned DataFrame
    """
    if has_nans(df):
        return df.interpolate(method='linear', limit_direction='forward', axis=0)

    return df

def calculate_rsi(df:pd.DataFrame, window:int=14) -> pd.Series:
    # Calculate the difference in closing prices
    delta = df['Close'].diff()

    # Separate the positive and negative changes
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)

    # Calculate the average gain and loss
    avg_gain = gain.rolling(window=window, min_periods=1).mean()
    avg_loss = loss.rolling(window=window, min_periods=1).mean()

    # Calculate the RSI
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_ibd_rsi(data:pd.DataFrame, periods:List[int]=[63, 126, 189, 252], weights:List[float]=[0.4, 0.2, 0.2, 0.2]) -> pd.Series:
# def calculate_ibd_rsi(data, periods=[13,26,39,52], weights=[0.4, 0.2, 0.2, 0.2]):
    """
    Calculate the IBD RSI for a given DataFrame.

    Parameters:
    data (pd.DataFrame): DataFrame containing at least the 'Close' column.
    periods (list of int): List of periods for the Rate of Change (ROC) calculation.
    weights (list of float): List of weights for each ROC period.

    Returns:
    pd.Series: Series containing the IBD RSI values.
    """
    roc_sum = pd.Series(0, index=data.index)

    for period, weight in zip(periods, weights):
        roc = data['Close'].pct_change(periods=period)
        roc_sum += weight * roc

    ibd_rsi = 100 - (100 / (1 + roc_sum))
    # ibd_rsi = roc_sum * 100
    return ibd_rsi

def calculate_ibd_rs_perplexity_1(stock_prices: pd.Series) -> pd.Series:
    """
    Calculate the IBD relative strength of a stock based on its closing prices,
    returning a ranking between 0 and 100 with the same length as the input series.
    Parameters:
    - stock_prices (pd.Series): A time series of the closing prices of a stock.
    Returns:
    - pd.Series: A time series representing the IBD relative strength ranking (0-100),
                  with NaN for days without sufficient data.

    Raises:
    - ValueError: If there are not enough data points to calculate IBD Relative Strength.
    """

    # Ensure there are enough data points
    if len(stock_prices) < 252:
        raise ValueError("Not enough data points to calculate IBD Relative Strength.")

    # Calculate past closing prices
    C = stock_prices
    C_63 = C.shift(63)
    C_126 = C.shift(126)
    C_189 = C.shift(189)
    C_252 = C.shift(252)
    # Calculate raw IBD Relative Strength
    ibd_rs_raw = (2 * (C / C_63)) + (C / C_126) + (C / C_189) + (C / C_252)
    # Normalize to a scale of 0-100
    ibd_rs_ranked = ibd_rs_raw.rank(pct=True) * 100

    # Create a full-length Series with NaNs for initial days
    ibd_rs_full_length = pd.Series(index=stock_prices.index, dtype=float)

    # Fill in calculated values starting from day 252
    ibd_rs_full_length.iloc[252:] = ibd_rs_ranked
    return ibd_rs_full_length


def calculate_ibd_rs_perplexity(stock_prices: pd.Series, sp500_prices: pd.Series) -> pd.Series:
    """
    Calculate the IBD relative strength of a stock compared to the S&P 500,
    returning a ranking between 0 and 100, with the same length as the input series.

    Parameters:
    - stock_prices (pd.Series): A time series of the closing prices of a stock.
    - sp500_prices (pd.Series): A time series of the S&P 500 closing prices.

    Returns:
    - pd.Series: A time series representing the IBD relative strength ranking (0-100),
                  with NaN for days without sufficient data.

    Raises:
    - ValueError: If input series do not cover the same time period or if there are
                  not enough data points to calculate IBD Relative Strength.
    """

    # Check if both Series have the same index (time period)
    if not stock_prices.index.equals(sp500_prices.index):
        raise ValueError("The stock prices and S&P 500 prices must cover the exact same time period.")

    # Ensure there are enough data points
    if len(stock_prices) < 252 or len(sp500_prices) < 252:
        raise ValueError("Not enough data points to calculate IBD Relative Strength.")

    # Calculate past closing prices for the stock
    C_stock = stock_prices
    C_63_stock = C_stock.shift(63)
    C_126_stock = C_stock.shift(126)
    C_189_stock = C_stock.shift(189)
    C_252_stock = C_stock.shift(252)

    # Calculate past closing prices for S&P 500
    C_sp500 = sp500_prices
    C_63_sp500 = C_sp500.shift(63)
    C_126_sp500 = C_sp500.shift(126)
    C_189_sp500 = C_sp500.shift(189)
    C_252_sp500 = C_sp500.shift(252)

    # Calculate relative performance for each time period
    rel_perf_63 = (C_stock / C_63_stock) / (C_sp500 / C_63_sp500)
    rel_perf_126 = (C_stock / C_126_stock) / (C_sp500 / C_126_sp500)
    rel_perf_189 = (C_stock / C_189_stock) / (C_sp500 / C_189_sp500)
    rel_perf_252 = (C_stock / C_252_stock) / (C_sp500 / C_252_sp500)

    # Calculate raw IBD Relative Strength
    ibd_rs_raw = (2 * rel_perf_63) + rel_perf_126 + rel_perf_189 + rel_perf_252

    # Normalize to a scale of 0-100
    ibd_rs_ranked = ibd_rs_raw.rank(pct=True) * 100

    # Create a full-length Series with NaNs for initial days
    ibd_rs_full_length = pd.Series(index=stock_prices.index, dtype=float)

    # Fill in calculated values starting from day 252
    ibd_rs_full_length.iloc[252:] = ibd_rs_ranked[252:]

    return ibd_rs_full_length

# Example usage (replace with your data)
# stock_prices = pd.Series([...], index=pd.date_range(start='2023-01-01', periods=365))
# sp500_prices = pd.Series([...], index=pd.date_range(start='2023-01-01', periods=365))
# ibd_rs = calculate_ibd_relative_strength(stock_prices, sp500_prices)

# def compute_ibd_rs_chatgpt(stock_df:pd.DataFrame, sp500_df:pd.DataFrame) -> pd.Series:
def compute_ibd_rs_chatgpt(stock_ticker:str, universe_tickers:List[str]) -> pd.Series:
    """Computes the IBD Relative Strength (RS) metric."""

    end_date = pd.to_datetime('today')
    start_date = end_date - pd.DateOffset(weeks=104)

    try:
        data = yf.download(universe_tickers, start=start_date, end=end_date)['Adj Close']
    except Exception as e:
        print(f"Error downloading data: {e}")
        return pd.Series([])

    if data.empty:
        print("No data downloaded.")
        return pd.Series([])

    data = data.fillna(method='ffill').fillna(method='bfill')

    quarterly_data = data.resample('Q').last()
    quarterly_returns = quarterly_data.pct_change()

    weighted_rs_values = []
    weights = [2, 1, 1, 1]

    for i in range(3, len(quarterly_returns)):
        last_4_quarters_stock = quarterly_returns.iloc[i-3:i+1][stock_ticker]

        # Correctly calculate equal-weighted universe return:
        last_4_quarters_universe = quarterly_returns.iloc[i-3:i+1].mean(axis=1)

        # Apply weights to the *difference* in returns
        weighted_diff = (last_4_quarters_stock - last_4_quarters_universe) * weights

        # Calculate RS as the sum of the weighted differences
        rs_value = weighted_diff.sum()

        weighted_rs_values.append(rs_value)

    rs_series = pd.Series(weighted_rs_values, index=quarterly_returns.index[3:])
    return rs_series

    # # Define the date range (2 years)
    # end_date = pd.to_datetime('today')
    # start_date = end_date - pd.DateOffset(years=2)

    # # Download historical data for the stock universe
    # data = yf.download(universe_tickers, start=start_date, end=end_date)['Adj Close']

    # # Ensure data is aligned and handle missing values
    # data = data.fillna(method='ffill').fillna(method='bfill')

    # # Calculate the daily returns for each stock
    # daily_returns = data.pct_change()

    # # Resample the data to quarterly frequency, summing the daily returns
    # quarterly_returns = daily_returns.resample('Q').sum()

    # # Initialize a list to store the RS values
    # rs_values = []

    # # For each quarter, calculate the RS
    # for i in range(3, len(quarterly_returns)):
    #     # Select the last 4 quarters of data for the stock
    #     last_4_quarters = quarterly_returns.iloc[i-3:i+1][stock_ticker]

    #     # Calculate the RS as the sum of the last 4 quarters' returns
    #     rs_value = last_4_quarters.sum()

    #     # Append the RS value to the list
    #     rs_values.append(rs_value)

    # # Create a pandas Series for the RS values, aligning with the original dates
    # rs_series = pd.Series(rs_values, index=quarterly_returns.index[3:])

    # return rs_series


def compute_ibd_rs_gemini(stock_df:pd.DataFrame, mrkt_df:pd.DataFrame) -> pd.Series:
    """
    Computes the IBD Relative Strength (RS) metric for a stock.

    Args:
        stock_df (pd.Series): Time series of the stock's closing prices.
        mrkt_df (pd.Series): Time series of the market (e.g., S&P 500) closing prices.

    Returns:
        pd.Series: Time series of the IBD Relative Strength percentile ranks (0-100).
                     Returns an empty series if input is invalid.
    """

    if not isinstance(stock_df, pd.Series) or not isinstance(mrkt_df, pd.Series):
        print("Error: Input must be pandas Series.")
        return pd.Series([])  # Return empty series for invalid input

    if stock_df.empty or mrkt_df.empty:
        print("Error: Input series cannot be empty.")
        return pd.Series([])

    if not stock_df.index.equals(mrkt_df.index):
      print("Error: Input series must have the same date index")
      return pd.Series([])

    # Calculate performance (percentage change)
    stock_quarterly_return = stock_df.pct_change(periods=63)
    stock_yearly_return = stock_df.pct_change(periods=252)
    mrkt_quarterly_return = mrkt_df.pct_change(periods=63)
    mrkt_yearly_return = mrkt_df.pct_change(periods=252)

    # Weighted performance (double weight on recent quarter)
    stock_weighted_return = stock_yearly_return + stock_quarterly_return
    mrkt_weighted_return = mrkt_yearly_return + mrkt_quarterly_return

    # Relative Strength (handling potential divide by zero)
    rs = stock_weighted_return.div(mrkt_weighted_return, fill_value=0)

    # Percentile rank (0-100) - only rank valid RS values
    valid_rs = rs.dropna()
    rs_pct = pd.Series(index=rs.index, dtype='float64')
    # print(f"********* {type(rs_pct)}, rs_pct:\n{rs_pct}")
    rs_pct.loc[valid_rs.index] = valid_rs.rank(pct=True) * 100
    # rs_pct = rs_pct.fillna(method="ffill")
    rs_pct = rs_pct.ffill()

    return rs_pct


def compute_ibd_relative_strength(stock_df:pd.DataFrame, sp500_df:pd.DataFrame) -> pd.Series:
    # Ensure the data is sorted by date
    stock_df = stock_df.sort_values(by=DATETIME)
    sp500_df = sp500_df.sort_values(by=DATETIME)

    # Calculate the quarterly performance for the stock and the S&P500
    stock_df['Quarterly_Performance'] = stock_df['Close'].pct_change(periods=63)
    sp500_df['Quarterly_Performance'] = sp500_df['Close'].pct_change(periods=63)

    # Calculate the yearly performance for the stock and the S&P500
    stock_df['Yearly_Performance'] = stock_df['Close'].pct_change(periods=252)
    sp500_df['Yearly_Performance'] = sp500_df['Close'].pct_change(periods=252)

    # Weight the most recent quarter double
    stock_df['Weighted_Performance'] = stock_df['Yearly_Performance'] + stock_df['Quarterly_Performance']
    sp500_df['Weighted_Performance'] = sp500_df['Yearly_Performance'] + sp500_df['Quarterly_Performance']

    # Calculate the relative strength
    relative_strength = stock_df['Weighted_Performance'] / sp500_df['Weighted_Performance']

    # Convert to percentile ranking (0-100)
    relative_strength_percentile = relative_strength.rank(pct=True) * 100

    return relative_strength_percentile

# Example usage
# stock_df and sp500_df should be pandas DataFrames with 'Date' and 'Close' columns
# stock_df = pd.read_csv('path_to_stock_data.csv')
# sp500_df = pd.read_csv('path_to_sp500_data.csv')
# relative_strength = compute_ibd_relative_strength(stock_df, sp500_df)
# print(relative_strength)


def create_trend_dataseries(df:pd.DataFrame, columnname:str) -> pd.Series:
    trendseries = pd.Series(0, index=df.index)

    # Iterate over the DataFrame starting from the 22nd row
    for i in range(22, len(df)):
        # Check if all the previous 22 days of 'SMA200_diff' are greater than 0
        if (df[columnname].iloc[i-22:i] > 0).all():
            # df.at[i, 'SMA200_trend'] = 1
            trendseries.at[i] = 1

    return trendseries

def append_sma200_slope_uptrending(df:pd.DataFrame) -> pd.DataFrame:
    """
    Create a new column 'SMA200_diff_indicator' that is 1 if the previous 22 days
    of 'SMA200_diff' are all > 0; otherwise, 0.

    Parameters:
    df (pd.DataFrame): DataFrame containing the 'SMA200_diff' column.

    Returns:
    pd.DataFrame: DataFrame with the new 'SMA200_diff_indicator' column.
    """
    # Initialize the new column with zeros
    df['SMA200_trend'] = 0

    # Iterate over the DataFrame starting from the 22nd row
    for i in range(22, len(df)):
        # Check if all the previous 22 days of 'SMA200_diff' are greater than 0
        if (df['SMA200_slope'].iloc[i-22:i] > 0).all():
            df.at[i, 'SMA200_trend'] = 1

    return df

def filterby_datetime(df:pd.DataFrame, dtime1:str, dtime2:str) -> pd.DataFrame:
    # mask = (df[DATETIME] > dtime1) & (df[DATETIME] <= dtime2)
    mask = (df.index > dtime1) & (df.index <= dtime2)
    return df.loc[mask]


if __name__ == "__main__":
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.width', 1000)

    # Example usage
    # symbol = 'AAPL'
    # symbol = 'TSLA'
    symbol = 'NVDA'
    fetcher = StockDataFetcher()

    # splits_data = fetcher.stock_splits(symbol)
    # print(splits_data)

    # ohlc = fetcher.ohlc_candles(symbol)
    interval = "1d"
    period = "5y"

    df = fetcher.history(symbol, interval, period)
    df = clean_nans(df)
    # df['RSI_IBD'] = calculate_ibd_rsi(df)

    indexto_datetime(df)

    dtime1 = '2021-02-01'
    dtime2 = '2024-05-01'
    mini_df = filterby_datetime(df, dtime1, dtime2)
    mini_file = get_jsonpath(symbol, interval, period)
    mini_file = mini_file.parent / f"{mini_file.stem}_{dtime1}_{dtime2}.json"
    save_json(mini_df, mini_file)
    print(f"Wrote JSON file '{mini_file}'")

    SMA50 = 'SMA50'
    SMA150 = 'SMA150'
    SMA200 = 'SMA200'
    SMA200_SLOPE = 'SMA200_slope'
    RSI = 'RSI'

    # Calculate the 50-period Simple Moving Average (SMA) for 'Close' prices
    df[RSI] = calculate_rsi(df, window=14)
    df[SMA50] = df[CLOSE].rolling(window=50).mean()
    df[SMA150] = df[CLOSE].rolling(window=150).mean()
    df[SMA200] = df[CLOSE].rolling(window=200).mean()
    # Create column 'SMA200_slope' that's the difference between current and previous 'SMA200' value
    df[SMA200_SLOPE] = df[SMA200].diff()

    # Mark Minervini SEPA criteria
    # SEPA first four = "Stage 2 analysis"
    SMA200_TREND = 'SMA200_trend'
    C_GT_150AND200 = "C>150&200"
    _150_GT_200 = "150>200"
    _50_GT_150AND200 = "50>150&200"
    # df = create_sma200_slope_uptrending(df)
    df[SMA200_TREND] = create_trend_dataseries(df, SMA200_SLOPE)
    df[C_GT_150AND200] = ((df[CLOSE] > df[SMA150]) & (df[CLOSE] > df[SMA200])).astype(int)
    df[_150_GT_200] = (df[SMA150] > df[SMA200]).astype(int)
    df[_50_GT_150AND200] = ((df[SMA50] > df[SMA150]) & (df[SMA50] > df[SMA200])).astype(int)

    # SEPA second four
    C_GT_50 = "C>50"
    C_GT_LOWPLUS25PCT = "C>Low+25%"
    C_GT_HIGHMINUS25PCT = "C>High-25%"
    RSI_GT_70 = "RSI>70"
    STAGE2_CRITERIA = "Stage2"
    SEPA = "SEPA"

    df[C_GT_50] = (df[CLOSE] > df[SMA50]).astype(int)

    low_1y = df[LOW].rolling(window=252).min()
    high_1y = df[HIGH].rolling(window=252).max()
    price_range = high_1y - low_1y
    low_plus25pct = low_1y + (0.25 * price_range)
    high_minus25pct = high_1y - (0.25 * price_range)
    df[C_GT_LOWPLUS25PCT] = (df[CLOSE] > low_plus25pct).astype(int)
    df[C_GT_HIGHMINUS25PCT] = (df[CLOSE] > high_minus25pct).astype(int)

    df[RSI_GT_70] = (df[RSI] > 70).astype(int)
    stage2_criteria = df[SMA200_TREND] + df[C_GT_150AND200] + df[_150_GT_200] + df[_50_GT_150AND200]

    df[STAGE2_CRITERIA]= stage2_criteria
    df[SEPA] = (stage2_criteria +
                df[C_GT_50] + df[C_GT_LOWPLUS25PCT] + df[C_GT_HIGHMINUS25PCT] + df[RSI_GT_70])

    df = df.round(decimals=2)
    print(f"After rename:\n{df.tail(10)}")
    # plot_candlesticks(df, symbol, interval, period)
    plot_minervini(df, symbol, interval, period)



