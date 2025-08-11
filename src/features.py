import numpy
import pandas as pd

def momentum(series, window):
    """    Calculate momentum over a given window.
    
    Parameters:
        series (pd.Series): Price series.
        window (int): Momentum window size (days).   
    Returns:
        pd.Series: Momentum values.
    """
    mom = series.pct_change(periods=window)
    return mom

def roll_volatility(series, window):
    """    Calculate rolling volatility (standard deviation of returns) over a given window.
    
    Parameters:
        series (pd.Series): Price or return series.
        window (int): Rolling window size (days).
    Returns:
        pd.Series: Rolling volatility.
    """
    returns = series.pct_change()
    return returns.rolling(window=window).std()

def MA_ratio(series, window):
    """    Calculate Moving Average Ratio over a given window.
    Parameters:
        series (pd.Series): Price series.
        window (int): Rolling window size (days).
    Returns:
        pd.Series: Moving Average Ratio values.
    """
    ratio = series / series.rolling(window=window).mean()
    return ratio

def z_score(series, window):
    """    Calculate z-score of a series over a given window.
    Parameters:
        series (pd.Series): Price or return series.
        window (int): Rolling window size (days).
    Returns:
        pd.Series: Z-score values.
    """
    mean = series.rolling(window=window).mean()
    std = series.rolling(window=window).std()
    z_scores = (series - mean) / std
    return z_scores

def RSI(series, window):
    """    Calculate Relative Strength Index (RSI) over a given window.
    Parameters:
        series (pd.Series): Price series.
        window (int): RSI window size (days).
    Returns:
        pd.Series: RSI values.
    """
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def vol_spike(series, window):
    """    Calculate volume spikes as a ratio of current volume to rolling mean volume.
    Parameters:
        series (pd.Series): Volume series.
        window (int): Rolling window size (days).
    Returns:
        pd.Series: Volume spike values.
    """
    mean_vol = series.rolling(window=window).mean()
    spikes = series / mean_vol
    return spikes