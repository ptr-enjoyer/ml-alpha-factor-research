import yfinance as yf
import pandas as pd
import numpy as np

def fetch_data(ticker, start_date, end_date):
    """Fetch historical stock data from Yahoo Finance."""
    data = yf.download(ticker, start=start_date, end=end_date)
    return data

def preprocess_data(data):
    """Preprocess the data by filling missing values and calculating returns."""
    data = data.fillna(method='ffill')
    data['Returns'] = data['Adj Close'].pct_change()
    return data



