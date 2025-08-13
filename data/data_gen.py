import yfinance as yf
import pandas as pd
import numpy as np
from data_fetch import fetch_data, preprocess_data

end_date = pd.Timestamp.today().normalize() # Get today's date
start_date = end_date - pd.DateOffset(years=15) # Define the date range for the data
date_range = pd.date_range(start=start_date, end=end_date, freq='D') # Create a date range for the last 15 years


# Fetch historical stock data for AAPL
ticker_appl = 'AAPL'
data_appl = fetch_data(ticker_appl, start_date, end_date)
data_appl.to_csv(r'C:\Users\leona\OneDrive\Documents\Alpha Factor Research\ml-alpha-factor-research\data\data_appl.csv', index=True)  # Save the data to a CSV file

# Fetch historical stock data for GOOGL
ticker_googl = 'GOOGL'
data_googl = fetch_data(ticker_googl, start_date, end_date)
data_googl.to_csv(r'C:\Users\leona\OneDrive\Documents\Alpha Factor Research\ml-alpha-factor-research\data\data_googl.csv', index=True) # Save the data to a CSV file

