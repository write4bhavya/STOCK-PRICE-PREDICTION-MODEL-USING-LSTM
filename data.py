import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def fetch_data(symbol, start_date, end_date, interval):
    df = yf.download(symbol, start=start_date, end=end_date, interval=interval)
    df = df[['Close']]
    df.rename(columns={'Close': 'Close'}, inplace=True)
    return df

def compute_technical_indicators(df):
    exp1 = df['Close'].ewm(span=12, adjust=False).mean()
    exp2 = df['Close'].ewm(span=26, adjust=False).mean()
    macd = exp1 - exp2
    signal = macd.ewm(span=9, adjust=False).mean()

    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = -delta.where(delta < 0, 0).rolling(14).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))

    sma = df['Close'].rolling(window=5).mean()
    ema = df['Close'].ewm(span=5, adjust=False).mean()

    df['MACD'] = macd
    df['Signal'] = signal
    df['RSI'] = rsi
    df['EMA_5'] = ema
    df['SMA_5'] = sma
    df.dropna(inplace=True)
    return df
