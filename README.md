# STOCK-PRICE-PREDICTION-MODEL-USING-LSTM
# 📈 Stock Price Prediction using LSTM and PyTorch (with CUDA Support)

This project focuses on predicting future stock prices using historical data and technical indicators through an LSTM (Long Short-Term Memory) neural network, implemented using PyTorch. The model is trained on normalized, time-sequenced data and optionally leverages GPU acceleration using CUDA for efficient computation.

---

## 🚀 Key Features

- Historical stock data retrieval using Yahoo Finance API
- Technical indicators: MACD, RSI, EMA, SMA
- LSTM-based deep learning model for time series forecasting
- CUDA/GPU acceleration (if available)
- Clean modular code structure for reusability
- Visualization of both training performance and forecasts

---

🔍 File Descriptions
data.py
Handles data-related operations:

fetch_data(): Downloads historical stock data using yfinance

compute_technical_indicators(): Calculates MACD, RSI, EMA, SMA

preprocess_data(): Normalizes features and generates training sequences (X, y) for LSTM

model.py
Defines the LSTM model:

LSTMmodel: A PyTorch neural network class with a single LSTM layer followed by a fully connected layer

utils.py
Supports visualization:

plot_technical_indicators(): Plots closing price, MACD with Signal Line, RSI, EMA, and SMA on a multi-panel graph

train.py
Main script to:

Accept user input for stock symbol, dates, and interval

Fetch and visualize data

Preprocess data and create training tensors

Train the LSTM model using MSE loss and Adam optimizer

Generate 15-day forecast recursively

Plot actual vs predicted vs forecasted prices

Calculate and display R² score on training data
