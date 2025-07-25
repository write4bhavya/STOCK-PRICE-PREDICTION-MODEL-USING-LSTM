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

## 🧠 Project Structure

```plaintext
📁 stock-price-prediction-lstm/
├── data.py            # Data fetching, technical indicators, preprocessing
├── model.py           # LSTM model architecture (PyTorch)
├── train.py           # Main script for training and prediction
├── utils.py           # Visualization of technical indicators
├── requirements.txt   # Python dependencies
└── README.md          # Project documentation


## ▶️ Run Instructions

```bash
pip install -r requirements.txt
python train.py
