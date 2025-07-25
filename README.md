# STOCK-PRICE-PREDICTION-MODEL-USING-LSTM

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

## 🔍 File Descriptions

### `data.py` and 'preprocess_data.py'
Handles all data-related operations:
- **`fetch_data()`**: Downloads historical stock price data using the `yfinance` API.
- **`compute_technical_indicators()`**: Computes popular technical indicators:
  - MACD (Moving Average Convergence Divergence)
  - RSI (Relative Strength Index)
  - EMA (Exponential Moving Average)
  - SMA (Simple Moving Average)
- **`preprocess_data()`**: 
  - Normalizes selected features using `MinMaxScaler`
  - Converts data into LSTM-compatible sequences (X, y)

---

### `model.py`
Defines the LSTM neural network architecture using PyTorch:
- **`LSTMmodel`**: 
  - Consists of one LSTM layer and one fully connected (linear) layer
  - Outputs a single predicted value for the next-day closing price

---

### `technical_indicators_plot.py`
Provides utility functions for data visualization:
- **`plot_technical_indicators()`**:
  - Plots the following technical indicators on subplots:
    - Closing Price
    - MACD & Signal Line
    - RSI
    - SMA and EMA

---

### `train.py`
Main execution script that orchestrates the end-to-end pipeline:
- Accepts user input (stock symbol, date range, interval)
- Fetches and visualizes historical data
- Computes indicators and preprocesses data
- Converts sequences to PyTorch tensors
- Initializes and trains the LSTM model using:
  - MSE loss (`nn.MSELoss`)
  - Adam optimizer (`torch.optim.Adam`)
- Generates 15-day forecast using recursive inference
- Visualizes:
  - Actual closing prices
  - Predicted prices on training data
  - 15-day forecast
- Calculates and prints the **R² score** as a performance metric

---

## 📚 Libraries Used

This project uses the following Python libraries:

### 🧮 Data Handling
- **pandas**: Efficient manipulation of time series and tabular data
- **numpy**: Numerical operations and array handling

### 📉 Financial Data
- **yfinance**: Fetches historical market data directly from Yahoo Finance

### 📊 Visualization
- **matplotlib**: Used to plot technical indicators and forecast results

### 🧠 Machine Learning
- **scikit-learn**: 
  - `MinMaxScaler` for feature normalization
  - `r2_score` to evaluate model performance

### 🔥 Deep Learning
- **PyTorch**: 
  - Model building (`nn.Module`)
  - GPU acceleration (`cuda`)
  - Loss functions and optimizers

---

## 👨‍💻 About the Author

**Bhavya Upadhyay**  
B.Tech Student, IIT Bombay | Data & AI Enthusiast | Tech Builder

Bhavya is an undergraduate student at the Indian Institute of Technology Bombay, deeply passionate about artificial intelligence, machine learning, and financial forecasting. As part of his academic and technical pursuits, he has built projects that fuse data science with real-world applications — including this LSTM-based stock prediction model.

In addition to his technical expertise, Bhavya is actively involved in leadership roles:
- 🎾 **Tennis Convener**, IIT Bombay 
- 🚀 **E-Cell Marketing Coordinator** 

His approach blends analytical thinking with creativity, often working on everything from predictive modeling to 3D design and prototyping in Fusion 360. Whether it’s forecasting markets, building hardware, or managing large-scale student events, Bhavya brings structure, curiosity, and execution to the table.

**Connect with Bhavya:**  
[LinkedIn](https://www.linkedin.com/in/bhavya-upadhyay/) | [GitHub](https://github.com/) *(Update with your actual handle if you'd like)*





