# STOCK-PRICE-PREDICTION-MODEL-USING-LSTM
# 📈 Stock Price Prediction using LSTM & PyTorch (CUDA)

This project implements an LSTM-based deep learning model to predict stock prices using historical data and technical indicators. The model is trained using PyTorch with optional CUDA support for faster computation.

---

## 🔁 Workflow

1. **Fetch stock data** using `yfinance`
2. **Compute technical indicators**: MACD, RSI, EMA, SMA
3. **Preprocess data**: Normalize and sequence using `MinMaxScaler`
4. **Train an LSTM model** on the sequences
5. **Predict and forecast prices**, then visualize results

---

## 📁 File Structure

- `data.py` → Data download, indicators, and preprocessing  
- `model.py` → LSTM architecture  
- `utils.py` → Plotting technical indicators  
- `train.py` → Model training, prediction, and visualization  
- `requirements.txt` → Dependencies  
- `README.md` → Project overview

---

## 🧰 Libraries

- `pandas`, `numpy`, `matplotlib`
- `yfinance`, `scikit-learn`
- `torch` (PyTorch)

---

## ▶️ Run Instructions

```bash
pip install -r requirements.txt
python train.py
