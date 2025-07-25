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

## ✅ Project Roadmap

### 📘 Week 1: Python Basics
Learned fundamentals of Python:
- Variables, data types, loops, functions
- Conditional statements and I/O operations
- Hands-on practice with simple programs

---

### 🛠️ Week 2: Python Libraries for Data Science
Explored essential libraries:
- **NumPy** – for numerical computing  
- **Pandas** – for data manipulation and analysis  
- **Matplotlib** & **Seaborn** – for data visualization  
- **Scikit-learn** – for machine learning tools  

Assignments:
- Integrated concepts from Week 1 and Week 2 in practical exercises.

---

### 🧠 Week 3: Introduction to Machine Learning + PyTorch Basics
What We Covered:
- What is Machine Learning?
- Core ML Math: Linear algebra, probability, and statistics
- Supervised Learning:
  - Linear Regression (Simple and MLR)
  - Logistic Regression (Binary Classification)
- Overfitting and Regularization (L2, cross-validation)
- Unsupervised Learning:  
  - K-Means Clustering  
  - K-Nearest Neighbors (KNN)

**Plus: Introduction to PyTorch**
- Tensors and operations
- Neural network layers and loss functions
- Autograd and backpropagation in PyTorch

---

### 🧬 Week 4: Introduction to Deep Learning – CNN & RNN
Learned the fundamentals of deep learning:
- **Neural Networks** – forward and backward propagation
- **Convolutional Neural Networks (CNNs)**:
  - Filters, pooling, and convolution operations
  - Applications in image tasks
- **Recurrent Neural Networks (RNNs)**:
  - Sequential modeling
  - Limitations of vanilla RNNs (vanishing gradients)
  - Introduction to LSTM and GRU

---

### 🔁 Week 5: Understanding LSTM Networks
Dived deep into LSTM architecture for time series prediction:
- Cell structure: input, forget, and output gates
- Sequence modeling with memory
- Input reshaping and tensor dimensions for LSTM
- Built LSTM networks in PyTorch:
  - Defined models with `nn.LSTM` and `nn.Linear`
  - Trained on sequential data (e.g., sine waves)

---

### 📊 Week 6: LSTM for Stock Price Prediction
Final implementation week – applying everything built so far:
- Used `yfinance` to fetch historical stock data
- Computed technical indicators:
  - MACD, RSI, EMA, SMA
- Preprocessed data and normalized it
- Generated LSTM training sequences
- Trained an LSTM model using:
  - MSE Loss  
  - Adam Optimizer  
  - CUDA acceleration for GPU training
- Forecasted future stock prices for 15 days
- Visualized:
  - Technical indicators
  - Actual vs Predicted vs Forecasted prices
- Evaluated performance using **R² score**

---

## 💻 Financial Data with yfinance
- What is the `yfinance` API
- How to:
  - Import the library
  - Download stock data using Python
  - Format and prepare data for modeling

---

## 🔧 Tools & Technologies
- **Python 3.x**
- **NumPy**, **Pandas**, **Matplotlib**, **Seaborn**
- **Scikit-learn**
- **PyTorch**
- **yfinance**
- **Jupyter Notebooks / VS Code**
- **CUDA / GPU acceleration** (for model training)

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





