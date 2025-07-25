import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score
import torch
import torch.nn as nn
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device : {device}')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score
import torch
import torch.nn as nn
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device : {device}')
def fetch_data (symbol, start_date, end_date, interval):
    df = yf.download(symbol, start=start_date, end=end_date, interval=interval)
    df=df[['Close']]
    df.rename(columns={'Close': 'Close'}, inplace=True)
    return df
def compute_technical_indicators(df):
    exp1 = df['Close'].ewm(span=12, adjust=False).mean()
    exp2 = df['Close'].ewm(span=26, adjust=False).mean()
    macd = exp1-exp2
    signal=macd.ewm(span=9, adjust=False).mean()

    delta = df['Close'].diff()
    gain=(delta.where(delta>0, 0)).rolling(14).mean()
    loss=(delta.where(delta<0, 0)).rolling(14).mean()
    rs=gain/loss
    rsi= 100-(100/(1+rs))

    sma=df['Close'].rolling(window=5).mean()
    ema=df['Close'].ewm(span=5, adjust=False).mean()

    df['MACD']=macd
    df['Signal']=signal
    df['RSI']=rsi
    df['EMA_5']=ema
    df['SMA_5']=sma

    df.dropna(inplace=True)
    return df
def plot_technial_indicators(df, symbol):
    plt.figure(figsize=(14,12))

    plt.subplot(4,1,1)
    plt.plot(df['Close'], label='Adjusted Close')
    plt.title(f'{symbol} - Adjusted Close Price')
    plt.legend()

    plt.subplot(4,1,2)
    plt.plot(df['MACD'], label='MACD')
    plt.plot(df['Signal'], label = 'Signal Line')
    plt.title('MACD')
    plt.legend()

    plt.subplot(4,1,3)
    plt.plot(df['RSI'], label='RSI')
    plt.title('RSI')
    plt.legend()

    plt.subplot(4,1,4)
    plt.plot(df['SMA_5'], label='SMA_5')
    plt.plot(df['EMA_5'], label='EMA_5')
    plt.title('SMA vs EMA')
    plt.legend()

    plt.tight_layout()
    plt.show()
def preprocess_data(df):
    df.dropna(inplace=True)

    features = df[['Close', 'MACD', 'RSI']].values  # shape: (N, 3)
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(features)

    def create_sequences(data, window=60):
        x, y = [], []
        for i in range(len(data) - window):
            x.append(data[i:i+window])      # shape: (60, 3)
            y.append(data[i+window, 0])     # 0th column = 'Close'
        return np.array(x), np.array(y)

    x, y = create_sequences(scaled_data)
    print(f"[DEBUG] Preprocessed shapes -> x: {x.shape}, y: {y.shape}")  # Should be (n, 60, 3) and (n,)
    return x, y, scaler, df

symbol = input('Enter stock symbol eg. ABC.NS : ')
start_date = input('Enter start date (YYYY-MM-DD) : ')
end_date = input('Enter end date (YYYY-MM-DD) : ')
interval = input('Enter timeframe eg. "1d, 1wk, 1mo" : ')

df=fetch_data(symbol, start_date, end_date, interval)
df=compute_technical_indicators(df)
plot_technial_indicators(df, symbol)
x, y, scaler, df = preprocess_data(df)

x_tensor=torch.tensor(x, dtype=torch.float32).to(device)
y_tensor=torch.tensor(y, dtype=torch.float32).view(-1,1).to(device)
class LSTMmodel(nn.Module):
    def __init__(self, input_size, hidden_size=50):
        super(LSTMmodel, self).__init__()
        self.lstm=nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc=nn.Linear(hidden_size, 1)

    def forward(self,x):
        out,_=self.lstm(x)
        out=out[:, -1, :]
        out=self.fc(out)
        return out
    
model=LSTMmodel(input_size=3).to(device)
learning_rate=0.001
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr= learning_rate)
num_epochs=300
for epoch in range(num_epochs):
    model.train()
    output=model(x_tensor)
    loss=criterion(output, y_tensor)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch%5 == 0:
        print(f'Epoch {epoch+1}/{num_epochs}, loss = {loss.item():.4f}')
model.eval()
predicted_train=model(x_tensor).detach().cpu().numpy()
predicted_train=scaler.inverse_transform(np.hstack((predicted_train, np.zeros((len(predicted_train), 2)))))[:, 0]
last_sequence=x_tensor[-1].unsqueeze(0).to(device)
forecast = []
num_days=15
for day in range(num_days):
    with torch.no_grad():
        prediction=model(last_sequence)
    forecast.append(prediction.item())
    new_row=torch.tensor([[prediction.item(), 0, 0]], dtype=torch.float32).to(device)
    last_sequence=torch.cat((last_sequence[:, 1:, :], new_row.unsqueeze(0)), dim=1)

forecasted_prices = scaler.inverse_transform(np.hstack((np.array(forecast).reshape(-1,1), np.zeros((15, 2)))))[:, 0]
plt.plot(df['Close'].values, label='Actual')
plt.plot(range(60, 60+len(predicted_train)), predicted_train, label='Predicted')
plt.plot(range(len(df), len(df)+num_days), forecasted_prices, label='Forecasted Prices', linestyle='--')
plt.legend()
plt.title('LSTM Stock Price Prediction')
plt.show()
print(forecasted_prices)
r2=r2_score(df['Close'].values[60:], predicted_train)
print(f'R^2 Score on training data = {r2:.4f} ')
def fetch_data (symbol, start_date, end_date, interval):
    df = yf.download(symbol, start=start_date, end=end_date, interval=interval)
    df=df[['Close']]
    df.rename(columns={'Close': 'Close'}, inplace=True)
    return df
def compute_technical_indicators(df):
    exp1 = df['Close'].ewm(span=12, adjust=False).mean()
    exp2 = df['Close'].ewm(span=26, adjust=False).mean()
    macd = exp1-exp2
    signal=macd.ewm(span=9, adjust=False).mean()

    delta = df['Close'].diff()
    gain=(delta.where(delta>0, 0)).rolling(14).mean()
    loss=(delta.where(delta<0, 0)).rolling(14).mean()
    rs=gain/loss
    rsi= 100-(100/(1+rs))

    sma=df['Close'].rolling(window=5).mean()
    ema=df['Close'].ewm(span=5, adjust=False).mean()

    df['MACD']=macd
    df['Signal']=signal
    df['RSI']=rsi
    df['EMA_5']=ema
    df['SMA_5']=sma

    df.dropna(inplace=True)
    return df
def plot_technial_indicators(df, symbol):
    plt.figure(figsize=(14,12))

    plt.subplot(4,1,1)
    plt.plot(df['Close'], label='Adjusted Close')
    plt.title(f'{symbol} - Adjusted Close Price')
    plt.legend()

    plt.subplot(4,1,2)
    plt.plot(df['MACD'], label='MACD')
    plt.plot(df['Signal'], label = 'Signal Line')
    plt.title('MACD')
    plt.legend()

    plt.subplot(4,1,3)
    plt.plot(df['RSI'], label='RSI')
    plt.title('RSI')
    plt.legend()

    plt.subplot(4,1,4)
    plt.plot(df['SMA_5'], label='SMA_5')
    plt.plot(df['EMA_5'], label='EMA_5')
    plt.title('SMA vs EMA')
    plt.legend()

    plt.tight_layout()
    plt.show()
def preprocess_data(df):
    df.dropna(inplace=True)

    features = df[['Close', 'MACD', 'RSI']].values  # shape: (N, 3)
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(features)

    def create_sequences(data, window=60):
        x, y = [], []
        for i in range(len(data) - window):
            x.append(data[i:i+window])      # shape: (60, 3)
            y.append(data[i+window, 0])     # 0th column = 'Close'
        return np.array(x), np.array(y)

    x, y = create_sequences(scaled_data)
    print(f"[DEBUG] Preprocessed shapes -> x: {x.shape}, y: {y.shape}")  # Should be (n, 60, 3) and (n,)
    return x, y, scaler, df

symbol = input('Enter stock symbol eg. ABC.NS : ')
start_date = input('Enter start date (YYYY-MM-DD) : ')
end_date = input('Enter end date (YYYY-MM-DD) : ')
interval = input('Enter timeframe eg. "1d, 1wk, 1mo" : ')

df=fetch_data(symbol, start_date, end_date, interval)
df=compute_technical_indicators(df)
plot_technial_indicators(df, symbol)
x, y, scaler, df = preprocess_data(df)

x_tensor=torch.tensor(x, dtype=torch.float32).to(device)
y_tensor=torch.tensor(y, dtype=torch.float32).view(-1,1).to(device)
class LSTMmodel(nn.Module):
    def __init__(self, input_size, hidden_size=50):
        super(LSTMmodel, self).__init__()
        self.lstm=nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc=nn.Linear(hidden_size, 1)

    def forward(self,x):
        out,_=self.lstm(x)
        out=out[:, -1, :]
        out=self.fc(out)
        return out
    
model=LSTMmodel(input_size=3).to(device)
learning_rate=0.001
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr= learning_rate)
num_epochs=300
for epoch in range(num_epochs):
    model.train()
    output=model(x_tensor)
    loss=criterion(output, y_tensor)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch%5 == 0:
        print(f'Epoch {epoch+1}/{num_epochs}, loss = {loss.item():.4f}')
model.eval()
predicted_train=model(x_tensor).detach().cpu().numpy()
predicted_train=scaler.inverse_transform(np.hstack((predicted_train, np.zeros((len(predicted_train), 2)))))[:, 0]
last_sequence=x_tensor[-1].unsqueeze(0).to(device)
forecast = []
num_days=15
for day in range(num_days):
    with torch.no_grad():
        prediction=model(last_sequence)
    forecast.append(prediction.item())
    new_row=torch.tensor([[prediction.item(), 0, 0]], dtype=torch.float32).to(device)
    last_sequence=torch.cat((last_sequence[:, 1:, :], new_row.unsqueeze(0)), dim=1)

forecasted_prices = scaler.inverse_transform(np.hstack((np.array(forecast).reshape(-1,1), np.zeros((15, 2)))))[:, 0]
plt.plot(df['Close'].values, label='Actual')
plt.plot(range(60, 60+len(predicted_train)), predicted_train, label='Predicted')
plt.plot(range(len(df), len(df)+num_days), forecasted_prices, label='Forecasted Prices', linestyle='--')
plt.legend()
plt.title('LSTM Stock Price Prediction')
plt.show()
print(forecasted_prices)
r2=r2_score(df['Close'].values[60:], predicted_train)
print(f'R^2 Score on training data = {r2:.4f} ')
