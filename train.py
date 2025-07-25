import torch
from data import fetch_data, compute_technical_indicators, preprocess_data
from model import LSTMmodel
from utils import plot_technical_indicators
from sklearn.metrics import r2_score
import numpy as np
import matplotlib.pyplot as plt

symbol = input('Enter stock symbol eg. ABC.NS : ')
start_date = input('Enter start date (YYYY-MM-DD) : ')
end_date = input('Enter end date (YYYY-MM-DD) : ')
interval = input('Enter timeframe eg. "1d, 1wk, 1mo" : ')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device : {device}')

df = fetch_data(symbol, start_date, end_date, interval)
df = compute_technical_indicators(df)
plot_technical_indicators(df, symbol)

x, y, scaler, df = preprocess_data(df)

x_tensor = torch.tensor(x, dtype=torch.float32).to(device)
y_tensor = torch.tensor(y, dtype=torch.float32).view(-1, 1).to(device)

model = LSTMmodel(input_size=3).to(device)
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

num_epochs = 300
for epoch in range(num_epochs):
    model.train()
    output = model(x_tensor)
    loss = criterion(output, y_tensor)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 5 == 0:
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}')

model.eval()
predicted_train = model(x_tensor).detach().cpu().numpy()
predicted_train = scaler.inverse_transform(np.hstack((predicted_train, np.zeros((len(predicted_train), 2)))))[:, 0]

last_sequence = x_tensor[-1].unsqueeze(0)
forecast = []
num_days = 15
for _ in range(num_days):
    with torch.no_grad():
        prediction = model(last_sequence)
    forecast.append(prediction.item())
    new_row = torch.tensor([[prediction.item(), 0, 0]], dtype=torch.float32).to(device)
    last_sequence = torch.cat((last_sequence[:, 1:, :], new_row.unsqueeze(0)), dim=1)

forecasted_prices = scaler.inverse_transform(np.hstack((np.array(forecast).reshape(-1,1), np.zeros((15, 2)))))[:, 0]

plt.plot(df['Close'].values, label='Actual')
plt.plot(range(60, 60+len(predicted_train)), predicted_train, label='Predicted')
plt.plot(range(len(df), len(df)+num_days), forecasted_prices, label='Forecasted Prices', linestyle='--')
plt.legend()
plt.title('LSTM Stock Price Prediction')
plt.show()

r2 = r2_score(df['Close'].values[60:], predicted_train)
print(f'R^2 Score on training data = {r2:.4f}')
