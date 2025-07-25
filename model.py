import torch
import torch.nn as nn

class LSTMmodel(nn.Module):
    def __init__(self, input_size, hidden_size=50):
        super(LSTMmodel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.fc(out)
        return out
