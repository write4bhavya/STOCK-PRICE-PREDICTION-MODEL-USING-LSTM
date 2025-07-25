import matplotlib.pyplot as plt

def plot_technical_indicators(df, symbol):
    plt.figure(figsize=(14, 12))

    plt.subplot(4, 1, 1)
    plt.plot(df['Close'], label='Close')
    plt.title(f'{symbol} - Adjusted Close Price')
    plt.legend()

    plt.subplot(4, 1, 2)
    plt.plot(df['MACD'], label='MACD')
    plt.plot(df['Signal'], label='Signal Line')
    plt.title('MACD')
    plt.legend()

    plt.subplot(4, 1, 3)
    plt.plot(df['RSI'], label='RSI')
    plt.title('RSI')
    plt.legend()

    plt.subplot(4, 1, 4)
    plt.plot(df['SMA_5'], label='SMA_5')
    plt.plot(df['EMA_5'], label='EMA_5')
    plt.title('SMA vs EMA')
    plt.legend()

    plt.tight_layout()
    plt.show()
