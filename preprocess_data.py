def preprocess_data(df):
    df.dropna(inplace=True)
    features = df[['Close', 'MACD', 'RSI']].values
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(features)

    def create_sequences(data, window=60):
        x, y = [], []
        for i in range(len(data) - window):
            x.append(data[i:i+window])
            y.append(data[i+window, 0])
        return np.array(x), np.array(y)

    x, y = create_sequences(scaled_data)
    return x, y, scaler, df
