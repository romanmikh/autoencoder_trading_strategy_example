import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Model
from keras.layers import Input, Dense
from keras.optimizers import Adam


# Step 1: Data Collection
# Fetch historical stock data using yfinance
ticker_symbol = 'AAPL'  # Example: Apple Inc.
data = yf.download(ticker_symbol, start='2020-01-01', end='2021-01-01')

# Step 2: Data Preprocessing
# For simplicity, using just 'Open', 'High', 'Low', 'Close', 'Volume'
df = data[['Open', 'High', 'Low', 'Close', 'Volume']]
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df)

# Step 3: Autoencoder Model Creation

# Model parameters
input_dim = scaled_data.shape[1]  # number of features
encoding_dim = 3  # dimension of encoded data

# Encoder
input_layer = Input(shape=(input_dim,))
encoder_layer = Dense(encoding_dim, activation='relu')(input_layer)

# Decoder
decoder_layer = Dense(input_dim, activation='sigmoid')(encoder_layer)

# Autoencoder
autoencoder = Model(inputs=input_layer, outputs=decoder_layer)
autoencoder.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')

# Display the autoencoder structure
autoencoder.summary()

# Step 4: Model Training
# For demonstration, using a small number of epochs
autoencoder.fit(scaled_data, scaled_data, epochs=10, batch_size=32, shuffle=True)





# Step 1: Visualize the Original and Reconstructed Data
reconstructed_data = autoencoder.predict(scaled_data)

# Plotting the original and reconstructed data for comparison
plt.figure(figsize=(15, 6))
plt.plot(scaled_data[:, 3], label='Original Close Prices')  # Assuming 3rd index is 'Close'
plt.plot(reconstructed_data[:, 3], label='Reconstructed Close Prices')
plt.title('Original vs Reconstructed Data')
plt.legend()
plt.show()

# Step 2: Feature Extraction
encoded_features = autoencoder.predict(scaled_data)

# Step 3: Visualize the Encoded Features
# This is a simple plot, more sophisticated methods might be required for deeper analysis
plt.figure(figsize=(15, 6))
plt.plot(encoded_features)
plt.title('Encoded Features Over Time')
plt.show()

# Step 4: Formulate a Basic Trading Strategy
# Example: A very simple strategy based on the moving average of encoded features
moving_average = pd.Series(encoded_features[:, 0]).rolling(window=5).mean()  # Simple moving average

# Generating buy/sell signals
buy_signals = moving_average > encoded_features[:, 0]
sell_signals = moving_average < encoded_features[:, 0]

# Step 5: Visualization
plt.figure(figsize=(15, 6))
plt.plot(df['Close'], label='Close Prices') # Plotting the actual close prices
plt.scatter(df.index[buy_signals], df['Close'][buy_signals], marker='^', color='g', label='Buy Signal', alpha=1)
plt.scatter(df.index[sell_signals], df['Close'][sell_signals], marker='v', color='r', label='Sell Signal', alpha=1)
plt.title('Trading Signals on Close Prices')
plt.legend()
plt.show()