import math
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pandas_datareader as pdr
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.models import Sequential

TRAIN_SPLIT = 0.8
TIME_STEPS = 100
FUTURE_PREDICTIONS = 30

key = os.getenv("TIINGO_KEY")
# Collect stock data --> AAPL
try:
    df = pdr.get_data_tiingo("AAPL", api_key=key)
except Exception as e:
    raise SystemExit(f"Data fetch failed: {e}")

# Preprocessing
df = df.reset_index()
priceDf = df["close"]
scaler = MinMaxScaler(feature_range=(0,1))
priceDf = scaler.fit_transform(np.array(priceDf).reshape(-1,1))

splitSize = int(len(priceDf) * TRAIN_SPLIT)
trainData = priceDf[:splitSize]
testData = priceDf[splitSize:]

def getDataset(data, timesteps=1):
    x, y = [], []
    for i in range(len(data)-timesteps-1):
        x.append(data[i:i+timesteps])
        y.append(data[i+timesteps])
    return np.array(x), np.array(y)

trainX, trainY = getDataset(trainData, TIME_STEPS)
testX, testY = getDataset(testData, TIME_STEPS)
print(trainX.shape, trainY.shape)

trainX = trainX.reshape(trainX.shape[0], trainX.shape[1], 1)
testX = testX.reshape(testX.shape[0], testX.shape[1], 1)

# Stacked LSTM model
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(TIME_STEPS, 1)))
model.add(LSTM(50, return_sequences=True))
model.add(LSTM(50))
model.add(Dense(1))
model.compile(loss="mean_squared_error", optimizer="adam")
model.summary()
model.fit(trainX, trainY, validation_data=(testX, testY), epochs=20, batch_size=64, verbose=1)

# Predict the test data
pred = model.predict(testX)
pred = scaler.inverse_transform(pred)

mse = math.sqrt(mean_squared_error(scaler.inverse_transform(testY.reshape(-1,1)), pred))
print("MSE: ", mse)

# Plotting
testPredictionsPlot = np.empty_like(priceDf)
testPredictionsPlot[:,:] = np.nan
testPredictionsPlot[splitSize+TIME_STEPS+1:] = pred
plt.plot(scaler.inverse_transform(priceDf))
plt.plot(testPredictionsPlot)
plt.show()

# Predict the future data
numFuturePredictions = 30

# Start with the last sequence from test data
dataX = testX[-1]
tempDataX = dataX.flatten().tolist()

newPreds = []
idx = 0

while idx < numFuturePredictions:
    if len(tempDataX) > TIME_STEPS:
        tempDataX = tempDataX[1:]

    # Prepare input for prediction
    inputX = np.array(tempDataX[-TIME_STEPS:]).reshape((1, TIME_STEPS, 1))

    dataY = model.predict(inputX, verbose=0)
    tempDataX.append(dataY[0][0])
    newPreds.append(dataY[0][0])

    idx += 1

# Plot results
newDays = np.arange(1, TIME_STEPS + 1)
newPredDays = np.arange(TIME_STEPS + 1, TIME_STEPS + 1 + numFuturePredictions)

plt.plot(newDays, scaler.inverse_transform(priceDf[-TIME_STEPS:]))
plt.plot(newPredDays, scaler.inverse_transform(np.array(newPreds).reshape(-1,1)))
plt.show()
