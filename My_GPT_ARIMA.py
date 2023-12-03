#My_GPT_ARIMA

"""
ARIMA (AutoRegressive Integrated Moving Average) is a popular time series forecasting algorithm that combines autoregression, differencing, and moving averages. You can use the statsmodels library in Python to implement ARIMA models. Here's a basic example:


python
Copy code

"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller

# Generate a sample time series data
np.random.seed(42)
time_series = pd.Series(np.cumsum(np.random.randn(100)), name='Time Series')

# Plot the time series
time_series.plot(title='Generated Time Series')
plt.show()

# Check for stationarity using the Augmented Dickey-Fuller test
result = adfuller(time_series)
print(f'ADF Statistic: {result[0]:.4f}')
print(f'p-value: {result[1]:.4f}')
print('Is the time series stationary? ', result[1] < 0.05)

# Plot autocorrelation and partial autocorrelation functions
plot_acf(time_series, lags=20)
plot_pacf(time_series, lags=20)
plt.show()

# Differencing to achieve stationarity
differenced_series = time_series.diff().dropna()

# Plot the differenced time series
differenced_series.plot(title='Differenced Time Series')
plt.show()

# Check for stationarity again
result_diff = adfuller(differenced_series)
print(f'ADF Statistic after differencing: {result_diff[0]:.4f}')
print(f'p-value after differencing: {result_diff[1]:.4f}')
print('Is the differenced time series stationary? ', result_diff[1] < 0.05)

# Split the data into training and testing sets
train_size = int(len(time_series) * 0.8)
train, test = time_series[:train_size], time_series[train_size:]

# Fit ARIMA model
order = (1, 1, 1)  # Example order, tune based on ACF and PACF plots
model = ARIMA(train, order=order)
result_arima = model.fit()

# Make predictions
predictions = result_arima.predict(start=len(train), end=len(train) + len(test) - 1, typ='levels')

# Plot actual vs predicted values
plt.plot(train, label='Train')
plt.plot(test, label='Test')
plt.plot(predictions, label='Predictions')
plt.legend()
plt.show()

"""
In this example:

We generate a sample time series and check for stationarity using the Augmented Dickey-Fuller test.
Autocorrelation and partial autocorrelation plots are used to determine the order of differencing (d in ARIMA).
The time series is differenced to achieve stationarity.
The differenced time series is split into training and testing sets.
An ARIMA model is fitted to the training set, and predictions are made on the test set.
Make sure to have statsmodels installed:

bash
Copy code
pip install statsmodels
You may need to adjust the hyperparameters (order) of the ARIMA model based on the characteristics of your specific time series data.

"""


