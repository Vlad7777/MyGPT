#MyGPT_SARIMA

"""
SARIMA (Seasonal AutoRegressive Integrated Moving Average) is a time series forecasting model that extends the ARIMA model by incorporating seasonality. SARIMA is particularly useful for datasets that exhibit both trend and seasonality. In Python, you can use the statsmodels library to implement SARIMA models. Below is a step-by-step guide on how to use SARIMA in Python:

python
Copy code

"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# Load your time series data
# For this example, let's assume you have a DataFrame 'df' with a datetime index and a column 'value'
# df = pd.read_csv('your_data.csv', parse_dates=True, index_col='date_column')

# Check for stationarity
def check_stationarity(timeseries):
    # Perform Dickey-Fuller test
    result = adfuller(timeseries)
    print('ADF Statistic:', result[0])
    print('p-value:', result[1])
    print('Critical Values:', result[4])
    
    if result[1] <= 0.05:
        print("The time series is stationary.")
    else:
        print("The time series is not stationary. Perform differencing.")

# Differencing to make the time series stationary if needed
# You may need to perform differencing multiple times until the series becomes stationary
df['stationary_value'] = df['value'].diff().dropna()

# Check stationarity after differencing
check_stationarity(df['stationary_value'])

# Plot ACF and PACF to determine the order of AR and MA terms
plot_acf(df['stationary_value'], lags=30)
plt.show()

plot_pacf(df['stationary_value'], lags=30)
plt.show()

# Determine the order of the seasonal components (SAR and SMA terms)
# For example, if you observe a seasonal pattern every 12 time points, set m=12
m = 12

# Fit the SARIMA model
order = (p, d, q)  # Specify the order of the ARIMA components
seasonal_order = (P, D, Q, m)  # Specify the order of the seasonal components
model = SARIMAX(df['value'], order=order, seasonal_order=seasonal_order)

# Fit the model to the data
results = model.fit()

# Make predictions
forecast_steps = 12  # Specify the number of steps to forecast
forecast = results.get_forecast(steps=forecast_steps)

# Plot the original data and the forecast
plt.figure(figsize=(12, 6))
plt.plot(df['value'], label='Original Data')
plt.plot(forecast.predicted_mean, label='Forecast', color='red')
plt.fill_between(forecast.index, forecast.conf_int()['lower value'], forecast.conf_int()['upper value'], color='red', alpha=0.2)
plt.title('SARIMA Forecast')
plt.legend()
plt.show()

"""
Note:

Replace 'your_data.csv' with the actual path to your dataset.
Adjust the parameters like p, d, q, P, D, Q, and m based on the characteristics of your data.
The check_stationarity function checks if the time series is stationary using the Augmented Dickey-Fuller (ADF) test. If the series is not stationary, differencing is performed.
ACF (AutoCorrelation Function) and PACF (Partial AutoCorrelation Function) plots help determine the order of the AR and MA terms.
The get_forecast method is used to make future predictions, and the results are plotted along with confidence intervals. Adjust the forecast_steps parameter based on the number of steps you want to forecast.
Remember to tailor the code to your specific dataset and requirements.

"""