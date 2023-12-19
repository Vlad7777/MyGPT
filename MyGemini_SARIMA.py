#MyGemini_SARIMA


import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

# Load the time series data
data = pd.read_csv('monthly_data.csv', index_col='Date', parse_dates=True)

# Select the desired time series variable
target_variable = data['Value']

# Fit the SARIMA model
model = SARIMAX(target_variable, order=(p, d, q), seasonal_order=(P, D, Q, m))
model_fit = model.fit()

# Forecast future values
forecast = model_fit.forecast(steps=12)  # Forecast for 12 periods (1 year)

